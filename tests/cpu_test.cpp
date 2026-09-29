#include "cpu/espresso/decoder.hpp"
#include "cpu/espresso/cafe_os_hle.hpp"
#include "cpu/espresso/elf_loader.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/guest_mutex.hpp"
#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/rpx_loader.hpp"
#include "cpu_state_test.hpp"
#include "emulator.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

namespace
{

using namespace affogato::cpu::espresso;

void set_be16(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint16_t value)
{
    bytes[offset] = static_cast<std::uint8_t>(value >> 8U);
    bytes[offset + 1] = static_cast<std::uint8_t>(value);
}

void set_be32(std::vector<std::uint8_t>& bytes, std::size_t offset, std::uint32_t value)
{
    bytes[offset] = static_cast<std::uint8_t>(value >> 24U);
    bytes[offset + 1] = static_cast<std::uint8_t>(value >> 16U);
    bytes[offset + 2] = static_cast<std::uint8_t>(value >> 8U);
    bytes[offset + 3] = static_cast<std::uint8_t>(value);
}

[[nodiscard]] std::vector<std::uint8_t> make_minimal_powerpc_elf()
{
    constexpr std::size_t elf_header_size = 52;
    constexpr std::size_t program_header_offset = elf_header_size;
    constexpr std::size_t program_header_size = 32;
    constexpr std::size_t code_offset = elf_header_size + program_header_size;
    constexpr std::uint32_t code_address = 0x1000U;

    std::vector<std::uint8_t> file(code_offset + 8, 0);
    file[0] = 0x7F;
    file[1] = 'E';
    file[2] = 'L';
    file[3] = 'F';
    file[4] = 1; // ELFCLASS32
    file[5] = 2; // ELFDATA2MSB
    file[6] = 1; // EV_CURRENT
    set_be16(file, 16, 2); // ET_EXEC
    set_be16(file, 18, 20); // EM_PPC
    set_be32(file, 20, 1); // e_version
    set_be32(file, 24, code_address); // e_entry
    set_be32(file, 28, static_cast<std::uint32_t>(program_header_offset));
    set_be16(file, 40, static_cast<std::uint16_t>(elf_header_size));
    set_be16(file, 42, static_cast<std::uint16_t>(program_header_size));
    set_be16(file, 44, 1); // e_phnum

    set_be32(file, program_header_offset, 1); // PT_LOAD
    set_be32(file, program_header_offset + 4, static_cast<std::uint32_t>(code_offset));
    set_be32(file, program_header_offset + 8, code_address); // p_vaddr
    set_be32(file, program_header_offset + 12, code_address); // p_paddr
    set_be32(file, program_header_offset + 16, 8); // p_filesz
    set_be32(file, program_header_offset + 20, 16); // p_memsz, including zero-fill tail
    set_be32(file, program_header_offset + 24, 5); // PF_R | PF_X
    set_be32(file, program_header_offset + 28, 4); // p_align

    // PPC code emitted by devkitPPC for tests/powerpc_leaf_function.c.
    set_be32(file, code_offset, 0x38630005U); // addi r3, r3, 5
    set_be32(file, code_offset + 4, 0x4E800020U); // blr
    return file;
}

struct RpxRelocationSymbol
{
    std::string name{"$UNDEF"};
    std::uint32_t value{};
    std::uint8_t binding{1}; // STB_GLOBAL
    std::uint8_t type{1}; // STT_OBJECT
    std::uint16_t section{}; // SHN_UNDEF
    std::int32_t addend{0x12345678};
};

struct RpxRelocationOptions
{
    std::uint32_t type{1};
    std::uint32_t offset{12};
    std::uint32_t branch_instruction{0x48000001U}; // bl
    std::uint32_t tramp_adjust{};
    std::uint32_t text_size{};
    std::uint32_t tramp_addition{};
    std::uint32_t sda_base{};
    std::uint32_t sda2_base{};
};

[[nodiscard]] std::vector<std::uint8_t> make_minimal_compressed_rpx(
    std::uint32_t code_address = 0x02000000U,
    std::size_t code_size = 16,
    const RpxRelocationSymbol& relocation_symbol = {},
    const RpxRelocationOptions& relocation_options = {})
{
    constexpr std::size_t header_size = 52;
    constexpr std::size_t section_header_size = 40;
    constexpr std::size_t section_count = 8;
    constexpr std::size_t section_table_offset = header_size;
    constexpr std::size_t text_offset = 384;
    assert(code_size >= relocation_options.offset + 4);
    std::vector<std::uint8_t> code(code_size, 0);
    code[0] = 0x38;
    code[1] = 0x63;
    code[2] = 0x00;
    code[3] = 0x05; // addi r3, r3, 5
    code[4] = 0x4E;
    code[5] = 0x80;
    code[6] = 0x00;
    code[7] = 0x20; // blr
    if (relocation_options.type == 10)
    {
        set_be32(code, relocation_options.offset, relocation_options.branch_instruction);
    }

    uLongf compressed_size = compressBound(code.size());
    std::vector<std::uint8_t> compressed(compressed_size);
    const int status = compress2(
        compressed.data(), &compressed_size, code.data(), code.size(), Z_BEST_COMPRESSION);
    assert(status == Z_OK);
    compressed.resize(compressed_size);

    const std::array<std::uint8_t, 68> names{
        '\0', '.', 't', 'e', 'x', 't', '\0', '.', 's', 'h', 's', 't', 'r', 't', 'a', 'b', '\0',
        '.', 'r', 'p', 'l', '_', 'c', 'r', 'c', 's', '\0', '.', 'r', 'p', 'l', '_', 'f', 'i', 'l', 'e', 'i', 'n', 'f', 'o', '\0',
        '.', 's', 'y', 'm', 't', 'a', 'b', '\0', '.', 's', 't', 'r', 't', 'a', 'b', '\0',
        '.', 'r', 'e', 'l', 'a', '.', 't', 'e', 'x', 't', '\0'};
    constexpr std::size_t inflated_size_prefix = 4;
    const std::size_t text_size = inflated_size_prefix + compressed.size();
    const std::size_t names_offset = text_offset + text_size;
    const std::size_t crcs_offset = names_offset + names.size();
    const std::size_t fileinfo_offset = crcs_offset + section_count * sizeof(std::uint32_t);
    const std::size_t symtab_offset = (fileinfo_offset + 0x60 + 3U) & ~std::size_t{3U};
    const std::size_t strtab_offset = symtab_offset + 32;
    std::vector<std::uint8_t> symbol_strings{0};
    symbol_strings.insert(symbol_strings.end(), relocation_symbol.name.begin(),
                          relocation_symbol.name.end());
    symbol_strings.push_back(0);
    const std::size_t rela_offset =
        (strtab_offset + symbol_strings.size() + 3U) & ~std::size_t{3U};
    const std::size_t relocation_table_size = 3 * 12;
    std::vector<std::uint8_t> file(rela_offset + relocation_table_size, 0);

    file[0] = 0x7F;
    file[1] = 'E';
    file[2] = 'L';
    file[3] = 'F';
    file[4] = 1;
    file[5] = 2;
    file[6] = 1;
    set_be16(file, 7, 0xCAFE); // EI_ABIVERSION / Cafe ABI marker
    set_be16(file, 16, 0xFE01); // ET_CAFE_RPL
    set_be16(file, 18, 20); // EM_PPC
    set_be32(file, 20, 1);
    set_be32(file, 24, code_address);
    set_be32(file, 32, static_cast<std::uint32_t>(section_table_offset));
    set_be16(file, 40, static_cast<std::uint16_t>(header_size));
    set_be16(file, 46, static_cast<std::uint16_t>(section_header_size));
    set_be16(file, 48, static_cast<std::uint16_t>(section_count));
    set_be16(file, 50, 2); // section-name string table

    const auto set_section = [&](std::size_t index, std::uint32_t name, std::uint32_t type,
                                 std::uint32_t flags, std::uint32_t address, std::uint32_t offset,
                                 std::uint32_t size, std::uint32_t alignment, std::uint32_t entry_size,
                                 std::uint32_t link = 0, std::uint32_t info = 0) {
        const std::size_t at = section_table_offset + index * section_header_size;
        set_be32(file, at, name);
        set_be32(file, at + 4, type);
        set_be32(file, at + 8, flags);
        set_be32(file, at + 12, address);
        set_be32(file, at + 16, offset);
        set_be32(file, at + 20, size);
        set_be32(file, at + 24, link);
        set_be32(file, at + 28, info);
        set_be32(file, at + 32, alignment);
        set_be32(file, at + 36, entry_size);
    };
    set_section(1, 1, 1, 0x08000006U, code_address, static_cast<std::uint32_t>(text_offset),
                static_cast<std::uint32_t>(text_size), 32, 0); // .text, SHF_DEFLATED
    set_section(2, 7, 3, 0, 0, static_cast<std::uint32_t>(names_offset),
                static_cast<std::uint32_t>(names.size()), 1, 0); // .shstrtab
    set_section(3, 17, 0x80000003U, 0, 0, static_cast<std::uint32_t>(crcs_offset),
                static_cast<std::uint32_t>(section_count * sizeof(std::uint32_t)), 4, 4);
    set_section(4, 27, 0x80000004U, 0, 0, static_cast<std::uint32_t>(fileinfo_offset), 0x60, 4, 0);
    set_section(5, 41, 2, 0, 0, static_cast<std::uint32_t>(symtab_offset), 32, 4, 16, 6, 1);
    set_section(6, 49, 3, 0, 0, static_cast<std::uint32_t>(strtab_offset),
                static_cast<std::uint32_t>(symbol_strings.size()), 1, 0);
    set_section(7, 57, 4, 0, 0, static_cast<std::uint32_t>(rela_offset),
                static_cast<std::uint32_t>(relocation_table_size), 4, 12, 5, 1);

    set_be32(file, text_offset, static_cast<std::uint32_t>(code.size()));
    std::copy(compressed.begin(), compressed.end(), file.begin() + text_offset + inflated_size_prefix);
    // Symbol zero is the ELF null symbol (S=0). Exercise Cafe's GHS relative
    // high/low relocations at the final word in .text.
    set_be32(file, rela_offset, code_address + 8);
    set_be32(file, rela_offset + 4, 252);
    set_be32(file, rela_offset + 8, 0);
    set_be32(file, rela_offset + 12, code_address + 10);
    set_be32(file, rela_offset + 16, 253);
    set_be32(file, rela_offset + 20, 0);
    const std::size_t symbol_offset = symtab_offset + 16;
    set_be32(file, symbol_offset, 1); // st_name
    set_be32(file, symbol_offset + 4, relocation_symbol.value); // st_value
    file[symbol_offset + 12] = static_cast<std::uint8_t>(
        (relocation_symbol.binding << 4U) | relocation_symbol.type); // st_info
    set_be16(file, symbol_offset + 14, relocation_symbol.section); // st_shndx
    const std::size_t synthetic_relocation_offset = rela_offset + 24;
    set_be32(file, synthetic_relocation_offset,
             code_address + relocation_options.offset); // r_offset
    set_be32(file, synthetic_relocation_offset + 4,
             (1U << 8U) | relocation_options.type); // symbol 1, relocation type
    set_be32(file, synthetic_relocation_offset + 8,
             static_cast<std::uint32_t>(relocation_symbol.addend)); // r_addend
    std::copy(symbol_strings.begin(), symbol_strings.end(), file.begin() + strtab_offset);
    std::copy(names.begin(), names.end(), file.begin() + names_offset);
    set_be32(file, fileinfo_offset, 0xCAFE0402U);
    set_be32(file, fileinfo_offset + 4, relocation_options.text_size);
    set_be32(file, fileinfo_offset + 0x20, relocation_options.tramp_adjust);
    set_be32(file, fileinfo_offset + 0x24, relocation_options.sda_base);
    set_be32(file, fileinfo_offset + 0x28, relocation_options.sda2_base);
    set_be32(file, fileinfo_offset + 0x48, relocation_options.tramp_addition);
    return file;
}

void decoder_tests()
{
    const DecodedInstruction xori = decode(0x68AC0001U); // xori r12, r5, 1
    assert(xori.opcode == Opcode::xor_immediate);
    assert(xori.destination == 12U);
    assert(xori.source == 5U);
    assert(xori.immediate == 1);

    const DecodedInstruction xoris = decode(0x6CACFFFFU); // xoris r12, r5, 0xFFFF
    assert(xoris.opcode == Opcode::xor_immediate_shifted);
    assert(xoris.destination == 12U);
    assert(xoris.source == 5U);
    assert(xoris.immediate == 0xFFFF);

    const DecodedInstruction mulli = decode(0x1D8B000CU); // mulli r12, r11, 12
    assert(mulli.opcode == Opcode::multiply_low_immediate);
    assert(mulli.destination == 12U);
    assert(mulli.base == 11U);
    assert(mulli.immediate == 12);

    const DecodedInstruction addi = decode(0x3860002AU); // addi r3, r0, 42
    assert(addi.opcode == Opcode::addi);
    assert(addi.destination == 3);
    assert(addi.base == 0);
    assert(addi.immediate == 42);

    const DecodedInstruction addis = decode(0x3C83FFFFU); // addis r4, r3, -1
    assert(addis.opcode == Opcode::addis);
    assert(addis.destination == 4);
    assert(addis.base == 3);
    assert(addis.immediate == -1);

    const DecodedInstruction addic_dot = decode(0x3529FFD2U); // addic. r9, r9, -46
    assert(addic_dot.opcode == Opcode::add_immediate_carry);
    assert(addic_dot.destination == 9);
    assert(addic_dot.base == 9);
    assert(addic_dot.immediate == -46);
    assert(addic_dot.record);

    const DecodedInstruction subfic = decode(0x213FFFefU); // subfic r9, r31, -17
    assert(subfic.opcode == Opcode::subtract_from_immediate_carry);
    assert(subfic.destination == 9);
    assert(subfic.base == 31);
    assert(subfic.immediate == -17);

    const DecodedInstruction add = decode(0x7C642A14U); // add r3, r4, r5
    assert(add.opcode == Opcode::add);
    assert(add.destination == 3);
    assert(add.base == 4);
    assert(add.source == 5);

    const DecodedInstruction subf = decode(0x7CC42850U); // subf r6, r4, r5
    assert(subf.opcode == Opcode::subtract_from);
    assert(subf.destination == 6);
    assert(subf.base == 4);
    assert(subf.source == 5);

    const DecodedInstruction ori = decode(0x60631234U); // ori r3, r3, 0x1234
    assert(ori.opcode == Opcode::ori);
    assert(ori.source == 3);
    assert(ori.destination == 3);
    assert(ori.immediate == 0x1234);

    const DecodedInstruction bit_or = decode(0x7C872B78U); // or r7, r4, r5
    assert(bit_or.opcode == Opcode::bitwise_or);
    assert(bit_or.source == 4);
    assert(bit_or.destination == 7);
    assert(bit_or.base == 5);

    const DecodedInstruction slw = decode(0x7C844830U); // slw r4, r4, r9
    assert(slw.opcode == Opcode::shift_left_word);
    assert(slw.source == 4);
    assert(slw.destination == 4);
    assert(slw.base == 9);

    const DecodedInstruction bit_and = decode(0x7C882838U); // and r8, r4, r5
    assert(bit_and.opcode == Opcode::bitwise_and);
    assert(bit_and.source == 4);
    assert(bit_and.destination == 8);
    assert(bit_and.base == 5);

    const DecodedInstruction andc = decode(0x7D293878U); // andc r9, r9, r7
    assert(andc.opcode == Opcode::bitwise_and_complement);
    assert(andc.source == 9);
    assert(andc.destination == 9);
    assert(andc.base == 7);

    const DecodedInstruction bit_xor = decode(0x7C892A78U); // xor r9, r4, r5
    assert(bit_xor.opcode == Opcode::bitwise_xor);
    assert(bit_xor.source == 4);
    assert(bit_xor.destination == 9);
    assert(bit_xor.base == 5);

    const DecodedInstruction bit_equivalence = decode(0x7CE65238U); // eqv r7, r6, r10
    assert(bit_equivalence.opcode == Opcode::bitwise_equivalence);
    assert(bit_equivalence.source == 7);
    assert(bit_equivalence.destination == 6);
    assert(bit_equivalence.base == 10);

    const DecodedInstruction andi = decode(0x708AFF00U); // andi. r10, r4, 0xFF00
    assert(andi.opcode == Opcode::and_immediate_record);
    assert(andi.source == 4);
    assert(andi.destination == 10);
    assert(andi.immediate == 0xFF00);
    assert(andi.record);

    const DecodedInstruction rlwinm = decode(0x548B2834U); // rlwinm r11, r4, 5, 0, 26
    assert(rlwinm.opcode == Opcode::rotate_left_word_and_mask);
    assert(rlwinm.source == 4);
    assert(rlwinm.destination == 11);
    assert(rlwinm.shift == 5);
    assert(rlwinm.mask_begin == 0);
    assert(rlwinm.mask_end == 26);

    const DecodedInstruction srawi = decode(0x7C691670U); // srawi r9, r3, 2
    assert(srawi.opcode == Opcode::arithmetic_shift_right_immediate);
    assert(srawi.source == 3);
    assert(srawi.destination == 9);
    assert(srawi.shift == 2);

    const DecodedInstruction branch = decode(0x48000009U); // bl +8
    assert(branch.opcode == Opcode::branch);
    assert(branch.immediate == 8);
    assert(!branch.absolute);
    assert(branch.link);

    const DecodedInstruction cmpwi = decode(0x2E830007U); // cmpwi cr5, r3, 7
    assert(cmpwi.opcode == Opcode::compare_signed_immediate);
    assert(cmpwi.cr_field == 5);
    assert(cmpwi.base == 3);
    assert(cmpwi.immediate == 7);

    const DecodedInstruction cmplwi = decode(0x28050003U); // cmplwi cr0, r5, 3
    assert(cmplwi.opcode == Opcode::compare_unsigned_immediate);
    assert(cmplwi.base == 5);
    assert(cmplwi.immediate == 3);

    const DecodedInstruction cmpw = decode(0x7E832000U); // cmpw cr5, r3, r4
    assert(cmpw.opcode == Opcode::compare_signed_register);
    assert(cmpw.cr_field == 5);
    assert(cmpw.base == 3);
    assert(cmpw.source == 4);

    const DecodedInstruction cmplw = decode(0x7C083040U); // cmplw cr0, r8, r6
    assert(cmplw.opcode == Opcode::compare_unsigned_register);
    assert(cmplw.base == 8);
    assert(cmplw.source == 6);

    const DecodedInstruction bc = decode(0x4182000CU); // beq +12
    assert(bc.opcode == Opcode::conditional_branch);
    assert(bc.branch_options == 12);
    assert(bc.condition_bit == 2);
    assert(bc.immediate == 12);

    const DecodedInstruction lwz = decode(0x8041FFFCU); // lwz r2, -4(r1)
    assert(lwz.opcode == Opcode::load_word_zero);
    assert(lwz.destination == 2);
    assert(lwz.base == 1);
    assert(lwz.immediate == -4);

    const DecodedInstruction lwzu = decode(0x853C0004U); // lwzu r9, 4(r28)
    assert(lwzu.opcode == Opcode::load_word_update);
    assert(lwzu.destination == 9);
    assert(lwzu.base == 28);

    const DecodedInstruction stw = decode(0x90610000U); // stw r3, 0(r1)
    assert(stw.opcode == Opcode::store_word);
    assert(stw.destination == 3);

    const DecodedInstruction stwx = decode(0x7F7A492EU); // stwx r27, r26, r9
    assert(stwx.opcode == Opcode::store_word_indexed);
    assert(stwx.destination == 27);
    assert(stwx.base == 26);
    assert(stwx.source == 9);

    const DecodedInstruction lbz = decode(0x88810002U); // lbz r4, 2(r1)
    assert(lbz.opcode == Opcode::load_byte_zero);
    assert(lbz.destination == 4);
    assert(lbz.base == 1);
    assert(lbz.immediate == 2);

    const DecodedInstruction lbzu = decode(0x8D090001U); // lbzu r8, 1(r9)
    assert(lbzu.opcode == Opcode::load_byte_update);
    assert(lbzu.destination == 8);
    assert(lbzu.base == 9);

    const DecodedInstruction stb = decode(0x98610004U); // stb r3, 4(r1)
    assert(stb.opcode == Opcode::store_byte);
    assert(stb.destination == 3);

    const DecodedInstruction stbu = decode(0x9CEA0001U); // stbu r7, 1(r10)
    assert(stbu.opcode == Opcode::store_byte_update);
    assert(stbu.destination == 7);
    assert(stbu.base == 10);

    const DecodedInstruction lhz = decode(0xA0A10002U); // lhz r5, 2(r1)
    assert(lhz.opcode == Opcode::load_halfword_zero);
    assert(lhz.destination == 5);

    const DecodedInstruction sth = decode(0xB0610006U); // sth r3, 6(r1)
    assert(sth.opcode == Opcode::store_halfword);
    assert(sth.destination == 3);

    const DecodedInstruction lfs = decode(0xC1AC0004U); // lfs f13, 4(r12)
    assert(lfs.opcode == Opcode::load_single);
    assert(lfs.fp_register == 13);
    assert(lfs.base == 12);
    assert(lfs.immediate == 4);

    const DecodedInstruction stfs = decode(0xD1ACF060U); // stfs f13, -4000(r12)
    assert(stfs.opcode == Opcode::store_single);
    assert(stfs.fp_register == 13);
    assert(stfs.base == 12);
    assert(stfs.immediate == -4000);

    const DecodedInstruction mflr = decode(0x7C0802A6U); // mflr r0
    assert(mflr.opcode == Opcode::move_from_link_register);
    assert(mflr.destination == 0);

    const DecodedInstruction mtlr = decode(0x7C0803A6U); // mtlr r0
    assert(mtlr.opcode == Opcode::move_to_link_register);
    assert(mtlr.destination == 0);

    const DecodedInstruction blr = decode(0x4E800020U); // blr
    assert(blr.opcode == Opcode::conditional_branch_to_link_register);
    assert(blr.branch_options == 20);
    assert(!blr.link);

    const DecodedInstruction mtctr = decode(0x7D8903A6U); // mtctr r12
    assert(mtctr.opcode == Opcode::move_to_count_register);
    assert(mtctr.destination == 12);

    const DecodedInstruction bctr = decode(0x4E800420U); // bctr
    assert(bctr.opcode == Opcode::conditional_branch_to_count_register);
    assert(bctr.branch_options == 20);

    const DecodedInstruction beqlr = decode(0x4D820020U); // beqlr
    assert(beqlr.opcode == Opcode::conditional_branch_to_link_register);
    assert(beqlr.branch_options == 12);
    assert(beqlr.condition_bit == 2);

    const DecodedInstruction blrl = decode(0x4E800021U); // blrl
    assert(blrl.opcode == Opcode::conditional_branch_to_link_register);
    assert(blrl.link);

    const DecodedInstruction stwu = decode(0x9421FFF0U); // stwu r1, -16(r1)
    assert(stwu.opcode == Opcode::store_word_update);
    assert(stwu.destination == 1);
    assert(stwu.base == 1);
    assert(stwu.immediate == -16);

    const DecodedInstruction stmw = decode(0xBFA1000CU); // stmw r29, 12(r1)
    assert(stmw.opcode == Opcode::store_multiple_word);
    assert(stmw.source == 29U);
    assert(stmw.base == 1U);
    assert(stmw.immediate == 12);

    const DecodedInstruction lmw = decode(0xBBA1FFF4U); // lmw r29, -12(r1)
    assert(lmw.opcode == Opcode::load_multiple_word);
    assert(lmw.destination == 29U);
    assert(lmw.base == 1U);
    assert(lmw.immediate == -12);

    assert(decode(0U).opcode == Opcode::unsupported);
    assert(decode(0x9400FFF0U).opcode == Opcode::unsupported); // stwu with rA=0
}

void guest_memory_tests()
{
    GuestMemory memory(8);

    memory.write32_be(0, 0x12345678U);

    assert(memory.read8(0) == 0x12);
    assert(memory.read8(1) == 0x34);
    assert(memory.read8(2) == 0x56);
    assert(memory.read8(3) == 0x78);
    assert(memory.read32_be(0) == 0x12345678U);

    memory.write16_be(4, 0xABCDU);
    assert(memory.read8(4) == 0xAB);
    assert(memory.read8(5) == 0xCD);
    assert(memory.read16_be(4) == 0xABCDU);

    memory.map_region(0xC0000000U, 0x20);
    memory.write32_be(0xC0000018U, 0x03F00024U);
    assert(memory.read32_be(0xC0000018U) == 0x03F00024U);
    memory.zero_fill(0xC0000018U, 4);
    assert(memory.read32_be(0xC0000018U) == 0);

    GuestMemory boundary_memory(8);
    boundary_memory.map_region(8, 8);
    const std::array<std::uint8_t, 6> block{0x10, 0x11, 0x12, 0x13, 0x14, 0x15};
    boundary_memory.write_bytes(5, block);
    for (std::size_t i = 0; i < block.size(); ++i)
    {
        assert(boundary_memory.read8(static_cast<std::uint32_t>(5 + i)) == block[i]);
    }
    std::array<std::uint8_t, 6> boundary_read{};
    boundary_memory.read_bytes(5U, boundary_read);
    assert(boundary_read == block);

    assert(boundary_memory.read16_be(7) == 0x1213U);
    assert(boundary_memory.read32_be(6) == 0x11121314U);

    boundary_memory.write16_be(7, 0xA1B2U);
    assert(boundary_memory.read8(7) == 0xA1U);
    assert(boundary_memory.read8(8) == 0xB2U);
    boundary_memory.write32_be(6, 0xC3D4E5F6U);
    assert(boundary_memory.read32_be(6) == 0xC3D4E5F6U);

    boundary_memory.zero_fill(6, 4);
    assert(boundary_memory.read32_be(6) == 0U);

    boundary_memory.fill_bytes(6, 4, 0x5AU);
    assert(boundary_memory.read8(5) == 0x10U);
    for (std::uint32_t address = 6; address < 10; ++address)
    {
        assert(boundary_memory.read8(address) == 0x5AU);
    }
    assert(boundary_memory.read8(10) == 0x15U);
    boundary_memory.fill_bytes(0xFFFFFFFFU, 0, 0xFFU);

    bool overlapping_mapping_rejected = false;
    try
    {
        boundary_memory.map_region(12, 4);
    }
    catch (const std::invalid_argument&)
    {
        overlapping_mapping_rejected = true;
    }
    assert(overlapping_mapping_rejected);
}

void interpreter_tests()
{
    EspressoCore history_core(0x1000U);
    history_core.memory.write32_be(0x00, 0x3C00FFFFU); // lis r0, -1
    history_core.memory.write32_be(0x04, 0x6000AC30U); // ori r0, r0, 0xAC30
    history_core.memory.write32_be(0x08, 0x7D06002EU); // lwzx r8, r6, r0
    const RunResult history_result = history_core.run(4);
    assert(history_result.reason == StopReason::memory_fault);
    assert(history_result.instruction_history.size() == 3U);
    assert(history_result.instruction_history[0].opcode_name == "addis");
    assert(history_result.instruction_history[1].opcode_name == "ori");
    assert(history_result.instruction_history[1].has_destination);
    assert(history_result.instruction_history[1].destination_register == 0U);
    assert(history_result.instruction_history[1].destination_value == 0xFFFFAC30U);
    assert(history_result.instruction_history[2].opcode_name == "lwzx");
    assert(history_result.instruction_history[2].source_count == 2U);
    assert(history_result.instruction_history[2].source_registers[0] == 6U);
    assert(history_result.instruction_history[2].source_registers[1] == 0U);
    assert(history_result.instruction_history[2].source_values[1] == 0xFFFFAC30U);
    assert(!history_result.instruction_history[2].completed);

    EspressoCore ring_core(0x100U);
    for (std::uint32_t i = 0; i < 35U; ++i)
    {
        ring_core.memory.write32_be(i * 4U, 0x38630001U); // addi r3, r3, 1
    }
    const RunResult ring_result = ring_core.run(35U);
    assert(ring_result.reason == StopReason::instruction_limit);
    assert(ring_result.instruction_history.size() == instruction_history_capacity);
    assert(ring_result.instruction_history.front().cia == 12U);
    assert(ring_result.instruction_history.back().cia == 136U);
    assert(ring_core.state.gpr[3] == 35U);

    EspressoCore fault_core(0x1000U);
    fault_core.memory.write32_be(0, 0x80640000U); // lwz r3, 0(r4)
    fault_core.state.gpr[4] = 0x3000U;
    const RunResult fault_result = fault_core.run(1);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.cia == 0U);
    assert(fault_result.has_instruction_word);
    assert(fault_result.instruction_word == 0x80640000U);
    assert(fault_result.detail.find("read 4 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("0x00003000") != std::string::npos);
    assert(fault_result.detail.find("CIA 0x00000000") != std::string::npos);
    assert(fault_result.detail.find("instruction 0x80640000") != std::string::npos);
    assert(fault_result.detail.find("rA=4 (0x00003000)") != std::string::npos);

    EspressoCore import_table_core(0x20);
    import_table_core.memory.map_region(0xC0000000U, 0x200);
    const auto import_address =
        import_table_core.hle.bind_import("coreinit", "MEMAllocFromDefaultHeap");
    import_table_core.memory.write32_be(0xC00001E8U, import_address);
    import_table_core.memory.write32_be(0x00, 0x3D20C000U); // lis r9, -16384
    import_table_core.memory.write32_be(0x04, 0x812901E8U); // lwz r9, 488(r9)
    import_table_core.memory.write32_be(0x08, 0x7D2903A6U); // mtctr r9
    import_table_core.memory.write32_be(0x0C, 0x4E800421U); // bctrl
    assert(import_table_core.step() == StepResult::executed);
    assert(import_table_core.step() == StepResult::executed);
    assert(import_table_core.state.gpr[9] == import_address);
    assert(import_table_core.step() == StepResult::executed);
    assert(import_table_core.state.ctr == import_address);
    assert(import_table_core.step() == StepResult::executed);
    assert(import_table_core.state.cia == import_address);
    assert(import_table_core.step() == StepResult::unimplemented_hle_call);

    EspressoCore core(0x20);

    // addi r3, r0, 5
    core.memory.write32_be(0x00, 0x38600005U);
    // ori r3, r3, 7
    core.memory.write32_be(0x04, 0x60630007U);
    // b +8: skip the unsupported word at 0x0c and continue at 0x10
    core.memory.write32_be(0x08, 0x48000008U);
    // addi r4, r3, 1
    core.memory.write32_be(0x10, 0x38830001U);

    const RunResult result = core.run(8);

    assert(result.steps == 4);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(result.cia == 0x14U);
    assert(result.instruction_word == 0U);
    assert(core.state.gpr[3] == 7);
    assert(core.state.gpr[4] == 8);
    assert(core.state.cia == 0x14);

    EspressoCore link_core(0x10);
    // bl +8
    link_core.memory.write32_be(0, 0x48000009U);

    assert(link_core.step() == StepResult::executed);
    assert(link_core.state.cia == 8);
    assert(link_core.state.lr == 4);

    EspressoCore count_branch_core(0x20);
    count_branch_core.state.gpr[12] = 0x12U;
    count_branch_core.memory.write32_be(0, 0x7D8903A6U); // mtctr r12
    count_branch_core.memory.write32_be(4, 0x4E800420U); // bctr
    const RunResult count_branch_result = count_branch_core.run(3);
    assert(count_branch_result.reason == StopReason::unsupported_instruction);
    assert(count_branch_result.cia == 0x10U);
    assert(count_branch_result.instruction_word == 0U);
    assert(count_branch_core.state.ctr == 0x12U);
}

void integer_alu_tests()
{
    const auto run_xor_immediate = [](bool shifted, std::uint8_t destination,
                                      std::uint8_t source, std::uint32_t input,
                                      std::uint16_t immediate)
    {
        EspressoCore xor_core(8);
        constexpr std::uint32_t initial_cr = 0xA5C36987U;
        constexpr std::uint32_t initial_xer = 0xE0000000U;
        xor_core.state.gpr[source] = input;
        xor_core.state.cr = initial_cr;
        xor_core.state.xer = initial_xer;
        const std::uint32_t primary_opcode = shifted ? 27U : 26U;
        const std::uint32_t word = (primary_opcode << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) | immediate;
        xor_core.memory.write32_be(0, word);

        const RunResult result = xor_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(xor_core.state.cia == 4U);
        assert(xor_core.state.cr == initial_cr);
        assert(xor_core.state.xer == initial_xer);
        assert(result.instruction_history.size() == 1U);
        return xor_core.state.gpr[destination];
    };

    assert(run_xor_immediate(false, 12, 5, 0U, 1U) == 1U); // zero source
    assert(run_xor_immediate(false, 12, 5, 0xF0F0F0F0U, 0x00FFU) ==
           0xF0F0F00FU); // nonzero source
    assert(run_xor_immediate(false, 12, 5, 0x12345678U, 0U) == 0x12345678U);
    assert(run_xor_immediate(false, 12, 5, 0U, 0xFFFFU) == 0x0000FFFFU);
    assert(run_xor_immediate(false, 6, 6, 0x12345678U, 0x00FFU) ==
           0x12345687U); // destination aliases source
    assert(run_xor_immediate(false, 12, 0, 0xA5A5A5A5U, 0xFFFFU) ==
           0xA5A55A5AU); // source r0 reads the actual GPR0
    assert(run_xor_immediate(true, 12, 5, 0x12345678U, 0xFFFFU) ==
           0xEDCB5678U);
    assert(run_xor_immediate(true, 12, 5, 0x12345678U, 0x0001U) ==
           0x12355678U); // xoris only changes the immediate's upper-word portion

    EspressoCore xori_trace_core(8);
    xori_trace_core.memory.write32_be(0, 0x68AC0001U);
    xori_trace_core.state.gpr[5] = 0U;
    const RunResult xori_trace = xori_trace_core.run(1);
    assert(format_instruction_history(xori_trace).find(
        "xori r5=0x00000000 imm=0x0001 -> r12=0x00000001") != std::string::npos);

    EspressoCore xoris_trace_core(8);
    xoris_trace_core.memory.write32_be(0, 0x6CACFFFFU);
    xoris_trace_core.state.gpr[5] = 0x12345678U;
    const RunResult xoris_trace = xoris_trace_core.run(1);
    assert(format_instruction_history(xoris_trace).find(
        "xoris r5=0x12345678 imm=0xFFFF -> r12=0xEDCB5678") != std::string::npos);

    const auto run_mulli = [](std::uint8_t destination, std::uint8_t source,
                              std::uint32_t input, std::int16_t immediate)
    {
        EspressoCore mulli_core(8);
        constexpr std::uint32_t initial_cr = 0xA5C36987U;
        constexpr std::uint32_t initial_xer = 0xE0000000U;
        mulli_core.state.gpr[source] = input;
        mulli_core.state.cr = initial_cr;
        mulli_core.state.xer = initial_xer;
        const std::uint32_t word = 0x1C000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 16U) |
            static_cast<std::uint16_t>(immediate);
        mulli_core.memory.write32_be(0, word);

        const RunResult result = mulli_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(mulli_core.state.cia == 4U);
        assert(mulli_core.state.cr == initial_cr);
        assert(mulli_core.state.xer == initial_xer);
        assert(result.instruction_history.size() == 1U);
        assert(result.instruction_history[0].has_destination);
        assert(result.instruction_history[0].destination_register == destination);
        return mulli_core.state.gpr[destination];
    };

    assert(run_mulli(3, 4, 7U, 3) == 21U); // positive * positive
    assert(run_mulli(3, 4, 0xFFFFFFF9U, 3) == 0xFFFFFFEBU); // negative * positive
    assert(run_mulli(3, 4, 7U, -3) == 0xFFFFFFEBU); // positive * negative
    assert(run_mulli(3, 4, 0xFFFFFFF9U, -3) == 21U); // negative * negative
    assert(run_mulli(3, 4, 0U, -123) == 0U);
    assert(run_mulli(12, 0, 9U, 4) == 36U); // RA=0 reads the real GPR0 value
    assert(run_mulli(3, 4, 0x40000000U, 4) == 0U); // low 32 bits of 2^32
    assert(run_mulli(6, 6, 5U, -2) == 0xFFFFFFF6U); // destination aliases source

    EspressoCore mulli_trace_core(8);
    mulli_trace_core.memory.write32_be(0, 0x1D8B000CU);
    const RunResult mulli_trace = mulli_trace_core.run(1);
    assert(format_instruction_history(mulli_trace).find(
        "mulli r11=0x00000000 imm=12 -> r12=0x00000000") != std::string::npos);

    EspressoCore addic_zero_base_core(8);
    addic_zero_base_core.state.gpr[0] = 0x10190000U;
    addic_zero_base_core.memory.write32_be(0, 0x3000AC30U); // addic r0, r0, 0xAC30
    const RunResult addic_zero_base_result = addic_zero_base_core.run(2);
    assert(addic_zero_base_result.reason == StopReason::unsupported_instruction);
    assert(addic_zero_base_core.state.gpr[0] == 0x1018AC30U);
    assert(addic_zero_base_result.instruction_history[0].source_count == 1U);
    assert(addic_zero_base_result.instruction_history[0].source_registers[0] == 0U);
    assert(addic_zero_base_result.instruction_history[0].source_values[0] == 0x10190000U);
    assert(addic_zero_base_result.instruction_history[0].destination_value == 0x1018AC30U);

    EspressoCore core(0x40);
    core.state.gpr[4] = 0xF0F00F0FU;
    core.state.gpr[5] = 0x0FF0FF00U;
    core.state.cr = 0x12345678U;
    core.state.xer = 0x80000000U;

    core.memory.write32_be(0x00, 0x7C642A14U); // add r3, r4, r5
    core.memory.write32_be(0x04, 0x7CC42850U); // subf r6, r4, r5
    core.memory.write32_be(0x08, 0x7C872B78U); // or r7, r4, r5
    core.memory.write32_be(0x0C, 0x7C882839U); // and. r8, r4, r5
    core.memory.write32_be(0x10, 0x7C892A78U); // xor r9, r4, r5
    core.memory.write32_be(0x14, 0x708AFF00U); // andi. r10, r4, 0xFF00
    core.memory.write32_be(0x18, 0x548B2834U); // rlwinm r11, r4, 5, 0, 26

    const RunResult result = core.run(8);

    assert(result.steps == 7);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.gpr[3] == 0x00E10E0FU);
    assert(core.state.gpr[6] == 0x1F00EFF1U);
    assert(core.state.gpr[7] == 0xFFF0FF0FU);
    assert(core.state.gpr[8] == 0x00F00F00U);
    assert(core.state.gpr[9] == 0xFF00F00FU);
    assert(core.state.gpr[10] == 0x00000F00U);
    assert(core.state.gpr[11] == 0x1E01E1E0U);
    // and. and andi. update only CR0, preserving the lower CR fields and XER.SO.
    assert(core.state.cr == 0x52345678U);

    EspressoCore subtract_carry_core(8);
    subtract_carry_core.state.gpr[4] = 2U;
    subtract_carry_core.memory.write32_be(0, 0x20640005U); // subfic r3, r4, 5
    assert(subtract_carry_core.step() == StepResult::executed);
    assert(subtract_carry_core.state.gpr[3] == 3U);
    assert((subtract_carry_core.state.xer & 0x20000000U) != 0);

    EspressoCore shift_left_core(8);
    shift_left_core.state.gpr[4] = 3U;
    shift_left_core.state.gpr[9] = 4U;
    shift_left_core.memory.write32_be(0, 0x7C844830U); // slw r4, r4, r9
    assert(shift_left_core.step() == StepResult::executed);
    assert(shift_left_core.state.gpr[4] == 48U);

    EspressoCore and_complement_core(8);
    and_complement_core.state.gpr[9] = 0xFFFF0000U;
    and_complement_core.state.gpr[7] = 0x00FF00FFU;
    and_complement_core.memory.write32_be(0, 0x7D293878U); // andc r9, r9, r7
    assert(and_complement_core.step() == StepResult::executed);
    assert(and_complement_core.state.gpr[9] == 0xFF000000U);

    // rlwinm masks may wrap across the word boundary: MB=28 through ME=3.
    EspressoCore wrapping_rotate_core(8);
    wrapping_rotate_core.state.gpr[4] = 0xF000000FU;
    // rlwinm r3, r4, 0, 28, 3
    wrapping_rotate_core.memory.write32_be(0, 0x54830706U);
    assert(wrapping_rotate_core.step() == StepResult::executed);
    assert(wrapping_rotate_core.state.gpr[3] == 0xF000000FU);

    EspressoCore arithmetic_shift_core(8);
    arithmetic_shift_core.state.gpr[3] = 0xFFFFFFFBU;
    arithmetic_shift_core.memory.write32_be(0, 0x7C691670U); // srawi r9, r3, 2
    assert(arithmetic_shift_core.step() == StepResult::executed);
    assert(arithmetic_shift_core.state.gpr[9] == 0xFFFFFFFEU);
    assert((arithmetic_shift_core.state.xer & 0x20000000U) != 0);
}

void leaf_function_abi_tests()
{
    EspressoCore core(0x20);

    // Compiler output for tests/powerpc_leaf_function.c, built with devkitPPC
    // GCC 16.1.0 using -O2 -mcpu=750 -fno-pic -fno-asynchronous-unwind-tables:
    // addi r3, r3, 5; blr.
    core.memory.write32_be(0x00, 0x38630005U);
    core.memory.write32_be(0x04, 0x4E800020U);
    core.state.gpr[3] = 37U; // PPC ABI: first integer argument arrives in r3.
    core.state.lr = 0x10U;   // Return to the test's unsupported sentinel.

    const RunResult result = core.run(4);

    assert(result.steps == 2);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.cia == 0x10U);
    assert(core.state.gpr[3] == 42U); // Integer result is returned in r3.
}

void elf_loader_tests()
{
    const std::vector<std::uint8_t> file = make_minimal_powerpc_elf();
    EspressoCore core(0x2000);
    core.memory.write32_be(0x1008, 0xDEADBEEFU);

    const ElfLoadResult load_result = load_elf32_powerpc(core, file);

    assert(load_result.entry_point == 0x1000U);
    assert(load_result.loaded_segments == 1);
    assert(core.state.cia == 0x1000U);
    assert(core.memory.read32_be(0x1000) == 0x38630005U);
    assert(core.memory.read32_be(0x1004) == 0x4E800020U);
    assert(core.memory.read32_be(0x1008) == 0U); // p_memsz - p_filesz is zero-filled.

    core.state.gpr[3] = 37U;
    core.state.lr = 0x1008U;
    const RunResult run_result = core.run(4);
    assert(run_result.steps == 2);
    assert(run_result.reason == StopReason::unsupported_instruction);
    assert(core.state.cia == 0x1008U);
    assert(core.state.gpr[3] == 42U);

    // Reject an invalid file without modifying guest memory or architectural state.
    std::vector<std::uint8_t> invalid_file = file;
    invalid_file[0] = 0;
    EspressoCore unchanged_core(0x2000);
    unchanged_core.state.cia = 0x40U;
    unchanged_core.memory.write32_be(0x1000, 0xAABBCCDDU);
    bool rejected = false;
    try
    {
        static_cast<void>(load_elf32_powerpc(unchanged_core, invalid_file));
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
    assert(unchanged_core.state.cia == 0x40U);
    assert(unchanged_core.memory.read32_be(0x1000) == 0xAABBCCDDU);

    std::vector<std::uint8_t> bad_segment = file;
    set_be32(bad_segment, 52 + 20, 4); // p_memsz < p_filesz
    EspressoCore bad_segment_core(0x2000);
    rejected = false;
    try
    {
        static_cast<void>(load_elf32_powerpc(bad_segment_core, bad_segment));
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
}

void rpx_loader_tests()
{
    const std::vector<std::uint8_t> file = make_minimal_compressed_rpx();
    constexpr std::uint32_t entry_point = 0x02000000U;
    EspressoCore core(static_cast<std::size_t>(entry_point) + 0x100U);
    const RpxLoadResult load_result = load_rpx32_powerpc(core, file);

    assert(load_result.entry_point == entry_point);
    assert(load_result.loaded_sections == 1);
    assert(load_result.sda_base == 0U);
    assert(load_result.sda2_base == 0U);
    assert(core.state.cia == entry_point);
    assert(core.memory.read32_be(entry_point) == 0x38630005U);
    assert(core.memory.read32_be(entry_point + 4) == 0x4E800020U);
    assert(core.memory.read32_be(entry_point + 8) == 0xFDFFFFF6U);
    assert(core.memory.read32_be(entry_point + 12) == 0x12345678U);

    // A different strong SHN_UNDEF symbol is still an unresolved import.
    const RpxRelocationSymbol unresolved_symbol{
        "MissingStrongSymbol", 0, 1, 1, 0, 0x2468};
    const std::vector<std::uint8_t> unresolved_file =
        make_minimal_compressed_rpx(entry_point, 16, unresolved_symbol);
    EspressoCore unresolved_core(static_cast<std::size_t>(entry_point) + 0x100U);
    bool strong_undefined_rejected = false;
    try
    {
        static_cast<void>(load_rpx32_powerpc(unresolved_core, unresolved_file));
    }
    catch (const std::invalid_argument& error)
    {
        strong_undefined_rejected =
            std::string(error.what()).find("MissingStrongSymbol") != std::string::npos;
    }
    assert(strong_undefined_rejected);

    // Undefined weak data symbols retain their prior zero-plus-addend behavior.
    const RpxRelocationSymbol weak_symbol{"OptionalWeakData", 0, 2, 1, 0, 0x76543210};
    const std::vector<std::uint8_t> weak_file =
        make_minimal_compressed_rpx(entry_point, 16, weak_symbol);
    EspressoCore weak_core(static_cast<std::size_t>(entry_point) + 0x100U);
    static_cast<void>(load_rpx32_powerpc(weak_core, weak_file));
    assert(weak_core.memory.read32_be(entry_point + 12) == 0x76543210U);

    constexpr std::uint32_t branch_site = 0x0200000CU;
    const RpxRelocationSymbol nearby_target{
        "NearbyTarget", 0x02001000U, 1, 1, 0xFFF1, 0};
    RpxRelocationOptions direct_options;
    direct_options.type = 10; // R_PPC_REL24
    direct_options.branch_instruction = 0x48000003U; // b with AA and LK set
    const auto direct_file = make_minimal_compressed_rpx(
        entry_point, 16, nearby_target, direct_options);
    EspressoCore direct_core(0x1000U);
    static_cast<void>(load_rpx32_powerpc(direct_core, direct_file));
    const std::uint32_t direct_branch = direct_core.memory.read32_be(branch_site);
    assert((direct_branch & 3U) == 3U); // AA and LK survive the direct patch.
    assert((direct_branch & 0x03FFFFFCU) == 0x00000FF4U);

    const auto assert_trampoline = [](const EspressoCore& trampoline_core,
                                      std::uint32_t trampoline_address,
                                      std::uint32_t branch_instruction,
                                      std::uint32_t branch_site_address,
                                      std::uint32_t absolute_target) {
        const std::uint32_t encoded_displacement = branch_instruction & 0x03FFFFFCU;
        const std::int32_t displacement = static_cast<std::int32_t>(
            (encoded_displacement & 0x02000000U) != 0
                ? (encoded_displacement | 0xFC000000U)
                : encoded_displacement);
        assert(static_cast<std::uint32_t>(branch_site_address + displacement) ==
               trampoline_address);
        assert((branch_instruction & 1U) == 1U); // The source branch remains a link.
        assert(trampoline_core.memory.read32_be(trampoline_address) ==
               (0x3D600000U | (absolute_target >> 16U)));
        assert(trampoline_core.memory.read32_be(trampoline_address + 4U) ==
               (0x616B0000U | (absolute_target & 0xFFFFU)));
        assert(trampoline_core.memory.read32_be(trampoline_address + 8U) == 0x7D6903A6U);
        assert(trampoline_core.memory.read32_be(trampoline_address + 12U) == 0x4E800420U);
    };

    RpxRelocationOptions positive_options;
    positive_options.type = 10;
    positive_options.tramp_adjust = 0x40;
    positive_options.text_size = 0x200;
    const RpxRelocationSymbol positive_far_target{
        "PositiveFarTarget", 0x08000000U, 1, 1, 0xFFF1, 0};
    const auto positive_file = make_minimal_compressed_rpx(
        entry_point, 16, positive_far_target, positive_options);
    EspressoCore positive_core(0x1000U);
    static_cast<void>(load_rpx32_powerpc(positive_core, positive_file));
    constexpr std::uint32_t post_trampoline = entry_point + 16U;
    const std::uint32_t positive_branch = positive_core.memory.read32_be(branch_site);
    assert_trampoline(positive_core, post_trampoline, positive_branch, branch_site, 0x08000000U);

    const RpxRelocationSymbol negative_far_target{
        "NegativeFarTarget", 0U, 1, 1, 0xFFF1, 0};
    RpxRelocationOptions negative_options = positive_options;
    negative_options.text_size = 0x50; // No post buffer; exercise Cafe's pre-buffer path.
    const auto negative_file = make_minimal_compressed_rpx(
        entry_point, 16, negative_far_target, negative_options);
    EspressoCore negative_core(0x1000U);
    static_cast<void>(load_rpx32_powerpc(negative_core, negative_file));
    constexpr std::uint32_t pre_trampoline = entry_point - 16U;
    const std::uint32_t negative_branch = negative_core.memory.read32_be(branch_site);
    assert_trampoline(negative_core, pre_trampoline, negative_branch, branch_site, 0U);

    const RpxRelocationSymbol undefined_rel24{"$UNDEF", 0, 1, 1, 0, 0};
    const auto undefined_rel24_file = make_minimal_compressed_rpx(
        entry_point, 16, undefined_rel24, positive_options);
    EspressoCore undefined_rel24_core(0x1000U);
    static_cast<void>(load_rpx32_powerpc(undefined_rel24_core, undefined_rel24_file));
    const std::uint32_t undefined_branch =
        undefined_rel24_core.memory.read32_be(branch_site);
    assert_trampoline(
        undefined_rel24_core, post_trampoline, undefined_branch, branch_site, 0U);

    RpxRelocationOptions exhausted_options;
    exhausted_options.type = 10;
    exhausted_options.text_size = 16;
    const auto exhausted_file = make_minimal_compressed_rpx(
        entry_point, 16, positive_far_target, exhausted_options);
    EspressoCore exhausted_core(0x1000U);
    bool trampoline_exhaustion_reported = false;
    try
    {
        static_cast<void>(load_rpx32_powerpc(exhausted_core, exhausted_file));
    }
    catch (const std::invalid_argument& error)
    {
        trampoline_exhaustion_reported =
            std::string(error.what()).find("no reachable free FILEINFO trampoline slot") !=
            std::string::npos;
    }
    assert(trampoline_exhaustion_reported);

    core.state.gpr[3] = 37;
    core.state.lr = entry_point + 8;
    const RunResult run_result = core.run(2);
    assert(run_result.steps == 2);
    assert(run_result.reason == StopReason::instruction_limit);
    assert(core.state.cia == entry_point + 8);
    assert(core.state.gpr[3] == 42);

    // Reject bad compressed data before changing CPU state or guest memory.
    std::vector<std::uint8_t> malformed = file;
    malformed[384 + 4] ^= 0xFF;
    core.state.cia = 0x80;
    core.memory.write32_be(entry_point, 0xAABBCCDDU);
    bool rejected = false;
    try
    {
        static_cast<void>(load_rpx32_powerpc(core, malformed));
    }
    catch (const std::invalid_argument&)
    {
        rejected = true;
    }
    assert(rejected);
    assert(core.state.cia == 0x80);
    assert(core.memory.read32_be(entry_point) == 0xAABBCCDDU);

    RpxRelocationOptions sda_options;
    sda_options.sda_base = 0x10008000U;
    sda_options.sda2_base = 0x10009000U;
    const auto sda_file = make_minimal_compressed_rpx(
        entry_point, 16, {}, sda_options);
    affogato::Emulator session;
    const auto session_image = session.load_rpx(sda_file);
    assert(session_image.sda_base == sda_options.sda_base);
    assert(session_image.sda2_base == sda_options.sda2_base);
    assert(session.core().state.gpr[2] == sda_options.sda2_base);
    assert(session.core().state.gpr[13] == sda_options.sda_base);
    const auto session_run = session.run(2);
    assert(session_image.entry_point == entry_point);
    assert(session_run.image.entry_point == entry_point);
    assert(session_run.execution.reason == StopReason::instruction_limit);
    assert(session_run.gpr3 == 5U);
    constexpr std::uint32_t thread_tag_offset = 0x320U;
    constexpr std::uint32_t thread_state_offset = 0x324U;
    constexpr std::uint32_t thread_attributes_offset = 0x325U;
    constexpr std::uint32_t thread_id_offset = 0x326U;
    constexpr std::uint32_t thread_stack_start_offset = 0x394U;
    constexpr std::uint32_t thread_stack_end_offset = 0x398U;
    constexpr std::uint32_t thread_type_offset = 0x5BCU;
    const std::uint32_t thread_address = session.core().current_thread_address;
    assert(thread_address != 0);
    assert(session.core().memory.read32_be(thread_address + thread_tag_offset) == 0x74487244U);
    assert(session.core().memory.read8(thread_address + thread_state_offset) == 2U);
    assert(session.core().memory.read8(thread_address + thread_attributes_offset) == 2U);
    assert(session.core().memory.read16_be(thread_address + thread_id_offset) != 0);
    assert(session.core().memory.read32_be(thread_address + thread_stack_start_offset) ==
           session.core().state.gpr[1]);
    assert(session.core().memory.read32_be(thread_address + thread_stack_end_offset) ==
           session.core().state.gpr[1] - affogato::Emulator::guest_stack_size);
    assert(session.core().memory.read32_be(thread_address + thread_type_offset) == 2U);

    // Allocated sections outside the flat backing vector use GuestMemory's
    // sparse map without changing their guest-visible addresses or bytes.
    constexpr std::uint32_t sparse_entry = 0x20000000U;
    const std::vector<std::uint8_t> sparse_file =
        make_minimal_compressed_rpx(sparse_entry);
    EspressoCore sparse_core(0x1000U);
    const RpxLoadResult sparse_result = load_rpx32_powerpc(sparse_core, sparse_file);
    assert(sparse_result.entry_point == sparse_entry);
    assert(sparse_result.loaded_sections == 1);
    assert(sparse_core.state.cia == sparse_entry);
    assert(sparse_core.memory.read32_be(sparse_entry) == 0x38630005U);
    assert(sparse_core.memory.read32_be(sparse_entry + 4) == 0x4E800020U);

    constexpr std::uint32_t wind_waker_section_address = 0x10000000U;
    constexpr std::size_t flat_memory_size = 0x10100000U;
    constexpr std::size_t wind_waker_section_size = 0x18C0C0U;
    const std::vector<std::uint8_t> boundary_section_file = make_minimal_compressed_rpx(
        wind_waker_section_address, wind_waker_section_size);
    EspressoCore boundary_section_core(flat_memory_size);
    const RpxLoadResult boundary_section_result =
        load_rpx32_powerpc(boundary_section_core, boundary_section_file);
    assert(boundary_section_result.entry_point == wind_waker_section_address);
    assert(boundary_section_core.state.cia == wind_waker_section_address);
    assert(boundary_section_core.memory.read32_be(0x100FFFFEU) == 0U);
    assert(boundary_section_core.memory.read32_be(0x10100000U) == 0U);
    assert(boundary_section_core.memory.read32_be(
               wind_waker_section_address + static_cast<std::uint32_t>(wind_waker_section_size) - 4U) ==
           0U);

    affogato::Emulator sparse_session(0x20000U);
    static_cast<void>(sparse_session.load_rpx(sparse_file));
    assert(sparse_session.core().state.gpr[2] == 0U);
    assert(sparse_session.core().state.gpr[13] == 0U);
    assert(sparse_session.core().state.gpr[1] == 0x1F000U);
    assert(sparse_session.core().current_thread_address == 0x1000U);
    assert(sparse_session.core().guest_heap_cursor > 0x1000U);

    // A loaded RPX section may not overwrite another section, whether the
    // address belongs to flat memory or a sparse mapping.
    std::vector<std::uint8_t> overlapping_file = sparse_file;
    constexpr std::size_t section_table_offset = 52;
    constexpr std::size_t section_header_size = 40;
    constexpr std::size_t rela_section_index = 7;
    const std::size_t rela_section_header =
        section_table_offset + rela_section_index * section_header_size;
    set_be32(overlapping_file, rela_section_header + 8, 0x2U); // SHF_ALLOC
    set_be32(overlapping_file, rela_section_header + 12, sparse_entry);
    bool overlap_rejected = false;
    try
    {
        EspressoCore overlapping_core(0x1000U);
        static_cast<void>(load_rpx32_powerpc(overlapping_core, overlapping_file));
    }
    catch (const std::invalid_argument& error)
    {
        overlap_rejected = std::string(error.what()).find("overlaps another allocated") !=
                           std::string::npos;
    }
    assert(overlap_rejected);

    // An RPX section may start in flat memory and continue into sparse memory.
    constexpr std::uint32_t boundary_entry = 0x1000U;
    const std::vector<std::uint8_t> boundary_file =
        make_minimal_compressed_rpx(boundary_entry);
    EspressoCore boundary_core(boundary_entry + 4U);
    const RpxLoadResult boundary_result = load_rpx32_powerpc(boundary_core, boundary_file);
    assert(boundary_result.entry_point == boundary_entry);
    assert(boundary_core.memory.read32_be(boundary_entry) == 0x38630005U);
    assert(boundary_core.memory.read32_be(boundary_entry + 4U) == 0x4E800020U);

    // Reject a section whose end would wrap beyond the 32-bit address space,
    // and include all useful range details in the diagnostic.
    constexpr std::uint32_t overflowing_entry = 0xFFFFFFFCU;
    const std::vector<std::uint8_t> overflowing_file =
        make_minimal_compressed_rpx(overflowing_entry);
    EspressoCore overflowing_core(0x1000U);
    bool overflow_rejected = false;
    try
    {
        static_cast<void>(load_rpx32_powerpc(overflowing_core, overflowing_file));
    }
    catch (const std::invalid_argument& error)
    {
        const std::string message = error.what();
        overflow_rejected = message.find("section 1") != std::string::npos &&
                            message.find("0xFFFFFFFC") != std::string::npos &&
                            message.find("section size 0x10") != std::string::npos &&
                            message.find("section end 0x10000000C") != std::string::npos;
    }
    assert(overflow_rejected);
}

void hle_dispatch_tests()
{
    EspressoCore core(0x100);
    register_coreinit_hle(core.hle);

    const std::uint32_t ghs_lock = core.hle.bind_import("coreinit", "__ghsLock");
    const std::uint32_t ghs_unlock = core.hle.bind_import("coreinit", "__ghsUnlock");
    for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
    {
        core.state.gpr[index] = 0x10203040U + static_cast<std::uint32_t>(index);
        core.state.fpr[index] = 0xA0B0C0D000000000ULL + index;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0x23456789U;
    core.state.ctr = 0x3456789AU;
    core.state.lr = 0x456789ABU;
    const auto gpr_before_ghs_calls = core.state.gpr;
    const auto fpr_before_ghs_calls = core.state.fpr;
    const std::uint32_t cr_before_ghs_calls = core.state.cr;
    const std::uint32_t xer_before_ghs_calls = core.state.xer;
    const std::uint32_t ctr_before_ghs_calls = core.state.ctr;

    core.state.cia = ghs_lock;
    core.state.lr = 0x60U;
    assert(core.step() == StepResult::executed);
    assert(core.state.cia == 0x60U);
    assert(core.state.gpr == gpr_before_ghs_calls);
    assert(core.state.fpr == fpr_before_ghs_calls);
    assert(core.state.cr == cr_before_ghs_calls);
    assert(core.state.xer == xer_before_ghs_calls);
    assert(core.state.ctr == ctr_before_ghs_calls);
    assert(core.state.lr == 0x60U);

    core.state.cia = ghs_unlock;
    core.state.lr = 0x64U;
    assert(core.step() == StepResult::executed);
    assert(core.state.cia == 0x64U);
    assert(core.state.gpr == gpr_before_ghs_calls);
    assert(core.state.fpr == fpr_before_ghs_calls);
    assert(core.state.cr == cr_before_ghs_calls);
    assert(core.state.xer == xer_before_ghs_calls);
    assert(core.state.ctr == ctr_before_ghs_calls);
    assert(core.state.lr == 0x64U);

    const std::uint32_t debugger_check =
        core.hle.bind_import("coreinit", "OSIsDebuggerInitialized");
    core.state.cia = debugger_check;
    core.state.lr = 0x40;
    core.state.gpr[3] = 1;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0);
    assert(core.state.cia == 0x40);

    const std::uint32_t device_info = core.hle.bind_import("coreinit", "FSAGetDeviceInfo");
    core.state.cia = device_info;
    core.state.lr = 0x48;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0);
    assert(core.state.cia == 0x48);

    core.configure_guest_heap(0x80U, 0x100U);
    const std::uint32_t allocate = core.hle.bind_import("coreinit", "MEMAllocFromDefaultHeap");
    core.state.cia = allocate;
    core.state.lr = 0x4C;
    core.state.gpr[3] = 44;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0x80U);
    assert(core.state.cia == 0x4C);

    const std::uint32_t mutex_init = core.hle.bind_import("coreinit", "OSFastMutex_Init");
    core.state.cia = mutex_init;
    core.state.lr = 0x50;
    core.state.gpr[3] = 0x90;
    core.state.gpr[4] = 0x1234;
    assert(core.step() == StepResult::executed);
    assert(core.memory.read32_be(0x90) == 0x664D7458U);
    assert(core.memory.read32_be(0x94) == 0x1234U);
    assert(core.memory.read32_be(0xB8) == 0);

    const std::uint32_t missing_import = core.hle.bind_import("coreinit", "OSFatal");
    core.state.cia = missing_import;
    core.state.lr = 0x44;
    assert(core.step() == StepResult::unimplemented_hle_call);
    assert(core.state.cia == missing_import);
    assert(core.hle.last_unimplemented_call() == "coreinit::OSFatal");

    EspressoCore thread_core(0x2000U);
    register_coreinit_hle(thread_core.hle);
    thread_core.configure_guest_heap(0x100U, 0x1000U);
    constexpr std::uint32_t stack_start = 0x1800U;
    constexpr std::uint32_t stack_end = 0x800U;
    const std::uint32_t thread_address = initialize_default_guest_thread(
        thread_core, stack_start, stack_end);
    assert(thread_address == 0x100U);

    const std::uint32_t get_current_thread =
        thread_core.hle.bind_import("coreinit", "OSGetCurrentThread");
    for (std::uint32_t call = 0; call < 2; ++call)
    {
        thread_core.state.cia = get_current_thread;
        thread_core.state.lr = 0x60U + call * 4U;
        assert(thread_core.step() == StepResult::executed);
        assert(thread_core.state.gpr[3] == thread_address);
        assert(thread_core.state.cia == thread_core.state.lr);
    }
    assert(thread_core.memory.read32_be(thread_address + 0x320U) == 0x74487244U);
    assert(thread_core.memory.read8(thread_address + 0x324U) == 2U);
    assert(thread_core.memory.read16_be(thread_address + 0x326U) == 1U);

    const std::uint32_t get_specific =
        thread_core.hle.bind_import("coreinit", "OSGetThreadSpecific");
    const std::uint32_t set_specific =
        thread_core.hle.bind_import("coreinit", "OSSetThreadSpecific");
    constexpr std::uint32_t specific_id = 5U;
    const std::uint32_t specific_slot = thread_address + 0x57CU + specific_id * 4U;
    thread_core.state.cia = set_specific;
    thread_core.state.lr = 0x68U;
    thread_core.state.gpr[3] = specific_id;
    thread_core.state.gpr[4] = 0xA1B2C3D4U;
    assert(thread_core.step() == StepResult::executed);
    assert(thread_core.memory.read32_be(specific_slot) == 0xA1B2C3D4U);

    thread_core.state.cia = get_specific;
    thread_core.state.lr = 0x6CU;
    thread_core.state.gpr[3] = specific_id;
    assert(thread_core.step() == StepResult::executed);
    assert(thread_core.state.gpr[3] == 0xA1B2C3D4U);

    // Direct guest writes and HLE reads share the same guest-visible slots.
    thread_core.memory.write32_be(specific_slot, 0x55667788U);
    thread_core.state.cia = get_specific;
    thread_core.state.lr = 0x70U;
    thread_core.state.gpr[3] = specific_id;
    assert(thread_core.step() == StepResult::executed);
    assert(thread_core.state.gpr[3] == 0x55667788U);

    thread_core.state.cia = get_specific;
    thread_core.state.lr = 0x74U;
    thread_core.state.gpr[3] = 16U;
    assert(thread_core.step() == StepResult::executed);
    assert(thread_core.state.gpr[3] == 0U);
    const std::uint32_t thread_type_before = thread_core.memory.read32_be(thread_address + 0x5BCU);
    thread_core.state.cia = set_specific;
    thread_core.state.lr = 0x78U;
    thread_core.state.gpr[3] = 16U;
    thread_core.state.gpr[4] = 0xFFFFFFFFU;
    assert(thread_core.step() == StepResult::executed);
    assert(thread_core.memory.read32_be(thread_address + 0x5BCU) == thread_type_before);
}

void guest_mutex_tests()
{
    EspressoCore core(0x2000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x100U, 0x1800U);
    constexpr std::uint32_t stack_start = 0x1F00U;
    constexpr std::uint32_t stack_end = 0x0F00U;
    const std::uint32_t thread =
        initialize_default_guest_thread(core, stack_start, stack_end);
    constexpr std::uint32_t wrapper = 0x40U;
    constexpr std::uint32_t return_address = 0x80U;
    const auto invoke = [&](std::uint32_t import, std::uint32_t argument) {
        core.state.cia = import;
        core.state.lr = return_address;
        core.state.gpr[3] = argument;
        assert(core.step() == StepResult::executed);
        assert(core.state.cia == return_address);
    };

    const std::uint32_t initialize = core.hle.bind_import("coreinit", "__ghs_mtx_init");
    const std::uint32_t lock = core.hle.bind_import("coreinit", "__ghs_mtx_lock");
    const std::uint32_t unlock = core.hle.bind_import("coreinit", "__ghs_mtx_unlock");
    const std::uint32_t destroy = core.hle.bind_import("coreinit", "__ghs_mtx_dst");
    invoke(initialize, wrapper);

    const std::uint32_t mutex = core.memory.read32_be(wrapper);
    assert(mutex != 0);
    assert((mutex & 7U) == 0);
    assert(mutex == 0x7A0U);
    assert(core.memory.read32_be(mutex) == os_mutex_tag);
    assert(core.memory.read32_be(mutex + 0x04U) == 0);
    assert(core.memory.read32_be(mutex + 0x0CU) == 0);
    assert(core.memory.read32_be(mutex + 0x1CU) == 0);
    assert(core.memory.read32_be(mutex + 0x20U) == 0);
    assert(core.memory.read32_be(mutex + 0x24U) == 0);

    invoke(lock, wrapper);
    assert(core.memory.read32_be(mutex + 0x1CU) == thread);
    assert(core.memory.read32_be(mutex + 0x20U) == 1U);
    invoke(lock, wrapper);
    assert(core.memory.read32_be(mutex + 0x1CU) == thread);
    assert(core.memory.read32_be(mutex + 0x20U) == 2U);
    invoke(unlock, wrapper);
    assert(core.memory.read32_be(mutex + 0x1CU) == thread);
    assert(core.memory.read32_be(mutex + 0x20U) == 1U);
    invoke(unlock, wrapper);
    assert(core.memory.read32_be(mutex + 0x1CU) == 0);
    assert(core.memory.read32_be(mutex + 0x20U) == 0);

    std::array<std::uint8_t, os_mutex_size> mutex_before_destroy{};
    for (std::size_t index = 0; index < mutex_before_destroy.size(); ++index)
    {
        mutex_before_destroy[index] = core.memory.read8(
            mutex + static_cast<std::uint32_t>(index));
    }
    invoke(destroy, wrapper);
    assert(core.memory.read32_be(wrapper) == 0);
    for (std::size_t index = 0; index < mutex_before_destroy.size(); ++index)
    {
        assert(core.memory.read8(mutex + static_cast<std::uint32_t>(index)) ==
               mutex_before_destroy[index]);
    }

    // The public Cafe mutex calls share the same guest layout and state logic.
    constexpr std::uint32_t direct_mutex = 0x900U;
    const std::uint32_t os_init = core.hle.bind_import("coreinit", "OSInitMutex");
    const std::uint32_t os_lock = core.hle.bind_import("coreinit", "OSLockMutex");
    const std::uint32_t os_unlock = core.hle.bind_import("coreinit", "OSUnlockMutex");
    core.state.cia = os_init;
    core.state.lr = return_address;
    core.state.gpr[3] = direct_mutex;
    core.state.gpr[4] = 0x12345678U;
    assert(core.step() == StepResult::executed);
    assert(core.memory.read32_be(direct_mutex) == os_mutex_tag);
    assert(core.memory.read32_be(direct_mutex + 4U) == 0x12345678U);
    invoke(os_lock, direct_mutex);
    assert(core.memory.read32_be(direct_mutex + 0x1CU) == thread);
    invoke(os_unlock, direct_mutex);
    assert(core.memory.read32_be(direct_mutex + 0x1CU) == 0);

    // A foreign owner cannot be treated as acquired without a scheduler.
    core.memory.write32_be(mutex + 0x1CU, 0xDEADBEEFU);
    core.memory.write32_be(mutex + 0x20U, 1U);
    core.state.cia = lock;
    core.state.lr = return_address;
    core.state.gpr[3] = wrapper;
    core.memory.write32_be(wrapper, mutex);
    const RunResult contention = core.run(1U);
    assert(contention.reason == StopReason::hle_error);
    assert(contention.detail.find("no scheduler") != std::string::npos);
    assert(core.memory.read32_be(mutex + 0x1CU) == 0xDEADBEEFU);
    assert(core.memory.read32_be(mutex + 0x20U) == 1U);

    EspressoCore exhausted_core(0x200U);
    register_coreinit_hle(exhausted_core.hle);
    exhausted_core.configure_guest_heap(0x100U, 0x100U);
    constexpr std::uint32_t failure_wrapper = 0x40U;
    constexpr std::uint32_t sentinel = 0xAABBCCDDU;
    exhausted_core.memory.write32_be(failure_wrapper, sentinel);
    const std::uint32_t exhausted_init =
        exhausted_core.hle.bind_import("coreinit", "__ghs_mtx_init");
    exhausted_core.state.cia = exhausted_init;
    exhausted_core.state.lr = return_address;
    exhausted_core.state.gpr[3] = failure_wrapper;
    assert(exhausted_core.step() == StepResult::executed);
    assert(exhausted_core.state.cia == return_address);
    assert(exhausted_core.memory.read32_be(failure_wrapper) == sentinel);
}

void memset_hle_tests()
{
    EspressoCore zero_core(0x20U);
    register_coreinit_hle(zero_core.hle);
    const std::uint32_t zero_memset = zero_core.hle.bind_import("coreinit", "memset");
    zero_core.memory.fill_bytes(0, zero_core.memory.size(), 0xA5U);
    zero_core.state.cia = zero_memset;
    zero_core.state.lr = 0x80U;
    zero_core.state.gpr[3] = 4U;
    zero_core.state.gpr[4] = 0x100U;
    zero_core.state.gpr[5] = 4U;
    assert(zero_core.step() == StepResult::executed);
    assert(zero_core.state.cia == 0x80U);
    assert(zero_core.state.gpr[3] == 4U);
    assert(zero_core.state.gpr[4] == 0x100U);
    assert(zero_core.state.gpr[5] == 4U);
    assert(zero_core.memory.read8(3U) == 0xA5U);
    for (std::uint32_t address = 4U; address < 8U; ++address)
    {
        assert(zero_core.memory.read8(address) == 0U);
    }
    assert(zero_core.memory.read8(8U) == 0xA5U);

    EspressoCore boundary_core(0x10U);
    register_coreinit_hle(boundary_core.hle);
    boundary_core.memory.map_region(0x10U, 0x10U);
    boundary_core.memory.fill_bytes(0U, 0x20U, 0xCCU);
    const std::uint32_t boundary_memset = boundary_core.hle.bind_import("coreinit", "memset");
    boundary_core.state.cia = boundary_memset;
    boundary_core.state.lr = 0x84U;
    boundary_core.state.gpr[3] = 0x0EU;
    boundary_core.state.gpr[4] = 0x1ABU;
    boundary_core.state.gpr[5] = 5U;
    assert(boundary_core.step() == StepResult::executed);
    assert(boundary_core.state.cia == 0x84U);
    assert(boundary_core.state.gpr[3] == 0x0EU);
    assert(boundary_core.state.gpr[4] == 0x1ABU);
    assert(boundary_core.state.gpr[5] == 5U);
    assert(boundary_core.memory.read8(0x0DU) == 0xCCU);
    for (std::uint32_t address = 0x0EU; address < 0x13U; ++address)
    {
        assert(boundary_core.memory.read8(address) == 0xABU);
    }
    assert(boundary_core.memory.read8(0x13U) == 0xCCU);

    // A zero-byte operation returns the destination without validating or
    // touching that otherwise-unmapped address.
    boundary_core.state.cia = boundary_memset;
    boundary_core.state.lr = 0x88U;
    boundary_core.state.gpr[3] = 0x20U;
    boundary_core.state.gpr[4] = 0xDEADBEEFU;
    boundary_core.state.gpr[5] = 0U;
    assert(boundary_core.step() == StepResult::executed);
    assert(boundary_core.state.cia == 0x88U);
    assert(boundary_core.state.gpr[3] == 0x20U);
    assert(boundary_core.state.gpr[4] == 0xDEADBEEFU);
    assert(boundary_core.state.gpr[5] == 0U);

    // Invalid nonempty ranges surface through the normal runtime memory-fault
    // result, without changing the return register or mapped neighbor bytes.
    boundary_core.state.cia = boundary_memset;
    boundary_core.state.lr = 0x8CU;
    boundary_core.state.gpr[3] = 0x20U;
    boundary_core.state.gpr[4] = 0x77U;
    boundary_core.state.gpr[5] = 1U;
    const RunResult fault = boundary_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 1 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x00000020") != std::string::npos);
    assert(boundary_core.memory.read8(0x1FU) == 0xCCU);
}

void memcpy_hle_tests()
{
    EspressoCore core(0x20U);
    register_coreinit_hle(core.hle);
    core.memory.map_region(0x20U, 0x20U);
    core.memory.fill_bytes(0U, 0x40U, 0xCCU);
    const std::uint32_t memcpy_import = core.hle.bind_import("coreinit", "memcpy");
    const auto invoke = [&](std::uint32_t destination, std::uint32_t source,
                            std::uint32_t size, std::uint32_t return_address) {
        core.state.cia = memcpy_import;
        core.state.lr = return_address;
        core.state.gpr[3] = destination;
        core.state.gpr[4] = source;
        core.state.gpr[5] = size;
        assert(core.step() == StepResult::executed);
        assert(core.state.cia == return_address);
        assert(core.state.gpr[3] == destination);
        assert(core.state.gpr[4] == source);
        assert(core.state.gpr[5] == size);
    };

    // One-byte and unaligned multi-byte copies leave adjacent bytes alone.
    core.memory.write8(1U, 0x5AU);
    invoke(3U, 1U, 1U, 0x80U);
    assert(core.memory.read8(2U) == 0xCCU);
    assert(core.memory.read8(3U) == 0x5AU);
    assert(core.memory.read8(4U) == 0xCCU);

    const std::array<std::uint8_t, 5> unaligned_source{0x10U, 0x20U, 0x30U, 0x40U, 0x50U};
    core.memory.write_bytes(7U, unaligned_source);
    invoke(0x0DU, 7U, static_cast<std::uint32_t>(unaligned_source.size()), 0x84U);
    std::array<std::uint8_t, 5> unaligned_result{};
    core.memory.read_bytes(0x0DU, unaligned_result);
    assert(unaligned_result == unaligned_source);
    assert(core.memory.read8(0x0CU) == 0xCCU);
    assert(core.memory.read8(0x12U) == 0xCCU);
    std::array<std::uint8_t, 5> source_after_copy{};
    core.memory.read_bytes(7U, source_after_copy);
    assert(source_after_copy == unaligned_source);

    // Source reads can span flat-to-sparse backing storage.
    const std::array<std::uint8_t, 6> sparse_source{1U, 2U, 3U, 4U, 5U, 6U};
    core.memory.write_bytes(0x1DU, sparse_source);
    invoke(2U, 0x1DU, static_cast<std::uint32_t>(sparse_source.size()), 0x88U);
    std::array<std::uint8_t, 6> sparse_source_result{};
    core.memory.read_bytes(2U, sparse_source_result);
    assert(sparse_source_result == sparse_source);
    std::array<std::uint8_t, 6> sparse_source_unchanged{};
    core.memory.read_bytes(0x1DU, sparse_source_unchanged);
    assert(sparse_source_unchanged == sparse_source);

    // Destination writes can independently span the flat-to-sparse boundary.
    const std::array<std::uint8_t, 6> destination_source{0xA1U, 0xA2U, 0xA3U,
                                                        0xA4U, 0xA5U, 0xA6U};
    core.memory.write_bytes(0x13U, destination_source);
    invoke(0x1DU, 0x13U, static_cast<std::uint32_t>(destination_source.size()), 0x8CU);
    std::array<std::uint8_t, 6> cross_boundary_result{};
    core.memory.read_bytes(0x1DU, cross_boundary_result);
    assert(cross_boundary_result == destination_source);
    assert(core.memory.read8(0x1CU) == 0xCCU);
    assert(core.memory.read8(0x23U) == 0xCCU);
    std::array<std::uint8_t, 6> destination_source_unchanged{};
    core.memory.read_bytes(0x13U, destination_source_unchanged);
    assert(destination_source_unchanged == destination_source);

    // A zero-length call returns dst without touching either unmapped address.
    invoke(0x40U, 0xFFFFFFFEU, 0U, 0x90U);

    // Invalid source is diagnosed as a guest read fault and leaves dst intact.
    core.state.cia = memcpy_import;
    core.state.lr = 0x94U;
    core.state.gpr[3] = 8U;
    core.state.gpr[4] = 0x40U;
    core.state.gpr[5] = 2U;
    const std::array<std::uint8_t, 2> destination_before_source_fault{
        core.memory.read8(8U), core.memory.read8(9U)};
    const RunResult source_fault = core.run(1U);
    assert(source_fault.reason == StopReason::memory_fault);
    assert(source_fault.detail.find("read 2 byte(s)") != std::string::npos);
    assert(source_fault.detail.find("0x00000040") != std::string::npos);
    assert(core.memory.read8(8U) == destination_before_source_fault[0]);
    assert(core.memory.read8(9U) == destination_before_source_fault[1]);
    assert(core.state.gpr[3] == 8U);
    assert(core.state.gpr[4] == 0x40U);
    assert(core.state.gpr[5] == 2U);

    // Invalid destination is validated before writes and is reported as write.
    core.state.cia = memcpy_import;
    core.state.lr = 0x98U;
    core.state.gpr[3] = 0x40U;
    core.state.gpr[4] = 0x13U;
    core.state.gpr[5] = 2U;
    const RunResult destination_fault = core.run(1U);
    assert(destination_fault.reason == StopReason::memory_fault);
    assert(destination_fault.detail.find("write 2 byte(s)") != std::string::npos);
    assert(destination_fault.detail.find("0x00000040") != std::string::npos);
    assert(core.memory.read8(0x13U) == destination_source[0]);
    assert(core.memory.read8(0x14U) == destination_source[1]);
    assert(core.state.gpr[3] == 0x40U);
    assert(core.state.gpr[4] == 0x13U);
    assert(core.state.gpr[5] == 2U);
}

void compare_and_conditional_branch_tests()
{
    EspressoCore unsigned_compare_core(8);
    unsigned_compare_core.state.gpr[5] = 4U;
    unsigned_compare_core.memory.write32_be(0, 0x28050003U); // cmplwi cr0, r5, 3
    assert(unsigned_compare_core.step() == StepResult::executed);
    assert(unsigned_compare_core.state.cr == 0x40000000U); // unsigned greater-than

    const auto run_if_else = [](std::uint32_t value) {
        EspressoCore core(0x20);
        // if (r3 == 7) r4 = 1; else r4 = 0;
        core.memory.write32_be(0x00, 0x38600000U | value); // addi r3, r0, value
        core.memory.write32_be(0x04, 0x2C030007U);         // cmpwi r0, r3, 7
        core.memory.write32_be(0x08, 0x4182000CU);         // beq +12 -> then
        core.memory.write32_be(0x0C, 0x38800000U);         // else: addi r4, r0, 0
        core.memory.write32_be(0x10, 0x48000008U);         // b +8 -> end
        core.memory.write32_be(0x14, 0x38800001U);         // then: addi r4, r0, 1

        const RunResult result = core.run(8);
        assert(result.reason == StopReason::unsupported_instruction);
        assert(core.state.cia == 0x18);
        return core.state.gpr[4];
    };

    assert(run_if_else(7) == 1);
    assert(run_if_else(6) == 0);

    // Compare writes only the selected CR field and copies XER[SO].
    EspressoCore compare_core(0x10);
    compare_core.state.gpr[3] = 0xFFFFFFFFU; // signed -1
    compare_core.state.xer = 0x80000000U;
    compare_core.state.cr = 0x12345678U;
    compare_core.memory.write32_be(0, 0x2E830001U); // cmpwi cr5, r3, 1
    assert(compare_core.step() == StepResult::executed);
    assert(compare_core.state.cr == 0x12345978U);

    // The CTR form decrements before testing; BO=16 is bdnz.
    EspressoCore count_core(0x10);
    count_core.state.ctr = 2;
    count_core.memory.write32_be(0, 0x42000004U); // bc 16, 0, +4
    assert(count_core.step() == StepResult::executed);
    assert(count_core.state.ctr == 1);
    assert(count_core.state.cia == 4);
}

void load_store_tests()
{
    EspressoCore core(0x100);

    // Set up a base pointer and a word value, then exercise each requested
    // load/store width against the same big-endian guest memory.
    core.memory.write32_be(0x00, 0x38200080U); // addi r1, r0, 0x80
    core.memory.write32_be(0x04, 0x38601234U); // addi r3, r0, 0x1234
    core.memory.write32_be(0x08, 0x90610000U); // stw r3, 0(r1)
    core.memory.write32_be(0x0C, 0x88810002U); // lbz r4, 2(r1)
    core.memory.write32_be(0x10, 0xA0A10002U); // lhz r5, 2(r1)
    core.memory.write32_be(0x14, 0x80C10000U); // lwz r6, 0(r1)
    core.memory.write32_be(0x18, 0x98610004U); // stb r3, 4(r1)
    core.memory.write32_be(0x1C, 0xB0610006U); // sth r3, 6(r1)
    core.memory.write32_be(0x20, 0x88E10004U); // lbz r7, 4(r1)
    core.memory.write32_be(0x24, 0xA1010006U); // lhz r8, 6(r1)

    const RunResult result = core.run(16);

    assert(result.reason == StopReason::unsupported_instruction);
    assert(result.steps == 10);
    assert(core.state.gpr[1] == 0x80U);
    assert(core.state.gpr[3] == 0x1234U);
    assert(core.memory.read8(0x82) == 0x12U);
    assert(core.state.gpr[4] == 0x12U);
    assert(core.state.gpr[5] == 0x1234U);
    assert(core.state.gpr[6] == 0x1234U);
    assert(core.state.gpr[7] == 0x34U);
    assert(core.state.gpr[8] == 0x1234U);
    assert(core.memory.read32_be(0x80) == 0x00001234U);
    assert(core.memory.read8(0x84) == 0x34U);
    assert(core.memory.read8(0x86) == 0x12U);
    assert(core.memory.read8(0x87) == 0x34U);

    // D-form displacement is signed, and rA=0 supplies a zero base even if
    // the stored contents of GPR0 are nonzero.
    EspressoCore address_core(0x100);
    address_core.state.gpr[1] = 0x54U;
    address_core.memory.write32_be(0x00, 0x8041FFFCU); // lwz r2, -4(r1)
    address_core.memory.write32_be(0x50, 0x89ABCDEFU);
    assert(address_core.step() == StepResult::executed);
    assert(address_core.state.gpr[2] == 0x89ABCDEFU);

    EspressoCore zero_base_core(0x100);
    zero_base_core.state.gpr[0] = 0x20U;
    zero_base_core.memory.write32_be(0x00, 0x81200080U); // lwz r9, 0x80(r0)
    zero_base_core.memory.write32_be(0x80, 0xCAFEBABEU);
    zero_base_core.memory.write32_be(0xA0, 0xDEADBEEFU);
    assert(zero_base_core.step() == StepResult::executed);
    assert(zero_base_core.state.gpr[9] == 0xCAFEBABEU);

    EspressoCore update_core(0x100);
    update_core.state.gpr[28] = 0x40U;
    update_core.memory.write32_be(0, 0x853C0004U); // lwzu r9, 4(r28)
    update_core.memory.write32_be(0x44, 0x12345678U);
    assert(update_core.step() == StepResult::executed);
    assert(update_core.state.gpr[9] == 0x12345678U);
    assert(update_core.state.gpr[28] == 0x44U);

    EspressoCore indexed_core(0x100);
    indexed_core.state.gpr[26] = 0x40U;
    indexed_core.state.gpr[9] = 4U;
    indexed_core.state.gpr[27] = 0xAABBCCDDU;
    indexed_core.memory.write32_be(0, 0x7F7A492EU); // stwx r27, r26, r9
    assert(indexed_core.step() == StepResult::executed);
    assert(indexed_core.memory.read32_be(0x44) == 0xAABBCCDDU);

    EspressoCore indexed_load_core(0x100);
    indexed_load_core.state.gpr[9] = 0x40U;
    indexed_load_core.state.gpr[10] = 4U;
    indexed_load_core.memory.write32_be(0x44, 0x13579BDFU);
    indexed_load_core.memory.write32_be(0, 0x7D09502EU); // lwzx r8, r9, r10
    assert(indexed_load_core.step() == StepResult::executed);
    assert(indexed_load_core.state.gpr[8] == 0x13579BDFU);

    EspressoCore indexed_byte_core(0x100);
    indexed_byte_core.state.gpr[24] = 0x40U;
    indexed_byte_core.state.gpr[3] = 2U;
    indexed_byte_core.state.gpr[9] = 0x123456ABU;
    indexed_byte_core.memory.write32_be(0, 0x7D3819AEU); // stbx r9, r24, r3
    assert(indexed_byte_core.step() == StepResult::executed);
    assert(indexed_byte_core.memory.read8(0x42) == 0xABU);

    EspressoCore byte_update_core(0x100);
    byte_update_core.state.gpr[9] = 0x40U;
    byte_update_core.memory.write32_be(0, 0x8D090001U); // lbzu r8, 1(r9)
    byte_update_core.memory.write8(0x41, 0xABU);
    assert(byte_update_core.step() == StepResult::executed);
    assert(byte_update_core.state.gpr[8] == 0xABU);
    assert(byte_update_core.state.gpr[9] == 0x41U);

    EspressoCore store_byte_update_core(0x100);
    store_byte_update_core.state.gpr[10] = 0x50U;
    store_byte_update_core.state.gpr[7] = 0xABU;
    store_byte_update_core.memory.write32_be(0, 0x9CEA0001U); // stbu r7, 1(r10)
    assert(store_byte_update_core.step() == StepResult::executed);
    assert(store_byte_update_core.memory.read8(0x51) == 0xABU);
    assert(store_byte_update_core.state.gpr[10] == 0x51U);
}

void multiple_word_load_store_tests()
{
    const auto encode_stmw = [](std::uint8_t first, std::uint8_t base,
                                std::int16_t displacement) {
        return 0xBC000000U |
            (static_cast<std::uint32_t>(first) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };
    const auto encode_lmw = [](std::uint8_t first, std::uint8_t base,
                               std::int16_t displacement) {
        return 0xB8000000U |
            (static_cast<std::uint32_t>(first) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    EspressoCore tail_store(0x200U);
    tail_store.state.gpr[1] = 0x80U;
    tail_store.state.gpr[29] = 0x11223344U;
    tail_store.state.gpr[30] = 0x55667788U;
    tail_store.state.gpr[31] = 0x99AABBCCU;
    tail_store.memory.write32_be(0, 0xBFA1000CU); // stmw r29, 12(r1)
    const RunResult tail_store_result = tail_store.run(1U);
    assert(tail_store_result.reason == StopReason::instruction_limit);
    assert(tail_store.state.gpr[1] == 0x80U);
    assert(tail_store.memory.read32_be(0x8CU) == 0x11223344U);
    assert(tail_store.memory.read32_be(0x90U) == 0x55667788U);
    assert(tail_store.memory.read32_be(0x94U) == 0x99AABBCCU);
    assert(tail_store.memory.read8(0x8CU) == 0x11U);
    const std::string tail_store_trace = format_instruction_history(tail_store_result);
    assert(tail_store_trace.find("stmw r29-r31, 12(r1) [0x0000008C]") !=
           std::string::npos);

    // Starting at r16 stores all sixteen remaining GPRs using a signed offset.
    EspressoCore full_store(0x200U);
    full_store.state.gpr[5] = 0x90U;
    for (std::uint32_t reg = 16U; reg <= 31U; ++reg)
    {
        full_store.state.gpr[reg] = 0xA0000000U + reg;
    }
    full_store.memory.write32_be(0, encode_stmw(16U, 5U, -0x10));
    assert(full_store.step() == StepResult::executed);
    assert(full_store.state.gpr[5] == 0x90U);
    for (std::uint32_t reg = 16U; reg <= 31U; ++reg)
    {
        assert(full_store.memory.read32_be(0x80U + (reg - 16U) * 4U) ==
               0xA0000000U + reg);
    }

    // rA=0 supplies a zero base even when the architectural GPR0 has data.
    EspressoCore zero_base_store(0x100U);
    zero_base_store.state.gpr[0] = 0x70U;
    zero_base_store.state.gpr[29] = 0x01020304U;
    zero_base_store.state.gpr[30] = 0x05060708U;
    zero_base_store.state.gpr[31] = 0x090A0B0CU;
    zero_base_store.memory.write32_be(0, encode_stmw(29U, 0U, 0x40));
    assert(zero_base_store.step() == StepResult::executed);
    assert(zero_base_store.state.gpr[0] == 0x70U);
    assert(zero_base_store.memory.read32_be(0x40U) == 0x01020304U);
    assert(zero_base_store.memory.read32_be(0x44U) == 0x05060708U);
    assert(zero_base_store.memory.read32_be(0x48U) == 0x090A0B0CU);

    // Store, clobber, and load the same register range back.
    EspressoCore round_trip(0x200U);
    round_trip.state.gpr[1] = 0x80U;
    round_trip.state.gpr[29] = 0xDEADBEEFU;
    round_trip.state.gpr[30] = 0x12345678U;
    round_trip.state.gpr[31] = 0xCAFEBABEU;
    round_trip.memory.write32_be(0, encode_stmw(29U, 1U, 0x10));
    round_trip.memory.write32_be(4, encode_lmw(29U, 1U, 0x10));
    assert(round_trip.step() == StepResult::executed);
    round_trip.state.gpr[29] = 0;
    round_trip.state.gpr[30] = 0;
    round_trip.state.gpr[31] = 0;
    round_trip.state.cia = 4U;
    assert(round_trip.step() == StepResult::executed);
    assert(round_trip.state.gpr[29] == 0xDEADBEEFU);
    assert(round_trip.state.gpr[30] == 0x12345678U);
    assert(round_trip.state.gpr[31] == 0xCAFEBABEU);
    assert(round_trip.state.gpr[1] == 0x80U);
    const RunResult round_trip_result = round_trip.run(0U);
    const std::string round_trip_trace = format_instruction_history(round_trip_result);
    assert(round_trip_trace.find("stmw r29-r31, 16(r1) [0x00000090]") !=
           std::string::npos);
    assert(round_trip_trace.find("lmw r29-r31, 16(r1) [0x00000090]") !=
           std::string::npos);

    // The first transfer commits before a later word faults; the failing word
    // is diagnosed at its own address and is not swallowed.
    EspressoCore store_fault(0x20U);
    store_fault.state.gpr[1] = 0x1CU;
    store_fault.state.gpr[30] = 0xAABBCCDDU;
    store_fault.state.gpr[31] = 0x11223344U;
    store_fault.memory.write32_be(0, encode_stmw(30U, 1U, 0));
    const RunResult store_fault_result = store_fault.run(1U);
    assert(store_fault_result.reason == StopReason::memory_fault);
    assert(store_fault_result.detail.find("write 4 byte(s)") != std::string::npos);
    assert(store_fault_result.detail.find("0x00000020") != std::string::npos);
    assert(store_fault.memory.read32_be(0x1CU) == 0xAABBCCDDU);
    assert(!store_fault_result.instruction_history.back().completed);

    EspressoCore load_fault(0x20U);
    load_fault.state.gpr[1] = 0x1CU;
    load_fault.state.gpr[30] = 0;
    load_fault.state.gpr[31] = 0xFFFFFFFFU;
    load_fault.memory.write32_be(0, encode_lmw(30U, 1U, 0));
    load_fault.memory.write32_be(0x1CU, 0x76543210U);
    const RunResult load_fault_result = load_fault.run(1U);
    assert(load_fault_result.reason == StopReason::memory_fault);
    assert(load_fault_result.detail.find("read 4 byte(s)") != std::string::npos);
    assert(load_fault_result.detail.find("0x00000020") != std::string::npos);
    assert(load_fault.state.gpr[30] == 0x76543210U);
    assert(load_fault.state.gpr[31] == 0xFFFFFFFFU);
    assert(!load_fault_result.instruction_history.back().completed);
}

void floating_point_load_tests()
{
    const auto encode_lfs = [](std::uint8_t destination, std::uint8_t base,
                               std::int16_t displacement) {
        return 0xC0000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    EspressoCore core(0x200U);
    core.state.gpr[12] = 0x100U;
    core.memory.write8(0x104U, 0x3FU);
    core.memory.write8(0x105U, 0xC0U);
    core.memory.write8(0x106U, 0x00U);
    core.memory.write8(0x107U, 0x00U); // Big-endian 1.5f.
    core.memory.write32_be(0, 0xC1AC0004U);
    const RunResult positive_result = core.run(1);
    assert(positive_result.reason == StopReason::instruction_limit);
    assert(core.state.fpr[13] == 0x3FF8000000000000ULL);
    assert(core.state.fpr[12] == 0U);
    const std::string positive_trace = format_instruction_history(positive_result);
    assert(positive_trace.find(
        "lfs r12=0x00000100 [0x00000104] -> f13=1.5 [0x3FF8000000000000]") !=
        std::string::npos);

    // A negative single value and negative D-form displacement.
    core.state.cia = 4U;
    core.state.gpr[12] = 0x120U;
    core.memory.write32_be(0x11CU, 0xC0100000U); // -2.25f
    core.memory.write32_be(4, encode_lfs(7, 12, -4));
    assert(core.step() == StepResult::executed);
    assert(core.state.fpr[7] == 0xC002000000000000ULL);

    // rA == 0 uses address zero as the base, and +0 remains exactly zero.
    core.state.cia = 8U;
    core.memory.write32_be(0x80U, 0U);
    core.memory.write32_be(8, encode_lfs(2, 0, 0x80));
    assert(core.step() == StepResult::executed);
    assert(core.state.fpr[2] == 0U);
    assert(core.state.fpr[13] == 0x3FF8000000000000ULL);

    // Reset clears all architectural FPR bits deterministically.
    core.state.fpr[31] = 0xFFFFFFFFFFFFFFFFULL;
    core.state.reset();
    for (const std::uint64_t value : core.state.fpr)
    {
        assert(value == 0U);
    }

    EspressoCore fault_core(0x100U);
    fault_core.memory.write32_be(0, 0xC1AC0004U);
    fault_core.state.gpr[12] = 0x1000U;
    const RunResult fault = fault_core.run(1);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x00001004") != std::string::npos);
    assert(fault.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed_lfs = fault.instruction_history.front();
    assert(failed_lfs.opcode_name == "lfs");
    assert(failed_lfs.has_effective_address);
    assert(failed_lfs.effective_address == 0x1004U);
    assert(failed_lfs.has_fp_destination);
    assert(failed_lfs.fp_destination_register == 13U);
    assert(!failed_lfs.completed);
    const std::string trace = format_instruction_history(fault);
    assert(trace.find(
        "lfs r12=0x00001000 [0x00001004] -> f13 (not written)") != std::string::npos);
}

void floating_point_store_tests()
{
    const auto encode_stfs = [](std::uint8_t source, std::uint8_t base,
                                std::int16_t displacement) {
        return 0xD0000000U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    EspressoCore positive_core(0x200U);
    positive_core.state.gpr[4] = 0x100U;
    positive_core.state.fpr[13] = 0x400A000000000000ULL; // 3.25
    positive_core.memory.write32_be(0, encode_stfs(13, 4, 4));
    const RunResult positive_result = positive_core.run(1);
    assert(positive_result.reason == StopReason::instruction_limit);
    assert(positive_core.memory.read32_be(0x104U) == 0x40500000U);
    assert(positive_core.memory.read8(0x104U) == 0x40U);
    assert(positive_core.memory.read8(0x105U) == 0x50U);
    assert(positive_core.memory.read8(0x106U) == 0x00U);
    assert(positive_core.memory.read8(0x107U) == 0x00U);
    assert(positive_core.state.fpr[13] == 0x400A000000000000ULL);
    const std::string positive_trace = format_instruction_history(positive_result);
    assert(positive_trace.find(
        "stfs r4=0x00000100 f13=3.25 [0x400A000000000000] "
        "[0x00000104] -> mem32=0x40500000") != std::string::npos);

    // Negative D-form displacement and negative finite source.
    EspressoCore negative_core(0x200U);
    negative_core.state.gpr[12] = 0x120U;
    negative_core.state.fpr[7] = 0xC002000000000000ULL; // -2.25
    negative_core.memory.write32_be(0, encode_stfs(7, 12, -4));
    assert(negative_core.step() == StepResult::executed);
    assert(negative_core.memory.read32_be(0x11CU) == 0xC0100000U);
    assert(negative_core.state.fpr[7] == 0xC002000000000000ULL);

    // rA == 0 addresses from zero; positive zero stores four zero bytes.
    EspressoCore zero_core(0x100U);
    zero_core.state.fpr[2] = 0U;
    zero_core.memory.write32_be(0, encode_stfs(2, 0, 0x80));
    assert(zero_core.step() == StepResult::executed);
    assert(zero_core.memory.read32_be(0x80U) == 0U);
    assert(zero_core.state.fpr[2] == 0U);

    // Loading a single and storing it again preserves the original -pi bits.
    EspressoCore round_trip_core(0x200U);
    round_trip_core.state.gpr[12] = 0x100U;
    round_trip_core.memory.write32_be(0, 0xC1AC0004U); // lfs f13, 4(r12)
    round_trip_core.memory.write32_be(4, encode_stfs(13, 12, -4));
    round_trip_core.memory.write32_be(0x104U, 0xC0490FDBU);
    const RunResult round_trip_result = round_trip_core.run(2);
    assert(round_trip_result.reason == StopReason::instruction_limit);
    assert(round_trip_core.memory.read32_be(0xFCU) == 0xC0490FDBU);
    assert(round_trip_core.state.fpr[13] == 0xC00921FB60000000ULL);

    // A failed write reports the address/width and retains source FPR details.
    EspressoCore fault_core(0x100U);
    fault_core.state.gpr[12] = 0x1000U;
    fault_core.state.fpr[13] = 0xC00921FB60000000ULL;
    fault_core.memory.write32_be(0, 0xD1AC0000U); // stfs f13, 0(r12)
    const RunResult fault = fault_core.run(1);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x00001000") != std::string::npos);
    assert(fault.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed_stfs = fault.instruction_history.front();
    assert(failed_stfs.opcode_name == "stfs");
    assert(failed_stfs.has_fp_source);
    assert(failed_stfs.fp_source_register == 13U);
    assert(failed_stfs.fp_source_value == 0xC00921FB60000000ULL);
    assert(!failed_stfs.completed);
    assert(!failed_stfs.has_stored_single_value);
    assert(fault_core.state.fpr[13] == 0xC00921FB60000000ULL);
}

void function_call_and_stack_tests()
{
    EspressoCore link_branch_core(0x20);
    link_branch_core.state.lr = 0x12U;
    link_branch_core.memory.write32_be(0, 0x4E800021U); // blrl
    assert(link_branch_core.step() == StepResult::executed);
    assert(link_branch_core.state.cia == 0x10U); // old LR target, aligned
    assert(link_branch_core.state.lr == 0x04U); // new link address

    EspressoCore conditional_link_branch_core(0x20);
    conditional_link_branch_core.state.lr = 0x10U;
    conditional_link_branch_core.state.cr = 0x20000000U; // CR0.EQ
    conditional_link_branch_core.memory.write32_be(0, 0x4D820020U); // beqlr
    assert(conditional_link_branch_core.step() == StepResult::executed);
    assert(conditional_link_branch_core.state.cia == 0x10U);

    EspressoCore core(0x100);

    core.memory.write32_be(0x00, 0x382000F0U); // addi r1, r0, 0xF0 (initial SP)
    core.memory.write32_be(0x04, 0x4800001DU); // bl +0x1C -> function at 0x20
    core.memory.write32_be(0x08, 0x38830000U); // after return: addi r4, r3, 0

    core.memory.write32_be(0x20, 0x7C0802A6U); // mflr r0
    core.memory.write32_be(0x24, 0x9421FFF0U); // stwu r1, -16(r1)
    core.memory.write32_be(0x28, 0x9001000CU); // stw r0, 12(r1): save LR
    core.memory.write32_be(0x2C, 0x3860002AU); // addi r3, r0, 42 (return value)
    core.memory.write32_be(0x30, 0x8001000CU); // lwz r0, 12(r1)
    core.memory.write32_be(0x34, 0x7C0803A6U); // mtlr r0
    core.memory.write32_be(0x38, 0x38210010U); // addi r1, r1, 16 (restore SP)
    core.memory.write32_be(0x3C, 0x4E800020U); // blr

    const RunResult result = core.run(16);

    assert(result.steps == 11);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.cia == 0x0CU);
    assert(core.state.gpr[1] == 0xF0U);
    assert(core.state.gpr[3] == 42U);
    assert(core.state.gpr[4] == 42U);
    assert(core.state.lr == 0x08U);
    assert(core.memory.read32_be(0xE0) == 0xF0U); // stwu saved the old SP
    assert(core.memory.read32_be(0xEC) == 0x08U); // function saved return LR
}

}

int main(int argc, char* argv[])
{
    if (argc == 2)
    {
        std::ifstream input(argv[1], std::ios::binary);
        if (!input)
        {
            std::cerr << "could not open RPX: " << argv[1] << '\n';
            return 1;
        }
        const std::vector<char> file_bytes{
            std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
        std::vector<std::uint8_t> file(file_bytes.begin(), file_bytes.end());

        try
        {
            affogato::Emulator emulator;
            const auto image = emulator.load_rpx(file);
            std::cout << "Loaded RPX entry point 0x" << std::hex << image.entry_point
                      << " (" << std::dec << image.loaded_sections << " sections)\n";
            const auto session_result = emulator.run(1'000'000);
            const auto& execution = session_result.execution;
            std::cout << "Stopped after " << execution.steps << " instructions at CIA 0x"
                      << std::hex << execution.cia << std::dec << ": ";
            switch (execution.reason)
            {
            case StopReason::instruction_limit:
                std::cout << "instruction limit";
                break;
            case StopReason::unsupported_instruction:
                std::cout << "unsupported PPC instruction 0x" << std::hex
                          << execution.instruction_word << std::dec;
                break;
            case StopReason::unimplemented_hle_call:
                std::cout << "unimplemented HLE " << execution.hle_call;
                break;
            case StopReason::hle_error:
                std::cout << "HLE error: " << execution.detail;
                break;
            case StopReason::memory_fault:
                std::cout << "guest memory fault: " << execution.detail;
                break;
            }
            std::cout << " (guest r3=" << session_result.gpr3 << ")\n";
            std::cout << affogato::cpu::espresso::format_instruction_history(execution) << '\n';
        }
        catch (const std::exception& error)
        {
            std::cerr << "RPX load failed: " << error.what() << '\n';
            return 1;
        }
        return 0;
    }

    affogato::tests::cpu_state_tests();
    decoder_tests();
    guest_memory_tests();
    interpreter_tests();
    integer_alu_tests();
    leaf_function_abi_tests();
    floating_point_load_tests();
    floating_point_store_tests();
    elf_loader_tests();
    rpx_loader_tests();
    hle_dispatch_tests();
    guest_mutex_tests();
    memset_hle_tests();
    memcpy_hle_tests();
    compare_and_conditional_branch_tests();
    load_store_tests();
    multiple_word_load_store_tests();
    function_call_and_stack_tests();
    return 0;
}
