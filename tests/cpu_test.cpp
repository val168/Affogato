#include "cpu/espresso/decoder.hpp"
#include "cpu/espresso/cafe_os_hle.hpp"
#include "cpu/espresso/elf_loader.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/guest_mutex.hpp"
#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/nn_olv_hle.hpp"
#include "cpu/espresso/nn_save_hle.hpp"
#include "cpu/espresso/rpx_loader.hpp"
#include "cpu_state_test.hpp"
#include "emulator.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
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

struct RpxImportOptions
{
    std::string library{"coreinit"};
    std::uint32_t section_address{0xC0000000U};
    std::uint32_t slot_offset{20U};
};

[[nodiscard]] std::vector<std::uint8_t> make_minimal_compressed_rpx(
    std::uint32_t code_address = 0x02000000U,
    std::size_t code_size = 16,
    const RpxRelocationSymbol& relocation_symbol = {},
    const RpxRelocationOptions& relocation_options = {},
    const std::optional<RpxImportOptions>& import_options = std::nullopt)
{
    constexpr std::size_t header_size = 52;
    constexpr std::size_t section_header_size = 40;
    const std::size_t section_count = import_options ? 9U : 8U;
    constexpr std::size_t section_table_offset = header_size;
    const std::size_t text_offset =
        (header_size + section_count * section_header_size + 15U) & ~std::size_t{15U};
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

    const std::array<std::uint8_t, 68> base_names{
        '\0', '.', 't', 'e', 'x', 't', '\0', '.', 's', 'h', 's', 't', 'r', 't', 'a', 'b', '\0',
        '.', 'r', 'p', 'l', '_', 'c', 'r', 'c', 's', '\0', '.', 'r', 'p', 'l', '_', 'f', 'i', 'l', 'e', 'i', 'n', 'f', 'o', '\0',
        '.', 's', 'y', 'm', 't', 'a', 'b', '\0', '.', 's', 't', 'r', 't', 'a', 'b', '\0',
        '.', 'r', 'e', 'l', 'a', '.', 't', 'e', 'x', 't', '\0'};
    std::vector<std::uint8_t> names(base_names.begin(), base_names.end());
    constexpr std::string_view import_section_name = ".dimport_coreinit";
    const std::uint32_t import_section_name_offset = static_cast<std::uint32_t>(names.size());
    names.insert(names.end(), import_section_name.begin(), import_section_name.end());
    names.push_back(0);
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
    constexpr std::size_t import_table_size = 24;
    const std::size_t import_offset = rela_offset + relocation_table_size;
    std::vector<std::uint8_t> file(
        import_offset + (import_options ? import_table_size : 0U), 0);

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
    if (import_options)
    {
        if (import_options->slot_offset > import_table_size - sizeof(std::uint32_t) ||
            import_options->slot_offset < 8U + import_options->library.size() + 1U)
        {
            throw std::invalid_argument("synthetic RPX import slot overlaps its library name");
        }
        set_section(8, import_section_name_offset, 0x80000002U, 0x2U,
                    import_options->section_address, static_cast<std::uint32_t>(import_offset),
                    static_cast<std::uint32_t>(import_table_size), 4, 0);
        const std::size_t library_name_offset = import_offset + 8U;
        std::copy(import_options->library.begin(), import_options->library.end(),
                  file.begin() + static_cast<std::ptrdiff_t>(library_name_offset));
        const std::size_t symbol_offset = symtab_offset + 16;
        set_be32(file, symbol_offset + 4,
                 import_options->section_address + import_options->slot_offset);
        set_be16(file, symbol_offset + 14, 8U);
    }
    else
    {
        set_section(8, import_section_name_offset, 0, 0, 0, 0, 0, 1, 0);
    }

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
    const DecodedInstruction cntlzw = decode(0x7D8C0034U); // cntlzw r12, r12
    assert(cntlzw.opcode == Opcode::count_leading_zeros);
    assert(cntlzw.source == 12U);
    assert(cntlzw.destination == 12U);
    assert(!cntlzw.record);

    const DecodedInstruction cntlzw_dot = decode(0x7D8C0035U); // cntlzw. r12, r12
    assert(cntlzw_dot.opcode == Opcode::count_leading_zeros);
    assert(cntlzw_dot.record);

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

    const DecodedInstruction addc = decode(0x7C845814U); // addc r4, r4, r11
    assert(addc.opcode == Opcode::add_carrying);
    assert(addc.destination == 4U);
    assert(addc.base == 4U);
    assert(addc.source == 11U);
    assert(((0x7C845814U >> 1U) & 0x3FFU) == 10U);
    assert(!addc.record);
    const DecodedInstruction addc_record = decode(0x7C845815U);
    assert(addc_record.opcode == Opcode::add_carrying);
    assert(addc_record.record);
    const std::uint32_t addco = (0x7C845814U & ~(0x3FFU << 1U)) | (522U << 1U);
    assert(decode(addco).opcode == Opcode::unsupported);

    const DecodedInstruction subf = decode(0x7CC42850U); // subf r6, r4, r5
    assert(subf.opcode == Opcode::subtract_from);
    assert(subf.destination == 6);
    assert(subf.base == 4);
    assert(subf.source == 5);

    const DecodedInstruction subfc = decode(0x7C6C5810U); // subfc r3, r12, r11
    assert(subfc.opcode == Opcode::subtract_from_carrying);
    assert(subfc.destination == 3);
    assert(subfc.base == 12);
    assert(subfc.source == 11);
    assert(!subfc.record);
    assert(decode(0x7C6C5C10U).opcode == Opcode::unsupported); // subfco remains unsupported

    const DecodedInstruction subfe = decode(0x7D8C6110U); // subfe r12, r12, r12
    assert(subfe.opcode == Opcode::subtract_from_extended);
    assert(subfe.destination == 12U);
    assert(subfe.base == 12U);
    assert(subfe.source == 12U);
    assert(((0x7D8C6110U >> 1U) & 0x3FFU) == 136U);
    assert(!subfe.record);
    assert(decode(0x7D8C6111U).opcode == Opcode::subtract_from_extended);
    assert(decode(0x7D8C6111U).record);
    assert(decode((0x7D8C6110U & ~(0x3FFU << 1U)) | (648U << 1U)).opcode ==
           Opcode::unsupported); // subfeo remains unsupported

    const DecodedInstruction mullw = decode(0x7FDFF1D6U); // mullw r30, r31, r30
    assert(mullw.opcode == Opcode::multiply_low_word);
    assert(mullw.destination == 30);
    assert(mullw.base == 31);
    assert(mullw.source == 30);
    assert(!mullw.record);
    assert(decode(0x7FDFF5D6U).opcode == Opcode::unsupported); // mullwo remains unsupported

    const DecodedInstruction neg = decode(0x7D6A00D0U); // neg r11, r10
    assert(neg.opcode == Opcode::negate);
    assert(neg.destination == 11);
    assert(neg.source == 10);
    assert(!neg.record);
    assert(decode(0x7D6A04D0U).opcode == Opcode::unsupported); // nego remains unsupported

    const DecodedInstruction ori = decode(0x60631234U); // ori r3, r3, 0x1234
    assert(ori.opcode == Opcode::ori);
    assert(ori.source == 3);
    assert(ori.destination == 3);
    assert(ori.immediate == 0x1234);

    const DecodedInstruction oris = decode(0x67FFFFFFU); // oris r31, r31, 0xFFFF
    assert(oris.opcode == Opcode::or_immediate_shifted);
    assert(oris.source == 31U);
    assert(oris.destination == 31U);
    assert(oris.immediate == 0xFFFF);

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

    const DecodedInstruction rlwimi = decode(0x53CC3632U);
    assert(rlwimi.opcode == Opcode::rotate_left_word_and_mask_insert);
    assert(rlwimi.source == 30);
    assert(rlwimi.destination == 12);
    assert(rlwimi.shift == 6);
    assert(rlwimi.mask_begin == 24);
    assert(rlwimi.mask_end == 25);
    assert(!rlwimi.record);

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

    const DecodedInstruction isync = decode(0x4C00012CU);
    assert(isync.opcode == Opcode::instruction_sync);
    assert(decode(0x4C00092CU).opcode == Opcode::unsupported);
    assert(decode(0x4C00012DU).opcode == Opcode::unsupported);

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

    const DecodedInstruction fcmpu = decode(0xFC01F800U);
    assert(fcmpu.opcode == Opcode::floating_compare_unordered);
    assert(fcmpu.cr_field == 0U);
    assert(fcmpu.fp_compare_a == 1U);
    assert(fcmpu.fp_compare_b == 31U);
    assert(decode(0xFC41F800U).opcode == Opcode::unsupported);
    assert(decode(0xFC01F801U).opcode == Opcode::unsupported);

    const DecodedInstruction fdivs = decode(0xEC016824U);
    assert(fdivs.opcode == Opcode::floating_divide_single);
    assert(fdivs.fp_register == 0U);
    assert(fdivs.fp_source_a == 1U);
    assert(fdivs.fp_source_b == 13U);
    assert(((0xEC016824U >> 1U) & 0x1FU) == 18U);
    assert(((0xEC016824U >> 6U) & 0x1FU) == 0U);
    assert((0xEC016824U & 1U) == 0U);
    assert(decode(0xEC016864U).opcode == Opcode::unsupported);
    assert(decode(0xEC016825U).opcode == Opcode::unsupported);

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

    const DecodedInstruction lbzx = decode(0x7D6900AEU); // lbzx r11, r9, r0
    assert(lbzx.opcode == Opcode::load_byte_zero_indexed);
    assert(lbzx.destination == 11U);
    assert(lbzx.base == 9U);
    assert(lbzx.source == 0U);
    assert(((0x7D6900AEU >> 1U) & 0x3FFU) == 87U);
    assert(decode(0x7D6900AFU).opcode == Opcode::unsupported);

    const DecodedInstruction lhzx = decode(0x7D7B422EU); // lhzx r11, r27, r8
    assert(lhzx.opcode == Opcode::load_halfword_zero_indexed);
    assert(lhzx.destination == 11U);
    assert(lhzx.base == 27U);
    assert(lhzx.source == 8U);
    assert(((0x7D7B422EU >> 1U) & 0x3FFU) == 279U);
    assert(decode(0x7D7B422FU).opcode == Opcode::unsupported);
    const DecodedInstruction lhax = decode(0x7D7B42AEU); // lhax r11, r27, r8
    assert(lhax.opcode == Opcode::load_halfword_algebraic_indexed);
    assert(lhax.destination == 11U && lhax.base == 27U && lhax.source == 8U);
    assert(((0x7D7B42AEU >> 1U) & 0x3FFU) == 343U);
    assert(decode(0x7D7B42AFU).opcode == Opcode::unsupported);

    const DecodedInstruction stw = decode(0x90610000U); // stw r3, 0(r1)
    assert(stw.opcode == Opcode::store_word);
    assert(stw.destination == 3);

    const DecodedInstruction stwx = decode(0x7F7A492EU); // stwx r27, r26, r9
    assert(stwx.opcode == Opcode::store_word_indexed);
    assert(stwx.destination == 27);
    assert(stwx.base == 26);
    assert(stwx.source == 9);

    const DecodedInstruction sthx = decode(0x7FA7DB2EU); // sthx r29, r7, r27
    assert(sthx.opcode == Opcode::store_halfword_indexed);
    assert(sthx.destination == 29U);
    assert(sthx.base == 7U);
    assert(sthx.source == 27U);
    assert(((0x7FA7DB2EU >> 1U) & 0x3FFU) == 407U);
    assert(decode(0x7FA7DB2FU).opcode == Opcode::unsupported);

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

    const DecodedInstruction lha = decode(0xA98A0000U); // lha r12, 0(r10)
    assert(lha.opcode == Opcode::load_halfword_algebraic);
    assert(lha.destination == 12);
    assert(lha.base == 10);
    assert(lha.immediate == 0);

    const DecodedInstruction sth = decode(0xB0610006U); // sth r3, 6(r1)
    assert(sth.opcode == Opcode::store_halfword);
    assert(sth.destination == 3);

    const DecodedInstruction sthu = decode(0xB7E80002U); // sthu r31, 2(r8)
    assert(sthu.opcode == Opcode::store_halfword_update);
    assert(sthu.destination == 31U);
    assert(sthu.base == 8U);
    assert(sthu.immediate == 2);
    assert(decode(0xB7E8FFFEU).immediate == -2);
    assert(decode(0xB7E00002U).opcode == Opcode::unsupported); // update RA=0 is invalid

    const DecodedInstruction lfs = decode(0xC1AC0004U); // lfs f13, 4(r12)
    assert(lfs.opcode == Opcode::load_single);
    assert(lfs.fp_register == 13);
    assert(lfs.base == 12);
    assert(lfs.immediate == 4);

    const DecodedInstruction lfsu = decode(0xC409FBA8U); // lfsu f0, -1112(r9)
    assert(lfsu.opcode == Opcode::load_single_update);
    assert(lfsu.fp_register == 0U);
    assert(lfsu.base == 9U);
    assert(lfsu.immediate == -1112);
    assert(decode(0xC4000000U).opcode == Opcode::unsupported);

    const DecodedInstruction lfd = decode(0xCBE10010U); // lfd f31, 16(r1)
    assert(lfd.opcode == Opcode::load_double);
    assert(lfd.fp_register == 31U);
    assert(lfd.base == 1U);
    assert(lfd.immediate == 16);

    const DecodedInstruction stfs = decode(0xD1ACF060U); // stfs f13, -4000(r12)
    assert(stfs.opcode == Opcode::store_single);
    assert(stfs.fp_register == 13);
    assert(stfs.base == 12);
    assert(stfs.immediate == -4000);

    const DecodedInstruction stfsu = decode(0xD5ABF504U); // stfsu f13, -2812(r11)
    assert(stfsu.opcode == Opcode::store_single_update);
    assert(stfsu.fp_register == 13);
    assert(stfsu.base == 11);
    assert(stfsu.immediate == -2812);
    assert(decode(0xD4000000U).opcode == Opcode::unsupported); // stfsu with rA=0

    const DecodedInstruction stfd = decode(0xDBE10010U); // stfd f31, 16(r1)
    assert(stfd.opcode == Opcode::store_double);
    assert(stfd.fp_register == 31U);
    assert(stfd.base == 1U);
    assert(stfd.immediate == 16);

    const DecodedInstruction stfdu = decode(0xDDA904E8U); // stfdu f13, 1256(r9)
    assert(stfdu.opcode == Opcode::store_double_update);
    assert(stfdu.fp_register == 13U);
    assert(stfdu.base == 9U);
    assert(stfdu.immediate == 1256);
    assert(decode(0xDC000040U).opcode == Opcode::unsupported); // update rA=0 is invalid

    const DecodedInstruction ps_merge10 = decode(0x13FFFCA0U);
    assert(ps_merge10.opcode == Opcode::paired_single_merge10);
    assert(ps_merge10.fp_register == 31U);
    assert(ps_merge10.fp_source_a == 31U);
    assert(ps_merge10.fp_source_b == 31U);
    assert(((0x13FFFCA0U >> 1U) & 0x3FFU) == 592U);
    assert((0x13FFFCA0U & 1U) == 0U);
    assert(decode(0x13FFFCA1U).opcode == Opcode::unsupported);

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

    GuestMemory double_boundary_memory(8U);
    double_boundary_memory.map_region(8U, 8U);
    double_boundary_memory.write64_be(4U, 0x0123456789ABCDEFULL);
    const std::array<std::uint8_t, 8> expected_double_bytes{
        0x01U, 0x23U, 0x45U, 0x67U, 0x89U, 0xABU, 0xCDU, 0xEFU};
    std::array<std::uint8_t, 8> actual_double_bytes{};
    double_boundary_memory.read_bytes(4U, actual_double_bytes);
    assert(actual_double_bytes == expected_double_bytes);
    assert(double_boundary_memory.read64_be(4U) == 0x0123456789ABCDEFULL);

    GuestMemory failed_double_memory(4U);
    failed_double_memory.fill_bytes(0U, 4U, 0xA5U);
    try
    {
        failed_double_memory.write64_be(0U, 0x0123456789ABCDEFULL);
        assert(false);
    }
    catch (const GuestMemoryFault& fault)
    {
        assert(fault.address() == 0U);
        assert(fault.width() == 8U);
        assert(fault.access() == GuestMemoryAccess::write);
    }
    for (std::uint32_t address = 0; address < 4U; ++address)
    {
        assert(failed_double_memory.read8(address) == 0xA5U);
    }
    try
    {
        static_cast<void>(failed_double_memory.read64_be(0U));
        assert(false);
    }
    catch (const GuestMemoryFault& fault)
    {
        assert(fault.address() == 0U);
        assert(fault.width() == 8U);
        assert(fault.access() == GuestMemoryAccess::read);
    }

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

void instruction_sync_tests()
{
    EspressoCore core(0x100U);
    for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
    {
        core.state.gpr[index] = static_cast<std::uint32_t>(0x10203040U + index);
        core.state.fpr[index] = 0x1111000000000000ULL + index;
        core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.lr = 0x10203040U;
    core.state.ctr = 0x50607080U;
    core.memory.write32_be(0U, 0x4C00012CU);

    const auto gpr_before = core.state.gpr;
    const auto fpr_ps0_before = core.state.fpr;
    const auto fpr_ps1_before = core.state.fpr_ps1;
    constexpr std::uint32_t expected_cr = 0x12345678U;
    constexpr std::uint32_t expected_xer = 0xA00000A5U;
    constexpr std::uint32_t expected_lr = 0x10203040U;
    constexpr std::uint32_t expected_ctr = 0x50607080U;
    std::array<std::uint8_t, 0x100U> memory_before{};
    core.memory.read_bytes(0U, memory_before);

    const RunResult result = core.run(1U);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.cia == 4U);
    assert(core.state.gpr == gpr_before);
    assert(core.state.fpr == fpr_ps0_before);
    assert(core.state.fpr_ps1 == fpr_ps1_before);
    assert(core.state.cr == expected_cr);
    assert(core.state.xer == expected_xer);
    assert(core.state.lr == expected_lr);
    assert(core.state.ctr == expected_ctr);
    std::array<std::uint8_t, 0x100U> memory_after{};
    core.memory.read_bytes(0U, memory_after);
    assert(memory_after == memory_before);
    assert(result.instruction_history.size() == 1U);
    assert(result.instruction_history.front().opcode_name == "isync");
    assert(format_instruction_history(result).find("isync") !=
           std::string::npos);
}

void floating_compare_unordered_tests()
{
    const auto encode_fcmpu = [](std::uint8_t cr_field, std::uint8_t fr_a,
                                 std::uint8_t fr_b) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(cr_field) << 23U) |
            (static_cast<std::uint32_t>(fr_a) << 16U) |
            (static_cast<std::uint32_t>(fr_b) << 11U);
    };
    struct ComparisonCase
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint8_t result;
        std::uint8_t cr_field;
    };
    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t positive_infinity = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_infinity = 0xFFF0000000000000ULL;
    const std::array<ComparisonCase, 7> cases{{
        {one, two, 0x8U, 0U},
        {two, one, 0x4U, 3U},
        {one, one, 0x2U, 0U},
        {positive_zero, negative_zero, 0x2U, 3U},
        {positive_infinity, one, 0x4U, 0U},
        {negative_infinity, one, 0x8U, 3U},
        {positive_infinity, positive_infinity, 0x2U, 0U},
    }};

    for (const ComparisonCase& test : cases)
    {
        EspressoCore core(16U);
        core.state.fpr[1] = test.a;
        core.state.fpr[31] = test.b;
        // Deliberately contradictory paired lanes prove fcmpu observes PS0 only.
        core.state.fpr_ps1[1] = negative_infinity;
        core.state.fpr_ps1[31] = positive_infinity;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x80000000U; // SO must not leak into the floating result.
        core.state.lr = 0x10203040U;
        core.state.ctr = 0x50607080U;
        core.state.fpscr = 0x0000A0A5U;
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const std::uint32_t old_cr = core.state.cr;
        const std::uint32_t old_xer = core.state.xer;
        core.memory.write32_be(0U, encode_fcmpu(test.cr_field, 1U, 31U));
        std::array<std::uint8_t, 16U> memory_before{};
        core.memory.read_bytes(0U, memory_before);

        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        const unsigned shift = (7U - test.cr_field) * 4U;
        const std::uint32_t cr_mask = 0xFU << shift;
        assert(core.state.cr == ((old_cr & ~cr_mask) |
               (static_cast<std::uint32_t>(test.result) << shift)));
        constexpr std::uint32_t fpcc_mask = 0x0000F000U;
        assert(core.state.fpscr == ((0x0000A0A5U & ~fpcc_mask) |
               (static_cast<std::uint32_t>(test.result) << 12U)));
        assert(core.state.gpr == gpr_before);
        assert(core.state.fpr == fpr_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        assert(core.state.xer == old_xer);
        assert(core.state.lr == 0x10203040U);
        assert(core.state.ctr == 0x50607080U);
        std::array<std::uint8_t, 16U> memory_after{};
        core.memory.read_bytes(0U, memory_after);
        assert(memory_after == memory_before);

        assert(result.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "fcmpu");
        assert(history.has_fp_compare);
        assert(history.fp_compare_cr_field == test.cr_field);
        assert(history.fp_compare_a_register == 1U);
        assert(history.fp_compare_b_register == 31U);
        assert(history.fp_compare_a_raw == test.a);
        assert(history.fp_compare_b_raw == test.b);
        assert(history.fp_compare_result == test.result);
    }

    // Current Wind Waker values compare greater-than and only alter CR0/FPCC.
    EspressoCore wind_waker(8U);
    wind_waker.state.fpr[1] = 0x40AA905E00000000ULL;
    wind_waker.state.fpr[31] = 0x0000000000000000ULL;
    const auto wind_waker_ps0_before = wind_waker.state.fpr;
    const auto wind_waker_ps1_before = wind_waker.state.fpr_ps1;
    wind_waker.memory.write32_be(0U, 0xFC01F800U);
    const RunResult wind_waker_result = wind_waker.run(1U);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert((wind_waker.state.cr >> 28U) == 0x4U);
    assert((wind_waker.state.fpscr & 0x0000F000U) == 0x00004000U);
    assert(wind_waker.state.fpr == wind_waker_ps0_before);
    assert(wind_waker.state.fpr_ps1 == wind_waker_ps1_before);
    const std::string trace = format_instruction_history(wind_waker_result);
    assert(trace.find("fcmpu cr0 f1=3400.18359375 [0x40AA905E00000000] ") !=
           std::string::npos);
    assert(trace.find("f31=0 [0x0000000000000000] -> GT (cr0=0x4, fpcc=0x4)") !=
           std::string::npos);

    constexpr std::uint32_t unrelated_fpscr = 0x00000025U;
    constexpr std::uint32_t exception_masks = fpscr::vxsnan_mask | fpscr::vx_mask |
        fpscr::fx_mask | fpscr::fex_mask;
    const auto run_nan_compare = [&](std::uint64_t nan_bits, bool enable_invalid) {
        EspressoCore core(8U);
        core.state.fpr[1] = nan_bits;
        core.state.fpr[31] = one;
        core.state.fpscr = unrelated_fpscr |
            (enable_invalid ? fpscr::ve_mask : 0U);
        core.memory.write32_be(0U, 0xFC01F800U);
        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert((core.state.cr >> 28U) == 0x1U);
        assert((core.state.fpscr & fpscr::fpcc_mask) == 0x00001000U);
        return core.state.fpscr;
    };

    constexpr std::uint64_t quiet_nan = 0x7FF8000000000001ULL;
    constexpr std::uint64_t signaling_nan = 0x7FF0000000000001ULL;
    const std::uint32_t quiet_fpscr = run_nan_compare(quiet_nan, false);
    assert((quiet_fpscr & exception_masks) == 0U);
    assert((quiet_fpscr & unrelated_fpscr) == unrelated_fpscr);

    const std::uint32_t signaling_disabled = run_nan_compare(signaling_nan, false);
    assert((signaling_disabled & fpscr::vxsnan_mask) != 0U);
    assert((signaling_disabled & fpscr::vx_mask) != 0U);
    assert((signaling_disabled & fpscr::fx_mask) != 0U);
    assert((signaling_disabled & fpscr::fex_mask) == 0U);
    assert((signaling_disabled & unrelated_fpscr) == unrelated_fpscr);

    const std::uint32_t signaling_enabled = run_nan_compare(signaling_nan, true);
    assert((signaling_enabled & fpscr::vxsnan_mask) != 0U);
    assert((signaling_enabled & fpscr::vx_mask) != 0U);
    assert((signaling_enabled & fpscr::fx_mask) != 0U);
    assert((signaling_enabled & fpscr::fex_mask) != 0U);
    assert((signaling_enabled & fpscr::ve_mask) != 0U);
}

void floating_divide_single_tests()
{
    const auto encode_fdivs = [](std::uint8_t destination, std::uint8_t source_a,
                                 std::uint8_t source_b, bool record = false,
                                 std::uint8_t source_c = 0U) {
        return 0xEC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (18U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint64_t a_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(4U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x100U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_b] = 0x4040000000000000ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.state.fpscr = fpscr_value;
        const auto gpr_before = core.state.gpr;
        core.memory.write32_be(0U, encode_fdivs(destination, source_a, source_b));

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.memory.read32_be(0U) == encode_fdivs(destination, source_a, source_b));
        execution.state = core.state;
        return execution;
    };

    const DecodedInstruction wind_waker_decode = decode(0xEC016824U);
    assert(wind_waker_decode.opcode == Opcode::floating_divide_single);
    assert(wind_waker_decode.fp_register == 0U);
    assert(wind_waker_decode.fp_source_a == 1U);
    assert(wind_waker_decode.fp_source_b == 13U);
    assert(((0xEC016824U >> 1U) & 0x1FU) == 18U);
    assert(((0xEC016824U >> 6U) & 0x1FU) == 0U);
    assert((0xEC016824U & 1U) == 0U);
    assert(decode(encode_fdivs(0U, 1U, 13U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fdivs(0U, 1U, 13U, true)).opcode == Opcode::unsupported);

    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t six = 0x4018000000000000ULL;
    constexpr std::uint64_t negative_six = 0xC018000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;

    const std::array<std::array<std::uint64_t, 3>, 4> ordinary_cases{{
        {six, two, three},
        {negative_six, two, negative_three},
        {six, negative_two, negative_three},
        {negative_six, negative_two, three},
    }};
    for (const auto& test : ordinary_cases)
    {
        const Execution execution = execute(0U, 1U, 13U, test[0], test[1]);
        assert(execution.state.fpr[0] == test[2]);
        assert(execution.state.fpr_ps1[0] == test[2]);
        assert(execution.state.fpr[1] == test[0]);
        assert(execution.state.fpr[13] == test[1]);
        assert(execution.state.fpr_ps1[1] == 0xC040000000000000ULL);
        assert(execution.state.fpr_ps1[13] == 0x4040000000000000ULL);
        assert((execution.state.fpscr & fpscr::fprf_mask) ==
               (test[2] == three ? 0x00004000U : 0x00008000U));
        assert((execution.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    }

    // The requested Wind Waker case rounds from binary64 to binary32, then widens.
    const Execution wind_waker = execute(
        0U, 1U, 13U, 0x40AA905E00000000ULL, 0x404CB204E0000000ULL,
        fpscr::xe_mask | 0x00000001U);
    assert(wind_waker.state.fpr[0] == 0x404D9F8280000000ULL);
    assert(wind_waker.state.fpr_ps1[0] == 0x404D9F8280000000ULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr & fpscr::fi_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fr_mask) == 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fx_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fex_mask) != 0U);
    assert((wind_waker.state.fpscr & 1U) != 0U);
    assert(wind_waker.result.instruction_history.size() == 1U);
    const std::string wind_trace = format_instruction_history(wind_waker.result);
    assert(wind_trace.find("fdivs f1=3400.18359375 [0x40AA905E00000000] ") !=
           std::string::npos);
    assert(wind_trace.find("f13=57.390773773193359 [0x404CB204E0000000]") !=
           std::string::npos);
    assert(wind_trace.find("-> f0=59.246170043945312 [0x404D9F8280000000] single=0x426CFC14") !=
           std::string::npos);

    const Execution exact_with_sticky_xx = execute(
        0U, 1U, 13U, six, two,
        fpscr::xx_mask | fpscr::xe_mask | fpscr::fi_mask | fpscr::fr_mask);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::fi_mask) == 0U);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::fr_mask) == 0U);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::xx_mask) != 0U);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution rounded_up = execute(0U, 1U, 13U, one, three);
    assert((rounded_up.state.fpscr & fpscr::fi_mask) != 0U);
    assert((rounded_up.state.fpscr & fpscr::fr_mask) != 0U);
    assert((rounded_up.state.fpscr & fpscr::xx_mask) != 0U);

    // Destination may alias either source; both operands are captured first.
    const Execution alias_a = execute(1U, 1U, 13U, six, two);
    assert(alias_a.state.fpr[1] == three);
    assert(alias_a.state.fpr_ps1[1] == three);
    const Execution alias_b = execute(13U, 1U, 13U, six, two);
    assert(alias_b.state.fpr[13] == three);
    assert(alias_b.state.fpr_ps1[13] == three);

    struct ClassCase
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint32_t fprf;
    };
    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t positive_min_single_subnormal = 0x36A0000000000000ULL;
    constexpr std::uint64_t negative_min_single_subnormal = 0xB6A0000000000000ULL;
    const std::array<ClassCase, 8> class_cases{{
        {positive_zero, one, 0x00002000U},
        {negative_zero, one, 0x00012000U},
        {positive_inf, one, 0x00005000U},
        {negative_inf, one, 0x00009000U},
        {positive_min_single_subnormal, one, 0x00014000U},
        {negative_min_single_subnormal, one, 0x00018000U},
        {0x7FF8000000000001ULL, one, 0x00011000U},
        {0x7FF0000000000000ULL, one, 0x00005000U},
    }};
    for (const ClassCase& test : class_cases)
    {
        const Execution execution = execute(0U, 1U, 13U, test.a, test.b, 1U);
        assert((execution.state.fpscr & fpscr::fprf_mask) == test.fprf);
        assert((execution.state.fpscr & 1U) != 0U);
    }

    const Execution positive_divide_zero = execute(0U, 1U, 13U, one, positive_zero);
    assert(positive_divide_zero.state.fpr[0] == positive_inf);
    assert((positive_divide_zero.state.fpscr & fpscr::zx_mask) != 0U);
    assert((positive_divide_zero.state.fpscr & fpscr::fx_mask) != 0U);
    assert((positive_divide_zero.state.fpscr & fpscr::fprf_mask) == 0x00005000U);

    const Execution negative_divide_zero = execute(0U, 1U, 13U, one, negative_zero);
    assert(negative_divide_zero.state.fpr[0] == negative_inf);
    assert((negative_divide_zero.state.fpscr & fpscr::zx_mask) != 0U);

    const Execution zero_divided_by_zero = execute(0U, 1U, 13U, positive_zero, positive_zero);
    assert((zero_divided_by_zero.state.fpr[0] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    assert((zero_divided_by_zero.state.fpscr & fpscr::vxzdz_mask) != 0U);
    assert((zero_divided_by_zero.state.fpscr & fpscr::vx_mask) != 0U);
    assert((zero_divided_by_zero.state.fpscr & fpscr::fx_mask) != 0U);

    const Execution infinity_divided_by_infinity = execute(
        0U, 1U, 13U, positive_inf, positive_inf);
    assert((infinity_divided_by_infinity.state.fpscr & fpscr::vxidi_mask) != 0U);
    assert((infinity_divided_by_infinity.state.fpscr & fpscr::vx_mask) != 0U);

    const Execution quiet_nan = execute(0U, 1U, 13U, 0x7FF8000000001234ULL, one);
    assert((quiet_nan.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet_nan.state.fpscr & fpscr::fprf_mask) == 0x00011000U);

    const Execution signaling_nan = execute(0U, 1U, 13U, 0x7FF0000000001234ULL, one);
    assert((signaling_nan.state.fpscr & fpscr::vxsnan_mask) != 0U);
    assert((signaling_nan.state.fpscr & fpscr::vx_mask) != 0U);
    assert((signaling_nan.state.fpscr & fpscr::fx_mask) != 0U);
    assert((signaling_nan.state.fpr[0] & 0x0008000000000000ULL) != 0U);

    const std::uint64_t destination_ps0 = 0x1111222233334444ULL;
    const std::uint64_t destination_ps1 = 0xAAAABBBBCCCCDDDDULL;
    const auto execute_suppressed = [&](std::uint64_t a, std::uint64_t b,
                                        std::uint32_t fpscr_value) {
        EspressoCore core(4U);
        core.state.fpr[1] = a;
        core.state.fpr[13] = b;
        core.state.fpr[0] = destination_ps0;
        core.state.fpr_ps1[0] = destination_ps1;
        core.state.fpscr = fpscr_value;
        core.memory.write32_be(0U, 0xEC016824U);
        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.fpr[0] == destination_ps0);
        assert(core.state.fpr_ps1[0] == destination_ps1);
        assert(!result.instruction_history.front().has_fp_arithmetic_result);
        return core.state.fpscr;
    };
    const std::uint32_t invalid_suppressed = execute_suppressed(
        positive_zero, positive_zero, fpscr::ve_mask);
    assert((invalid_suppressed & fpscr::vxzdz_mask) != 0U);
    assert((invalid_suppressed & fpscr::vx_mask) != 0U);
    assert((invalid_suppressed & fpscr::fex_mask) != 0U);
    assert((invalid_suppressed & fpscr::fx_mask) != 0U);

    const std::uint32_t zero_suppressed = execute_suppressed(
        one, positive_zero, fpscr::ze_mask);
    assert((zero_suppressed & fpscr::zx_mask) != 0U);
    assert((zero_suppressed & fpscr::fex_mask) != 0U);

    const std::uint32_t exact_wind_fprf = 0x00004000U;
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == exact_wind_fprf);
}

void floating_divide_double_tests()
{
    const auto encode_fdiv = [](std::uint8_t destination, std::uint8_t source_a,
                                std::uint8_t source_b, bool record = false,
                                std::uint8_t reserved_c = 0U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(reserved_c) << 6U) | (18U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFC210024U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_divide_double);
    assert(decoded.fp_register == 1U);
    assert(decoded.fp_source_a == 1U);
    assert(decoded.fp_source_b == 0U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 18U);
    assert(((wind_waker_word >> 6U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fdiv(1U, 1U, 0U) == wind_waker_word);
    assert(decode(encode_fdiv(1U, 1U, 0U, true)).opcode == Opcode::unsupported);
    assert(decode(encode_fdiv(1U, 1U, 0U, false, 1U)).opcode ==
           Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    constexpr std::uint64_t destination_sentinel = 0x1111222233334444ULL;
    constexpr std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL;
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint64_t a_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x500U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[destination] = destination_sentinel;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U, encode_fdiv(destination, source_a, source_b));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t next_one = 0x3FF0000000000001ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t six = 0x4018000000000000ULL;
    constexpr std::uint64_t negative_one = 0xBFF0000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t default_qnan = 0x7FF8000000000000ULL;
    constexpr std::uint64_t qnan_a = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t qnan_b = 0xFFF823456789ABCDULL;
    constexpr std::uint64_t snan_a = 0x7FF0000000004321ULL;
    constexpr std::uint64_t snan_b = 0xFFF0000000005432ULL;

    assert(execute(3U, 1U, 2U, six, three).state.fpr[3] == two);
    assert(execute(3U, 1U, 2U, one, two).state.fpr[3] == 0x3FE0000000000000ULL);
    assert(execute(3U, 1U, 2U, 0xC018000000000000ULL, three).state.fpr[3] ==
           negative_two);
    assert(execute(3U, 1U, 2U, six, negative_three).state.fpr[3] == negative_two);
    assert(execute(3U, 1U, 2U, 0xC018000000000000ULL, negative_three).state.fpr[3] ==
           two);
    assert(execute(3U, 1U, 2U, next_one, one).state.fpr[3] == next_one);

    // Signed zero and infinity-denominator cases are ordinary non-invalid divisions.
    assert(execute(3U, 1U, 2U, positive_zero, one).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, negative_zero, one).state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, positive_zero, negative_one).state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, negative_zero, negative_one).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, one, positive_inf).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, negative_one, positive_inf).state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, one, negative_inf).state.fpr[3] == negative_zero);

    // frD may alias A, B, or both; all calculations use the old values.
    assert(execute(1U, 1U, 2U, six, three).state.fpr[1] == two);
    assert(execute(2U, 1U, 2U, 0x4028000000000000ULL, six).state.fpr[2] == two);
    assert(execute(5U, 5U, 5U, six, six).state.fpr[5] == one);

    const Execution zero_by_zero = execute(3U, 1U, 2U, positive_zero, negative_zero);
    assert(zero_by_zero.state.fpr[3] == default_qnan);
    assert((zero_by_zero.state.fpscr &
            (fpscr::vxzdz_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vxzdz_mask | fpscr::vx_mask | fpscr::fx_mask));
    assert((zero_by_zero.state.fpscr & fpscr::zx_mask) == 0U);
    const Execution zero_suppressed = execute(
        3U, 1U, 2U, positive_zero, positive_zero, fpscr::ve_mask);
    assert(zero_suppressed.state.fpr[3] == destination_sentinel);
    assert(zero_suppressed.state.fpr_ps1[3] == destination_ps1);
    assert((zero_suppressed.state.fpscr & fpscr::fex_mask) != 0U);
    assert(!zero_suppressed.result.instruction_history.front().has_fp_arithmetic_result);

    const Execution infinity_by_infinity = execute(3U, 1U, 2U, positive_inf, negative_inf);
    assert(infinity_by_infinity.state.fpr[3] == default_qnan);
    assert((infinity_by_infinity.state.fpscr & fpscr::vxidi_mask) != 0U);
    const Execution infinity_suppressed = execute(
        3U, 1U, 2U, positive_inf, positive_inf, fpscr::ve_mask);
    assert(infinity_suppressed.state.fpr[3] == destination_sentinel);
    assert(infinity_suppressed.state.fpr_ps1[3] == destination_ps1);

    for (const auto& [a, expected] : {
             std::pair{one, positive_inf},
             std::pair{negative_one, negative_inf},
             std::pair{positive_inf, positive_inf}})
    {
        const Execution divide_zero = execute(3U, 1U, 2U, a, positive_zero);
        assert(divide_zero.state.fpr[3] == expected);
        assert((divide_zero.state.fpscr & fpscr::zx_mask) != 0U);
        assert((divide_zero.state.fpscr & fpscr::fx_mask) != 0U);
    }
    const Execution divide_zero_negative = execute(3U, 1U, 2U, one, negative_zero);
    assert(divide_zero_negative.state.fpr[3] == negative_inf);
    const Execution ze_suppressed = execute(
        3U, 1U, 2U, one, positive_zero, fpscr::ze_mask);
    assert(ze_suppressed.state.fpr[3] == destination_sentinel);
    assert(ze_suppressed.state.fpr_ps1[3] == destination_ps1);
    assert((ze_suppressed.state.fpscr & (fpscr::zx_mask | fpscr::fex_mask)) ==
           (fpscr::zx_mask | fpscr::fex_mask));

    const Execution quiet_nan = execute(3U, 1U, 2U, qnan_a, positive_zero);
    assert(quiet_nan.state.fpr[3] == qnan_a);
    assert((quiet_nan.state.fpscr & (fpscr::vxsnan_mask | fpscr::zx_mask)) == 0U);
    const Execution quiet_nan_priority = execute(3U, 1U, 2U, qnan_a, qnan_b);
    assert(quiet_nan_priority.state.fpr[3] == qnan_a);
    const Execution signaling_nan = execute(3U, 1U, 2U, one, snan_b);
    assert(signaling_nan.state.fpr[3] == (snan_b | 0x0008000000000000ULL));
    assert((signaling_nan.state.fpscr & fpscr::vxsnan_mask) != 0U);
    const Execution signaling_suppressed = execute(
        3U, 1U, 2U, snan_a, one, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[3] == destination_sentinel);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution wind_waker = execute(
        1U, 1U, 0U, 0x3F50000000000000ULL, 0x40B3C1C6035AEA81ULL);
    assert(wind_waker.state.fpr[1] == 0x3E89EA3AD959E987ULL);
    assert(wind_waker.state.fpr_ps1[1] == destination_ps1);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr &
            (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask)) ==
           (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask));
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fdiv f1=0.0009765625 [0x3F50000000000000]") !=
           std::string::npos);
    assert(trace.find("f0=5057.7734886953849 [0x40B3C1C6035AEA81]") !=
           std::string::npos);
    assert(trace.find("[0x3E89EA3AD959E987]") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_add_single_tests()
{
    const auto encode_fadds = [](std::uint8_t destination, std::uint8_t source_a,
                                 std::uint8_t source_b, bool record = false,
                                 std::uint8_t source_c = 0U) {
        return 0xEC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (21U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint64_t a_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(4U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x200U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_b] = 0x4040000000000000ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.state.fpscr = fpscr_value;
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        core.memory.write32_be(0U, encode_fadds(destination, source_a, source_b));

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.memory.read32_be(0U) == encode_fadds(destination, source_a, source_b));
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
                assert(core.state.fpr_ps1[index] == fpr_ps1_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    const DecodedInstruction wind_waker_decode = decode(0xEDAD002AU);
    assert(wind_waker_decode.opcode == Opcode::floating_add_single);
    assert(wind_waker_decode.fp_register == 13U);
    assert(wind_waker_decode.fp_source_a == 13U);
    assert(wind_waker_decode.fp_source_b == 0U);
    assert(((0xEDAD002AU >> 1U) & 0x1FU) == 21U);
    assert(((0xEDAD002AU >> 6U) & 0x1FU) == 0U);
    assert((0xEDAD002AU & 1U) == 0U);
    assert(encode_fadds(13U, 13U, 0U) == 0xEDAD002AU);
    assert(decode(encode_fadds(13U, 13U, 0U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fadds(13U, 13U, 0U, true)).opcode == Opcode::unsupported);

    constexpr std::uint64_t positive_one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t positive_two = 0x4000000000000000ULL;
    constexpr std::uint64_t positive_three = 0x4008000000000000ULL;
    constexpr std::uint64_t negative_one = 0xBFF0000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;

    for (const auto& test : std::array<std::array<std::uint64_t, 3>, 5>{{
             {positive_one, positive_two, positive_three},
             {positive_two, negative_one, positive_one},
             {negative_one, negative_two, negative_three},
             {positive_zero, positive_zero, positive_zero},
             {positive_one, positive_one, positive_two},
         }})
    {
        const Execution execution = execute(3U, 1U, 2U, test[0], test[1]);
        assert(execution.state.fpr[3] == test[2]);
        assert(execution.state.fpr_ps1[3] == test[2]);
        assert((execution.state.fpscr & fpscr::fprf_mask) ==
               (test[2] == negative_three ? 0x00008000U
                   : test[2] == positive_zero ? 0x00002000U
                                               : 0x00004000U));
        assert((execution.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    }
    const Execution signed_zero = execute(3U, 1U, 2U, negative_zero, negative_zero);
    assert(signed_zero.state.fpr[3] == negative_zero);
    assert(signed_zero.state.fpr_ps1[3] == negative_zero);
    assert((signed_zero.state.fpscr & fpscr::fprf_mask) == 0x00012000U);

    // The destination aliases either source; operands are captured before writing.
    assert(execute(1U, 1U, 2U, positive_one, positive_two).state.fpr[1] == positive_three);
    assert(execute(2U, 1U, 2U, positive_one, positive_two).state.fpr[2] == positive_three);
    assert(execute(1U, 1U, 1U, positive_one, positive_one).state.fpr[1] == positive_two);

    // Wind Waker exact regression: add PS0 doubles, then round once to binary32.
    const Execution wind_waker = execute(
        13U, 13U, 0U, 0x404CB204E0000000ULL, 0x404D9F8280000000ULL,
        0x00000001U);
    assert(wind_waker.state.fpr[13] == 0x405D28C3C0000000ULL);
    assert(wind_waker.state.fpr_ps1[13] == 0x405D28C3C0000000ULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr & fpscr::fi_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fr_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fx_mask) != 0U);
    assert((wind_waker.state.fpscr & 1U) != 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fadds f13=57.390773773193359 [0x404CB204E0000000]") !=
           std::string::npos);
    assert(trace.find("f0=59.246170043945312 [0x404D9F8280000000]") !=
           std::string::npos);
    assert(trace.find("-> f13=116.63694763183594 [0x405D28C3C0000000] single=0x42E9461E") !=
           std::string::npos);

    // Arithmetic must use PS0, never the deliberately contradictory PS1 values.
    const Execution ps0_only = execute(3U, 1U, 2U, positive_one, positive_two);
    assert(ps0_only.state.fpr[3] == positive_three);
    const Execution rounded_down = execute(3U, 1U, 2U, 0x3FF0000000000000ULL,
                                           0x3E70000000000000ULL);
    assert(rounded_down.state.fpr[3] == positive_one);
    assert((rounded_down.state.fpscr & fpscr::fi_mask) != 0U);
    assert((rounded_down.state.fpscr & fpscr::fr_mask) == 0U);
    const Execution exact_with_sticky_xx = execute(
        3U, 1U, 2U, positive_one, positive_two,
        fpscr::xx_mask | fpscr::fi_mask | fpscr::fr_mask);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::fi_mask) == 0U);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::fr_mask) == 0U);
    assert((exact_with_sticky_xx.state.fpscr & fpscr::xx_mask) != 0U);

    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t quiet_nan = 0x7FF8000000001234ULL;
    constexpr std::uint64_t signaling_nan = 0x7FF0000000001234ULL;
    const Execution infinity_sum = execute(3U, 1U, 2U, positive_inf, positive_one);
    assert(infinity_sum.state.fpr[3] == positive_inf);
    assert((infinity_sum.state.fpscr & fpscr::fprf_mask) == 0x00005000U);
    const Execution same_infinity = execute(3U, 1U, 2U, negative_inf, negative_inf);
    assert(same_infinity.state.fpr[3] == negative_inf);
    const Execution invalid_infinities = execute(3U, 1U, 2U, positive_inf, negative_inf);
    assert((invalid_infinities.state.fpr[3] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    assert((invalid_infinities.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert((invalid_infinities.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vx_mask | fpscr::fx_mask));
    const Execution invalid_reverse = execute(3U, 1U, 2U, negative_inf, positive_inf);
    assert((invalid_reverse.state.fpscr & fpscr::vxisi_mask) != 0U);
    const Execution invalid_enabled = execute(
        3U, 1U, 2U, positive_inf, negative_inf, fpscr::ve_mask);
    assert(invalid_enabled.state.fpr[3] == 0x1111000000000003ULL);
    assert(invalid_enabled.state.fpr_ps1[3] == 0x2222000000000003ULL);
    assert((invalid_enabled.state.fpscr & fpscr::fex_mask) != 0U);
    const Execution signaling = execute(3U, 1U, 2U, signaling_nan, positive_one);
    assert((signaling.state.fpscr & fpscr::vxsnan_mask) != 0U);
    assert((signaling.state.fpscr & fpscr::vx_mask) != 0U);
    assert((signaling.state.fpr[3] & 0x0008000000000000ULL) != 0U);
    const Execution signaling_enabled = execute(
        3U, 1U, 2U, signaling_nan, positive_one, fpscr::ve_mask);
    assert(signaling_enabled.state.fpr[3] == 0x1111000000000003ULL);
    const Execution quiet = execute(3U, 1U, 2U, quiet_nan, positive_one);
    assert((quiet.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet.state.fpr[3] & 0x7FF8000000000000ULL) == 0x7FF8000000000000ULL);
}

void floating_multiply_add_single_tests()
{
    const auto encode_fmadds = [](std::uint8_t destination, std::uint8_t source_a,
                                  std::uint8_t source_c, std::uint8_t source_b,
                                  bool record = false) {
        return 0xEC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (29U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_c, std::uint8_t source_b,
                             std::uint64_t a_raw, std::uint64_t c_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(4U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x300U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_c] = c_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_c] = 0x4040000000000000ULL;
        core.state.fpr_ps1[source_b] = 0xC008000000000000ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.state.fpscr = fpscr_value;
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        core.memory.write32_be(
            0U, encode_fmadds(destination, source_a, source_c, source_b));

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.memory.read32_be(0U) ==
               encode_fmadds(destination, source_a, source_c, source_b));
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
                assert(core.state.fpr_ps1[index] == fpr_ps1_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint32_t wind_waker_word = 0xEFED083AU;
    const DecodedInstruction wind_decode = decode(wind_waker_word);
    assert(wind_decode.opcode == Opcode::floating_multiply_add_single);
    assert(wind_decode.fp_register == 31U);
    assert(wind_decode.fp_source_a == 13U);
    assert(wind_decode.fp_source_b == 1U);
    assert(wind_decode.fp_source_c == 0U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 29U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fmadds(31U, 13U, 0U, 1U) == wind_waker_word);
    assert(decode(encode_fmadds(4U, 5U, 7U, 6U)).opcode ==
           Opcode::floating_multiply_add_single);
    assert(decode(encode_fmadds(31U, 13U, 0U, 1U, true)).opcode == Opcode::unsupported);

    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t four = 0x4010000000000000ULL;
    constexpr std::uint64_t ten = 0x4024000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_four = 0xC010000000000000ULL;
    constexpr std::uint64_t negative_eight = 0xC020000000000000ULL;
    constexpr std::uint64_t positive_two = 0x4000000000000000ULL;
    const Execution ordinary = execute(7U, 1U, 2U, 3U, two, three, four);
    assert(ordinary.state.fpr[7] == ten);
    assert(ordinary.state.fpr_ps1[7] == ten);
    assert((ordinary.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((ordinary.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert(execute(7U, 1U, 2U, 3U, negative_two, three, negative_two)
               .state.fpr[7] == negative_eight);
    assert(execute(7U, 1U, 2U, 3U, two, three, negative_four)
               .state.fpr[7] == positive_two);

    // frD may alias A, C, or B; all inputs must be read before committing.
    assert(execute(1U, 1U, 2U, 3U, two, three, four).state.fpr[1] == ten);
    assert(execute(2U, 1U, 2U, 3U, two, three, four).state.fpr[2] == ten);
    assert(execute(3U, 1U, 2U, 3U, two, three, four).state.fpr[3] == ten);

    // This case differs from a separately rounded binary64 multiply then add.
    constexpr std::uint64_t a_near_one = 0x3FF0000000800000ULL; // 1 + 2^-29
    constexpr std::uint64_t c_near_one = 0x3FF0000010000000ULL; // 1 + 2^-24, Force25Bit-stable
    constexpr std::uint64_t negative_rounded_product = 0xBFF0000010800000ULL;
    const double a_value = std::bit_cast<double>(a_near_one);
    const double c_value = std::bit_cast<double>(c_near_one);
    const double b_value = std::bit_cast<double>(negative_rounded_product);
    assert((a_value * c_value) + b_value == 0.0);
    const double fused = std::fma(a_value, c_value, b_value);
    const float fused_single = static_cast<float>(fused);
    const std::uint64_t fused_raw = std::bit_cast<std::uint64_t>(
        static_cast<double>(fused_single));
    assert(fused_raw == 0x3CA0000000000000ULL);
    const Execution fused_execution = execute(
        7U, 1U, 2U, 3U, a_near_one, c_near_one, negative_rounded_product);
    assert(fused_execution.state.fpr[7] == fused_raw);
    assert(fused_execution.state.fpr_ps1[7] == fused_raw);

    // fmadds and fmuls share Espresso's frC operand precision reduction.
    constexpr std::uint64_t precision_a = 0x4070000000000000ULL;
    constexpr std::uint64_t precision_c = 0x3FFA794410E24F4BULL;
    const double rounded_precision_c = std::bit_cast<double>(0x3FFA794410000000ULL);
    const double precision_product = std::fma(
        std::bit_cast<double>(precision_a), rounded_precision_c, 0.0);
    const std::uint64_t expected_shared_precision = std::bit_cast<std::uint64_t>(
        static_cast<double>(static_cast<float>(precision_product)));
    const std::uint64_t unrounded_precision = std::bit_cast<std::uint64_t>(
        static_cast<double>(static_cast<float>(
            std::fma(std::bit_cast<double>(precision_a),
                     std::bit_cast<double>(precision_c), 0.0))));
    assert(expected_shared_precision == 0x407A794400000000ULL);
    assert(unrounded_precision == 0x407A794420000000ULL);
    const Execution shared_precision = execute(
        7U, 1U, 2U, 3U, precision_a, precision_c, 0U);
    assert(shared_precision.state.fpr[7] == expected_shared_precision);

    // Wind Waker regression, with the source operands deliberately placed in PS0.
    const Execution wind_waker = execute(
        31U, 13U, 0U, 1U,
        0x405D28C3C0000000ULL, 0x3FD0000000000000ULL,
        0x403D26E020000000ULL, 1U);
    assert(wind_waker.state.fpr[31] == 0x404D27D200000000ULL);
    assert(wind_waker.state.fpr_ps1[31] == 0x404D27D200000000ULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr & fpscr::fi_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fr_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    assert((wind_waker.state.fpscr & fpscr::fx_mask) != 0U);
    assert((wind_waker.state.fpscr & 1U) != 0U);
    const std::string wind_trace = format_instruction_history(wind_waker.result);
    assert(wind_trace.find("fmadds f13=116.63694763183594 [0x405D28C3C0000000] ") !=
           std::string::npos);
    assert(wind_trace.find("f0=0.25 [0x3FD0000000000000] ") != std::string::npos);
    assert(wind_trace.find("f1=29.151857376098633 [0x403D26E020000000]") !=
           std::string::npos);
    assert(wind_trace.find("-> f31=58.31109619140625 [0x404D27D200000000] single=0x42693E90") !=
           std::string::npos);

    constexpr std::uint64_t positive_zero = 0U;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t quiet_nan = 0x7FF8000000001234ULL;
    constexpr std::uint64_t signaling_nan = 0x7FF0000000001234ULL;
    const Execution infinity_times_zero = execute(
        7U, 1U, 2U, 3U, positive_inf, positive_zero, one);
    assert((infinity_times_zero.state.fpscr & fpscr::vximz_mask) != 0U);
    assert((infinity_times_zero.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vx_mask | fpscr::fx_mask));
    assert((infinity_times_zero.state.fpr[7] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    const Execution zero_times_infinity = execute(
        7U, 1U, 2U, 3U, positive_zero, positive_inf, one);
    assert((zero_times_infinity.state.fpscr & fpscr::vximz_mask) != 0U);

    const Execution positive_inf_plus_negative_inf = execute(
        7U, 1U, 2U, 3U, positive_inf, one, negative_inf);
    assert((positive_inf_plus_negative_inf.state.fpscr & fpscr::vxisi_mask) != 0U);
    const Execution negative_inf_plus_positive_inf = execute(
        7U, 1U, 2U, 3U, negative_inf, one, positive_inf);
    assert((negative_inf_plus_positive_inf.state.fpscr & fpscr::vxisi_mask) != 0U);
    const Execution same_sign_infinities = execute(
        7U, 1U, 2U, 3U, positive_inf, one, positive_inf);
    assert(same_sign_infinities.state.fpr[7] == positive_inf);

    for (std::size_t nan_source = 0; nan_source < 3U; ++nan_source)
    {
        const std::uint64_t a = nan_source == 0U ? signaling_nan : one;
        const std::uint64_t c = nan_source == 1U ? signaling_nan : one;
        const std::uint64_t b = nan_source == 2U ? signaling_nan : one;
        const Execution signaling = execute(7U, 1U, 2U, 3U, a, c, b);
        assert((signaling.state.fpscr & fpscr::vxsnan_mask) != 0U);
        assert((signaling.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vx_mask | fpscr::fx_mask));
        assert((signaling.state.fpr[7] & 0x0008000000000000ULL) != 0U);
    }
    const Execution quiet = execute(7U, 1U, 2U, 3U, quiet_nan, one, one);
    assert((quiet.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet.state.fpr[7] & 0x7FF8000000000000ULL) == 0x7FF8000000000000ULL);

    constexpr std::uint64_t sentinel_ps0 = 0x1111000000000007ULL;
    constexpr std::uint64_t sentinel_ps1 = 0x2222000000000007ULL;
    const Execution vximz_enabled = execute(
        7U, 1U, 2U, 3U, positive_inf, positive_zero, one, fpscr::ve_mask);
    assert(vximz_enabled.state.fpr[7] == sentinel_ps0);
    assert(vximz_enabled.state.fpr_ps1[7] == sentinel_ps1);
    assert((vximz_enabled.state.fpscr & fpscr::fex_mask) != 0U);
    const Execution vxisi_enabled = execute(
        7U, 1U, 2U, 3U, positive_inf, one, negative_inf, fpscr::ve_mask);
    assert(vxisi_enabled.state.fpr[7] == sentinel_ps0);
    assert(vxisi_enabled.state.fpr_ps1[7] == sentinel_ps1);
    const Execution snan_enabled = execute(
        7U, 1U, 2U, 3U, one, signaling_nan, one, fpscr::ve_mask);
    assert(snan_enabled.state.fpr[7] == sentinel_ps0);
    assert(snan_enabled.state.fpr_ps1[7] == sentinel_ps1);
}

void floating_multiply_single_tests()
{
    const auto encode_fmuls = [](std::uint8_t destination, std::uint8_t source_a,
                                 std::uint8_t source_c, std::uint8_t reserved_b = 0U,
                                 bool record = false) {
        return 0xEC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(reserved_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (25U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_c, std::uint64_t a_raw,
                             std::uint64_t c_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(4U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x400U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_c] = c_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_c] = 0x4040000000000000ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.state.fpscr = fpscr_value;
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const std::uint32_t instruction = encode_fmuls(destination, source_a, source_c);
        core.memory.write32_be(0U, instruction);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.memory.read32_be(0U) == instruction);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
                assert(core.state.fpr_ps1[index] == fpr_ps1_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint32_t wind_waker_word = 0xED2B0032U;
    const DecodedInstruction wind_decode = decode(wind_waker_word);
    assert(wind_decode.opcode == Opcode::floating_multiply_single);
    assert(wind_decode.fp_register == 9U);
    assert(wind_decode.fp_source_a == 11U);
    assert(wind_decode.fp_source_c == 0U);
    assert(((wind_waker_word >> 11U) & 0x1FU) == 0U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 25U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fmuls(9U, 11U, 0U) == wind_waker_word);
    assert(decode(encode_fmuls(3U, 4U, 7U)).opcode == Opcode::floating_multiply_single);
    assert(decode(encode_fmuls(9U, 11U, 0U, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fmuls(9U, 11U, 0U, 0U, true)).opcode == Opcode::unsupported);

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t six = 0x4018000000000000ULL;
    constexpr std::uint64_t negative_six = 0xC018000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t quiet_nan = 0x7FF8000000001234ULL;
    constexpr std::uint64_t signaling_nan = 0x7FF0000000001234ULL;

    const Execution ordinary = execute(3U, 1U, 2U, two, three);
    assert(ordinary.state.fpr[3] == six);
    assert(ordinary.state.fpr_ps1[3] == six);
    assert((ordinary.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((ordinary.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert(execute(3U, 1U, 2U, two, negative_three).state.fpr[3] == negative_six);
    assert(execute(3U, 1U, 2U, negative_two, negative_three).state.fpr[3] == six);
    assert(execute(3U, 1U, 2U, positive_zero, one).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, negative_zero, one).state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, positive_zero, negative_two).state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, negative_zero, negative_two).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, positive_inf, two).state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, negative_inf, two).state.fpr[3] == negative_inf);
    assert(execute(3U, 1U, 2U, two, positive_inf).state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, two, negative_inf).state.fpr[3] == negative_inf);
    assert(execute(1U, 1U, 2U, two, three).state.fpr[1] == six);
    assert(execute(2U, 1U, 2U, two, three).state.fpr[2] == six);
    assert(ordinary.state.fpr_ps1[1] == 0xC040000000000000ULL);
    assert(ordinary.state.fpr_ps1[2] == 0x4040000000000000ULL);

    const auto force25_reference = [](std::uint64_t raw) {
        const std::uint64_t exponent = raw & 0x7FF0000000000000ULL;
        const std::uint64_t fraction = raw & 0x000FFFFFFFFFFFFFULL;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep = static_cast<std::uint64_t>(signed_mask >> shift);
            return (raw & keep) + (raw & (0x8000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x8000000ULL);
    };
    const auto expected_product = [&](std::uint64_t a_raw, std::uint64_t c_raw) {
        const double product = std::bit_cast<double>(a_raw) *
            std::bit_cast<double>(force25_reference(c_raw));
        return std::bit_cast<std::uint64_t>(static_cast<double>(static_cast<float>(product)));
    };

    constexpr std::uint64_t precision_a = 0x4070000000000000ULL;
    constexpr std::uint64_t precision_c = 0x3FFA794410E24F4BULL;
    const std::uint64_t rounded_expected = expected_product(precision_a, precision_c);
    const std::uint64_t unrounded_expected = std::bit_cast<std::uint64_t>(
        static_cast<double>(static_cast<float>(
            std::bit_cast<double>(precision_a) * std::bit_cast<double>(precision_c))));
    assert(force25_reference(precision_c) == 0x3FFA794410000000ULL);
    assert(rounded_expected == 0x407A794400000000ULL);
    assert(unrounded_expected == 0x407A794420000000ULL);
    assert(execute(3U, 1U, 2U, precision_a, precision_c).state.fpr[3] == rounded_expected);

    constexpr std::uint64_t subnormal_a = 0x79B0000000000000ULL;
    constexpr std::uint64_t subnormal_c = 0x000522DCBAAABE53ULL;
    assert(force25_reference(subnormal_c) == 0x000522DCBC000000ULL);
    const std::uint64_t subnormal_expected = expected_product(subnormal_a, subnormal_c);
    assert(subnormal_expected == 0x39B48B7300000000ULL);
    assert(execute(3U, 1U, 2U, subnormal_a, subnormal_c).state.fpr[3] ==
           subnormal_expected);
    assert(force25_reference(positive_zero) == positive_zero);
    assert(force25_reference(negative_zero) == negative_zero);
    assert(force25_reference(positive_inf) == positive_inf);
    assert(force25_reference(negative_inf) == negative_inf);

    const Execution inexact = execute(3U, 1U, 2U, precision_a, precision_c, 1U);
    assert((inexact.state.fpscr & fpscr::fi_mask) != 0U);
    assert((inexact.state.fpscr & fpscr::xx_mask) != 0U);
    assert((inexact.state.fpscr & 1U) != 0U);
    assert((inexact.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    const Execution exact_after_inexact = execute(
        3U, 1U, 2U, two, three, fpscr::xx_mask | 0x1U);
    assert((exact_after_inexact.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((exact_after_inexact.state.fpscr & fpscr::xx_mask) != 0U);
    assert((exact_after_inexact.state.fpscr & 1U) != 0U);

    const Execution inf_zero = execute(3U, 1U, 2U, positive_inf, positive_zero);
    assert((inf_zero.state.fpscr & fpscr::vximz_mask) != 0U);
    assert((inf_zero.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vx_mask | fpscr::fx_mask));
    assert((inf_zero.state.fpr[3] & 0x7FF8000000000000ULL) == 0x7FF8000000000000ULL);
    assert((execute(3U, 1U, 2U, positive_zero, positive_inf).state.fpscr &
            fpscr::vximz_mask) != 0U);
    const Execution enabled_invalid = execute(
        3U, 1U, 2U, positive_inf, positive_zero, fpscr::ve_mask);
    assert(enabled_invalid.state.fpr[3] == 0x1111000000000003ULL);
    assert(enabled_invalid.state.fpr_ps1[3] == 0x2222000000000003ULL);
    assert((enabled_invalid.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution signaling = execute(3U, 1U, 2U, signaling_nan, one);
    assert((signaling.state.fpscr & fpscr::vxsnan_mask) != 0U);
    assert((signaling.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vx_mask | fpscr::fx_mask));
    assert((signaling.state.fpr[3] & 0x0008000000000000ULL) != 0U);
    const Execution signaling_enabled = execute(
        3U, 1U, 2U, one, signaling_nan, fpscr::ve_mask);
    assert(signaling_enabled.state.fpr[3] == 0x1111000000000003ULL);
    assert(signaling_enabled.state.fpr_ps1[3] == 0x2222000000000003ULL);
    const Execution quiet = execute(3U, 1U, 2U, quiet_nan, one);
    assert((quiet.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet.state.fpr[3] & 0x7FF8000000000000ULL) == 0x7FF8000000000000ULL);

    constexpr std::uint64_t wind_reciprocal = 0x3FC24924A0000000ULL;
    const Execution wind_waker = execute(9U, 11U, 0U, positive_zero, wind_reciprocal);
    assert(wind_waker.state.fpr[9] == positive_zero);
    assert(wind_waker.state.fpr_ps1[9] == positive_zero);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00002000U);
    assert((wind_waker.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fmuls f11=0 [0x0000000000000000] f0=0.1428571492433548 ") !=
           std::string::npos);
    assert(trace.find("-> f9=0 [0x0000000000000000] single=0x00000000") !=
           std::string::npos);
    assert(trace.find("f0=0.1428571492433548") ==
           trace.rfind("f0=0.1428571492433548"));
}

void floating_move_register_tests()
{
    constexpr std::uint32_t wind_waker_word = 0xFC20F890U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_move_register);
    assert(decoded.fp_register == 1U);
    assert(decoded.fp_source_b == 31U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 72U);
    assert((wind_waker_word & 1U) == 0U);
    assert(decode(wind_waker_word | 0x00010000U).opcode == Opcode::unsupported);
    assert(decode(wind_waker_word | 1U).opcode == Opcode::unsupported);
    assert(decode(wind_waker_word ^ 0x2U).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
        std::uint32_t memory_word{};
    };
    const auto execute = [](std::uint8_t destination, std::uint8_t source,
                            std::uint64_t source_raw,
                            std::uint64_t destination_ps1 = 0xAABBCCDDEEFF0011ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x400U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source] = source_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = 0xD5C3BEEF;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        const std::uint32_t instruction_word = 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 11U) | (72U << 1U);
        core.memory.write32_be(0U, instruction_word);
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const std::uint32_t fpscr_before = core.state.fpscr;

        Execution execution{.state = {}, .result = core.run(1U), .memory_word = 0U};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.state.fpr[destination] == source_raw);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.memory_word = core.memory.read32_be(8U);
        assert(execution.memory_word == 0xDEADBEEFU);
        execution.state = core.state;
        return execution;
    };

    constexpr std::array<std::uint64_t, 7> raw_values{{
        0x3FF123456789ABCDULL, // finite value
        0x0000000000000000ULL, // +0
        0x8000000000000000ULL, // -0
        0x7FF0000000000000ULL, // +infinity
        0xFFF0000000000000ULL, // -infinity
        0x7FF8000000001234ULL, // quiet NaN payload
        0x7FF0000000001234ULL, // signaling NaN payload
    }};
    for (const std::uint64_t raw : raw_values)
    {
        const Execution result = execute(1U, 31U, raw);
        assert(result.state.fpr[1] == raw);
        assert(result.state.fpr_ps1[1] == 0xAABBCCDDEEFF0011ULL);
    }

    // Same-register move is an exact PS0 no-op and never touches PS1.
    const Execution alias = execute(5U, 5U, 0x7FF0000000004321ULL,
                                    0x123456789ABCDEF0ULL);
    assert(alias.state.fpr[5] == 0x7FF0000000004321ULL);
    assert(alias.state.fpr_ps1[5] == 0x123456789ABCDEF0ULL);

    // Wind Waker's fmr must preserve the destination's prior PS1 independently.
    constexpr std::uint64_t wind_source = 0x404D27D200000000ULL;
    constexpr std::uint64_t old_destination_ps1 = 0x403D26E020000000ULL;
    const Execution wind_waker = execute(1U, 31U, wind_source, old_destination_ps1);
    assert(wind_waker.state.fpr[1] == wind_source);
    assert(wind_waker.state.fpr_ps1[1] == old_destination_ps1);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fmr f31.ps0=0x404D27D200000000 -> f1.ps0=0x404D27D200000000") !=
           std::string::npos);
    assert(trace.find("ps1 unchanged=0x403D26E020000000") != std::string::npos);
}

void floating_negate_tests()
{
    constexpr std::uint32_t wind_waker_word = 0xFD805050U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_negate);
    assert(decoded.fp_register == 12U);
    assert(decoded.fp_source_b == 10U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 40U);
    assert(((wind_waker_word >> 16U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(decode(wind_waker_word | 0x00010000U).opcode == Opcode::unsupported);
    assert(decode(wind_waker_word | 1U).opcode == Opcode::unsupported);
    assert(decode(wind_waker_word ^ 0x2U).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
        std::uint32_t untouched_memory{};
    };
    const auto execute = [](std::uint8_t destination, std::uint8_t source,
                            std::uint64_t source_raw,
                            std::uint64_t destination_ps1 = 0xAABBCCDDEEFF0011ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x400U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source] = source_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = 0xD5C3BEEF;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        const std::uint32_t instruction = 0xFC000050U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 11U);
        core.memory.write32_be(0U, instruction);
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const std::uint32_t fpscr_before = core.state.fpscr;

        Execution execution{.state = {}, .result = core.run(1U), .untouched_memory = 0U};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.state.fpr[destination] == (source_raw ^ 0x8000000000000000ULL));
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.untouched_memory = core.memory.read32_be(8U);
        assert(execution.untouched_memory == 0xDEADBEEFU);
        execution.state = core.state;
        return execution;
    };

    constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 9> values{{
        {0x3FF0000000000000ULL, 0xBFF0000000000000ULL}, // +1 -> -1
        {0xBFF0000000000000ULL, 0x3FF0000000000000ULL}, // -1 -> +1
        {0x3FF123456789ABCDULL, 0xBFF123456789ABCDULL}, // positive finite
        {0xBFF123456789ABCDULL, 0x3FF123456789ABCDULL}, // negative finite
        {0x0000000000000000ULL, 0x8000000000000000ULL}, // +0 -> -0
        {0x8000000000000000ULL, 0x0000000000000000ULL}, // -0 -> +0
        {0x7FF0000000000000ULL, 0xFFF0000000000000ULL}, // +inf -> -inf
        {0xFFF0000000000000ULL, 0x7FF0000000000000ULL}, // -inf -> +inf
        {0x7FF8123456789ABCULL, 0xFFF8123456789ABCULL}, // quiet NaN payload
    }};
    for (const auto& [source_raw, expected_raw] : values)
    {
        const Execution result = execute(12U, 10U, source_raw);
        assert(result.state.fpr[12] == expected_raw);
        assert(result.state.fpr_ps1[12] == 0xAABBCCDDEEFF0011ULL);
    }

    constexpr std::uint64_t signaling_nan = 0x7FF0000000004321ULL;
    const Execution signaling = execute(12U, 10U, signaling_nan);
    assert(signaling.state.fpr[12] == 0xFFF0000000004321ULL);
    assert((signaling.state.fpr[12] & 0x0008000000000000ULL) == 0U);
    assert(signaling.state.fpscr == 0xD5C3BEEFU);

    const Execution alias = execute(5U, 5U, signaling_nan, 0x123456789ABCDEF0ULL);
    assert(alias.state.fpr[5] == 0xFFF0000000004321ULL);
    assert(alias.state.fpr_ps1[5] == 0x123456789ABCDEF0ULL);

    // Wind Waker's exact instruction and incoming register value.
    const Execution wind_waker = execute(
        12U, 10U, 0x40620ACD80000000ULL, 0x4030123456789ABCULL);
    assert(wind_waker.state.fpr[12] == 0xC0620ACD80000000ULL);
    assert(wind_waker.state.fpr_ps1[12] == 0x4030123456789ABCULL);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fneg f10.ps0=0x40620ACD80000000 -> f12.ps0=0xC0620ACD80000000") !=
           std::string::npos);
    assert(trace.find("ps1 unchanged=0x4030123456789ABC") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_absolute_value_tests()
{
    constexpr std::uint32_t wind_waker_word = 0xFC200A10U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_absolute_value);
    assert(decoded.fp_register == 1U);
    assert(decoded.fp_source_b == 1U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 264U);
    assert(((wind_waker_word >> 16U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(decode(wind_waker_word | 0x00010000U).opcode == Opcode::unsupported);
    assert(decode(wind_waker_word | 1U).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [](std::uint8_t destination, std::uint8_t source,
                            std::uint64_t source_raw,
                            std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x400U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source] = source_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = 0xD4C3BEEFU;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        const std::uint32_t instruction = 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 11U) | (264U << 1U);
        core.memory.write32_be(0U, instruction);
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.fpscr == 0xD4C3BEEFU);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.state.fpr[destination] ==
               (source_raw & 0x7FFFFFFFFFFFFFFFULL));
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        assert(core.memory.read32_be(8U) == memory_before);
        if (destination != source)
        {
            assert(core.state.fpr[source] == source_raw);
        }
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 9> cases{{
        {0x0000000000000000ULL, 0x0000000000000000ULL}, // +0
        {0x8000000000000000ULL, 0x0000000000000000ULL}, // -0 -> +0
        {0x3FF0000000000000ULL, 0x3FF0000000000000ULL}, // +1
        {0xBFF0000000000000ULL, 0x3FF0000000000000ULL}, // -1 -> +1
        {0xBFF8000000000000ULL, 0x3FF8000000000000ULL}, // -1.5 -> +1.5
        {0x7FF0000000000000ULL, 0x7FF0000000000000ULL}, // +inf
        {0xFFF0000000000000ULL, 0x7FF0000000000000ULL}, // -inf -> +inf
        {0xFFF8123456789ABCULL, 0x7FF8123456789ABCULL}, // negative qNaN
        {0xFFF0123456789ABCULL, 0x7FF0123456789ABCULL}, // negative sNaN
    }};
    for (const auto& [source_raw, expected_raw] : cases)
    {
        const Execution result = execute(1U, 1U, source_raw);
        assert(result.state.fpr[1] == expected_raw);
        assert(result.state.fpr_ps1[1] == 0xDEADBEEF01234567ULL);
        assert(result.state.fpscr == 0xD4C3BEEFU);
    }

    // An sNaN remains signaling and does not raise an arithmetic exception.
    const Execution signaling_nan = execute(1U, 1U, 0xFFF0123456789ABCULL);
    assert((signaling_nan.state.fpr[1] & 0x0008000000000000ULL) == 0U);
    assert((signaling_nan.state.fpscr & fpscr::vxsnan_mask) == 0U);

    const Execution distinct = execute(3U, 4U, 0xBFF8000000000000ULL,
                                        0x0123456789ABCDEFULL);
    assert(distinct.state.fpr[3] == 0x3FF8000000000000ULL);
    assert(distinct.state.fpr[4] == 0xBFF8000000000000ULL);
    assert(distinct.state.fpr_ps1[3] == 0x0123456789ABCDEFULL);
    assert(distinct.state.fpr_ps1[4] == 0x2222000000000004ULL);

    // Wind Waker's exact aliasing case keeps +0 and the prior PS1 lane intact.
    constexpr std::uint64_t wind_waker_ps1 = 0x3FF6666660000000ULL;
    const Execution wind_waker = execute(1U, 1U, 0x0000000000000000ULL,
                                          wind_waker_ps1);
    assert(wind_waker.state.fpr[1] == 0x0000000000000000ULL);
    assert(wind_waker.state.fpr_ps1[1] == wind_waker_ps1);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fabs f1.ps0=0x0000000000000000 -> f1.ps0=0x0000000000000000") !=
           std::string::npos);
    assert(trace.find("ps1 unchanged=0x3FF6666660000000") != std::string::npos);
}

void floating_subtract_double_tests()
{
    const auto encode_fsub = [](std::uint8_t destination, std::uint8_t source_a,
                                std::uint8_t source_b, bool record = false,
                                std::uint8_t reserved_c = 0U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(reserved_c) << 6U) | (20U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFD8B4828U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_subtract_double);
    assert(decoded.fp_register == 12U);
    assert(decoded.fp_source_a == 11U);
    assert(decoded.fp_source_b == 9U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 20U);
    assert(((wind_waker_word >> 6U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(decode(encode_fsub(12U, 11U, 9U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fsub(12U, 11U, 9U, true)).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint64_t a_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x500U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_b] = 0x4040000000000000ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U, encode_fsub(destination, source_a, source_b));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t five = 0x4014000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t negative_five = 0xC014000000000000ULL;
    assert(execute(3U, 1U, 2U, five, two).state.fpr[3] == three);
    assert(execute(3U, 1U, 2U, two, five).state.fpr[3] == negative_three);
    const Execution equal = execute(3U, 1U, 2U, five, five);
    assert(equal.state.fpr[3] == 0x0000000000000000ULL);
    assert((equal.state.fpscr & fpscr::fprf_mask) == 0x00002000U);
    assert(execute(3U, 1U, 2U, negative_five, negative_two).state.fpr[3] ==
           negative_three);

    // Keep meaningful binary64 bits that a binary32 intermediate would erase.
    constexpr std::uint64_t two_to_minus_53 = 0x3CA0000000000000ULL;
    const Execution precision = execute(3U, 1U, 2U, one, two_to_minus_53);
    assert(precision.state.fpr[3] == 0x3FEFFFFFFFFFFFFFULL);
    assert(precision.state.fpr[3] != std::bit_cast<std::uint64_t>(
        static_cast<double>(static_cast<float>(std::bit_cast<double>(precision.state.fpr[3])))));
    const Execution inexact = execute(
        3U, 1U, 2U, one, 0x3C90000000000000ULL);
    assert(inexact.state.fpr[3] == one);
    assert((inexact.state.fpscr & fpscr::fi_mask) != 0U);
    assert((inexact.state.fpscr & fpscr::fr_mask) != 0U);
    assert((inexact.state.fpscr & (fpscr::xx_mask | fpscr::fx_mask)) ==
           (fpscr::xx_mask | fpscr::fx_mask));

    assert(execute(1U, 1U, 2U, five, two).state.fpr[1] == three);
    assert(execute(2U, 1U, 2U, five, two).state.fpr[2] == three);

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t qnan = 0x7FF8000000001234ULL;
    constexpr std::uint64_t snan = 0x7FF0000000001234ULL;
    const Execution positive_zero_result = execute(3U, 1U, 2U, positive_zero, negative_zero);
    assert(positive_zero_result.state.fpr[3] == positive_zero);
    assert((positive_zero_result.state.fpscr & fpscr::fprf_mask) == 0x00002000U);
    const Execution negative_zero_result = execute(3U, 1U, 2U, negative_zero, positive_zero);
    assert(negative_zero_result.state.fpr[3] == negative_zero);
    assert((negative_zero_result.state.fpscr & fpscr::fprf_mask) == 0x00012000U);
    assert((execute(3U, 1U, 2U, positive_inf, one).state.fpr[3]) == positive_inf);
    assert((execute(3U, 1U, 2U, negative_inf, one).state.fpr[3]) == negative_inf);
    const Execution positive_subnormal = execute(
        3U, 1U, 2U, 0x0000000000000001ULL, positive_zero);
    assert((positive_subnormal.state.fpscr & fpscr::fprf_mask) == 0x00014000U);
    const Execution negative_subnormal = execute(
        3U, 1U, 2U, positive_zero, 0x0000000000000001ULL);
    assert((negative_subnormal.state.fpscr & fpscr::fprf_mask) == 0x00018000U);

    // Binary64 FPRF classification covers signed normal, zero, infinity, NaN, subnormal.
    assert((execute(3U, 1U, 2U, five, two).state.fpscr & fpscr::fprf_mask) ==
           0x00004000U);
    assert((execute(3U, 1U, 2U, two, five).state.fpscr & fpscr::fprf_mask) ==
           0x00008000U);
    assert((execute(3U, 1U, 2U, positive_inf, negative_inf).state.fpscr &
            fpscr::fprf_mask) == 0x00005000U);
    assert((execute(3U, 1U, 2U, negative_inf, positive_inf).state.fpscr &
            fpscr::fprf_mask) == 0x00009000U);
    const Execution quiet_nan = execute(3U, 1U, 2U, qnan, one);
    assert((quiet_nan.state.fpr[3] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    assert((quiet_nan.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet_nan.state.fpscr & fpscr::fprf_mask) == 0x00011000U);

    const Execution signaling_nan = execute(3U, 1U, 2U, snan, one);
    assert(signaling_nan.state.fpr[3] == (snan | 0x0008000000000000ULL));
    assert((signaling_nan.state.fpscr & (fpscr::vxsnan_mask | fpscr::vx_mask |
                                         fpscr::fx_mask)) ==
           (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    const Execution signaling_suppressed = execute(
        3U, 1U, 2U, snan, one, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution same_positive_infinity = execute(3U, 1U, 2U, positive_inf, positive_inf);
    assert((same_positive_infinity.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert((same_positive_infinity.state.fpr[3] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    assert((same_positive_infinity.state.fpscr & (fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vx_mask | fpscr::fx_mask));
    const Execution same_negative_infinity = execute(3U, 1U, 2U, negative_inf, negative_inf);
    assert((same_negative_infinity.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert(execute(3U, 1U, 2U, positive_inf, negative_inf).state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, negative_inf, positive_inf).state.fpr[3] == negative_inf);
    const Execution invalid_suppressed = execute(
        3U, 1U, 2U, positive_inf, positive_inf, fpscr::ve_mask);
    assert(invalid_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((invalid_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    // A prior FI/FR is cleared by this exactly representable result; sticky XX stays sticky.
    const Execution wind_waker = execute(
        12U, 11U, 9U, 0x4330000080000000ULL, 0x4330000080000000ULL,
        fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask | 1U);
    assert(wind_waker.state.fpr[12] == 0x0000000000000000ULL);
    assert(wind_waker.state.fpr_ps1[12] == 0x222200000000000CULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00002000U);
    assert((wind_waker.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    assert((wind_waker.state.fpscr & 1U) != 0U);
    const Execution exact_wind_waker = execute(
        12U, 11U, 9U, 0x4330000080000000ULL, 0x4330000080000000ULL, 1U);
    assert((exact_wind_waker.state.fpscr &
            (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask)) == 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fsub f11=4503601774854144 [0x4330000080000000] ") !=
           std::string::npos);
    assert(trace.find("f9=4503601774854144 [0x4330000080000000]") !=
           std::string::npos);
    assert(trace.find("-> f12=0 [0x0000000000000000]") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_add_double_tests()
{
    const auto encode_fadd = [](std::uint8_t destination, std::uint8_t source_a,
                                std::uint8_t source_b, bool record = false,
                                std::uint8_t reserved_c = 0U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(reserved_c) << 6U) | (21U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFDAA682AU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_add_double);
    assert(decoded.fp_register == 13U);
    assert(decoded.fp_source_a == 10U);
    assert(decoded.fp_source_b == 13U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 21U);
    assert(((wind_waker_word >> 6U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fadd(13U, 10U, 13U) == wind_waker_word);
    assert(decode(encode_fadd(13U, 10U, 13U, false, 1U)).opcode ==
           Opcode::unsupported);
    assert(decode(encode_fadd(13U, 10U, 13U, true)).opcode ==
           Opcode::unsupported);
    assert(decode((encode_fadd(13U, 10U, 13U, false, 0U) & ~(0x1FU << 1U)) |
                  (20U << 1U)).opcode == Opcode::floating_subtract_double);
    assert(decode((encode_fadd(13U, 10U, 13U, false, 0U) & ~(0x1FU << 1U)) |
                  (25U << 1U)).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint64_t a_raw,
                             std::uint64_t b_raw, std::uint32_t fpscr_value = 0U,
                             std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x500U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U, encode_fadd(destination, source_a, source_b));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t next_one = 0x3FF0000000000001ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t negative_one = 0xBFF0000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t negative_five = 0xC014000000000000ULL;
    constexpr std::uint64_t four = 0x4010000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t qnan_a = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t qnan_b = 0xFFF823456789ABCDULL;
    constexpr std::uint64_t snan_a = 0x7FF0000000004321ULL;
    constexpr std::uint64_t snan_b = 0xFFF0000000005432ULL;

    assert(execute(3U, 1U, 2U, one, two).state.fpr[3] == three);
    assert(execute(3U, 1U, 2U, negative_one, two).state.fpr[3] == one);
    assert(execute(3U, 1U, 2U, negative_two, negative_three)
               .state.fpr[3] == negative_five);
    assert(execute(3U, 1U, 2U, 0x3FE0000000000000ULL,
                   0x3FD0000000000000ULL).state.fpr[3] == 0x3FE8000000000000ULL);

    assert(execute(3U, 1U, 2U, positive_zero, positive_zero)
               .state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, negative_zero, negative_zero)
               .state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, positive_zero, negative_zero)
               .state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, next_one, positive_zero).state.fpr[3] == next_one);

    const auto half_ulp = std::bit_cast<std::uint64_t>(std::ldexp(1.0, -54));
    const auto three_quarters_ulp =
        std::bit_cast<std::uint64_t>(std::ldexp(3.0, -54));
    const Execution inexact_down = execute(3U, 1U, 2U, one, half_ulp);
    assert(inexact_down.state.fpr[3] == one);
    assert((inexact_down.state.fpscr &
            (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask)) ==
           (fpscr::fi_mask | fpscr::xx_mask));
    const Execution inexact_up = execute(3U, 1U, 2U, one, three_quarters_ulp);
    assert(inexact_up.state.fpr[3] == next_one);
    assert((inexact_up.state.fpscr &
            (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask)) ==
           (fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask));

    // Aliasing destination A or B must use the original operands.
    assert(execute(1U, 1U, 2U, one, two).state.fpr[1] == three);
    assert(execute(2U, 1U, 2U, one, two).state.fpr[2] == three);
    assert(execute(5U, 5U, 5U, two, two).state.fpr[5] == four);

    assert(execute(3U, 1U, 2U, positive_inf, positive_inf).state.fpr[3] ==
           positive_inf);
    assert(execute(3U, 1U, 2U, negative_inf, negative_inf).state.fpr[3] ==
           negative_inf);
    assert(execute(3U, 1U, 2U, positive_inf, one).state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, negative_inf, one).state.fpr[3] == negative_inf);
    for (const auto& [a, b] : {std::pair{positive_inf, negative_inf},
                               std::pair{negative_inf, positive_inf}})
    {
        const Execution invalid = execute(3U, 1U, 2U, a, b);
        assert(invalid.state.fpr[3] == 0x7FF8000000000000ULL);
        assert((invalid.state.fpscr &
                (fpscr::vxisi_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vxisi_mask | fpscr::vx_mask | fpscr::fx_mask));
    }
    const Execution infinity_suppressed = execute(
        3U, 1U, 2U, positive_inf, negative_inf, fpscr::ve_mask);
    assert(infinity_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((infinity_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    assert(execute(3U, 1U, 2U, qnan_a, one).state.fpr[3] == qnan_a);
    assert(execute(3U, 1U, 2U, one, qnan_b).state.fpr[3] == qnan_b);
    const Execution qnan_priority = execute(3U, 1U, 2U, qnan_a, qnan_b);
    assert(qnan_priority.state.fpr[3] == qnan_a);
    assert((qnan_priority.state.fpscr & fpscr::vxsnan_mask) == 0U);
    for (const auto& [a, b, expected] : {
             std::tuple{snan_a, one, snan_a},
             std::tuple{one, snan_b, snan_b},
             std::tuple{snan_a, snan_b, snan_a}})
    {
        const Execution signaling = execute(3U, 1U, 2U, a, b);
        assert(signaling.state.fpr[3] == (expected | 0x0008000000000000ULL));
        assert((signaling.state.fpscr &
                (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    }
    const Execution signaling_suppressed = execute(
        2U, 1U, 2U, one, snan_b, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[2] == snan_b);
    assert(signaling_suppressed.state.fpr_ps1[2] == 0xDEADBEEF01234567ULL);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    constexpr std::uint64_t wind_a = 0x3EB0000000000000ULL;
    constexpr std::uint64_t wind_b = 0x404AA17F41ECFE91ULL;
    const Execution wind_waker = execute(
        13U, 10U, 13U, wind_a, wind_b,
        fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask | 1U,
        0xDEADBEEF01234567ULL);
    assert(wind_waker.state.fpr[13] == 0x404AA17F49ECFE91ULL);
    assert(wind_waker.state.fpr_ps1[13] == 0xDEADBEEF01234567ULL);
    assert(wind_waker.state.fpr[10] == wind_a);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    assert((wind_waker.state.fpscr & 1U) != 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fadd f10=") != std::string::npos);
    assert(trace.find("[0x3EB0000000000000]") != std::string::npos);
    assert(trace.find("f13=53.261696091384813 [0x404AA17F41ECFE91]") !=
           std::string::npos);
    assert(trace.find("[0x404AA17F49ECFE91]") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_multiply_double_tests()
{
    const auto encode_fmul = [](std::uint8_t destination, std::uint8_t source_a,
                                std::uint8_t source_c, bool record = false,
                                std::uint8_t reserved_b = 0U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(reserved_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (25U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFC1E02F2U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_multiply_double);
    assert(decoded.fp_register == 0U);
    assert(decoded.fp_source_a == 30U);
    assert(decoded.fp_source_c == 11U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 25U);
    assert(((wind_waker_word >> 11U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(decode(encode_fmul(0U, 30U, 11U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fmul(0U, 30U, 11U, true)).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_c, std::uint64_t a_raw,
                             std::uint64_t c_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x500U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_c] = c_raw;
        core.state.fpr_ps1[source_a] = 0xC040000000000000ULL;
        core.state.fpr_ps1[source_c] = 0x4040000000000000ULL;
        core.state.fpr_ps1[destination] = 0xDEADBEEF01234567ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U, encode_fmul(destination, source_a, source_c));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t one_plus_ulp = 0x3FF0000000000001ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t six = 0x4018000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_six = 0xC018000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t qnan_a = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t snan_a = 0xFFF0000000004321ULL;

    assert(execute(3U, 1U, 2U, two, three).state.fpr[3] == six);
    assert(execute(3U, 1U, 2U, negative_two, three).state.fpr[3] == negative_six);
    assert(execute(3U, 1U, 2U, 0x3FE0000000000000ULL,
                   0x3FD0000000000000ULL).state.fpr[3] ==
           0x3FC0000000000000ULL);
    assert(execute(3U, 1U, 2U, one_plus_ulp, one).state.fpr[3] == one_plus_ulp);

    // fmul uses the exact binary64 C operand, not fmuls' reduced-precision path.
    constexpr std::uint64_t low_bits_c = 0x3FF0000008000001ULL;
    const Execution no_25bit_rounding = execute(3U, 1U, 2U, one, low_bits_c);
    assert(no_25bit_rounding.state.fpr[3] == low_bits_c);
    const std::uint64_t force25_c =
        (low_bits_c & 0xFFFFFFFFF8000000ULL) + (low_bits_c & 0x0000000008000000ULL);
    assert(no_25bit_rounding.state.fpr[3] != force25_c);

    const Execution signed_zero = execute(3U, 1U, 2U, negative_zero, two);
    assert(signed_zero.state.fpr[3] == negative_zero);
    assert((signed_zero.state.fpscr & fpscr::fprf_mask) == 0x00012000U);
    assert(execute(3U, 1U, 2U, negative_zero, negative_two).state.fpr[3] == positive_zero);
    assert(execute(3U, 1U, 2U, positive_inf, two).state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, negative_inf, two).state.fpr[3] == negative_inf);

    // Destination aliases either source, including the all-identical form.
    assert(execute(1U, 1U, 2U, two, three).state.fpr[1] == six);
    assert(execute(2U, 1U, 2U, two, three).state.fpr[2] == six);
    assert(execute(5U, 5U, 5U, two, two).state.fpr[5] ==
           0x4010000000000000ULL);

    const Execution inexact = execute(3U, 1U, 2U, one_plus_ulp, one_plus_ulp);
    assert((inexact.state.fpscr & fpscr::fi_mask) != 0U);
    assert((inexact.state.fpscr & (fpscr::xx_mask | fpscr::fx_mask)) ==
           (fpscr::xx_mask | fpscr::fx_mask));
    assert((inexact.state.fpscr & fpscr::fprf_mask) == 0x00004000U);

    const Execution quiet_nan = execute(3U, 1U, 2U, qnan_a, one);
    assert(quiet_nan.state.fpr[3] == qnan_a);
    assert((quiet_nan.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet_nan.state.fpscr & fpscr::fprf_mask) == 0x00011000U);

    const Execution signaling_nan = execute(3U, 1U, 2U, snan_a, one);
    assert(signaling_nan.state.fpr[3] == (snan_a | 0x0008000000000000ULL));
    assert((signaling_nan.state.fpscr & (fpscr::vxsnan_mask | fpscr::vx_mask |
                                         fpscr::fx_mask)) ==
           (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    const Execution signaling_suppressed = execute(
        3U, 1U, 2U, snan_a, one, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert(signaling_suppressed.state.fpr_ps1[3] == 0xDEADBEEF01234567ULL);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution inf_zero = execute(3U, 1U, 2U, positive_inf, negative_zero);
    assert(inf_zero.state.fpr[3] == 0x7FF8000000000000ULL);
    assert((inf_zero.state.fpscr & (fpscr::vximz_mask | fpscr::vx_mask |
                                    fpscr::fx_mask)) ==
           (fpscr::vximz_mask | fpscr::vx_mask | fpscr::fx_mask));
    const Execution inf_zero_suppressed = execute(
        3U, 1U, 2U, positive_inf, positive_zero, fpscr::ve_mask);
    assert(inf_zero_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((inf_zero_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    // Wind Waker's exact blocker: fmul f0,f30,f11 with a zero second operand.
    const Execution wind_waker = execute(
        0U, 30U, 11U, 0x401921FB60000000ULL, positive_zero);
    assert(wind_waker.state.fpr[0] == positive_zero);
    assert(wind_waker.state.fpr_ps1[0] == 0xDEADBEEF01234567ULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00002000U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fmul f30=6.2831854820251465 [0x401921FB60000000]") !=
           std::string::npos);
    assert(trace.find("f11=0 [0x0000000000000000]") != std::string::npos);
    assert(trace.find("-> f0=0 [0x0000000000000000]") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_multiply_add_double_tests()
{
    const auto encode_fmadd = [](std::uint8_t destination, std::uint8_t source_a,
                                 std::uint8_t source_b, std::uint8_t source_c,
                                 bool record = false, std::uint32_t xo = 29U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (xo << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFDA06B3AU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_multiply_add_double);
    assert(decoded.fp_register == 13U);
    assert(decoded.fp_source_a == 0U);
    assert(decoded.fp_source_b == 13U);
    assert(decoded.fp_source_c == 12U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 29U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fmadd(13U, 0U, 13U, 12U) == wind_waker_word);
    assert(decode(encode_fmadd(13U, 0U, 13U, 12U, true)).opcode ==
           Opcode::unsupported);
    assert(decode(encode_fmadd(13U, 0U, 13U, 12U, false, 30U)).opcode ==
           Opcode::floating_negative_multiply_subtract_double);
    for (const std::uint32_t sibling_xo : {28U, 31U})
    {
        assert(decode(encode_fmadd(13U, 0U, 13U, 12U, false, sibling_xo)).opcode ==
               Opcode::unsupported);
    }

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint8_t source_c,
                             std::uint64_t a_raw, std::uint64_t b_raw,
                             std::uint64_t c_raw, std::uint32_t fpscr_value = 0U,
                             std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x700U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr[source_c] = c_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U,
            encode_fmadd(destination, source_a, source_b, source_c));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t four = 0x4010000000000000ULL;
    constexpr std::uint64_t negative_one = 0xBFF0000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_three = 0xC008000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t quiet_nan_a = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t quiet_nan_b = 0xFFF823456789ABCDULL;
    constexpr std::uint64_t quiet_nan_c = 0x7FF83456789ABCDEULL;
    constexpr std::uint64_t signaling_nan_a = 0x7FF0000000004321ULL;
    constexpr std::uint64_t signaling_nan_b = 0xFFF0000000005432ULL;
    constexpr std::uint64_t signaling_nan_c = 0x7FF0000000006543ULL;

    assert(execute(3U, 1U, 2U, 4U, two, four, three).state.fpr[3] == 0x4024000000000000ULL);
    assert(execute(3U, 1U, 2U, 4U, negative_two, four, three).state.fpr[3] == negative_two);
    assert(execute(3U, 1U, 2U, 4U, two, four, negative_three).state.fpr[3] == negative_two);
    assert(execute(3U, 1U, 2U, 4U, positive_zero, four, three).state.fpr[3] == four);
    assert(execute(3U, 1U, 2U, 4U, one, one, one).state.fpr[3] == two);

    // The rounded product is 1.0, but a fused operation preserves its residual.
    const double a_fused = 1.0 + std::ldexp(1.0, -27);
    const double c_fused = 1.0 - std::ldexp(1.0, -27);
    assert(a_fused * c_fused == 1.0);
    const auto a_fused_raw = std::bit_cast<std::uint64_t>(a_fused);
    const auto c_fused_raw = std::bit_cast<std::uint64_t>(c_fused);
    assert(execute(3U, 1U, 2U, 4U, a_fused_raw, negative_one, c_fused_raw)
               .state.fpr[3] == 0xBC90000000000000ULL);

    // Destination aliases each source; all operands are read before the write.
    assert(execute(1U, 1U, 2U, 4U, two, four, three).state.fpr[1] ==
           0x4024000000000000ULL);
    assert(execute(2U, 1U, 2U, 4U, two, four, three).state.fpr[2] ==
           0x4024000000000000ULL);
    assert(execute(4U, 1U, 2U, 4U, two, four, three).state.fpr[4] ==
           0x4024000000000000ULL);
    assert(execute(5U, 5U, 5U, 5U, two, two, two).state.fpr[5] ==
           0x4018000000000000ULL);

    // NaN selection follows architectural source priority A, C, then B.
    assert(execute(3U, 1U, 2U, 4U, quiet_nan_a, quiet_nan_b, quiet_nan_c)
               .state.fpr[3] == quiet_nan_a);
    assert(execute(3U, 1U, 2U, 4U, one, quiet_nan_b, quiet_nan_c)
               .state.fpr[3] == quiet_nan_c);
    assert(execute(3U, 1U, 2U, 4U, one, quiet_nan_b, one)
               .state.fpr[3] == quiet_nan_b);
    const auto quiet_with_snan = execute(
        3U, 1U, 2U, 4U, quiet_nan_a, signaling_nan_b, quiet_nan_c);
    assert(quiet_with_snan.state.fpr[3] ==
           (signaling_nan_b | 0x0008000000000000ULL));
    assert((quiet_with_snan.state.fpscr & fpscr::vxsnan_mask) != 0U);

    for (const auto& [a_raw, b_raw, c_raw, expected] : {
             std::tuple{signaling_nan_a, one, one, signaling_nan_a},
             std::tuple{one, signaling_nan_b, one, signaling_nan_b},
             std::tuple{one, one, signaling_nan_c, signaling_nan_c}})
    {
        const auto invalid = execute(3U, 1U, 2U, 4U, a_raw, b_raw, c_raw);
        assert(invalid.state.fpr[3] ==
               (expected | 0x0008000000000000ULL));
        assert((invalid.state.fpscr &
                (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    }
    const auto signaling_suppressed = execute(
        2U, 1U, 2U, 4U, one, signaling_nan_b, one, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[2] == signaling_nan_b);
    assert(signaling_suppressed.state.fpr_ps1[2] == 0xDEADBEEF01234567ULL);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const auto vximz = execute(3U, 1U, 2U, 4U, positive_inf, four, positive_zero);
    assert(vximz.state.fpr[3] == 0x7FF8000000000000ULL);
    assert((vximz.state.fpscr &
            (fpscr::vximz_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
           (fpscr::vximz_mask | fpscr::vx_mask | fpscr::fx_mask));
    assert((execute(3U, 1U, 2U, 4U, positive_zero, four, positive_inf)
                .state.fpscr & fpscr::vximz_mask) != 0U);
    const auto vximz_suppressed = execute(
        3U, 1U, 2U, 4U, positive_inf, four, positive_zero, fpscr::ve_mask);
    assert(vximz_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((vximz_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const auto vxisi_positive = execute(3U, 1U, 2U, 4U,
                                         positive_inf, negative_inf, one);
    const auto vxisi_negative = execute(3U, 1U, 2U, 4U,
                                         negative_inf, positive_inf, one);
    assert((vxisi_positive.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert((vxisi_negative.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert(execute(3U, 1U, 2U, 4U, positive_inf, positive_inf, one)
               .state.fpr[3] == positive_inf);
    assert(execute(3U, 1U, 2U, 4U, negative_inf, negative_inf, one)
               .state.fpr[3] == negative_inf);

    // Wind Waker regression: destination aliases B, while the double result writes PS0 only.
    constexpr std::uint64_t wind_a = 0xBD6AA924B640F52DULL;
    constexpr std::uint64_t wind_b = 0x3DE612064D31FD1CULL;
    constexpr std::uint64_t wind_c = 0x3EA3BD3CDC2CAB20ULL;
    const auto wind_waker = execute(13U, 0U, 13U, 12U, wind_a, wind_b, wind_c,
                                    0x00000001U, 0xAABBCCDDEEFF0011ULL);
    assert(wind_waker.state.fpr[13] == 0x3DE612064C2ADBA4ULL);
    assert(wind_waker.state.fpr_ps1[13] == 0xAABBCCDDEEFF0011ULL);
    assert(wind_waker.state.fpr[0] == wind_a);
    assert(wind_waker.state.fpr[12] == wind_c);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fmadd f0=-7.5774313556780005E-13 [0xBD6AA924B640F52D]") !=
           std::string::npos);
    assert(trace.find("f12=5.8827429714100916E-07 [0x3EA3BD3CDC2CAB20]") !=
           std::string::npos);
    assert(trace.find("f13=1.605833578937085E-10 [0x3DE612064D31FD1C]") !=
           std::string::npos);
    assert(trace.find("-> f13=") != std::string::npos);
    assert(trace.find("[0x3DE612064C2ADBA4]") != std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_negative_multiply_subtract_double_tests()
{
    const auto encode_fnmsub = [](std::uint8_t destination, std::uint8_t source_a,
                                  std::uint8_t source_b, std::uint8_t source_c,
                                  bool record = false) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (30U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFFC0F07CU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_negative_multiply_subtract_double);
    assert(decoded.fp_register == 30U);
    assert(decoded.fp_source_a == 0U);
    assert(decoded.fp_source_b == 30U);
    assert(decoded.fp_source_c == 1U);
    assert(((wind_waker_word >> 1U) & 0x1FU) == 30U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fnmsub(30U, 0U, 30U, 1U) == wind_waker_word);
    assert(decode(encode_fnmsub(30U, 0U, 30U, 1U, true)).opcode == Opcode::unsupported);
    const std::uint32_t fmadd_encoding =
        (wind_waker_word & ~(0x1FU << 1U)) | (29U << 1U);
    assert(decode(fmadd_encoding).opcode == Opcode::floating_multiply_add_double);
    for (const std::uint32_t sibling_xo : {28U, 31U})
    {
        const std::uint32_t sibling_encoding =
            (wind_waker_word & ~(0x1FU << 1U)) | (sibling_xo << 1U);
        assert(decode(sibling_encoding).opcode == Opcode::unsupported);
    }

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source_a,
                             std::uint8_t source_b, std::uint8_t source_c,
                             std::uint64_t a_raw, std::uint64_t b_raw,
                             std::uint64_t c_raw, std::uint32_t fpscr_value = 0U,
                             std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x700U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source_a] = a_raw;
        core.state.fpr[source_b] = b_raw;
        core.state.fpr[source_c] = c_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U,
            encode_fnmsub(destination, source_a, source_b, source_c));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };

    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t two = 0x4000000000000000ULL;
    constexpr std::uint64_t three = 0x4008000000000000ULL;
    constexpr std::uint64_t four = 0x4010000000000000ULL;
    constexpr std::uint64_t negative_two = 0xC000000000000000ULL;
    constexpr std::uint64_t negative_five = 0xC014000000000000ULL;
    constexpr std::uint64_t seven = 0x401C000000000000ULL;
    constexpr std::uint64_t positive_inf = 0x7FF0000000000000ULL;
    constexpr std::uint64_t negative_inf = 0xFFF0000000000000ULL;
    constexpr std::uint64_t quiet_nan_a = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t quiet_nan_b = 0xFFF823456789ABCDULL;
    constexpr std::uint64_t quiet_nan_c = 0x7FF83456789ABCDEULL;
    constexpr std::uint64_t signaling_nan = 0xFFF0000000004321ULL;

    assert(execute(3U, 1U, 2U, 4U, two, 0x4024000000000000ULL, three)
               .state.fpr[3] == four); // 10 - 2*3
    assert(execute(3U, 1U, 2U, 4U, two, one, three).state.fpr[3] == negative_five);
    assert(execute(3U, 1U, 2U, 4U, negative_two, one, three).state.fpr[3] == seven);
    assert(execute(3U, 1U, 2U, 4U, positive_zero, four, three)
               .state.fpr[3] == four);

    // Mandatory fused regression: separately rounded a*c becomes 1, but FMA retains -2^-54.
    const double a_fused = 1.0 + std::ldexp(1.0, -27);
    const double c_fused = 1.0 - std::ldexp(1.0, -27);
    assert(a_fused * c_fused == 1.0);
    const std::uint64_t a_fused_raw = std::bit_cast<std::uint64_t>(a_fused);
    const std::uint64_t c_fused_raw = std::bit_cast<std::uint64_t>(c_fused);
    const Execution fused = execute(3U, 1U, 2U, 4U, a_fused_raw, one, c_fused_raw);
    assert(fused.state.fpr[3] == 0x3C90000000000000ULL);

    // Destination aliases each source independently, and all inputs are snapshotted.
    assert(execute(1U, 1U, 2U, 4U, two, 0x4024000000000000ULL, three)
               .state.fpr[1] == four);
    assert(execute(2U, 1U, 2U, 4U, two, 0x4024000000000000ULL, three)
               .state.fpr[2] == four);
    assert(execute(4U, 1U, 2U, 4U, two, 0x4024000000000000ULL, three)
               .state.fpr[4] == four);
    assert(execute(5U, 5U, 5U, 5U, two, two, two).state.fpr[5] == negative_two);

    // Fused zero result is sign-negated; PS1 and all non-FP state remain unchanged.
    assert(execute(3U, 1U, 2U, 4U, positive_zero, positive_zero, one)
               .state.fpr[3] == negative_zero);
    assert(execute(3U, 1U, 2U, 4U, negative_zero, positive_zero, one)
               .state.fpr[3] == positive_zero);

    // NaNs use the established A, C, B priority and are never sign-negated.
    assert(execute(3U, 1U, 2U, 4U, quiet_nan_a, one, one).state.fpr[3] == quiet_nan_a);
    assert(execute(3U, 1U, 2U, 4U, one, quiet_nan_b, one).state.fpr[3] == quiet_nan_b);
    assert(execute(3U, 1U, 2U, 4U, one, one, quiet_nan_c).state.fpr[3] == quiet_nan_c);
    const Execution quiet_nan_result = execute(3U, 1U, 2U, 4U, quiet_nan_a, one, one);
    assert((quiet_nan_result.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet_nan_result.state.fpscr & fpscr::fprf_mask) == 0x00011000U);
    const Execution signaling_a = execute(3U, 1U, 2U, 4U, signaling_nan, one, one);
    const Execution signaling_b = execute(3U, 1U, 2U, 4U, one, signaling_nan, one);
    const Execution signaling_c = execute(3U, 1U, 2U, 4U, one, one, signaling_nan);
    for (const Execution* signaling : {&signaling_a, &signaling_b, &signaling_c})
    {
        assert(signaling->state.fpr[3] ==
               (signaling_nan | 0x0008000000000000ULL));
        assert((signaling->state.fpscr &
                (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    }
    const Execution signaling_suppressed = execute(
        2U, 1U, 2U, 4U, one, signaling_nan, one, fpscr::ve_mask);
    assert(signaling_suppressed.state.fpr[2] == signaling_nan);
    assert(signaling_suppressed.state.fpr_ps1[2] == 0xDEADBEEF01234567ULL);
    assert((signaling_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    // 0*infinity is VXIMZ; infinite product minus same-sign infinity is VXISI.
    const Execution vximz = execute(3U, 1U, 2U, 4U, positive_inf, one, positive_zero);
    assert(vximz.state.fpr[3] == 0x7FF8000000000000ULL);
    assert((vximz.state.fpscr & (fpscr::vximz_mask | fpscr::vx_mask |
                                 fpscr::fx_mask)) ==
           (fpscr::vximz_mask | fpscr::vx_mask | fpscr::fx_mask));
    const Execution vximz_reverse = execute(3U, 1U, 2U, 4U, positive_zero, one,
                                            negative_inf);
    assert((vximz_reverse.state.fpscr & fpscr::vximz_mask) != 0U);
    const Execution vximz_suppressed = execute(
        3U, 1U, 2U, 4U, positive_inf, one, positive_zero, fpscr::ve_mask);
    assert(vximz_suppressed.state.fpr[3] == 0x1111000000000003ULL);
    assert((vximz_suppressed.state.fpscr & fpscr::fex_mask) != 0U);

    const Execution vxisi_positive = execute(3U, 1U, 2U, 4U,
                                              positive_inf, positive_inf, one);
    assert((vxisi_positive.state.fpscr & fpscr::vxisi_mask) != 0U);
    const Execution vxisi_negative = execute(3U, 1U, 2U, 4U,
                                              negative_inf, negative_inf, one);
    assert((vxisi_negative.state.fpscr & fpscr::vxisi_mask) != 0U);
    assert(execute(3U, 1U, 2U, 4U, positive_inf, negative_inf, one)
               .state.fpr[3] == negative_inf);
    assert(execute(3U, 1U, 2U, 4U, negative_inf, positive_inf, one)
               .state.fpr[3] == positive_inf);

    // Wind Waker's destination aliases B; C=0 makes the result exactly B.
    constexpr std::uint64_t wind_b = 0x3F4921FB60000000ULL;
    constexpr std::uint64_t wind_c = 0x0000000000000000ULL;
    const Execution wind_waker = execute(
        30U, 0U, 30U, 1U, 0x3FF921FB54442D18ULL, wind_b, wind_c,
        fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask | 1U,
        0x401921FB60000000ULL);
    assert(wind_waker.state.fpr[30] == wind_b);
    assert(wind_waker.state.fpr_ps1[30] == 0x401921FB60000000ULL);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert((wind_waker.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((wind_waker.state.fpscr & fpscr::xx_mask) != 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fnmsub f0=1.5707963267948966 [0x3FF921FB54442D18]") !=
           std::string::npos);
    assert(trace.find("f1=0 [0x0000000000000000]") != std::string::npos);
    assert(trace.find("f30=0.00076699041528627276 [0x3F4921FB60000000]") !=
           std::string::npos);
    assert(trace.find("-> f30=0.00076699041528627276 [0x3F4921FB60000000]") !=
           std::string::npos);
    assert(trace.find("single=") == std::string::npos);
}

void floating_round_to_single_tests()
{
    const auto encode_frsp = [](std::uint8_t destination, std::uint8_t source,
                                bool record = false, std::uint8_t reserved_a = 0U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(reserved_a) << 16U) |
            (static_cast<std::uint32_t>(source) << 11U) | (12U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFD606018U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_round_to_single);
    assert(decoded.fp_register == 11U);
    assert(decoded.fp_source_a == 12U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 12U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_frsp(11U, 12U) == wind_waker_word);
    assert(decode(encode_frsp(11U, 12U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_frsp(11U, 12U, true)).opcode == Opcode::unsupported);
    assert(decode(0xFC00001AU).opcode == Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint64_t source_raw, std::uint32_t fpscr_value = 0U) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x600U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source] = source_raw;
        core.state.fpr_ps1[source] = 0xC040000000000000ULL; // deliberately irrelevant
        core.state.fpr_ps1[destination] = 0xAABBCCDDEEFF0011ULL;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = fpscr_value;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        const std::uint32_t instruction_word = encode_frsp(destination, source);
        core.memory.write32_be(0U, instruction_word);
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
                assert(core.state.fpr_ps1[index] == fpr_ps1_before[index]);
            }
        }
        assert(core.memory.read32_be(8U) == memory_before);
        execution.state = core.state;
        return execution;
    };
    const auto expected_extended_single = [](std::uint64_t raw) {
        const float rounded = static_cast<float>(std::bit_cast<double>(raw));
        return std::pair{
            std::bit_cast<std::uint64_t>(static_cast<double>(rounded)),
            std::bit_cast<std::uint32_t>(rounded)};
    };

    const Execution positive_zero = execute(11U, 12U, 0x0000000000000000ULL);
    assert(positive_zero.result.reason == StopReason::instruction_limit);
    assert(positive_zero.state.fpr[11] == 0x0000000000000000ULL);
    assert(positive_zero.state.fpr_ps1[11] == 0x0000000000000000ULL);
    assert((positive_zero.state.fpscr & fpscr::fprf_mask) == 0x00002000U);

    const Execution negative_zero = execute(11U, 12U, 0x8000000000000000ULL);
    assert(negative_zero.state.fpr[11] == 0x8000000000000000ULL);
    assert(negative_zero.state.fpr_ps1[11] == 0x8000000000000000ULL);
    assert((negative_zero.state.fpscr & fpscr::fprf_mask) == 0x00012000U);

    constexpr std::uint64_t exact_positive = 0x3FF8000000000000ULL; // 1.5
    constexpr std::uint64_t exact_negative = 0xBFF8000000000000ULL; // -1.5
    const Execution exact = execute(11U, 12U, exact_positive,
                                    fpscr::fi_mask | fpscr::fr_mask | fpscr::xx_mask);
    assert(exact.state.fpr[11] == exact_positive);
    assert(exact.state.fpr_ps1[11] == exact_positive);
    assert((exact.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((exact.state.fpscr & fpscr::xx_mask) != 0U);
    const Execution negative = execute(11U, 12U, exact_negative);
    assert(negative.state.fpr[11] == exact_negative);
    assert((negative.state.fpscr & fpscr::fprf_mask) == 0x00008000U);

    const Execution rounded_down = execute(
        11U, 12U,
        std::bit_cast<std::uint64_t>(1.0 + std::ldexp(1.0, -24)));
    assert(rounded_down.state.fpr[11] == 0x3FF0000000000000ULL);
    assert((rounded_down.state.fpscr & fpscr::fi_mask) != 0U);
    assert((rounded_down.state.fpscr & fpscr::fr_mask) == 0U);
    assert((rounded_down.state.fpscr & fpscr::xx_mask) != 0U);
    const Execution rounded_up = execute(
        11U, 12U,
        std::bit_cast<std::uint64_t>(1.0 + 3.0 * std::ldexp(1.0, -24)));
    const auto rounded_up_expected = expected_extended_single(
        std::bit_cast<std::uint64_t>(1.0 + 3.0 * std::ldexp(1.0, -24)));
    assert(rounded_up.state.fpr[11] == rounded_up_expected.first);
    assert((rounded_up.state.fpscr & fpscr::fr_mask) != 0U);
    assert((rounded_up.state.fpscr & fpscr::fprf_mask) == 0x00004000U);

    constexpr std::uint64_t non_exact_double = 0x3FF3C0CA428C59FBULL;
    const Execution rounded = execute(11U, 12U, non_exact_double);
    const auto rounded_expected = expected_extended_single(non_exact_double);
    assert(rounded.state.fpr[11] == rounded_expected.first);
    assert(rounded.state.fpr_ps1[11] == rounded_expected.first);
    assert(rounded.state.fpr[11] != non_exact_double);
    assert((rounded.state.fpscr & fpscr::fi_mask) != 0U);
    assert((rounded.state.fpscr & fpscr::fprf_mask) == 0x00004000U);
    assert(rounded_expected.second != 0U);

    const Execution positive_infinity = execute(11U, 12U, 0x7FF0000000000000ULL);
    assert(positive_infinity.state.fpr[11] == 0x7FF0000000000000ULL);
    assert((positive_infinity.state.fpscr & fpscr::fprf_mask) == 0x00005000U);
    const Execution negative_infinity = execute(11U, 12U, 0xFFF0000000000000ULL);
    assert(negative_infinity.state.fpr[11] == 0xFFF0000000000000ULL);
    assert((negative_infinity.state.fpscr & fpscr::fprf_mask) == 0x00009000U);

    constexpr std::uint64_t quiet_nan = 0x7FF923456789ABCDULL;
    constexpr std::uint64_t signaling_nan = 0xFFF0123456789ABCULL;
    const Execution quiet = execute(11U, 12U, quiet_nan);
    const std::uint32_t quiet_single = static_cast<std::uint32_t>(
        0x7F800000U | static_cast<std::uint32_t>((quiet_nan & 0x000FFFFFFFFFFFFFULL) >> 29U) |
        0x00400000U);
    const std::uint64_t quiet_extended = std::bit_cast<std::uint64_t>(
        static_cast<double>(std::bit_cast<float>(quiet_single)));
    assert(quiet.state.fpr[11] == quiet_extended);
    assert(quiet.state.fpr_ps1[11] == quiet_extended);
    assert((quiet.state.fpscr & fpscr::vxsnan_mask) == 0U);
    assert((quiet.state.fpscr & fpscr::fprf_mask) == 0x00011000U);
    const Execution signaling = execute(11U, 12U, signaling_nan);
    assert((signaling.state.fpr[11] & 0x7FF8000000000000ULL) ==
           0x7FF8000000000000ULL);
    assert(signaling.state.fpr_ps1[11] == signaling.state.fpr[11]);
    assert((signaling.state.fpscr & (fpscr::vxsnan_mask | fpscr::vx_mask |
                                     fpscr::fx_mask)) ==
           (fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask));
    const Execution signaling_enabled = execute(
        11U, 12U, signaling_nan, fpscr::ve_mask);
    assert(signaling_enabled.state.fpr[11] == 0x111100000000000BULL);
    assert(signaling_enabled.state.fpr_ps1[11] == 0xAABBCCDDEEFF0011ULL);
    assert((signaling_enabled.state.fpscr & fpscr::fex_mask) != 0U);

    // Destination/source aliasing still rounds from the old PS0 value.
    const Execution alias = execute(5U, 5U, non_exact_double);
    assert(alias.state.fpr[5] == rounded_expected.first);
    assert(alias.state.fpr_ps1[5] == rounded_expected.first);

    // Wind Waker's zero conversion overwrites both lanes and reports a unary trace.
    const Execution wind_waker = execute(11U, 12U, 0U);
    assert(wind_waker.state.fpr[11] == 0U);
    assert(wind_waker.state.fpr_ps1[11] == 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("frsp f12=0 [0x0000000000000000] -> f11=0 [0x0000000000000000] single=0x00000000") !=
           std::string::npos);
}

void floating_convert_to_integer_word_zero_tests()
{
    const auto encode_fctiwz = [](std::uint8_t destination, std::uint8_t source,
                                  bool record = false, std::uint8_t reserved_a = 0U,
                                  std::uint32_t xo = 15U) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(reserved_a) << 16U) |
            (static_cast<std::uint32_t>(source) << 11U) | (xo << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0xFC00081EU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::floating_convert_to_integer_word_zero);
    assert(decoded.fp_register == 0U);
    assert(decoded.fp_source_b == 1U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 15U);
    assert(((wind_waker_word >> 16U) & 0x1FU) == 0U);
    assert((wind_waker_word & 1U) == 0U);
    assert(encode_fctiwz(0U, 1U) == wind_waker_word);
    assert(decode(encode_fctiwz(0U, 1U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_fctiwz(0U, 1U, true)).opcode == Opcode::unsupported);
    assert(decode(encode_fctiwz(0U, 1U, false, 0U, 14U)).opcode ==
           Opcode::unsupported);

    struct Execution
    {
        CpuState state;
        RunResult result;
    };
    constexpr std::uint32_t fpscr_sentinel = 0x00015001U;
    constexpr std::uint64_t destination_ps1 = 0xDEADBEEF01234567ULL;
    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint64_t source_raw,
                             std::uint32_t initial_fpscr = 0x00015001U) {
        EspressoCore core(16U);
        for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
        {
            core.state.gpr[index] = static_cast<std::uint32_t>(0x900U + index);
            core.state.fpr[index] = 0x1111000000000000ULL + index;
            core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
        }
        core.state.fpr[source] = source_raw;
        core.state.fpr_ps1[destination] = destination_ps1;
        core.state.cr = 0xA5C36E91U;
        core.state.xer = 0x800000A5U;
        core.state.fpscr = initial_fpscr;
        core.state.lr = 0x12345678U;
        core.state.ctr = 0x87654321U;
        core.memory.write32_be(0U, encode_fctiwz(destination, source));
        core.memory.write32_be(8U, 0xDEADBEEFU);
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto memory_before = core.memory.read32_be(8U);

        Execution execution{.state = {}, .result = core.run(1U)};
        assert(execution.result.reason == StopReason::instruction_limit);
        assert(execution.result.steps == 1U);
        assert(core.state.cia == 4U);
        assert(core.state.gpr == gpr_before);
        assert(core.state.cr == 0xA5C36E91U);
        assert(core.state.xer == 0x800000A5U);
        assert(core.state.lr == 0x12345678U);
        assert(core.state.ctr == 0x87654321U);
        assert(core.memory.read32_be(8U) == memory_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        assert((core.state.fpscr & fpscr::fprf_mask) ==
               (initial_fpscr & fpscr::fprf_mask));
        if (source != destination)
        {
            assert(core.state.fpr[source] == source_raw);
        }
        for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
        {
            if (index != destination)
            {
                assert(core.state.fpr[index] == fpr_before[index]);
            }
        }
        execution.state = core.state;
        return execution;
    };
    const auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };

    const std::array<std::pair<double, std::uint64_t>, 7> exact_values{{
        {0.0, 0xFFF8000000000000ULL},
        {1.0, 0xFFF8000000000001ULL},
        {123.0, 0xFFF800000000007BULL},
        {-1.0, 0xFFF80000FFFFFFFFULL},
        {-2.0, 0xFFF80000FFFFFFFEULL},
        {2147483647.0, 0xFFF800007FFFFFFFULL},
        {-2147483648.0, 0xFFF8000080000000ULL},
    }};
    for (const auto& [source, expected] : exact_values)
    {
        const Execution result = execute(0U, 1U, raw(source));
        assert(result.state.fpr[0] == expected);
        assert((result.state.fpscr & (fpscr::vxcvi_mask | fpscr::vxsnan_mask)) == 0U);
        assert((result.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    }

    const Execution positive_fraction = execute(0U, 1U, raw(3.9));
    assert(positive_fraction.state.fpr[0] == 0xFFF8000000000003ULL);
    assert((positive_fraction.state.fpscr &
            (fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask)) ==
           (fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask));
    assert((positive_fraction.state.fpscr & fpscr::fr_mask) == 0U);
    const Execution negative_fraction = execute(0U, 1U, raw(-3.9));
    assert(negative_fraction.state.fpr[0] == 0xFFF80000FFFFFFFDULL);
    assert((negative_fraction.state.fpscr & fpscr::fi_mask) != 0U);
    assert((negative_fraction.state.fpscr & fpscr::fr_mask) == 0U);
    assert(execute(0U, 1U, raw(0.5)).state.fpr[0] ==
           0xFFF8000000000000ULL);
    assert((execute(0U, 1U, raw(0.5)).state.fpscr & fpscr::fi_mask) != 0U);
    assert(execute(0U, 1U, raw(-0.5)).state.fpr[0] ==
           0xFFF8000100000000ULL);
    assert(execute(0U, 1U, raw(-0.0)).state.fpr[0] ==
           0xFFF8000100000000ULL);
    assert((execute(0U, 1U, raw(-0.0)).state.fpscr & fpscr::fi_mask) == 0U);

    const Execution positive_overflow = execute(
        0U, 1U, raw(2147483648.0));
    assert(positive_overflow.state.fpr[0] == 0xFFF800007FFFFFFFULL);
    const Execution above_positive_overflow = execute(
        0U, 1U, raw(2147483649.0));
    assert(above_positive_overflow.state.fpr[0] == 0xFFF800007FFFFFFFULL);
    const Execution negative_overflow = execute(
        0U, 1U, raw(-2147483649.0));
    assert(negative_overflow.state.fpr[0] == 0xFFF8000080000000ULL);
    const Execution positive_infinity = execute(0U, 1U, 0x7FF0000000000000ULL);
    assert(positive_infinity.state.fpr[0] == 0xFFF800007FFFFFFFULL);
    const Execution negative_infinity = execute(0U, 1U, 0xFFF0000000000000ULL);
    assert(negative_infinity.state.fpr[0] == 0xFFF8000080000000ULL);
    for (const Execution* invalid : {&positive_overflow, &above_positive_overflow,
                                     &negative_overflow, &positive_infinity,
                                     &negative_infinity})
    {
        assert((invalid->state.fpscr &
                (fpscr::vxcvi_mask | fpscr::vx_mask | fpscr::fx_mask)) ==
               (fpscr::vxcvi_mask | fpscr::vx_mask | fpscr::fx_mask));
        assert((invalid->state.fpscr & (fpscr::fi_mask | fpscr::fr_mask |
                                        fpscr::xx_mask)) == 0U);
    }

    constexpr std::uint64_t quiet_nan = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t signaling_nan = 0x7FF0000000004321ULL;
    const Execution qnan = execute(0U, 1U, quiet_nan);
    assert(qnan.state.fpr[0] == 0xFFF8000080000000ULL);
    assert((qnan.state.fpscr & fpscr::vxcvi_mask) != 0U);
    assert((qnan.state.fpscr & fpscr::vxsnan_mask) == 0U);
    const Execution snan = execute(0U, 1U, signaling_nan);
    assert(snan.state.fpr[0] == 0xFFF8000080000000ULL);
    assert((snan.state.fpscr & (fpscr::vxcvi_mask | fpscr::vxsnan_mask)) ==
           (fpscr::vxcvi_mask | fpscr::vxsnan_mask));

    const std::uint32_t enabled_fpscr = fpscr_sentinel | fpscr::ve_mask;
    const Execution qnan_suppressed = execute(0U, 1U, quiet_nan, enabled_fpscr);
    assert(qnan_suppressed.state.fpr[0] == 0x1111000000000000ULL);
    assert(qnan_suppressed.state.fpr_ps1[0] == destination_ps1);
    assert((qnan_suppressed.state.fpscr & fpscr::fex_mask) != 0U);
    const Execution snan_suppressed = execute(0U, 1U, signaling_nan, enabled_fpscr);
    assert(snan_suppressed.state.fpr[0] == 0x1111000000000000ULL);
    assert((snan_suppressed.state.fpscr & (fpscr::vxcvi_mask | fpscr::vxsnan_mask |
                                            fpscr::vx_mask | fpscr::fx_mask |
                                            fpscr::fex_mask)) ==
           (fpscr::vxcvi_mask | fpscr::vxsnan_mask | fpscr::vx_mask |
            fpscr::fx_mask | fpscr::fex_mask));

    const Execution sticky = execute(0U, 1U, raw(3.0),
                                     fpscr_sentinel | fpscr::xx_mask |
                                     fpscr::fi_mask | fpscr::fr_mask);
    assert((sticky.state.fpscr & (fpscr::fi_mask | fpscr::fr_mask)) == 0U);
    assert((sticky.state.fpscr & fpscr::xx_mask) != 0U);
    const Execution alias = execute(1U, 1U, raw(-3.75));
    assert(alias.state.fpr[1] == 0xFFF80000FFFFFFFDULL);
    assert(alias.state.fpr_ps1[1] == destination_ps1);

    // Wind Waker: fctiwz f0,f1 converts +0 to the hardware FPR integer encoding.
    const Execution wind_waker = execute(0U, 1U, 0U);
    assert(wind_waker.state.fpr[0] == 0xFFF8000000000000ULL);
    assert(wind_waker.state.fpr_ps1[0] == destination_ps1);
    assert((wind_waker.state.fpscr & fpscr::fprf_mask) ==
           (fpscr_sentinel & fpscr::fprf_mask));
    assert((wind_waker.state.fpscr & (fpscr::vxcvi_mask | fpscr::fi_mask |
                                      fpscr::fr_mask)) == 0U);
    const std::string trace = format_instruction_history(wind_waker.result);
    assert(trace.find("fctiwz f1=0 [0x0000000000000000] -> f0=int32(0) raw=0xFFF8000000000000") !=
           std::string::npos);

    const Execution suppressed_trace = execute(
        0U, 1U, quiet_nan, enabled_fpscr);
    const std::string suppressed_history = format_instruction_history(suppressed_trace.result);
    assert(suppressed_history.find("f0 (write suppressed; int32(-2147483648) raw=0xFFF8000080000000 invalid)") !=
           std::string::npos);
}

void integer_alu_tests()
{
    const auto run_neg = [](std::uint8_t destination, std::uint8_t source,
                            std::uint32_t input, bool record = false)
    {
        EspressoCore neg_core(8);
        constexpr std::uint32_t initial_cr = 0x12345678U;
        constexpr std::uint32_t initial_xer = 0xA00000A5U;
        neg_core.state.gpr[source] = input;
        neg_core.state.cr = initial_cr;
        neg_core.state.xer = initial_xer;
        const std::uint32_t word = (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 16U) | (104U << 1U) |
            static_cast<std::uint32_t>(record);
        neg_core.memory.write32_be(0, word);
        const RunResult result = neg_core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(neg_core.state.cia == 4U);
        const std::uint32_t expected = 0U - input;
        assert(neg_core.state.gpr[destination] == expected);
        assert(neg_core.state.xer == initial_xer);
        if (record)
        {
            const std::uint32_t cr0 = expected == 0U ? 0x3U
                : (expected & 0x80000000U) != 0 ? 0x9U : 0x5U;
            assert(neg_core.state.cr == ((initial_cr & 0x0FFFFFFFU) | (cr0 << 28U)));
        }
        else
        {
            assert(neg_core.state.cr == initial_cr);
        }
        return std::pair{expected, result};
    };

    assert(run_neg(11, 10, 0U).first == 0U);
    assert(run_neg(11, 10, 1U).first == 0xFFFFFFFFU);
    assert(run_neg(11, 10, 0x20U).first == 0xFFFFFFE0U);
    assert(run_neg(11, 10, 0xFFFFFFFFU).first == 1U);
    assert(run_neg(11, 10, 0x80000000U).first == 0x80000000U);
    assert(run_neg(9, 9, 0x20U).first == 0xFFFFFFE0U); // destination aliases source
    assert(run_neg(5, 0, 0x20U).first == 0xFFFFFFE0U); // r0 is a normal GPR source
    const auto neg_record = run_neg(11, 10, 0x20U, true);
    assert(neg_record.first == 0xFFFFFFE0U);
    assert(neg_record.second.instruction_history[0].opcode_name == "neg.");

    EspressoCore neg_trace_core(8);
    neg_trace_core.memory.write32_be(0, 0x7D6A00D0U);
    neg_trace_core.state.gpr[10] = 0x20U;
    const RunResult neg_trace = neg_trace_core.run(1U);
    assert(format_instruction_history(neg_trace).find(
        "neg r10=0x00000020 -> r11=0xFFFFFFE0") != std::string::npos);

    const auto run_mullw = [](std::uint8_t destination, std::uint8_t ra, std::uint8_t rb,
                              std::uint32_t a, std::uint32_t b, bool record = false)
    {
        EspressoCore multiply_core(8);
        constexpr std::uint32_t initial_cr = 0x12345678U;
        constexpr std::uint32_t initial_xer = 0xA00000A5U;
        multiply_core.state.gpr[ra] = a;
        multiply_core.state.gpr[rb] = b;
        multiply_core.state.cr = initial_cr;
        multiply_core.state.xer = initial_xer;
        const std::uint32_t word = (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) | (235U << 1U) |
            static_cast<std::uint32_t>(record);
        multiply_core.memory.write32_be(0, word);
        const RunResult result = multiply_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(multiply_core.state.cia == 4U);
        const std::int64_t product =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) *
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(b));
        const std::uint32_t expected = static_cast<std::uint32_t>(product);
        assert(multiply_core.state.gpr[destination] == expected);
        assert(multiply_core.state.xer == initial_xer);
        if (record)
        {
            const std::uint32_t cr0 = expected == 0U ? 0x3U
                : (expected & 0x80000000U) != 0 ? 0x9U : 0x5U;
            assert(multiply_core.state.cr == ((initial_cr & 0x0FFFFFFFU) | (cr0 << 28U)));
        }
        else
        {
            assert(multiply_core.state.cr == initial_cr);
        }
        return std::pair{expected, result};
    };

    assert(run_mullw(30, 31, 30, 2U, 0x100U).first == 0x200U);
    assert(run_mullw(3, 4, 5, 7U, 9U).first == 63U); // positive * positive
    assert(run_mullw(3, 4, 5, 0xFFFFFFFEU, 3U).first == 0xFFFFFFFAU);
    assert(run_mullw(3, 4, 5, 0xFFFFFFFEU, 0xFFFFFFFDU).first == 6U);
    assert(run_mullw(3, 4, 5, 0U, 0x87654321U).first == 0U);
    assert(run_mullw(3, 4, 5, 0x40000000U, 4U).first == 0U); // low-word truncation
    assert(run_mullw(4, 4, 5, 7U, 3U).first == 21U); // rD aliases rA
    assert(run_mullw(5, 4, 5, 7U, 3U).first == 21U); // rD aliases rB
    assert(run_mullw(6, 0, 4, 6U, 7U).first == 42U); // r0 is a normal source
    const auto mullw_record = run_mullw(3, 4, 5, 0x100U, 2U, true);
    assert(mullw_record.first == 0x200U);
    assert(mullw_record.second.instruction_history[0].opcode_name == "mullw.");

    EspressoCore mullw_trace_core(8);
    mullw_trace_core.memory.write32_be(0, 0x7FDFF1D6U);
    mullw_trace_core.state.gpr[31] = 2U;
    mullw_trace_core.state.gpr[30] = 0x100U;
    const RunResult mullw_trace = mullw_trace_core.run(1);
    assert(format_instruction_history(mullw_trace).find(
        "mullw r31=0x00000002 r30=0x00000100 -> r30=0x00000200")
        != std::string::npos);

    constexpr std::uint32_t xer_ca = 0x20000000U;
    const auto run_subfc = [xer_ca](std::uint8_t destination, std::uint8_t ra, std::uint8_t rb,
                              std::uint32_t a, std::uint32_t b, bool record = false)
    {
        EspressoCore subfc_core(8);
        constexpr std::uint32_t initial_cr = 0x12345678U;
        constexpr std::uint32_t initial_xer = 0xA00000A5U;
        subfc_core.state.gpr[ra] = a;
        subfc_core.state.gpr[rb] = b;
        subfc_core.state.cr = initial_cr;
        subfc_core.state.xer = initial_xer;
        const std::uint32_t word = (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) | (8U << 1U) |
            static_cast<std::uint32_t>(record);
        subfc_core.memory.write32_be(0, word);
        const RunResult result = subfc_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(subfc_core.state.cia == 4U);
        const std::uint32_t expected = b - a;
        const bool expected_ca = b >= a;
        assert(subfc_core.state.gpr[destination] == expected);
        assert(((subfc_core.state.xer & xer_ca) != 0) == expected_ca);
        assert((subfc_core.state.xer & ~xer_ca) == (initial_xer & ~xer_ca));
        if (record)
        {
            const std::uint32_t cr0 = expected == 0U ? 0x3U
                : (expected & 0x80000000U) != 0 ? 0x9U : 0x5U;
            assert(subfc_core.state.cr == ((initial_cr & 0x0FFFFFFFU) | (cr0 << 28U)));
        }
        else
        {
            assert(subfc_core.state.cr == initial_cr);
        }
        return std::pair{expected, result};
    };

    assert(run_subfc(3, 12, 11, 3U, 5U).first == 2U);
    assert(run_subfc(3, 12, 11, 5U, 3U).first == 0xFFFFFFFEU);
    assert(run_subfc(3, 12, 11, 5U, 5U).first == 0U);
    assert(run_subfc(3, 12, 11, 1U, 0U).first == 0xFFFFFFFFU);
    assert(run_subfc(3, 12, 11, 0U, 0xFFFFFFFFU).first == 0xFFFFFFFFU);
    assert(run_subfc(12, 12, 11, 3U, 5U).first == 2U); // destination aliases rA
    assert(run_subfc(11, 12, 11, 3U, 5U).first == 2U); // destination aliases rB
    const auto subfc_record = run_subfc(3, 12, 11, 0U, 0U, true);
    assert(subfc_record.first == 0U);
    assert(subfc_record.second.instruction_history[0].has_carry_result);
    assert(subfc_record.second.instruction_history[0].carry_result);

    EspressoCore subfc_trace_core(8);
    subfc_trace_core.memory.write32_be(0, 0x7C6C5810U);
    const RunResult subfc_trace = subfc_trace_core.run(1);
    assert(format_instruction_history(subfc_trace).find(
        "subfc r12=0x00000000 r11=0x00000000 -> r3=0x00000000 CA=1")
        != std::string::npos);

    const auto run_rlwimi = [](std::uint8_t destination, std::uint8_t source,
                               std::uint32_t source_value, std::uint32_t old_destination,
                               std::uint8_t shift, std::uint8_t mask_begin,
                               std::uint8_t mask_end, bool record = false)
    {
        EspressoCore rotate_core(8);
        constexpr std::uint32_t initial_cr = 0x12345678U;
        constexpr std::uint32_t initial_xer = 0xE0000000U;
        rotate_core.state.gpr[source] = source_value;
        rotate_core.state.gpr[destination] = old_destination;
        rotate_core.state.cr = initial_cr;
        rotate_core.state.xer = initial_xer;
        const std::uint32_t word = (20U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) |
            (static_cast<std::uint32_t>(shift) << 11U) |
            (static_cast<std::uint32_t>(mask_begin) << 6U) |
            (static_cast<std::uint32_t>(mask_end) << 1U) | static_cast<std::uint32_t>(record);
        rotate_core.memory.write32_be(0, word);
        const RunResult result = rotate_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(rotate_core.state.cia == 4U);
        assert(rotate_core.state.xer == initial_xer);
        assert(result.instruction_history.size() == 1U);
        assert(result.instruction_history[0].has_old_destination);
        assert(result.instruction_history[0].old_destination_value == old_destination);
        const std::uint32_t expected_cr = record
            ? ((initial_cr & 0x0FFFFFFFU) | 0x50000000U)
            : initial_cr;
        assert(rotate_core.state.cr == expected_cr);
        return std::pair{rotate_core.state.gpr[destination], result};
    };

    // MB=24..25 selects bits 6..7 after rotating the source left by six.
    const auto sample_rlwimi = run_rlwimi(12, 30, 3U, 0x1234563FU, 6, 24, 25);
    assert(sample_rlwimi.first == 0x123456FFU);
    assert((sample_rlwimi.second.instruction_history[0].instruction_word == 0x53CC3632U));
    assert(format_instruction_history(sample_rlwimi.second).find(
        "rlwimi r30=0x00000003 old-r12=0x1234563F sh=6 mb=24 me=25 -> r12=0x123456FF")
        != std::string::npos);
    assert(run_rlwimi(3, 4, 0xFFFFFFFFU, 0U, 0, 24, 25).first == 0x000000C0U);

    assert(run_rlwimi(3, 4, 0U, 0xDEADBEEFU, 0, 24, 31).first == 0xDEADBE00U);
    assert(run_rlwimi(5, 5, 0xAAAAAAAAU, 0xAAAAAAAAU, 1, 31, 31).first ==
           0xAAAAAAABU); // aliased source reads the original register value
    assert(run_rlwimi(3, 4, 1U, 0U, 31, 0, 0).first == 0x80000000U);
    assert(run_rlwimi(3, 4, 0x80000001U, 0x0F0000F0U, 0, 28, 3).first ==
           0x8F0000F1U); // wrapping mask
    const auto record_rlwimi = run_rlwimi(12, 30, 3U, 0x1234563FU, 6, 24, 25, true);
    assert(record_rlwimi.first == 0x123456FFU);

    const auto run_cntlzw = [](std::uint32_t input, std::uint8_t destination,
                               std::uint8_t source, bool record,
                               std::uint32_t initial_cr, std::uint32_t initial_xer)
    {
        EspressoCore cntlzw_core(8);
        cntlzw_core.state.gpr[source] = input;
        cntlzw_core.state.cr = initial_cr;
        cntlzw_core.state.xer = initial_xer;
        const std::uint32_t word = 0x7C000034U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) |
            static_cast<std::uint32_t>(record);
        cntlzw_core.memory.write32_be(0, word);
        const RunResult result = cntlzw_core.run(1);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(cntlzw_core.state.cia == 4U);
        assert(cntlzw_core.state.xer == initial_xer);
        assert(result.instruction_history.size() == 1U);
        return std::pair{cntlzw_core.state.gpr[destination], cntlzw_core.state.cr};
    };

    constexpr std::uint32_t initial_cr = 0x12345678U;
    constexpr std::uint32_t initial_xer = 0x80000000U;
    assert((run_cntlzw(0x80000000U, 12, 5, false, initial_cr, initial_xer) ==
            std::pair{0U, initial_cr}));
    assert((run_cntlzw(0x40000000U, 12, 5, false, initial_cr, initial_xer) ==
            std::pair{1U, initial_cr}));
    assert((run_cntlzw(0x00000001U, 12, 5, false, initial_cr, initial_xer) ==
            std::pair{31U, initial_cr}));
    assert((run_cntlzw(0x00000000U, 12, 5, false, initial_cr, initial_xer) ==
            std::pair{32U, initial_cr}));
    assert((run_cntlzw(0x00100000U, 12, 5, false, initial_cr, initial_xer) ==
            std::pair{11U, initial_cr}));
    assert((run_cntlzw(0x00000001U, 12, 12, false, initial_cr, initial_xer) ==
            std::pair{31U, initial_cr})); // source == destination
    assert((run_cntlzw(0x00000001U, 7, 5, false, initial_cr, initial_xer) ==
            std::pair{31U, initial_cr})); // separate source and destination
    assert((run_cntlzw(0x00000001U, 12, 5, true, initial_cr, initial_xer) ==
            std::pair{31U, 0x52345678U})); // positive result and XER[SO]

    EspressoCore cntlzw_trace_core(8);
    cntlzw_trace_core.memory.write32_be(0, 0x7D8C0034U);
    cntlzw_trace_core.state.gpr[12] = 1U;
    const RunResult cntlzw_trace = cntlzw_trace_core.run(1);
    assert(format_instruction_history(cntlzw_trace).find(
        "cntlzw r12=0x00000001 -> r12=0x0000001F") != std::string::npos);

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

void subtract_from_extended_tests()
{
    constexpr std::uint32_t xer_ca = 0x20000000U;
    constexpr std::uint32_t xer_ov = 0x40000000U;
    constexpr std::uint32_t xer_so = 0x80000000U;
    const auto encode_subfe = [](std::uint8_t destination, std::uint8_t ra,
                                 std::uint8_t rb, bool record = false,
                                 bool overflow_enable = false) {
        const std::uint32_t xo = 136U + (overflow_enable ? 512U : 0U);
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) |
            (xo << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D8C6110U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::subtract_from_extended);
    assert(wind_waker.destination == 12U);
    assert(wind_waker.base == 12U);
    assert(wind_waker.source == 12U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 136U);
    assert(!wind_waker.record);
    const DecodedInstruction record_form = decode(encode_subfe(3U, 4U, 5U, true));
    assert(record_form.opcode == Opcode::subtract_from_extended);
    assert(record_form.record);
    assert(decode(encode_subfe(3U, 4U, 5U, false, true)).opcode ==
           Opcode::unsupported); // subfeo remains unsupported

    const auto execute = [&](std::uint8_t destination, std::uint8_t ra,
                             std::uint8_t rb, std::uint32_t a, std::uint32_t b,
                             bool carry_in, bool record = false, bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[ra] = a;
        core.state.gpr[rb] = b;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = 0x400000A5U |
            (so ? xer_so : 0U) | (carry_in ? xer_ca : 0U);
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_subfe(destination, ra, rb, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint64_t sum =
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(~a)) + b +
            static_cast<std::uint32_t>(carry_in);
        const std::uint32_t expected = static_cast<std::uint32_t>(sum);
        const bool expected_carry = (sum >> 32U) != 0U;

        const RunResult run_result = core.run(1U);
        assert(run_result.steps == 1U);
        assert(run_result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(((core.state.xer & xer_ca) != 0U) == expected_carry);
        assert((core.state.xer & ~xer_ca) == (initial_xer & ~xer_ca));
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        if (record)
        {
            const std::uint32_t expected_cr0 =
                (expected & 0x80000000U) != 0U ? 0x8U
                : expected == 0U ? 0x2U : 0x4U;
            assert(((core.state.cr >> 28U) & 0xFU) ==
                   (expected_cr0 | static_cast<std::uint32_t>(so)));
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        return std::pair{expected, run_result};
    };

    assert(execute(3U, 4U, 5U, 5U, 5U, true).first == 0U);
    assert(execute(3U, 4U, 5U, 5U, 5U, false).first == 0xFFFFFFFFU);
    assert(execute(3U, 4U, 5U, 5U, 6U, false).first == 0U);
    assert(execute(3U, 4U, 5U, 6U, 5U, true).first == 0xFFFFFFFFU);
    assert(execute(3U, 4U, 5U, 0U, 0U, false).first == 0xFFFFFFFFU);
    assert(execute(3U, 4U, 5U, 0U, 0U, true).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU, true).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0U, true).first == 1U);
    assert(execute(3U, 4U, 5U, 0U, 0xFFFFFFFFU, false).first == 0xFFFFFFFEU);

    assert(execute(4U, 4U, 5U, 5U, 6U, false).first == 0U); // rD == rA
    assert(execute(5U, 4U, 5U, 5U, 6U, false).first == 0U); // rD == rB
    assert(execute(5U, 5U, 5U, 5U, 5U, false).first == 0xFFFFFFFFU);
    (void)execute(6U, 4U, 5U, 5U, 5U, true, true); // EQ
    (void)execute(6U, 4U, 5U, 5U, 6U, false, true); // EQ, CA out set
    (void)execute(6U, 4U, 5U, 5U, 7U, false, true); // GT
    (void)execute(6U, 4U, 5U, 5U, 5U, false, true); // LT
    (void)execute(6U, 4U, 5U, 5U, 5U, false, true, true); // LT | SO
    (void)xer_ov;

    // The exact Wind Waker subfc -> subfe carry chain.
    EspressoCore wind_chain(8U);
    wind_chain.state.gpr[10] = 0xFFFFF800U;
    wind_chain.state.gpr[7] = 0x60B7A437U;
    wind_chain.state.gpr[12] = 0xDEADBEEFU;
    wind_chain.memory.write32_be(0U, 0x7D8A3810U); // subfc r12,r10,r7
    wind_chain.memory.write32_be(4U, wind_waker_word); // subfe r12,r12,r12
    const RunResult chain_result = wind_chain.run(2U);
    assert(chain_result.steps == 2U);
    assert(wind_chain.state.gpr[12] == 0xFFFFFFFFU);
    assert((wind_chain.state.xer & xer_ca) == 0U);
    const std::string chain_trace = format_instruction_history(chain_result);
    assert(chain_trace.find("subfc r10=0xFFFFF800 r7=0x60B7A437 -> r12=0x60B7AC37 CA=0") !=
           std::string::npos);
    assert(chain_trace.find("subfe r12=0x60B7AC37 CAin=0 -> r12=0xFFFFFFFF CA=0") !=
           std::string::npos);

    EspressoCore record_trace_core(8U);
    record_trace_core.state.gpr[4] = 5U;
    record_trace_core.state.gpr[5] = 5U;
    record_trace_core.state.xer = xer_ca;
    record_trace_core.memory.write32_be(0U, encode_subfe(3U, 4U, 5U, true));
    const RunResult record_trace = record_trace_core.run(1U);
    assert(record_trace.instruction_history[0].opcode_name == "subfe.");
    assert(format_instruction_history(record_trace).find("CAin=1") != std::string::npos);

    // Verify subfc/subfe form a correct 64-bit unsigned borrow chain.
    const auto run_64bit_subtract = [&encode_subfe](std::uint64_t a, std::uint64_t b) {
        EspressoCore core(8U);
        core.state.gpr[4] = static_cast<std::uint32_t>(a);
        core.state.gpr[7] = static_cast<std::uint32_t>(a >> 32U);
        core.state.gpr[5] = static_cast<std::uint32_t>(b);
        core.state.gpr[8] = static_cast<std::uint32_t>(b >> 32U);
        core.memory.write32_be(0U, (31U << 26U) | (3U << 21U) |
            (4U << 16U) | (5U << 11U) | (8U << 1U)); // subfc r3,r4,r5
        core.memory.write32_be(4U, encode_subfe(6U, 7U, 8U)); // subfe r6,r7,r8
        const RunResult run = core.run(2U);
        assert(run.steps == 2U);
        return (static_cast<std::uint64_t>(core.state.gpr[6]) << 32U) |
            core.state.gpr[3];
    };
    for (const auto [a, b] : std::array<std::pair<std::uint64_t, std::uint64_t>, 4>{
             std::pair{0x0000000100000000ULL, 0x0000000200000001ULL},
             std::pair{0x0000000100000001ULL, 0x0000000200000000ULL},
             std::pair{0xFEDCBA9876543210ULL, 0x123456789ABCDEF0ULL},
             std::pair{0xFFFFFFFFFFFFFFFFULL, 0ULL}})
    {
        assert(run_64bit_subtract(a, b) == b - a);
    }
}

void add_carrying_tests()
{
    constexpr std::uint32_t xer_ca = 0x20000000U;
    constexpr std::uint32_t xer_so = 0x80000000U;
    const auto encode_addc = [](std::uint8_t destination, std::uint8_t ra,
                               std::uint8_t rb, bool record = false,
                               std::uint32_t xo = 10U) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) | (xo << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0x7C845814U;
    assert(decode(wind_waker_word).opcode == Opcode::add_carrying);

    const auto execute = [&](std::uint8_t destination, std::uint8_t ra,
                             std::uint8_t rb, std::uint32_t a, std::uint32_t b,
                             bool old_ca, bool record = false, bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[ra] = a;
        core.state.gpr[rb] = b;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = 0x400000A5U |
            (so ? xer_so : 0U) | (old_ca ? xer_ca : 0U);
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_addc(destination, ra, rb, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + b;
        const std::uint32_t expected = static_cast<std::uint32_t>(sum);
        const bool expected_carry = (sum >> 32U) != 0U;

        const RunResult run_result = core.run(1U);
        assert(run_result.steps == 1U);
        assert(run_result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(((core.state.xer & xer_ca) != 0U) == expected_carry);
        assert((core.state.xer & ~xer_ca) == (initial_xer & ~xer_ca));
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        if (record)
        {
            const std::uint32_t expected_cr0 =
                (expected & 0x80000000U) != 0U ? 0x8U
                : expected == 0U ? 0x2U : 0x4U;
            assert(((core.state.cr >> 28U) & 0xFU) ==
                   (expected_cr0 | static_cast<std::uint32_t>(so)));
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        return std::pair{expected, run_result};
    };

    // CA is produced from this addition and is never consumed as an input.
    assert(execute(3U, 4U, 5U, 0U, 1U, false).first == 1U);
    assert(execute(3U, 4U, 5U, 0U, 1U, true).first == 1U);
    assert(execute(3U, 4U, 5U, 5U, 7U, false).first == 12U);
    assert(execute(3U, 4U, 5U, 5U, 7U, true).first == 12U);
    assert(execute(3U, 4U, 5U, 0U, 0U, true).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0U, true).first == 0xFFFFFFFFU);

    assert(execute(3U, 4U, 5U, 0U, 0U, false).first == 0U);
    assert(execute(3U, 4U, 5U, 0U, 1U, false).first == 1U);
    assert(execute(3U, 4U, 5U, 1U, 1U, false).first == 2U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 1U, false).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU, false).first ==
           0xFFFFFFFEU);
    assert(execute(3U, 4U, 5U, 0x80000000U, 0x80000000U, false).first == 0U);
    assert(execute(3U, 4U, 5U, 0x7FFFFFFFU, 1U, false).first == 0x80000000U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0U, true).first == 0xFFFFFFFFU);

    assert(execute(4U, 4U, 5U, 5U, 7U, false).first == 12U); // rD == rA
    assert(execute(5U, 4U, 5U, 5U, 7U, false).first == 12U); // rD == rB
    assert(execute(5U, 5U, 5U, 0x80000000U, 0x80000000U, false).first == 0U);
    (void)execute(6U, 4U, 5U, 1U, 1U, false, true); // GT
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 1U, false, true); // EQ with carry
    (void)execute(6U, 4U, 5U, 0x80000000U, 0U, false, true); // LT
    (void)execute(6U, 4U, 5U, 0x80000000U, 0U, true, true, true); // LT | SO

    EspressoCore wind_waker(8U);
    wind_waker.state.gpr[4] = 0U;
    wind_waker.state.gpr[11] = 1U;
    wind_waker.memory.write32_be(0U, wind_waker_word);
    const RunResult wind_result = wind_waker.run(1U);
    assert(wind_result.steps == 1U);
    assert(wind_waker.state.gpr[4] == 1U);
    assert((wind_waker.state.xer & xer_ca) == 0U);
    assert(wind_result.instruction_history[0].opcode_name == "addc");
    assert(wind_result.instruction_history[0].has_carry_result);
    assert(!wind_result.instruction_history[0].has_carry_input);
    assert(format_instruction_history(wind_result).find(
               "addc r4=0x00000000 r11=0x00000001 -> r4=0x00000001 CA=0") !=
           std::string::npos);

    EspressoCore record_trace(8U);
    record_trace.state.gpr[4] = 0x80000000U;
    record_trace.state.gpr[5] = 0U;
    record_trace.state.xer = xer_so;
    record_trace.memory.write32_be(0U, encode_addc(6U, 4U, 5U, true));
    const RunResult record_result = record_trace.run(1U);
    assert(record_result.instruction_history[0].opcode_name == "addc.");
    assert(((record_trace.state.cr >> 28U) & 0xFU) == 0x9U);
    assert((record_trace.state.xer & xer_so) != 0U);
}

void add_to_zero_extended_tests()
{
    constexpr std::uint32_t xer_ca = 0x20000000U;
    constexpr std::uint32_t xer_so = 0x80000000U;
    const auto encode_addze = [](std::uint8_t destination, std::uint8_t ra,
                                bool record = false, std::uint8_t reserved_rb = 0U,
                                std::uint32_t xo = 202U) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(reserved_rb) << 11U) | (xo << 1U) |
            static_cast<std::uint32_t>(record);
    };
    constexpr std::uint32_t wind_waker_word = 0x7C630194U;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::add_to_zero_extended);
    assert(decoded.destination == 3U);
    assert(decoded.base == 3U);
    assert(!decoded.record);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 202U);
    assert(((wind_waker_word >> 11U) & 0x1FU) == 0U);
    assert(decode(encode_addze(3U, 3U, true)).opcode == Opcode::add_to_zero_extended);
    assert(decode(encode_addze(3U, 3U, true)).record);
    assert(decode(encode_addze(3U, 3U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode(encode_addze(3U, 3U, false, 0U, 714U)).opcode ==
           Opcode::unsupported);

    const auto execute = [&](std::uint8_t destination, std::uint8_t ra,
                             std::uint32_t a, bool old_ca,
                             bool record = false, bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[ra] = a;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = 0x400000A5U |
            (so ? xer_so : 0U) | (old_ca ? xer_ca : 0U);
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_addze(destination, ra, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) +
            static_cast<std::uint32_t>(old_ca);
        const std::uint32_t expected = static_cast<std::uint32_t>(sum);
        const bool expected_carry = (sum >> 32U) != 0U;

        const RunResult run_result = core.run(1U);
        assert(run_result.steps == 1U);
        assert(run_result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(((core.state.xer & xer_ca) != 0U) == expected_carry);
        assert((core.state.xer & ~xer_ca) == (initial_xer & ~xer_ca));
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        if (record)
        {
            const std::uint32_t expected_cr0 =
                (expected & 0x80000000U) != 0U ? 0x8U
                : expected == 0U ? 0x2U : 0x4U;
            assert(((core.state.cr >> 28U) & 0xFU) ==
                   (expected_cr0 | static_cast<std::uint32_t>(so)));
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        return std::pair{expected, run_result};
    };

    assert(execute(3U, 4U, 0U, false).first == 0U);
    assert(execute(3U, 4U, 0U, true).first == 1U);
    assert(execute(3U, 4U, 5U, false).first == 5U);
    assert(execute(3U, 4U, 5U, true).first == 6U);
    assert(execute(3U, 4U, 0xFFFFFFFEU, true).first == 0xFFFFFFFFU);
    assert(execute(3U, 4U, 0xFFFFFFFFU, false).first == 0xFFFFFFFFU);
    assert(execute(3U, 4U, 0xFFFFFFFFU, true).first == 0U);
    assert(execute(3U, 4U, 0x7FFFFFFFU, true).first == 0x80000000U);

    (void)execute(3U, 3U, 0U, false, true); // EQ
    (void)execute(3U, 3U, 0U, true, true); // GT
    (void)execute(3U, 3U, 0x7FFFFFFFU, true, true); // LT despite signed overflow
    (void)execute(3U, 3U, 0x7FFFFFFFU, true, true, true); // LT | SO

    EspressoCore wind_waker(8U);
    wind_waker.state.gpr[3] = 0U;
    wind_waker.state.xer = 0x400000A5U;
    wind_waker.memory.write32_be(0U, wind_waker_word);
    const std::uint32_t initial_cr = wind_waker.state.cr;
    const std::uint32_t initial_xer = wind_waker.state.xer;
    const RunResult wind_result = wind_waker.run(1U);
    assert(wind_result.steps == 1U);
    assert(wind_waker.state.gpr[3] == 0U);
    assert(wind_waker.state.xer == initial_xer);
    assert(wind_waker.state.cr == initial_cr);
    assert(wind_result.instruction_history[0].opcode_name == "addze");
    assert(wind_result.instruction_history[0].has_carry_input);
    assert(wind_result.instruction_history[0].has_carry_result);
    assert(!wind_result.instruction_history[0].carry_input);
    assert(format_instruction_history(wind_result).find(
               "addze r3=0x00000000 CAin=0 -> r3=0x00000000 CA=0") !=
           std::string::npos);

    EspressoCore carry_chain(8U);
    carry_chain.state.gpr[4] = 0xFFFFFFFFU;
    carry_chain.state.gpr[11] = 1U;
    carry_chain.state.gpr[3] = 5U;
    carry_chain.memory.write32_be(0U, 0x7C845814U); // addc r4,r4,r11
    carry_chain.memory.write32_be(4U, wind_waker_word); // addze r3,r3
    const RunResult chain_result = carry_chain.run(2U);
    assert(chain_result.steps == 2U);
    assert(carry_chain.state.gpr[4] == 0U);
    assert(carry_chain.state.gpr[3] == 6U);
    assert((carry_chain.state.xer & xer_ca) == 0U);
    const std::string chain_trace = format_instruction_history(chain_result);
    assert(chain_trace.find("addc r4=0xFFFFFFFF r11=0x00000001 -> r4=0x00000000 CA=1") !=
           std::string::npos);
    assert(chain_trace.find("addze r3=0x00000005 CAin=1 -> r3=0x00000006 CA=0") !=
           std::string::npos);
}

void multiply_high_word_unsigned_tests()
{
    const auto encode_mulhwu = [](std::uint8_t destination, std::uint8_t ra,
                                 std::uint8_t rb, bool record = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) |
            (11U << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7C605816U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::multiply_high_word_unsigned);
    assert(wind_waker.destination == 3U);
    assert(wind_waker.base == 0U);
    assert(wind_waker.source == 11U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 11U);
    assert(!wind_waker.record);
    assert(decode(encode_mulhwu(3U, 0U, 11U, true)).opcode ==
           Opcode::multiply_high_word_unsigned);
    assert(decode(encode_mulhwu(3U, 0U, 11U, true)).record);

    const auto execute = [&](std::uint8_t destination, std::uint8_t ra,
                             std::uint8_t rb, std::uint32_t lhs,
                             std::uint32_t rhs, bool record = false,
                             bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[ra] = lhs;
        core.state.gpr[rb] = rhs;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = so ? 0xA00000A5U : 0x200000A5U;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_mulhwu(destination, ra, rb, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint64_t product = static_cast<std::uint64_t>(lhs) * rhs;
        const std::uint32_t expected = static_cast<std::uint32_t>(product >> 32U);

        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        if (record)
        {
            const std::uint32_t expected_cr0 = (expected & 0x80000000U) != 0U
                ? (0x8U | static_cast<std::uint32_t>(so))
                : expected == 0U ? (0x2U | static_cast<std::uint32_t>(so))
                                 : (0x4U | static_cast<std::uint32_t>(so));
            assert(((core.state.cr >> 28U) & 0xFU) == expected_cr0);
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        assert(core.state.xer == initial_xer);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        return std::pair{expected, result};
    };

    assert(execute(3U, 0U, 11U, 1U, 16807U).first == 0U);
    assert(execute(3U, 4U, 5U, 0U, 0xFFFFFFFFU).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 2U).first == 0x00000001U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU).first == 0xFFFFFFFEU);
    assert(execute(3U, 4U, 5U, 0x80000000U, 2U).first == 0x00000001U);
    assert(execute(3U, 4U, 5U, 0x80000000U, 0x80000000U).first == 0x40000000U);

    // Destination aliases either input or both, with operands captured first.
    assert(execute(3U, 3U, 4U, 0xFFFFFFFFU, 2U).first == 1U);
    assert(execute(4U, 3U, 4U, 0xFFFFFFFFU, 2U).first == 1U);
    assert(execute(5U, 5U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU).first == 0xFFFFFFFEU);

    (void)execute(6U, 4U, 5U, 1U, 0U, true, false); // zero -> EQ
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 2U, true, false); // +1 -> GT
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU, true, false); // negative -> LT
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU, true, true); // LT | SO

    const auto wind_result = execute(3U, 0U, 11U, 1U, 0x41A7U);
    assert(wind_result.first == 0U);
    const std::string trace = format_instruction_history(wind_result.second);
    assert(trace.find("mulhwu r0=0x00000001 r11=0x000041A7 -> r3=0x00000000") !=
           std::string::npos);
}

void divide_word_unsigned_tests()
{
    const auto encode_divwu = [](std::uint8_t destination, std::uint8_t ra,
                                std::uint8_t rb, bool record = false,
                                bool overflow_enable = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) |
            ((459U + (overflow_enable ? 512U : 0U)) << 1U) |
            static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D1DFB96U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::divide_word_unsigned);
    assert(wind_waker.destination == 8U);
    assert(wind_waker.base == 29U);
    assert(wind_waker.source == 31U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 459U);
    assert(!wind_waker.record);

    const DecodedInstruction record_form = decode(encode_divwu(8U, 29U, 31U, true));
    assert(record_form.opcode == Opcode::divide_word_unsigned);
    assert(record_form.record);
    assert(decode(encode_divwu(8U, 29U, 31U, false, true)).opcode ==
           Opcode::unsupported);

    const auto execute = [&](std::uint8_t destination, std::uint8_t ra,
                             std::uint8_t rb, std::uint32_t dividend,
                             std::uint32_t divisor, bool record = false,
                             bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[ra] = dividend;
        core.state.gpr[rb] = divisor;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = so ? 0xA00000A5U : 0x200000A5U;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_divwu(destination, ra, rb, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_xer = core.state.xer;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint32_t expected = divisor == 0U ? 0U : dividend / divisor;

        const RunResult run_result = core.run(1U);
        assert(run_result.steps == 1U);
        assert(run_result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        if (record)
        {
            const std::uint32_t expected_cr0 = (expected & 0x80000000U) != 0U
                ? (0x8U | static_cast<std::uint32_t>(so))
                : expected == 0U ? (0x2U | static_cast<std::uint32_t>(so))
                                 : (0x4U | static_cast<std::uint32_t>(so));
            assert(((core.state.cr >> 28U) & 0xFU) == expected_cr0);
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        assert(core.state.xer == original_xer);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        return std::pair{expected, run_result};
    };

    assert(execute(3U, 4U, 5U, 10U, 2U).first == 5U);
    assert(execute(3U, 4U, 5U, 10U, 3U).first == 3U);
    assert(execute(3U, 4U, 5U, 0U, 123U).first == 0U);
    assert(execute(3U, 4U, 5U, 123U, 1U).first == 123U);
    assert(execute(3U, 4U, 5U, 5U, 10U).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 2U).first == 0x7FFFFFFFU);
    assert(execute(3U, 4U, 5U, 0x80000000U, 2U).first == 0x40000000U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 0xFFFFFFFFU).first == 1U);
    assert(execute(3U, 4U, 5U, 0x80000000U, 0xFFFFFFFFU).first == 0U);
    assert(execute(3U, 4U, 5U, 0x12345678U, 0U).first == 0U);

    // Destination aliases either input or both; source operands are snapshotted.
    assert(execute(3U, 3U, 4U, 10U, 2U).first == 5U);
    assert(execute(4U, 3U, 4U, 10U, 2U).first == 5U);
    assert(execute(5U, 5U, 5U, 123U, 123U).first == 1U);

    (void)execute(6U, 4U, 5U, 10U, 0U, true); // zero -> EQ
    (void)execute(6U, 4U, 5U, 10U, 2U, true); // positive -> GT
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 1U, true); // high-bit result -> LT
    (void)execute(6U, 4U, 5U, 0xFFFFFFFFU, 1U, true, true); // LT | SO

    const auto wind_result = execute(8U, 29U, 31U, 0x41A7U, 0x7FFFFFFFU);
    assert(wind_result.first == 0U);
    const std::string trace = format_instruction_history(wind_result.second);
    assert(trace.find("divwu r29=0x000041A7 r31=0x7FFFFFFF -> r8=0x00000000") !=
           std::string::npos);
}

void shift_right_word_tests()
{
    const auto encode_srw = [](std::uint8_t destination, std::uint8_t source,
                               std::uint8_t shift_register, bool record = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) |
            (static_cast<std::uint32_t>(shift_register) << 11U) |
            (536U << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7CC95430U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::shift_right_word);
    assert(wind_waker.source == 6U);
    assert(wind_waker.destination == 9U);
    assert(wind_waker.base == 10U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 536U);
    assert(!wind_waker.record);
    const DecodedInstruction record_form = decode(encode_srw(9U, 6U, 10U, true));
    assert(record_form.opcode == Opcode::shift_right_word);
    assert(record_form.record);

    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint8_t shift_register, std::uint32_t value,
                             std::uint32_t shift_amount, bool record = false,
                             bool so = false) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[source] = value;
        core.state.gpr[shift_register] = shift_amount;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = so ? 0xA00000A5U : 0x200000A5U;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_srw(destination, source, shift_register, record);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint32_t amount = shift_amount & 0x3FU;
        const std::uint32_t expected = amount >= 32U ? 0U : value >> amount;

        const RunResult run_result = core.run(1U);
        assert(run_result.steps == 1U);
        assert(run_result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        if (record)
        {
            const std::uint32_t expected_cr0 = (expected & 0x80000000U) != 0U
                ? (0x8U | static_cast<std::uint32_t>(so))
                : expected == 0U ? (0x2U | static_cast<std::uint32_t>(so))
                                 : (0x4U | static_cast<std::uint32_t>(so));
            assert(((core.state.cr >> 28U) & 0xFU) == expected_cr0);
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        assert(core.state.xer == initial_xer);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        return std::pair{expected, run_result};
    };

    assert(execute(9U, 6U, 10U, 0x7FFFFFFFU, 20U).first == 0x000007FFU);
    assert(execute(3U, 4U, 5U, 0x80000000U, 1U).first == 0x40000000U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 4U).first == 0x0FFFFFFFU);
    assert(execute(3U, 4U, 5U, 0x12345678U, 0U).first == 0x12345678U);
    assert(execute(3U, 4U, 5U, 1U, 1U).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 31U).first == 1U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 32U).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 33U).first == 0U);
    assert(execute(3U, 4U, 5U, 0xFFFFFFFFU, 63U).first == 0U);
    assert(execute(3U, 4U, 5U, 0x12345678U, 64U).first == 0x12345678U);
    assert(execute(3U, 4U, 5U, 0x80000000U, 65U).first == 0x40000000U);

    // Destination/source and destination/shift-register aliasing use old values.
    assert(execute(5U, 5U, 6U, 0x80000000U, 1U).first == 0x40000000U);
    assert(execute(6U, 5U, 6U, 0x80000000U, 1U).first == 0x40000000U);
    assert(execute(5U, 5U, 5U, 4U, 4U).first == 0U);

    (void)execute(7U, 4U, 5U, 0U, 0U, true); // EQ
    (void)execute(7U, 4U, 5U, 2U, 1U, true); // GT
    (void)execute(7U, 4U, 5U, 0xFFFFFFFFU, 0U, true); // LT
    (void)execute(7U, 4U, 5U, 0xFFFFFFFFU, 0U, true, true); // LT | SO

    const auto wind_result = execute(9U, 6U, 10U, 0x7FFFFFFFU, 0x14U);
    assert(wind_result.first == 0x000007FFU);
    const std::string trace = format_instruction_history(wind_result.second);
    assert(trace.find("srw r6=0x7FFFFFFF r10=0x00000014 -> r9=0x000007FF") !=
           std::string::npos);
    const auto record_trace = execute(7U, 4U, 5U, 8U, 1U, true);
    assert(format_instruction_history(record_trace.second).find("srw.") !=
           std::string::npos);
}

void or_immediate_shifted_tests()
{
    const auto encode_oris = [](std::uint8_t destination, std::uint8_t source,
                                std::uint16_t immediate) {
        return (25U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) | immediate;
    };

    constexpr std::uint32_t wind_waker_word = 0x67FFFFFFU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::or_immediate_shifted);
    assert(decoded.source == 31U);
    assert(decoded.destination == 31U);
    assert(decoded.immediate == 0xFFFF);

    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint32_t source_value, std::uint16_t immediate) {
        EspressoCore core(8U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[source] = source_value;
        core.state.cr = 0x12345678U;
        core.state.xer = 0xA00000A5U;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const std::uint32_t word = encode_oris(destination, source, immediate);
        core.memory.write32_be(0U, word);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const RunResult result = core.run(1U);
        const std::uint32_t expected = source_value |
            (static_cast<std::uint32_t>(immediate) << 16U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(core.state.cr == 0x12345678U);
        assert(core.state.xer == 0xA00000A5U);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == word);
        return std::pair{expected, result};
    };

    assert(execute(6U, 5U, 0x12345678U, 0x0000U).first == 0x12345678U);
    assert(execute(6U, 5U, 0x12345678U, 0x0001U).first == 0x12355678U);
    assert(execute(6U, 5U, 0x12345678U, 0x8000U).first == 0x92345678U);
    assert(execute(6U, 5U, 0x12345678U, 0xFFFFU).first == 0xFFFF5678U);
    assert(execute(5U, 5U, 0x12345678U, 0xFFFFU).first == 0xFFFF5678U);

    // Exact Wind Waker instruction and generic source/destination aliasing.
    const auto wind_waker = execute(31U, 31U, 0x12345678U, 0xFFFFU);
    assert(wind_waker.first == 0xFFFF5678U);
    const std::string trace = format_instruction_history(wind_waker.second);
    assert(trace.find("oris r31=0x12345678 imm=0xFFFF -> r31=0xFFFF5678") !=
           std::string::npos);
    assert(execute(31U, 31U, 0xA5A51234U, 0xFFFFU).first == 0xFFFF1234U);
}

void extend_sign_byte_tests()
{
    const auto encode_extsb = [](std::uint8_t destination, std::uint8_t source,
                                 bool record = false, std::uint8_t reserved = 0U) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) |
            (static_cast<std::uint32_t>(reserved & 0x1FU) << 11U) |
            (954U << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D8C0775U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::extend_sign_byte);
    assert(wind_waker.source == 12U);
    assert(wind_waker.destination == 12U);
    assert(wind_waker.record);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 954U);
    assert(decode(encode_extsb(6U, 5U)).opcode == Opcode::extend_sign_byte);
    assert(!decode(encode_extsb(6U, 5U)).record);
    assert(decode(encode_extsb(6U, 5U, false, 1U)).opcode == Opcode::unsupported);
    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint32_t input, bool record, bool so) {
        EspressoCore core(8U);
        core.state.gpr.fill(0x76543210U);
        core.state.gpr[source] = input;
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = so ? 0xA00000A5U : 0x200000A5U;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        core.memory.write32_be(0U, encode_extsb(destination, source, record));
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const auto original_memory = core.memory.read32_be(0U);

        const std::uint32_t byte = input & 0xFFU;
        const std::uint32_t expected = byte |
            ((byte & 0x80U) != 0U ? 0xFFFFFF00U : 0U);
        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        if (record)
        {
            const std::uint32_t cr0 = ((core.state.cr >> 28U) & 0xFU);
            const std::uint32_t expected_cr0 = (expected & 0x80000000U) != 0U
                ? (0x8U | static_cast<std::uint32_t>(so))
                : expected == 0U ? (0x2U | static_cast<std::uint32_t>(so))
                                 : (0x4U | static_cast<std::uint32_t>(so));
            assert(cr0 == expected_cr0);
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        assert(core.state.xer == initial_xer);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == original_memory);
        return result;
    };

    assert(execute(6U, 5U, 0x00000000U, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x00000001U, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x0000007FU, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x00000080U, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x000000FFU, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x1234567FU, false, false).steps == 1U);
    assert(execute(6U, 5U, 0x12345680U, false, false).steps == 1U);
    assert(execute(6U, 5U, 0xABCDEF81U, false, false).steps == 1U);

    // Rc=1 updates only CR0 and propagates the existing XER.SO bit.
    (void)execute(6U, 5U, 1U, true, false);          // GT
    (void)execute(6U, 5U, 0U, true, false);          // EQ
    (void)execute(6U, 5U, 0x80U, true, false);       // LT
    (void)execute(6U, 5U, 0x80U, true, true);        // LT | SO
    (void)execute(6U, 5U, 1U, true, true);           // GT | SO

    // Source/destination aliasing is the Wind Waker case; record form changes
    // CR0 even though the integer value remains one.
    const RunResult wind_waker_result = execute(12U, 12U, 1U, true, false);
    const std::string trace = format_instruction_history(wind_waker_result);
    assert(trace.find("extsb. r12=0x00000001 -> r12=0x00000001") != std::string::npos);
}

void extend_sign_halfword_tests()
{
    const auto encode_extsh = [](std::uint8_t destination, std::uint8_t source,
                                 bool record = false, std::uint8_t reserved = 0U) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(destination) << 16U) |
            (static_cast<std::uint32_t>(reserved & 0x1FU) << 11U) |
            (922U << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D2B0734U;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::extend_sign_halfword);
    assert(wind_waker.source == 9U);
    assert(wind_waker.destination == 11U);
    assert(!wind_waker.record);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 922U);
    assert(((wind_waker_word >> 11U) & 0x1FU) == 0U);
    assert(decode(encode_extsh(6U, 5U, true)).record);
    assert(decode(encode_extsh(6U, 5U, false, 1U)).opcode == Opcode::unsupported);
    assert(decode((31U << 26U) | (954U << 1U)).opcode == Opcode::extend_sign_byte);

    struct Case
    {
        std::uint32_t input;
        std::uint32_t expected;
    };
    constexpr std::array<Case, 9> cases{{
        {0x00000000U, 0x00000000U},
        {0x00000001U, 0x00000001U},
        {0x00007FFFU, 0x00007FFFU},
        {0x00008000U, 0xFFFF8000U},
        {0x0000FFFFU, 0xFFFFFFFFU},
        {0x12347FFFU, 0x00007FFFU},
        {0x12348000U, 0xFFFF8000U},
        {0xABCD8001U, 0xFFFF8001U},
        {0xFFFF0001U, 0x00000001U},
    }};

    const auto execute = [&](std::uint8_t destination, std::uint8_t source,
                             std::uint32_t input, bool record, bool so) {
        EspressoCore core(8U);
        core.state.gpr.fill(0x76543210U);
        core.state.gpr[source] = input;
        if (destination != source)
        {
            core.state.gpr[destination] = 0xDEADBEEFU;
        }
        constexpr std::uint32_t initial_cr = 0x12345678U;
        const std::uint32_t initial_xer = so ? 0xA00000A5U : 0x200000A5U;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        core.memory.write32_be(0U, encode_extsh(destination, source, record));
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint32_t instruction_word = core.memory.read32_be(0U);

        const std::uint32_t low_half = input & 0xFFFFU;
        const std::uint32_t expected = low_half |
            ((low_half & 0x8000U) != 0U ? 0xFFFF0000U : 0U);
        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == expected);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        if (record)
        {
            const std::uint32_t cr0 = (core.state.cr >> 28U) & 0xFU;
            const std::uint32_t expected_cr0 = (expected & 0x80000000U) != 0U
                ? (0x8U | static_cast<std::uint32_t>(so))
                : expected == 0U ? (0x2U | static_cast<std::uint32_t>(so))
                                 : (0x4U | static_cast<std::uint32_t>(so));
            assert(cr0 == expected_cr0);
            assert((core.state.cr & 0x0FFFFFFFU) == (initial_cr & 0x0FFFFFFFU));
        }
        else
        {
            assert(core.state.cr == initial_cr);
        }
        assert(core.state.xer == initial_xer);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read32_be(0U) == instruction_word);
        return result;
    };

    for (const Case& test : cases)
    {
        const RunResult result = execute(6U, 5U, test.input, false, false);
        assert(result.instruction_history.size() == 1U);
        assert(result.instruction_history.front().opcode_name == "extsh");
        assert(result.instruction_history.front().source_count == 1U);
        assert(result.instruction_history.front().source_values[0] == test.input);
    }

    const RunResult alias = execute(9U, 9U, 0x12348001U, false, false);
    assert(alias.instruction_history.front().destination_value == 0xFFFF8001U);
    (void)execute(6U, 5U, 1U, true, false);           // GT
    (void)execute(6U, 5U, 0U, true, false);           // EQ
    (void)execute(6U, 5U, 0x00008000U, true, false); // LT
    (void)execute(6U, 5U, 0x00008000U, true, true);  // LT | SO
    const RunResult record_alias = execute(9U, 9U, 0x12348001U, true, true);
    assert(record_alias.instruction_history.front().opcode_name == "extsh.");
    assert(format_instruction_history(record_alias).find(
        "extsh. r9=0x12348001 -> r9=0xFFFF8001") != std::string::npos);

    // Wind Waker's Rc=0 case writes +1 and leaves CR/XER and its source alone.
    EspressoCore wind_waker_core(8U);
    wind_waker_core.state.gpr[9] = 1U;
    wind_waker_core.state.gpr[11] = 0xAABBCCDDU;
    wind_waker_core.state.cr = 0x12345678U;
    wind_waker_core.state.xer = 0xA00000A5U;
    wind_waker_core.memory.write32_be(0U, wind_waker_word);
    const RunResult wind_waker_result = wind_waker_core.run(1U);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert(wind_waker_core.state.gpr[9] == 1U);
    assert(wind_waker_core.state.gpr[11] == 1U);
    assert(wind_waker_core.state.cr == 0x12345678U);
    assert(wind_waker_core.state.xer == 0xA00000A5U);
    assert(format_instruction_history(wind_waker_result).find(
        "extsh r9=0x00000001 -> r11=0x00000001") != std::string::npos);
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
    constexpr std::uint32_t imported_data_slot = 0xC0000014U;
    const RpxImportOptions import_options{};
    const RpxRelocationSymbol imported_data_symbol{
        "TestData", imported_data_slot, 1, 1, 8, 0};
    const auto imported_data_file = make_minimal_compressed_rpx(
        0x02000000U, 16, imported_data_symbol, {}, import_options);
    EspressoCore imported_data_core(0x1000U);
    imported_data_core.hle.register_data(
        "coreinit", "TestData", 4, 4,
        [](GuestMemory& memory, std::uint32_t address) {
            memory.write32_be(address, 0xA1B2C3D4U);
        });
    static_cast<void>(load_rpx32_powerpc(imported_data_core, imported_data_file));
    const auto imported_data_address =
        imported_data_core.hle.data_address("coreinit", "TestData");
    assert(imported_data_address.has_value());
    assert(*imported_data_address >= HleDispatcher::first_data_address);
    assert(*imported_data_address + 4U <= HleDispatcher::data_address_limit);
    assert(imported_data_core.memory.read32_be(0x0200000CU) == *imported_data_address);
    assert(imported_data_core.memory.read32_be(imported_data_slot) == *imported_data_address);
    assert(imported_data_core.memory.read32_be(*imported_data_address) == 0xA1B2C3D4U);

    const RpxRelocationSymbol imported_function_symbol{
        "TestFunction", imported_data_slot, 1, 2, 8, 0};
    const auto imported_function_file = make_minimal_compressed_rpx(
        0x02000000U, 16, imported_function_symbol, {}, import_options);
    EspressoCore imported_function_core(0x1000U);
    imported_function_core.hle.register_function(
        "coreinit", "TestFunction", [](EspressoCore& core) { core.state.gpr[3] = 42U; });
    static_cast<void>(load_rpx32_powerpc(imported_function_core, imported_function_file));
    const std::uint32_t function_address =
        imported_function_core.hle.bind_import("coreinit", "TestFunction");
    assert(imported_function_core.memory.read32_be(0x0200000CU) == function_address);
    assert(imported_function_core.memory.read32_be(imported_data_slot) == function_address);
    imported_function_core.state.cia = function_address;
    imported_function_core.state.lr = 0x80U;
    assert(imported_function_core.step() == StepResult::executed);
    assert(imported_function_core.state.gpr[3] == 42U);
    assert(imported_function_core.state.cia == 0x80U);

    const RpxRelocationSymbol unregistered_data_symbol{
        "UnregisteredData", imported_data_slot, 1, 1, 8, 0};
    const auto unregistered_data_file = make_minimal_compressed_rpx(
        0x02000000U, 16, unregistered_data_symbol, {}, import_options);
    EspressoCore unregistered_data_core(0x1000U);
    static_cast<void>(load_rpx32_powerpc(unregistered_data_core, unregistered_data_file));
    const std::uint32_t fallback_import_address =
        unregistered_data_core.hle.bind_import("coreinit", "UnregisteredData");
    assert(unregistered_data_core.memory.read32_be(0x0200000CU) == imported_data_slot);
    assert(unregistered_data_core.memory.read32_be(imported_data_slot) == fallback_import_address);

    EspressoCore reserved_section_core(0x1000U);
    bool reserved_data_section_rejected = false;
    try
    {
        const auto reserved_file = make_minimal_compressed_rpx(
            HleDispatcher::first_data_address);
        static_cast<void>(load_rpx32_powerpc(reserved_section_core, reserved_file));
    }
    catch (const std::invalid_argument& error)
    {
        reserved_data_section_rejected =
            std::string(error.what()).find("reserved HLE data address range") !=
            std::string::npos;
    }
    assert(reserved_data_section_rejected);

    EspressoCore reserved_heap_core(0x04010000U);
    static_cast<void>(load_rpx32_powerpc(
        reserved_heap_core, make_minimal_compressed_rpx()));
    assert(reserved_heap_core.guest_heap_limit <= HleDispatcher::first_data_address ||
           reserved_heap_core.guest_heap_cursor >= HleDispatcher::data_address_limit);

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

    const auto iob = core.hle.bind_data_import("coreinit", "_iob", core.memory);
    const auto iob_lock = core.hle.bind_data_import("coreinit", "_iob_lock", core.memory);
    const auto fopen_max =
        core.hle.bind_data_import("coreinit", "__gh_FOPEN_MAX", core.memory);
    assert(iob && iob_lock && fopen_max);
    assert((*iob % 8U) == 0);
    assert(*iob_lock - *iob == 20U * 0x10U);
    assert(*fopen_max - *iob_lock == 21U * sizeof(std::uint32_t));
    assert(core.hle.bind_data_import("coreinit", "_iob", core.memory) == iob);
    assert(core.memory.read32_be(*iob) == 0U);
    assert(core.memory.read32_be(*iob + 4U) == 0U);
    assert(core.memory.read32_be(*iob + 8U) == 0U);
    assert(core.memory.read32_be(*iob + 0x0CU) == (1U << 2U));
    assert(core.memory.read32_be(*iob + 0x1CU) == ((1U << 18U) | (1U << 1U)));
    assert(core.memory.read32_be(*iob + 0x2CU) == ((2U << 18U) | (1U << 1U)));
    for (std::uint32_t index = 0; index < 21U; ++index)
    {
        assert(core.memory.read32_be(*iob_lock + index * 4U) == 0U);
    }
    assert(core.memory.read16_be(*fopen_max) == 20U);

    const std::uint32_t flock_ptr = core.hle.bind_import("coreinit", "__ghs_flock_ptr");
    const auto flock_index = [&](std::uint32_t index, std::uint32_t expected_index) {
        core.state.cia = flock_ptr;
        core.state.lr = 0x70U;
        core.state.gpr[3] = *iob + index * 0x10U;
        assert(core.step() == StepResult::executed);
        assert(core.state.gpr[3] == *iob_lock + expected_index * 4U);
        assert(core.state.cia == 0x70U);
    };
    flock_index(0U, 0U);
    flock_index(1U, 1U);
    flock_index(2U, 2U);
    flock_index(20U, 20U);
    flock_index(30U, 20U);

    core.state.cia = flock_ptr;
    core.state.lr = 0x74U;
    core.state.gpr[3] = *iob + 1U;
    const RunResult misaligned_iob_result = core.run(1);
    assert(misaligned_iob_result.reason == StopReason::hle_error);
    assert(misaligned_iob_result.detail.find("misaligned pointer") != std::string::npos);

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
    assert(thread_core.memory.read32_be(thread_address + 0x32CU) == 80U);
    assert(thread_core.memory.read32_be(thread_address + 0x330U) == 80U);
    assert(thread_core.memory.read32_be(thread_address + 0x394U) == stack_start);
    assert(thread_core.memory.read32_be(thread_address + 0x398U) == stack_end);
    assert(thread_core.memory.read32_be(thread_address + 0x5BCU) == 2U);

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

void nn_olv_hle_tests()
{
    constexpr std::string_view library = "nn_olv.rpl";
    constexpr std::string_view symbol = "__ct__Q3_2nn3olv18DownloadedPostDataFv";
    constexpr std::uint32_t first_object = 0x100U;
    constexpr std::uint32_t return_address = 0x80U;
    constexpr std::uint8_t untouched = 0xA5U;

    EspressoCore adjacent_core(downloaded_post_data_size * 2U + 0x1000U);
    register_nn_olv_hle(adjacent_core.hle);
    const std::uint32_t constructor = adjacent_core.hle.bind_import(
        std::string(library), std::string(symbol));
    const std::uint32_t second_object = first_object + downloaded_post_data_size;
    adjacent_core.memory.fill_bytes(
        first_object, downloaded_post_data_size * 2U, untouched);

    const auto construct = [&](EspressoCore& core, std::uint32_t self) {
        core.state.cia = constructor;
        core.state.lr = return_address;
        core.state.gpr[3] = self;
        assert(core.step() == StepResult::executed);
        assert(core.state.cia == return_address);
        assert(core.state.gpr[3] == self);
    };

    construct(adjacent_core, first_object);
    for (std::uint32_t offset = 0; offset < downloaded_data_base_size; ++offset)
    {
        assert(adjacent_core.memory.read8(first_object + offset) == 0U);
    }
    assert(adjacent_core.memory.read32_be(
        first_object + downloaded_data_base_vtable_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        first_object + downloaded_post_data_community_id_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        first_object + downloaded_post_data_empathy_count_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        first_object + downloaded_post_data_comment_count_offset) == 0U);
    for (std::uint32_t offset = downloaded_post_data_initialized_size;
         offset < downloaded_post_data_size; ++offset)
    {
        assert(adjacent_core.memory.read8(first_object + offset) == untouched);
    }
    for (std::uint32_t offset = 0; offset < downloaded_post_data_size; ++offset)
    {
        assert(adjacent_core.memory.read8(second_object + offset) == untouched);
    }

    construct(adjacent_core, second_object);
    assert(adjacent_core.memory.read32_be(
        second_object + downloaded_data_base_vtable_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        second_object + downloaded_post_data_community_id_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        second_object + downloaded_post_data_empathy_count_offset) == 0U);
    assert(adjacent_core.memory.read32_be(
        second_object + downloaded_post_data_comment_count_offset) == 0U);
    for (std::uint32_t offset = downloaded_post_data_initialized_size;
         offset < downloaded_post_data_size; ++offset)
    {
        assert(adjacent_core.memory.read8(second_object + offset) == untouched);
    }

    EspressoCore allocating_core(downloaded_post_data_size + 0x1000U);
    register_nn_olv_hle(allocating_core.hle);
    const std::uint32_t allocating_constructor = allocating_core.hle.bind_import(
        std::string(library), std::string(symbol));
    allocating_core.configure_guest_heap(
        first_object, first_object + downloaded_post_data_size);
    allocating_core.memory.fill_bytes(first_object, downloaded_post_data_size, untouched);
    allocating_core.state.cia = allocating_constructor;
    allocating_core.state.lr = return_address;
    allocating_core.state.gpr[3] = 0U;
    assert(allocating_core.step() == StepResult::executed);
    assert(allocating_core.state.gpr[3] == first_object);
    assert((allocating_core.state.gpr[3] & 7U) == 0U);
    assert(allocating_core.guest_heap_cursor == first_object + downloaded_post_data_size);
    assert(allocating_core.memory.read32_be(
        first_object + downloaded_post_data_comment_count_offset) == 0U);
    for (std::uint32_t offset = downloaded_post_data_initialized_size;
         offset < downloaded_post_data_size; ++offset)
    {
        assert(allocating_core.memory.read8(first_object + offset) == untouched);
    }

    EspressoCore allocation_failure_core(0x100U);
    register_nn_olv_hle(allocation_failure_core.hle);
    const std::uint32_t failed_constructor = allocation_failure_core.hle.bind_import(
        std::string(library), std::string(symbol));
    allocation_failure_core.configure_guest_heap(0x20U, 0x20U);
    allocation_failure_core.state.cia = failed_constructor;
    allocation_failure_core.state.lr = return_address;
    allocation_failure_core.state.gpr[3] = 0U;
    assert(allocation_failure_core.step() == StepResult::executed);
    assert(allocation_failure_core.state.gpr[3] == 0U);
    assert(allocation_failure_core.state.cia == return_address);

    EspressoCore invalid_memory_core(0x100U);
    register_nn_olv_hle(invalid_memory_core.hle);
    const std::uint32_t invalid_constructor = invalid_memory_core.hle.bind_import(
        std::string(library), std::string(symbol));
    invalid_memory_core.state.cia = invalid_constructor;
    invalid_memory_core.state.lr = return_address;
    invalid_memory_core.state.gpr[3] = 0x100U;
    const RunResult invalid_result = invalid_memory_core.run(1U);
    assert(invalid_result.reason == StopReason::memory_fault);
    assert(invalid_result.detail.find("write") != std::string::npos);
    assert(invalid_result.detail.find("0x00000100") != std::string::npos);
    assert(invalid_memory_core.state.gpr[3] == 0x100U);

    constexpr std::string_view topic_symbol =
        "__ct__Q3_2nn3olv19DownloadedTopicDataFv";
    constexpr std::uint32_t topic_object = 0x200U;
    constexpr std::uint8_t topic_sentinel = 0x6BU;

    EspressoCore topic_core(downloaded_topic_data_size + 0x1000U);
    register_nn_olv_hle(topic_core.hle);
    const std::uint32_t topic_constructor = topic_core.hle.bind_import(
        std::string(library), std::string(topic_symbol));
    topic_core.memory.fill_bytes(topic_object, downloaded_topic_data_size, topic_sentinel);
    topic_core.state.cia = topic_constructor;
    topic_core.state.lr = return_address;
    topic_core.state.gpr[3] = topic_object;
    assert(topic_core.step() == StepResult::executed);
    assert(topic_core.state.cia == return_address);
    assert(topic_core.state.gpr[3] == topic_object);
    assert(topic_core.memory.read32_be(
        topic_object + downloaded_topic_data_unk1_offset) == 0U);
    assert(topic_core.memory.read32_be(
        topic_object + downloaded_topic_data_community_id_offset) == 0U);
    for (std::uint32_t offset = downloaded_topic_data_initialized_size;
         offset < downloaded_topic_data_size; ++offset)
    {
        assert(topic_core.memory.read8(topic_object + offset) == topic_sentinel);
    }

    EspressoCore topic_allocating_core(downloaded_topic_data_size + 0x1000U);
    register_nn_olv_hle(topic_allocating_core.hle);
    const std::uint32_t allocating_topic_constructor =
        topic_allocating_core.hle.bind_import(
            std::string(library), std::string(topic_symbol));
    topic_allocating_core.configure_guest_heap(
        topic_object, topic_object + downloaded_topic_data_size);
    topic_allocating_core.state.cia = allocating_topic_constructor;
    topic_allocating_core.state.lr = return_address;
    topic_allocating_core.state.gpr[3] = 0U;
    assert(topic_allocating_core.step() == StepResult::executed);
    const std::uint32_t allocated_topic = topic_allocating_core.state.gpr[3];
    assert(allocated_topic == topic_object);
    assert(allocated_topic != 0U);
    assert((allocated_topic & 7U) == 0U);
    assert(topic_allocating_core.guest_heap_cursor ==
           topic_object + downloaded_topic_data_size);
    assert(topic_allocating_core.memory.read32_be(
        allocated_topic + downloaded_topic_data_unk1_offset) == 0U);
    assert(topic_allocating_core.memory.read32_be(
        allocated_topic + downloaded_topic_data_community_id_offset) == 0U);

    EspressoCore topic_allocation_failure_core(0x100U);
    register_nn_olv_hle(topic_allocation_failure_core.hle);
    const std::uint32_t failed_topic_constructor =
        topic_allocation_failure_core.hle.bind_import(
            std::string(library), std::string(topic_symbol));
    topic_allocation_failure_core.configure_guest_heap(0x20U, 0x20U);
    topic_allocation_failure_core.state.cia = failed_topic_constructor;
    topic_allocation_failure_core.state.lr = return_address;
    topic_allocation_failure_core.state.gpr[3] = 0U;
    assert(topic_allocation_failure_core.step() == StepResult::executed);
    assert(topic_allocation_failure_core.state.gpr[3] == 0U);
    assert(topic_allocation_failure_core.state.cia == return_address);

    EspressoCore invalid_topic_core(0x100U);
    register_nn_olv_hle(invalid_topic_core.hle);
    const std::uint32_t invalid_topic_constructor = invalid_topic_core.hle.bind_import(
        std::string(library), std::string(topic_symbol));
    constexpr std::uint32_t partially_mapped_topic = 0xFCU;
    invalid_topic_core.memory.fill_bytes(
        partially_mapped_topic, 4U, topic_sentinel);
    invalid_topic_core.state.cia = invalid_topic_constructor;
    invalid_topic_core.state.lr = return_address;
    invalid_topic_core.state.gpr[3] = partially_mapped_topic;
    const RunResult invalid_topic_result = invalid_topic_core.run(1U);
    assert(invalid_topic_result.reason == StopReason::memory_fault);
    assert(invalid_topic_result.detail.find("write") != std::string::npos);
    assert(invalid_topic_result.detail.find("0x000000FC") != std::string::npos);
    assert(invalid_topic_core.state.gpr[3] == partially_mapped_topic);
    assert(invalid_topic_core.memory.read32_be(partially_mapped_topic) ==
           0x6B6B6B6BU);

    constexpr std::string_view list_param_symbol =
        "__ct__Q3_2nn3olv25DownloadPostDataListParamFv";
    constexpr std::uint32_t list_param_object = 0x1000U;
    constexpr std::uint32_t adjacent_list_param =
        list_param_object + download_post_data_list_param_size;
    constexpr std::uint8_t list_param_sentinel = 0xD3U;

    EspressoCore list_param_core(
        list_param_object + 2U * download_post_data_list_param_size + 0x1000U);
    register_nn_olv_hle(list_param_core.hle);
    const std::uint32_t list_param_constructor = list_param_core.hle.bind_import(
        std::string(library), std::string(list_param_symbol));
    list_param_core.memory.fill_bytes(
        list_param_object - 1U,
        2U * download_post_data_list_param_size + 2U,
        list_param_sentinel);
    list_param_core.state.cia = list_param_constructor;
    list_param_core.state.lr = return_address;
    list_param_core.state.gpr[3] = list_param_object;
    assert(list_param_core.step() == StepResult::executed);
    assert(list_param_core.state.cia == return_address);
    assert(list_param_core.state.gpr[3] == list_param_object);
    assert(list_param_core.memory.read8(list_param_object - 1U) ==
           list_param_sentinel);
    for (std::uint32_t offset = 0; offset < download_post_data_list_param_size; ++offset)
    {
        assert(list_param_core.memory.read8(list_param_object + offset) == 0U);
    }
    for (std::uint32_t offset = 0; offset < download_post_data_list_param_size; ++offset)
    {
        assert(list_param_core.memory.read8(adjacent_list_param + offset) ==
               list_param_sentinel);
    }
    assert(list_param_core.memory.read8(
        adjacent_list_param + download_post_data_list_param_size) == list_param_sentinel);

    EspressoCore invalid_list_param_core(0x2000U);
    register_nn_olv_hle(invalid_list_param_core.hle);
    const std::uint32_t invalid_list_param_constructor =
        invalid_list_param_core.hle.bind_import(
            std::string(library), std::string(list_param_symbol));
    constexpr std::uint32_t partially_mapped_list_param = 0x1800U;
    invalid_list_param_core.memory.fill_bytes(
        partially_mapped_list_param,
        0x2000U - partially_mapped_list_param,
        list_param_sentinel);
    invalid_list_param_core.state.cia = invalid_list_param_constructor;
    invalid_list_param_core.state.lr = return_address;
    invalid_list_param_core.state.gpr[3] = partially_mapped_list_param;
    const RunResult invalid_list_param_result = invalid_list_param_core.run(1U);
    assert(invalid_list_param_result.reason == StopReason::memory_fault);
    assert(invalid_list_param_result.detail.find("write") != std::string::npos);
    assert(invalid_list_param_result.detail.find("0x00001800") != std::string::npos);
    assert(invalid_list_param_core.state.gpr[3] == partially_mapped_list_param);
    for (std::uint32_t address = partially_mapped_list_param;
         address < 0x2000U; ++address)
    {
        assert(invalid_list_param_core.memory.read8(address) == list_param_sentinel);
    }
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

void os_block_move_hle_tests()
{
    EspressoCore core(0x20U);
    register_coreinit_hle(core.hle);
    core.memory.map_region(0x20U, 0x20U);
    core.memory.fill_bytes(0U, 0x40U, 0xCCU);
    const std::uint32_t import = core.hle.bind_import("coreinit", "OSBlockMove");

    const auto invoke = [&](std::uint32_t destination, std::uint32_t source,
                            std::uint32_t size, std::uint32_t flush,
                            std::uint32_t return_address) {
        core.state.cia = import;
        core.state.lr = return_address;
        core.state.gpr[3] = destination;
        core.state.gpr[4] = source;
        core.state.gpr[5] = size;
        core.state.gpr[6] = flush;
        assert(core.step() == StepResult::executed);
        assert(core.state.cia == return_address);
        assert(core.state.gpr[3] == destination);
        assert(core.state.gpr[4] == source);
        assert(core.state.gpr[5] == size);
        assert(core.state.gpr[6] == flush);
    };

    core.memory.write8(1U, 0x5AU);
    invoke(3U, 1U, 1U, 0U, 0x80U);
    assert(core.memory.read8(2U) == 0xCCU);
    assert(core.memory.read8(3U) == 0x5AU);
    assert(core.memory.read8(4U) == 0xCCU);

    const std::array<std::uint8_t, 5> unaligned{0x10U, 0x20U, 0x30U, 0x40U, 0x50U};
    core.memory.write_bytes(7U, unaligned);
    invoke(0x0DU, 7U, static_cast<std::uint32_t>(unaligned.size()), 0U, 0x84U);
    std::array<std::uint8_t, 5> unaligned_result{};
    core.memory.read_bytes(0x0DU, unaligned_result);
    assert(unaligned_result == unaligned);
    assert(core.memory.read8(0x0CU) == 0xCCU);
    assert(core.memory.read8(0x12U) == 0xCCU);

    const auto reset_overlap_bytes = [&]() {
        constexpr std::array<std::uint8_t, 8> bytes{0, 1, 2, 3, 4, 5, 6, 7};
        core.memory.write_bytes(0x10U, bytes);
    };
    std::array<std::uint8_t, 8> overlap_result{};
    reset_overlap_bytes();
    invoke(0x12U, 0x10U, 6U, 0U, 0x88U);
    core.memory.read_bytes(0x10U, overlap_result);
    assert((overlap_result == std::array<std::uint8_t, 8>{0, 1, 0, 1, 2, 3, 4, 5}));

    reset_overlap_bytes();
    invoke(0x10U, 0x12U, 6U, 0U, 0x8CU);
    core.memory.read_bytes(0x10U, overlap_result);
    assert((overlap_result == std::array<std::uint8_t, 8>{2, 3, 4, 5, 6, 7, 6, 7}));

    reset_overlap_bytes();
    invoke(0x10U, 0x10U, 8U, 0U, 0x90U);
    core.memory.read_bytes(0x10U, overlap_result);
    assert((overlap_result == std::array<std::uint8_t, 8>{0, 1, 2, 3, 4, 5, 6, 7}));

    // Both source and destination can independently cross flat -> sparse storage.
    const std::array<std::uint8_t, 6> cross_source{0x20U, 0x26U, 0x31U,
                                                  0x42U, 0x53U, 0x64U};
    core.memory.write_bytes(0x1DU, cross_source);
    invoke(0x08U, 0x1DU, 6U, 0U, 0x94U);
    std::array<std::uint8_t, 6> cross_result{};
    core.memory.read_bytes(0x08U, cross_result);
    assert(cross_result == cross_source);

    const std::array<std::uint8_t, 6> destination_cross_source{1, 3, 5, 7, 9, 11};
    core.memory.write_bytes(0x0AU, destination_cross_source);
    invoke(0x1DU, 0x0AU, 6U, 1U, 0x98U);
    core.memory.read_bytes(0x1DU, cross_result);
    assert(cross_result == destination_cross_source);

    // Overlap across the backing-store boundary still stages original bytes.
    const std::array<std::uint8_t, 8> spanning_overlap{0xA0U, 0xA1U, 0xA2U, 0xA3U,
                                                       0xA4U, 0xA5U, 0xA6U, 0xA7U};
    core.memory.write_bytes(0x1CU, spanning_overlap);
    invoke(0x1EU, 0x1CU, 6U, 0U, 0x9CU);
    std::array<std::uint8_t, 8> spanning_result{};
    core.memory.read_bytes(0x1CU, spanning_result);
    assert((spanning_result == std::array<std::uint8_t, 8>{
        0xA0U, 0xA1U, 0xA0U, 0xA1U, 0xA2U, 0xA3U, 0xA4U, 0xA5U}));

    // Zero length never touches invalid guest addresses but still returns dst.
    invoke(0xFFFFFFFFU, 0xFFFFFFFEU, 0U, 0U, 0xA0U);

    // Invalid source: no destination byte changes.
    core.memory.write_bytes(0x10U, std::array<std::uint8_t, 4>{0xB1U, 0xB2U, 0xB3U, 0xB4U});
    core.state.cia = import;
    core.state.lr = 0xA4U;
    core.state.gpr[3] = 0x10U;
    core.state.gpr[4] = 0x40U;
    core.state.gpr[5] = 2U;
    core.state.gpr[6] = 0U;
    const RunResult invalid_source = core.run(1U);
    assert(invalid_source.reason == StopReason::memory_fault);
    assert(invalid_source.detail.find("read 2 byte(s)") != std::string::npos);
    assert(core.memory.read32_be(0x10U) == 0xB1B2B3B4U);

    // A partly mapped source also faults transactionally.
    core.state.cia = import;
    core.state.lr = 0xA8U;
    core.state.gpr[3] = 0x10U;
    core.state.gpr[4] = 0x3EU;
    core.state.gpr[5] = 4U;
    core.state.gpr[6] = 0U;
    const RunResult partial_source = core.run(1U);
    assert(partial_source.reason == StopReason::memory_fault);
    assert(partial_source.detail.find("read 4 byte(s)") != std::string::npos);
    assert(core.memory.read32_be(0x10U) == 0xB1B2B3B4U);

    // Invalid/partly mapped destination writes nothing, including its mapped prefix.
    core.memory.write8(0x3EU, 0xD1U);
    core.memory.write8(0x3FU, 0xD2U);
    core.state.cia = import;
    core.state.lr = 0xACU;
    core.state.gpr[3] = 0x3EU;
    core.state.gpr[4] = 0x10U;
    core.state.gpr[5] = 4U;
    core.state.gpr[6] = 1U;
    const RunResult invalid_destination = core.run(1U);
    assert(invalid_destination.reason == StopReason::memory_fault);
    assert(invalid_destination.detail.find("write 4 byte(s)") != std::string::npos);
    assert(core.memory.read8(0x3EU) == 0xD1U);
    assert(core.memory.read8(0x3FU) == 0xD2U);

    // flush=0 and flush=1 have the same coherent GuestMemory behavior for now.
    core.memory.write_bytes(0x10U, std::array<std::uint8_t, 3>{0x20U, 0x26U, 0x30U});
    invoke(0x18U, 0x10U, 2U, 0U, 0xB0U);
    assert(core.memory.read16_be(0x18U) == 0x2026U);
    core.memory.zero_fill(0x18U, 2U);
    invoke(0x18U, 0x10U, 2U, 1U, 0xB4U);
    assert(core.memory.read16_be(0x18U) == 0x2026U);

    // Wind Waker regression: copy the two-byte UTF-16 value 0x2026.
    core.memory.write16_be(0x14U, 0x2026U);
    invoke(0x16U, 0x14U, 2U, 0U, 0xB8U);
    assert(core.memory.read16_be(0x16U) == 0x2026U);
}

void os_get_system_info_hle_tests()
{
    using Clock = std::chrono::system_clock;
    constexpr std::uint32_t bus_clock_speed = 248625000U;
    constexpr std::uint32_t core_clock_speed = 1243125000U;
    constexpr std::uint32_t timer_clock_speed = 62156250U;
    constexpr std::int64_t unix_epoch_offset = 946684800;

    const auto base_ticks_at = [](Clock::time_point time) {
        using namespace std::chrono;
        const auto elapsed = time.time_since_epoch();
        const auto unix_seconds = duration_cast<seconds>(elapsed).count();
        if (unix_seconds < unix_epoch_offset)
        {
            return std::uint64_t{0};
        }
        const auto fractional_nanoseconds =
            duration_cast<nanoseconds>(elapsed - seconds(unix_seconds)).count();
        return static_cast<std::uint64_t>(unix_seconds - unix_epoch_offset) *
                timer_clock_speed +
            static_cast<std::uint64_t>(fractional_nanoseconds) * timer_clock_speed /
                1000000000ULL;
    };

    EspressoCore core(0x400U);
    register_coreinit_hle(core.hle);
    const std::uint32_t import = core.hle.bind_import("coreinit", "OSGetSystemInfo");
    core.memory.fill_bytes(0x100U, 0x20U, 0xA5U);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA0000000U + reg * 0x10101U;
    }
    const auto gpr_before = core.state.gpr;
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    const auto ps0_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = 0x80U;

    const auto host_time_before = Clock::now();
    core.state.cia = import;
    assert(core.step() == StepResult::executed);
    const auto host_time_after = Clock::now();

    const std::uint32_t address = core.state.gpr[3];
    assert(address != 0U);
    assert(address >= HleDispatcher::first_data_address);
    assert(address + 0x20U <= HleDispatcher::data_address_limit);
    assert((address & 7U) == 0U);
    assert(core.state.cia == 0x80U);
    assert(core.state.lr == 0x80U);
    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        if (reg != 3U)
        {
            assert(core.state.gpr[reg] == gpr_before[reg]);
        }
    }
    assert(core.state.cr == 0x12345678U);
    assert(core.state.xer == 0xA00000A5U);
    assert(core.state.ctr == 0xCAFEBABEU);
    assert(core.state.fpscr == 0x5A5AA55AU);
    assert(core.state.fpr == ps0_before);
    assert(core.state.fpr_ps1 == ps1_before);
    assert(core.memory.read8(0x100U) == 0xA5U);
    assert(core.memory.read8(0x11FU) == 0xA5U);

    assert(core.memory.read32_be(address + 0x00U) == bus_clock_speed);
    assert(core.memory.read32_be(address + 0x04U) == core_clock_speed);
    const std::uint64_t base_time = core.memory.read64_be(address + 0x08U);
    assert(base_time != 0U);
    assert(core.memory.read32_be(address + 0x10U) == 0x00080000U);
    assert(core.memory.read32_be(address + 0x14U) == 0x00200000U);
    assert(core.memory.read32_be(address + 0x18U) == 0x00080000U);
    assert(core.memory.read32_be(address + 0x1CU) == 5U);
    const std::uint64_t earliest_base = base_ticks_at(host_time_before);
    const std::uint64_t latest_base = base_ticks_at(host_time_after) + timer_clock_speed;
    assert(base_time >= earliest_base && base_time <= latest_base);

    // Every guest field is big-endian and occupies exactly its documented bytes.
    assert(core.memory.read8(address + 0x00U) == 0x0EU);
    assert(core.memory.read8(address + 0x01U) == 0xD1U);
    assert(core.memory.read8(address + 0x02U) == 0xB7U);
    assert(core.memory.read8(address + 0x03U) == 0x68U);
    assert(core.memory.read8(address + 0x04U) == 0x4AU);
    assert(core.memory.read8(address + 0x05U) == 0x18U);
    assert(core.memory.read8(address + 0x06U) == 0x95U);
    assert(core.memory.read8(address + 0x07U) == 0x08U);

    // A second call returns the same data object and does not refresh baseTime.
    core.state.gpr[3] = 0x101F8BA0U; // stale caller value is not an input
    core.state.lr = 0x84U;
    core.state.cia = import;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == address);
    assert(core.memory.read64_be(address + 0x08U) == base_time);
    assert(core.state.cia == 0x84U);

    // A separate core receives its own guest backing, even though the virtual
    // address can be identical in its independent address space.
    EspressoCore other(0x200U);
    register_coreinit_hle(other.hle);
    const std::uint32_t other_import =
        other.hle.bind_import("coreinit", "OSGetSystemInfo");
    other.state.cia = other_import;
    other.state.lr = 0x88U;
    assert(other.step() == StepResult::executed);
    const std::uint32_t other_address = other.state.gpr[3];
    assert(other_address == address);
    assert(other.memory.read32_be(other_address) == bus_clock_speed);
    other.memory.write32_be(other_address, 0xDEADBEEFU);
    assert(core.memory.read32_be(address) == bus_clock_speed);
}

void fs_init_hle_tests()
{
    constexpr std::uint32_t return_address = 0x0274223CU;
    constexpr std::uint32_t thread_stack_start = 0x3F00U;
    constexpr std::uint32_t thread_stack_end = 0x2F00U;

    EspressoCore core(0x4000U);
    assert(!core.fs_initialized);
    register_coreinit_hle(core.hle);
    const std::uint32_t import = core.hle.bind_import("coreinit", "FSInit");
    core.configure_guest_heap(0x100U, 0x1000U);
    const std::uint32_t thread = initialize_default_guest_thread(
        core, thread_stack_start, thread_stack_end);
    constexpr std::uint32_t queue = 0x1800U;
    core.memory.fill_bytes(queue, 0x3CU, 0xD7U);
    core.memory.fill_bytes(0x2000U, 0x100U, 0xA6U);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.current_thread_address = thread;

    const auto invoke = [&](std::uint32_t stale_r3) {
        core.state.gpr[3] = stale_r3;
        core.state.lr = return_address;
        core.state.cia = import;
        const auto gprs_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto ps1_before = core.state.fpr_ps1;
        const std::uint32_t cr_before = core.state.cr;
        const std::uint32_t xer_before = core.state.xer;
        const std::uint32_t ctr_before = core.state.ctr;
        const std::uint32_t fpscr_before = core.state.fpscr;
        const std::uint32_t lr_before = core.state.lr;
        const std::uint32_t thread_before = core.current_thread_address;
        const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
        const std::uint32_t heap_limit_before = core.guest_heap_limit;
        const auto base_handles_before = core.base_heap_handles;
        const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
        const std::uint32_t mem2_end_before = core.mem2_heap_region_end;
        std::vector<std::uint8_t> memory_before(core.memory.size());
        core.memory.read_bytes(0U, memory_before);

        assert(core.step() == StepResult::executed);
        assert(core.fs_initialized);
        assert(core.state.cia == return_address);
        assert(core.state.gpr == gprs_before);
        assert(core.state.fpr == fpr_before);
        assert(core.state.fpr_ps1 == ps1_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.lr == lr_before);
        assert(core.current_thread_address == thread_before);
        assert(core.guest_heap_cursor == heap_cursor_before);
        assert(core.guest_heap_limit == heap_limit_before);
        assert(core.base_heap_handles == base_handles_before);
        assert(core.mem2_heap_region_begin == mem2_begin_before);
        assert(core.mem2_heap_region_end == mem2_end_before);
        std::vector<std::uint8_t> memory_after(core.memory.size());
        core.memory.read_bytes(0U, memory_after);
        assert(memory_after == memory_before);
    };

    assert(!core.fs_initialized);
    invoke(0x04021078U);
    assert(core.state.gpr[3] == 0x04021078U);
    invoke(0xDEADBEEFU);
    assert(core.state.gpr[3] == 0xDEADBEEFU);

    // Heap configuration is allocator setup, not a new Cafe process/session.
    core.configure_guest_heap(0x2000U, 0x3000U);
    assert(core.fs_initialized);

    core.reset();
    assert(!core.fs_initialized);

    EspressoCore first(0x100U);
    EspressoCore second(0x100U);
    register_coreinit_hle(first.hle);
    register_coreinit_hle(second.hle);
    const std::uint32_t first_import = first.hle.bind_import("coreinit", "FSInit");
    const std::uint32_t second_import = second.hle.bind_import("coreinit", "FSInit");
    assert(!first.fs_initialized);
    assert(!second.fs_initialized);

    first.state.cia = first_import;
    first.state.lr = 0x80U;
    first.state.gpr[3] = 0x11111111U;
    assert(first.step() == StepResult::executed);
    assert(first.fs_initialized);
    assert(!second.fs_initialized);

    second.state.cia = second_import;
    second.state.lr = 0x84U;
    second.state.gpr[3] = 0x22222222U;
    assert(second.step() == StepResult::executed);
    assert(first.fs_initialized);
    assert(second.fs_initialized);

    first.reset();
    assert(!first.fs_initialized);
    assert(second.fs_initialized);
}

void nn_save_init_hle_tests()
{
    constexpr std::uint32_t return_address = 0x0274226CU;
    constexpr std::uint32_t memory_size = 0x10000U;
    constexpr std::uint32_t client_address = 0x2003U;
    constexpr std::uint32_t client_body = 0x2040U;
    constexpr std::uint32_t client_size = 0x1700U;

    const auto invoke = [](EspressoCore& core,
                           std::uint32_t import,
                           std::uint32_t stale_r3,
                           std::uint32_t stale_r4,
                           std::uint32_t lr) {
        core.state.cia = import;
        core.state.lr = lr;
        core.state.gpr[3] = stale_r3;
        core.state.gpr[4] = stale_r4;
        return core.run(1U);
    };
    const auto snapshot_memory = [](const EspressoCore& core) {
        std::vector<std::uint8_t> bytes(core.memory.size());
        core.memory.read_bytes(0U, bytes);
        return bytes;
    };

    EspressoCore core(memory_size);
    register_nn_save_hle(core.hle);
    assert(!core.nn_save_initialized);
    const std::uint32_t save_init = core.hle.bind_import("nn_save", "SAVEInit");

    // Populate unrelated per-session state and guest objects so the test can
    // prove SAVEInit changes only its lifecycle flag and status return.
    core.memory.fill_bytes(0U, memory_size, 0xA5U);
    core.memory.fill_bytes(client_address, client_size, 0x5AU);
    core.memory.fill_bytes(0x5000U, 0x6A0U, 0x3CU);
    core.memory.fill_bytes(0x6000U, 0x3CU, 0xC3U);
    core.fs_initialized = true;
    core.fs_clients.push_back({client_address, client_body, 1U});
    core.next_fs_client_handle = 2U;
    core.current_thread_address = 0x5000U;
    core.configure_guest_heap(0x8000U, 0xA000U);
    core.base_heap_handles[1] = 0x8100U;
    core.mem2_heap_region_begin = 0x8200U;
    core.mem2_heap_region_end = 0x9000U;

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;

    constexpr std::uint32_t application_client_callback_offset =
        0x1448U + 0x14U;
    core.memory.write32_be(
        client_body + application_client_callback_offset, 0x02742028U);
    core.memory.write32_be(
        client_body + application_client_callback_offset + 4U, 0U);
    core.memory.write32_be(
        client_body + application_client_callback_offset + 8U, 0U);
    core.memory.write32_be(
        client_body + application_client_callback_offset + 0x0CU,
        client_body + application_client_callback_offset);
    core.memory.write32_be(
        client_body + application_client_callback_offset + 0x18U, 11U);
    core.memory.write32_be(
        client_body + application_client_callback_offset + 0x1CU, client_address);

    const auto initial_gprs = core.state.gpr;
    const auto initial_fprs = core.state.fpr;
    const auto initial_ps1 = core.state.fpr_ps1;
    const auto initial_fs_clients = core.fs_clients;
    const auto initial_memory = snapshot_memory(core);
    const std::uint32_t initial_next_handle = core.next_fs_client_handle;
    const std::uint32_t initial_thread = core.current_thread_address;
    const std::uint32_t initial_cr = core.state.cr;
    const std::uint32_t initial_xer = core.state.xer;
    const std::uint32_t initial_ctr = core.state.ctr;
    const std::uint32_t initial_fpscr = core.state.fpscr;
    const std::uint32_t initial_heap_cursor = core.guest_heap_cursor;
    const std::uint32_t initial_heap_limit = core.guest_heap_limit;
    const auto initial_base_heap_handles = core.base_heap_handles;
    const std::uint32_t initial_mem2_begin = core.mem2_heap_region_begin;
    const std::uint32_t initial_mem2_end = core.mem2_heap_region_end;

    // Current Wind Waker values are stale registers, not SAVEInit arguments.
    const RunResult first = invoke(
        core, save_init, 0xFFFFFFFFU, 0xFFFFFFFCU, return_address);
    assert(first.reason == StopReason::instruction_limit);
    assert(first.steps == 1U);
    assert(core.nn_save_initialized);
    assert(core.state.gpr[3] == 0U);
    assert(core.state.gpr[4] == 0xFFFFFFFCU);
    assert(core.state.cia == return_address);
    assert(core.state.lr == return_address);
    assert(core.state.cr == initial_cr);
    assert(core.state.xer == initial_xer);
    assert(core.state.ctr == initial_ctr);
    assert(core.state.fpscr == initial_fpscr);
    assert(core.state.fpr == initial_fprs);
    assert(core.state.fpr_ps1 == initial_ps1);
    for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        if (reg != 3U && reg != 4U)
        {
            assert(core.state.gpr[reg] == initial_gprs[reg]);
        }
    }
    assert(snapshot_memory(core) == initial_memory);
    assert(core.fs_initialized);
    assert(core.fs_clients.size() == initial_fs_clients.size());
    assert(core.fs_clients[0].client_address == initial_fs_clients[0].client_address);
    assert(core.fs_clients[0].body_address == initial_fs_clients[0].body_address);
    assert(core.fs_clients[0].synthetic_handle == initial_fs_clients[0].synthetic_handle);
    assert(core.next_fs_client_handle == initial_next_handle);
    assert(core.current_thread_address == initial_thread);
    assert(core.guest_heap_cursor == initial_heap_cursor);
    assert(core.guest_heap_limit == initial_heap_limit);
    assert(core.base_heap_handles == initial_base_heap_handles);
    assert(core.mem2_heap_region_begin == initial_mem2_begin);
    assert(core.mem2_heap_region_end == initial_mem2_end);
    assert(core.memory.read32_be(client_body + 0x1444U) == 0x5A5A5A5AU);
    assert(core.memory.read32_be(client_body + application_client_callback_offset) ==
           0x02742028U);

    // Repeated initialization succeeds and always returns SAVE_STATUS_OK,
    // irrespective of the stale value supplied in r3.
    for (const std::uint32_t stale_r3 : {0xDEADBEEFU, 0x12345678U, client_address})
    {
        const RunResult repeated = invoke(
            core, save_init, stale_r3, 0x100FEE68U, return_address + 4U);
        assert(repeated.reason == StopReason::instruction_limit);
        assert(repeated.steps == 1U);
        assert(core.nn_save_initialized);
        assert(core.state.gpr[3] == 0U);
        assert(core.state.gpr[4] == 0x100FEE68U);
        assert(core.state.cia == return_address + 4U);
        assert(snapshot_memory(core) == initial_memory);
        assert(core.fs_clients.size() == 1U);
        assert(core.next_fs_client_handle == initial_next_handle);
    }

    core.configure_guest_heap(0x9000U, 0xA000U);
    assert(core.nn_save_initialized);
    core.reset();
    assert(!core.nn_save_initialized);
    assert(!core.fs_initialized);
    assert(core.fs_clients.empty());
    assert(core.next_fs_client_handle == 1U);

    // Reset does not discard registered HLE handlers, so the same core can
    // initialize a fresh nn_save session without re-registering exports.
    const RunResult after_reset = invoke(
        core, save_init, 0xFFFFFFFFU, 0xFFFFFFFCU, return_address + 8U);
    assert(after_reset.reason == StopReason::instruction_limit);
    assert(core.nn_save_initialized);
    assert(core.state.gpr[3] == 0U);
    assert(core.state.gpr[4] == 0xFFFFFFFCU);
    assert(core.state.cia == return_address + 8U);

    EspressoCore other(0x1000U);
    register_nn_save_hle(other.hle);
    const std::uint32_t other_save_init =
        other.hle.bind_import("nn_save", "SAVEInit");
    assert(!other.nn_save_initialized);
    const RunResult other_result = invoke(
        other, other_save_init, 0x04021164U, 0x100FEE68U, return_address);
    assert(other_result.reason == StopReason::instruction_limit);
    assert(other.nn_save_initialized);
    assert(core.nn_save_initialized);
    other.reset();
    assert(!other.nn_save_initialized);
    assert(core.nn_save_initialized);

    // Exercise the normal Emulator registration and RPX import-binding path:
    // the synthetic guest calls the exact nn_save::SAVEInit import.
    RpxImportOptions import_options;
    import_options.library = "nn_save";
    RpxRelocationSymbol import_symbol{
        "SAVEInit", import_options.section_address + import_options.slot_offset,
        1U, 2U, 8U, 0};
    RpxRelocationOptions relocation;
    relocation.type = 10U;
    relocation.offset = 4U;
    relocation.branch_instruction = 0x48000001U;
    const auto import_rpx = make_minimal_compressed_rpx(
        0x02000000U, 16U, import_symbol, relocation, import_options);
    affogato::Emulator emulator(0x40000U);
    assert(!emulator.core().nn_save_initialized);
    static_cast<void>(emulator.load_rpx(import_rpx));
    const auto emulator_result = emulator.run(3U);
    assert(emulator_result.execution.reason == StopReason::instruction_limit);
    assert(emulator_result.execution.steps == 3U);
    assert(emulator.core().nn_save_initialized);
    assert(emulator.core().state.gpr[3] == 0U);
    assert(emulator.core().state.cia == 0x02000008U);

    affogato::Emulator independent_emulator(0x40000U);
    assert(!independent_emulator.core().nn_save_initialized);
    static_cast<void>(independent_emulator.load_rpx(import_rpx));
    assert(!independent_emulator.core().nn_save_initialized);
    const auto independent_result = independent_emulator.run(3U);
    assert(independent_result.execution.reason == StopReason::instruction_limit);
    assert(independent_emulator.core().nn_save_initialized);
    assert(emulator.core().nn_save_initialized);
}

void fs_add_client_hle_tests()
{
    constexpr std::uint32_t client_size = 0x1700U;
    constexpr std::uint32_t fatal_status = 0xFFFFFC00U;
    constexpr std::uint32_t fast_mutex_tag = 0x664D7458U;
    constexpr std::uint32_t alarm_tag = 0x614C724DU;
    constexpr std::uint32_t return_address = 0x02742248U;

    const auto aligned_body = [](std::uint32_t client) {
        return static_cast<std::uint32_t>(
            (static_cast<std::uint64_t>(client) + 0x3FU) & ~std::uint64_t{0x3F});
    };
    const auto write_expected_u32 = [](std::vector<std::uint8_t>& bytes,
                                       std::uint32_t offset,
                                       std::uint32_t value) {
        bytes[offset] = static_cast<std::uint8_t>(value >> 24U);
        bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 16U);
        bytes[offset + 2U] = static_cast<std::uint8_t>(value >> 8U);
        bytes[offset + 3U] = static_cast<std::uint8_t>(value);
    };

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    const std::uint32_t import = core.hle.bind_import("coreinit", "FSAddClient");
    core.fs_initialized = true;
    constexpr std::uint32_t client_a = 0x2003U;
    constexpr std::uint32_t body_a = 0x2040U;
    core.memory.write8(client_a - 1U, 0x91U);
    core.memory.write8(client_a + client_size, 0x92U);
    core.memory.fill_bytes(client_a, client_size, 0xA5U);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    core.state.cia = import;
    core.state.gpr[3] = client_a;
    core.state.gpr[4] = 0x12345678U;
    const auto gprs_before = core.state.gpr;
    const auto fpr_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const std::uint32_t cr_before = core.state.cr;
    const std::uint32_t xer_before = core.state.xer;
    const std::uint32_t ctr_before = core.state.ctr;
    const std::uint32_t fpscr_before = core.state.fpscr;
    const std::uint32_t lr_before = core.state.lr;
    const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0U);
    assert(core.state.cia == return_address);
    assert(core.state.lr == lr_before);
    assert(core.state.gpr[4] == gprs_before[4]);
    for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        if (reg != 3U)
        {
            assert(core.state.gpr[reg] == gprs_before[reg]);
        }
    }
    assert(core.state.fpr == fpr_before);
    assert(core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before);
    assert(core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before);
    assert(core.state.fpscr == fpscr_before);
    assert(core.guest_heap_cursor == heap_cursor_before);
    assert(core.fs_clients.size() == 1U);
    assert(core.fs_clients[0].client_address == client_a);
    assert(core.fs_clients[0].body_address == body_a);
    assert(core.fs_clients[0].synthetic_handle == 1U);
    assert(core.next_fs_client_handle == 2U);
    assert(core.memory.read8(client_a - 1U) == 0x91U);
    assert(core.memory.read8(client_a + client_size) == 0x92U);

    // Compare the entire FSClient against a zero-filled expected image with
    // only the known Cafe fields populated.
    std::vector<std::uint8_t> expected_client(client_size, 0U);
    const auto put = [&](std::uint32_t body_offset, std::uint32_t value) {
        write_expected_u32(expected_client, body_a - client_a + body_offset, value);
    };
    put(0x1444U, 1U);
    put(0x1448U, 1U);
    put(0x144CU, 1U);
    put(0x1488U, fast_mutex_tag);
    put(0x14BCU, 1U);
    put(0x1560U, fast_mutex_tag);
    put(0x1590U, alarm_tag);
    put(0x1614U, body_a);
    put(0x1618U, body_a);
    put(0x161CU, client_a);
    std::vector<std::uint8_t> actual_client(client_size);
    core.memory.read_bytes(client_a, actual_client);
    assert(actual_client == expected_client);

    // Second and third clients extend a circular doubly-linked guest list and
    // receive distinct stable per-core service handles.
    constexpr std::array<std::uint32_t, 3> clients{
        client_a, 0x4007U, 0x600BU};
    for (std::size_t index = 1; index < clients.size(); ++index)
    {
        core.state.cia = import;
        core.state.lr = return_address + static_cast<std::uint32_t>(index * 4U);
        core.state.gpr[3] = clients[index];
        core.state.gpr[4] = 0U;
        assert(core.step() == StepResult::executed);
        assert(core.state.gpr[3] == 0U);
    }
    assert(core.fs_clients.size() == clients.size());
    for (std::size_t index = 0; index < clients.size(); ++index)
    {
        const std::uint32_t body = aligned_body(clients[index]);
        const std::uint32_t next_body = aligned_body(
            clients[(index + 1U) % clients.size()]);
        const std::uint32_t previous_body = aligned_body(
            clients[(index + clients.size() - 1U) % clients.size()]);
        assert(core.fs_clients[index].client_address == clients[index]);
        assert(core.fs_clients[index].body_address == body);
        assert(core.fs_clients[index].synthetic_handle == index + 1U);
        assert(core.memory.read32_be(body + 0x1444U) == index + 1U);
        assert(core.memory.read32_be(body + 0x1614U) == next_body);
        assert(core.memory.read32_be(body + 0x1618U) == previous_body);
        assert(core.memory.read32_be(next_body + 0x1618U) == body);
        assert(core.memory.read32_be(previous_body + 0x1614U) == body);
        assert(core.memory.read32_be(body + 0x161CU) == clients[index]);
    }

    // A duplicate pointer or body must fail without zeroing or allocating.
    const auto client_a_snapshot = [&]() {
        std::vector<std::uint8_t> result(client_size);
        core.memory.read_bytes(client_a, result);
        return result;
    }();
    const std::uint32_t next_handle_before_duplicate = core.next_fs_client_handle;
    core.state.cia = import;
    core.state.lr = return_address;
    core.state.gpr[3] = client_a;
    core.state.gpr[4] = 0U;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == fatal_status);
    assert(core.fs_clients.size() == 3U);
    assert(core.next_fs_client_handle == next_handle_before_duplicate);
    std::vector<std::uint8_t> client_a_after_duplicate(client_size);
    core.memory.read_bytes(client_a, client_a_after_duplicate);
    assert(client_a_after_duplicate == client_a_snapshot);

    // Same aligned body (and overlapping storage) is also a duplicate.
    core.state.cia = import;
    core.state.gpr[3] = 0x203FU;
    core.state.gpr[4] = 0U;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == fatal_status);
    assert(core.fs_clients.size() == 3U);
    assert(core.next_fs_client_handle == next_handle_before_duplicate);
    std::vector<std::uint8_t> client_a_after_alias(client_size);
    core.memory.read_bytes(client_a, client_a_after_alias);
    assert(client_a_after_alias == client_a_snapshot);

    // A fresh core rejects FSAddClient before initialization without touching
    // the valid client storage or consuming a handle.
    EspressoCore uninitialized(0x8000U);
    register_coreinit_hle(uninitialized.hle);
    const std::uint32_t uninitialized_import =
        uninitialized.hle.bind_import("coreinit", "FSAddClient");
    constexpr std::uint32_t uninitialized_client = 0x1000U;
    uninitialized.memory.fill_bytes(uninitialized_client, client_size, 0xA5U);
    std::vector<std::uint8_t> uninitialized_before(client_size);
    uninitialized.memory.read_bytes(uninitialized_client, uninitialized_before);
    uninitialized.state.cia = uninitialized_import;
    uninitialized.state.lr = return_address;
    uninitialized.state.gpr[3] = uninitialized_client;
    uninitialized.state.gpr[4] = 0U;
    assert(uninitialized.step() == StepResult::executed);
    assert(uninitialized.state.gpr[3] == fatal_status);
    assert(uninitialized.fs_clients.empty());
    assert(uninitialized.next_fs_client_handle == 1U);
    std::vector<std::uint8_t> uninitialized_after(client_size);
    uninitialized.memory.read_bytes(uninitialized_client, uninitialized_after);
    assert(uninitialized_after == uninitialized_before);

    // Null client and exhausted synthetic handle namespace fail without
    // touching guest memory or registration state.
    core.state.cia = import;
    core.state.gpr[3] = 0U;
    core.state.gpr[4] = 0U;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == fatal_status);
    assert(core.fs_clients.size() == 3U);
    const std::uint32_t saved_next_handle = core.next_fs_client_handle;
    core.next_fs_client_handle = 0x80000000U;
    core.state.cia = import;
    core.state.gpr[3] = 0x8003U;
    core.state.gpr[4] = 0U;
    core.memory.fill_bytes(0x8003U, client_size, 0x5AU);
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == fatal_status);
    assert(core.fs_clients.size() == 3U);
    assert(core.next_fs_client_handle == 0x80000000U);
    assert(core.memory.read8(0x8003U) == 0x5AU);
    core.next_fs_client_handle = saved_next_handle;

    // Inaccessible ranges stop cleanly before any partial initialization.
    EspressoCore unmapped(0x1000U);
    register_coreinit_hle(unmapped.hle);
    const std::uint32_t unmapped_import =
        unmapped.hle.bind_import("coreinit", "FSAddClient");
    unmapped.fs_initialized = true;
    unmapped.memory.fill_bytes(0x800U, 0x800U, 0xA5U);
    unmapped.state.cia = unmapped_import;
    unmapped.state.lr = return_address;
    unmapped.state.gpr[3] = 0x800U;
    unmapped.state.gpr[4] = 0U;
    const RunResult unmapped_result = unmapped.run(1U);
    assert(unmapped_result.reason == StopReason::memory_fault);
    assert(unmapped.fs_clients.empty());
    assert(unmapped.next_fs_client_handle == 1U);
    for (std::uint32_t address = 0x800U; address < 0x1000U; ++address)
    {
        assert(unmapped.memory.read8(address) == 0xA5U);
    }

    EspressoCore wrapping(0x1000U);
    register_coreinit_hle(wrapping.hle);
    const std::uint32_t wrapping_import =
        wrapping.hle.bind_import("coreinit", "FSAddClient");
    wrapping.fs_initialized = true;
    wrapping.state.cia = wrapping_import;
    wrapping.state.lr = return_address;
    wrapping.state.gpr[3] = 0xFFFFFFF0U;
    wrapping.state.gpr[4] = 0U;
    const RunResult wrapping_result = wrapping.run(1U);
    assert(wrapping_result.reason == StopReason::hle_error);
    assert(wrapping.fs_clients.empty());
    assert(wrapping.next_fs_client_handle == 1U);

    // Independent cores have independent registries and handle namespaces;
    // reset clears only its own registrations and restarts handle numbering.
    EspressoCore other(0x8000U);
    register_coreinit_hle(other.hle);
    const std::uint32_t other_import = other.hle.bind_import("coreinit", "FSAddClient");
    other.fs_initialized = true;
    other.state.cia = other_import;
    other.state.lr = return_address;
    other.state.gpr[3] = 0x2003U;
    other.state.gpr[4] = 0U;
    assert(other.step() == StepResult::executed);
    assert(other.state.gpr[3] == 0U);
    assert(other.fs_clients.size() == 1U);
    assert(other.fs_clients[0].synthetic_handle == 1U);
    assert(core.fs_clients.size() == 3U);
    core.reset();
    assert(!core.fs_initialized);
    assert(core.fs_clients.empty());
    assert(core.next_fs_client_handle == 1U);
    assert(other.fs_clients.size() == 1U);
    assert(other.fs_clients[0].synthetic_handle == 1U);

    // The unsigned raw error value is the signed FS_STATUS_FATAL_ERROR ABI.
    assert(static_cast<std::uint32_t>(static_cast<std::int32_t>(-0x400)) ==
           fatal_status);
}

void fs_set_state_change_notification_hle_tests()
{
    constexpr std::uint32_t client_size = 0x1700U;
    constexpr std::uint32_t client_a = 0x2003U;
    constexpr std::uint32_t client_b = 0x4007U;
    constexpr std::uint32_t body_a = 0x2040U;
    constexpr std::uint32_t body_b = 0x4040U;
    constexpr std::uint32_t fsm_offset = 0x1448U;
    constexpr std::uint32_t enabled_offset = 0x10U;
    constexpr std::uint32_t info_offset = 0x14U;
    constexpr std::uint32_t info_size = 0x24U;
    constexpr std::uint32_t message_data_offset = 0x0CU;
    constexpr std::uint32_t message_type_offset = 0x18U;
    constexpr std::uint32_t client_pointer_offset = 0x1CU;
    constexpr std::uint32_t state_offset = 0x20U;
    constexpr std::uint32_t params = 0x6000U;
    constexpr std::uint32_t queue = 0x8000U;
    constexpr std::uint32_t queue_messages = 0x8100U;
    constexpr std::uint32_t return_address = 0x02742268U;
    constexpr std::uint32_t notification_message_type = 11U;

    const auto invoke = [&](EspressoCore& core,
                            std::uint32_t import,
                            std::uint32_t client,
                            std::uint32_t info) {
        core.state.cia = import;
        core.state.lr = return_address;
        core.state.gpr[3] = client;
        core.state.gpr[4] = info;
        return core.run(1U);
    };
    const auto snapshot = [](const EspressoCore& core,
                             std::uint32_t address,
                             std::uint32_t size) {
        std::vector<std::uint8_t> bytes(size);
        core.memory.read_bytes(address, bytes);
        return bytes;
    };
    const auto put_be32 = [](std::vector<std::uint8_t>& bytes,
                             std::size_t offset,
                             std::uint32_t value) {
        bytes[offset] = static_cast<std::uint8_t>(value >> 24U);
        bytes[offset + 1U] = static_cast<std::uint8_t>(value >> 16U);
        bytes[offset + 2U] = static_cast<std::uint8_t>(value >> 8U);
        bytes[offset + 3U] = static_cast<std::uint8_t>(value);
    };
    const auto write_params = [](EspressoCore& core,
                                 std::uint32_t address,
                                 std::uint32_t callback,
                                 std::uint32_t param,
                                 std::uint32_t queue_pointer) {
        core.memory.write32_be(address, callback);
        core.memory.write32_be(address + 4U, param);
        core.memory.write32_be(address + 8U, queue_pointer);
    };

    EspressoCore core(0x20000U);
    register_coreinit_hle(core.hle);
    const std::uint32_t fs_init = core.hle.bind_import("coreinit", "FSInit");
    const std::uint32_t fs_add = core.hle.bind_import("coreinit", "FSAddClient");
    const std::uint32_t initialize_queue =
        core.hle.bind_import("coreinit", "OSInitMessageQueue");
    const std::uint32_t set_notification = core.hle.bind_import(
        "coreinit", "FSSetStateChangeNotification");

    core.state.cia = fs_init;
    core.state.lr = return_address;
    assert(core.step() == StepResult::executed);
    assert(core.fs_initialized);
    for (const std::uint32_t client : {client_a, client_b})
    {
        core.state.cia = fs_add;
        core.state.lr = return_address;
        core.state.gpr[3] = client;
        core.state.gpr[4] = 0U;
        assert(core.step() == StepResult::executed);
        assert(core.state.gpr[3] == 0U);
    }
    assert(core.fs_clients.size() == 2U);
    assert(core.fs_clients[0].body_address == body_a);
    assert(core.fs_clients[1].body_address == body_b);

    core.state.cia = initialize_queue;
    core.state.lr = return_address;
    core.state.gpr[3] = queue;
    core.state.gpr[4] = queue_messages;
    core.state.gpr[5] = 8U;
    assert(core.step() == StepResult::executed);
    const auto queue_before = snapshot(core, queue, 0x3CU);

    const std::uint32_t body_fsm_a = body_a + fsm_offset;
    const std::uint32_t state_info_a = body_fsm_a + info_offset;
    const std::uint32_t body_fsm_b = body_b + fsm_offset;
    const std::uint32_t state_info_b = body_fsm_b + info_offset;
    core.memory.write32_be(state_info_a + 0x10U, 0xA1A2A3A4U);
    core.memory.write32_be(state_info_a + 0x14U, 0xB1B2B3B4U);
    core.memory.write32_be(state_info_a + state_offset, 0xC1C2C3C4U);
    core.memory.write32_be(state_info_b + 0x10U, 0xD1D2D3D4U);
    core.memory.write32_be(state_info_b + 0x14U, 0xE1E2E3E4U);
    core.memory.write32_be(state_info_b + state_offset, 0xF1F2F3F4U);

    // The real Wind Waker argument block registers a callback and leaves both
    // the user parameter and queue null.
    write_params(core, params, 0x02742028U, 0U, 0U);
    const auto input_before = snapshot(core, params, 12U);
    auto expected_a = snapshot(core, client_a, client_size);
    put_be32(expected_a, body_a - client_a + fsm_offset + enabled_offset, 1U);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset,
             0x02742028U);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset + 4U, 0U);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset + 8U, 0U);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset +
                              message_data_offset,
             state_info_a);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset +
                              message_type_offset,
             notification_message_type);
    put_be32(expected_a, body_a - client_a + fsm_offset + info_offset +
                              client_pointer_offset,
             client_a);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.gpr[3] = client_a;
    core.state.gpr[4] = params;
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const auto registrations_before = core.fs_clients;
    const std::uint32_t next_handle_before = core.next_fs_client_handle;
    const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
    const std::uint32_t cr_before = core.state.cr;
    const std::uint32_t xer_before = core.state.xer;
    const std::uint32_t ctr_before = core.state.ctr;
    const std::uint32_t fpscr_before = core.state.fpscr;
    const std::uint32_t lr_before = core.state.lr;

    const RunResult registered =
        invoke(core, set_notification, client_a, params);
    assert(registered.reason == StopReason::instruction_limit);
    assert(registered.steps == 1U);
    assert(core.state.cia == return_address);
    assert(core.state.lr == lr_before);
    assert(core.state.gpr == gprs_before);
    assert(core.state.fpr == fprs_before);
    assert(core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before);
    assert(core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before);
    assert(core.state.fpscr == fpscr_before);
    assert(core.guest_heap_cursor == heap_cursor_before);
    assert(core.next_fs_client_handle == next_handle_before);
    assert(core.fs_clients.size() == registrations_before.size());
    for (std::size_t index = 0; index < registrations_before.size(); ++index)
    {
        assert(core.fs_clients[index].client_address ==
               registrations_before[index].client_address);
        assert(core.fs_clients[index].body_address ==
               registrations_before[index].body_address);
        assert(core.fs_clients[index].synthetic_handle ==
               registrations_before[index].synthetic_handle);
    }
    assert(snapshot(core, client_a, client_size) == expected_a);
    assert(snapshot(core, params, 12U) == input_before);
    assert(snapshot(core, queue, 0x3CU) == queue_before);
    assert(core.memory.read32_be(body_fsm_a) == 1U);
    assert(core.memory.read32_be(body_fsm_a + 4U) == 1U);
    assert(core.memory.read32_be(body_fsm_a + 0x0CU) == 0U);
    assert(core.memory.read32_be(state_info_a + state_offset) == 0xC1C2C3C4U);

    // Replacement refreshes the registered metadata but preserves the opaque
    // embedded-message words and state field. The supplied real queue pointer
    // is stored as-is and is not modified or used for delivery.
    constexpr std::uint32_t replacement_params = params + 0x20U;
    write_params(core, replacement_params, 0x02001000U, 0x11223344U, queue);
    const auto replacement_input = snapshot(core, replacement_params, 12U);
    auto expected_replacement = snapshot(core, client_a, client_size);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset, 0x02001000U);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset + 4U, 0x11223344U);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset + 8U, queue);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset + message_data_offset,
             state_info_a);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset + message_type_offset,
             notification_message_type);
    put_be32(expected_replacement,
             body_a - client_a + fsm_offset + info_offset + client_pointer_offset,
             client_a);
    const RunResult replaced =
        invoke(core, set_notification, client_a, replacement_params);
    assert(replaced.reason == StopReason::instruction_limit);
    assert(core.state.cia == return_address);
    assert(snapshot(core, client_a, client_size) == expected_replacement);
    assert(snapshot(core, replacement_params, 12U) == replacement_input);
    assert(snapshot(core, queue, 0x3CU) == queue_before);
    assert(core.memory.read32_be(state_info_a + state_offset) == 0xC1C2C3C4U);

    // A null parameter pointer only clears the enable flag. All 0x24 bytes of
    // the previous state-change info remain byte-for-byte unchanged.
    const auto info_before_disable = snapshot(core, state_info_a, info_size);
    auto expected_disabled = snapshot(core, client_a, client_size);
    put_be32(expected_disabled, body_a - client_a + fsm_offset + enabled_offset, 0U);
    const RunResult disabled = invoke(core, set_notification, client_a, 0U);
    assert(disabled.reason == StopReason::instruction_limit);
    assert(snapshot(core, client_a, client_size) == expected_disabled);
    assert(snapshot(core, state_info_a, info_size) == info_before_disable);

    // Re-enable with an all-zero parameter block: a non-null info pointer
    // enables notifications even when callback, param, and queue are null.
    constexpr std::uint32_t zero_params = params + 0x40U;
    write_params(core, zero_params, 0U, 0U, 0U);
    const auto zero_input = snapshot(core, zero_params, 12U);
    auto expected_reenabled = snapshot(core, client_a, client_size);
    put_be32(expected_reenabled, body_a - client_a + fsm_offset + enabled_offset, 1U);
    put_be32(expected_reenabled, body_a - client_a + fsm_offset + info_offset, 0U);
    put_be32(expected_reenabled, body_a - client_a + fsm_offset + info_offset + 4U, 0U);
    put_be32(expected_reenabled, body_a - client_a + fsm_offset + info_offset + 8U, 0U);
    put_be32(expected_reenabled,
             body_a - client_a + fsm_offset + info_offset + message_data_offset,
             state_info_a);
    put_be32(expected_reenabled,
             body_a - client_a + fsm_offset + info_offset + message_type_offset,
             notification_message_type);
    put_be32(expected_reenabled,
             body_a - client_a + fsm_offset + info_offset + client_pointer_offset,
             client_a);
    const RunResult reenabled = invoke(core, set_notification, client_a, zero_params);
    assert(reenabled.reason == StopReason::instruction_limit);
    assert(snapshot(core, client_a, client_size) == expected_reenabled);
    assert(snapshot(core, zero_params, 12U) == zero_input);
    assert(core.memory.read32_be(state_info_a + state_offset) == 0xC1C2C3C4U);

    // The second registered FSClient has independent state-change metadata.
    constexpr std::uint32_t params_b = params + 0x60U;
    write_params(core, params_b, 0x03002000U, 0x55667788U, 0x88776655U);
    const auto client_a_before_b = snapshot(core, client_a, client_size);
    auto expected_b = snapshot(core, client_b, client_size);
    put_be32(expected_b, body_b - client_b + fsm_offset + enabled_offset, 1U);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset, 0x03002000U);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset + 4U,
             0x55667788U);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset + 8U,
             0x88776655U);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset +
                             message_data_offset,
             state_info_b);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset +
                             message_type_offset,
             notification_message_type);
    put_be32(expected_b, body_b - client_b + fsm_offset + info_offset +
                             client_pointer_offset,
             client_b);
    const RunResult registered_b = invoke(core, set_notification, client_b, params_b);
    assert(registered_b.reason == StopReason::instruction_limit);
    assert(snapshot(core, client_a, client_size) == client_a_before_b);
    assert(snapshot(core, client_b, client_size) == expected_b);

    // Invalid parameter memory faults before any client-side mutation.
    const auto before_unmapped = snapshot(core, client_a, client_size);
    core.state.cia = set_notification;
    core.state.lr = return_address;
    core.state.gpr[3] = client_a;
    core.state.gpr[4] = 0x1FFF8U;
    const RunResult unmapped_params = core.run(1U);
    assert(unmapped_params.reason == StopReason::memory_fault);
    assert(snapshot(core, client_a, client_size) == before_unmapped);

    // Address-space wrap in the full 12-byte parameter read has the same
    // transactional behavior.
    core.state.cia = set_notification;
    core.state.lr = return_address;
    core.state.gpr[3] = client_a;
    core.state.gpr[4] = 0xFFFFFFFCU;
    const RunResult wrapping_params = core.run(1U);
    assert(wrapping_params.reason == StopReason::memory_fault);
    assert(snapshot(core, client_a, client_size) == before_unmapped);

    // Null and unregistered client pointers are HLE contract errors and leave
    // both registered clients untouched.
    for (const std::uint32_t invalid_client : {0U, 0x7003U})
    {
        core.state.cia = set_notification;
        core.state.lr = return_address;
        core.state.gpr[3] = invalid_client;
        core.state.gpr[4] = params;
        const RunResult invalid = core.run(1U);
        assert(invalid.reason == StopReason::hle_error);
        assert(snapshot(core, client_a, client_size) == before_unmapped);
        assert(snapshot(core, client_b, client_size) == expected_b);
    }

    EspressoCore uninitialized(0x8000U);
    register_coreinit_hle(uninitialized.hle);
    const std::uint32_t uninitialized_import = uninitialized.hle.bind_import(
        "coreinit", "FSSetStateChangeNotification");
    uninitialized.state.cia = uninitialized_import;
    uninitialized.state.lr = return_address;
    uninitialized.state.gpr[3] = client_a;
    uninitialized.state.gpr[4] = 0U;
    assert(uninitialized.run(1U).reason == StopReason::hle_error);
    assert(uninitialized.memory.read8(0x100U) == 0U);
}

void os_get_system_time_hle_tests()
{
    using Clock = std::chrono::system_clock;
    constexpr std::uint32_t timer_clock_speed = 62156250U;
    constexpr std::int64_t unix_epoch_offset = 946684800;
    constexpr std::uint32_t return_address = 0x027616D4U;
    constexpr std::uint32_t stack_sentinel_address = 0x100U;

    const auto cafe_epoch_ticks_at = [](Clock::time_point time) {
        using namespace std::chrono;
        const auto elapsed = time.time_since_epoch();
        const auto unix_seconds = duration_cast<seconds>(elapsed).count();
        if (unix_seconds < unix_epoch_offset)
        {
            return std::uint64_t{0};
        }
        const auto fractional_nanoseconds =
            duration_cast<nanoseconds>(elapsed - seconds(unix_seconds)).count();
        return static_cast<std::uint64_t>(unix_seconds - unix_epoch_offset) *
                timer_clock_speed +
            static_cast<std::uint64_t>(fractional_nanoseconds) * timer_clock_speed /
                1000000000ULL;
    };
    assert(cafe_epoch_ticks_at(Clock::time_point{
               std::chrono::seconds(unix_epoch_offset + 1)}) == timer_clock_speed);
    assert(cafe_epoch_ticks_at(Clock::time_point{
               std::chrono::milliseconds(unix_epoch_offset * 1000 + 500)}) ==
           timer_clock_speed / 2U);
    assert(cafe_epoch_ticks_at(Clock::time_point{
               std::chrono::milliseconds(unix_epoch_offset * 1000 + 1)}) == 62156U);

    EspressoCore core(0x400U);
    register_coreinit_hle(core.hle);
    const std::uint32_t import = core.hle.bind_import("coreinit", "OSGetSystemTime");
    core.configure_guest_heap(0x20U, 0x300U);
    core.current_thread_address = 0x12345678U;
    core.base_heap_handles.fill(0xAABBCCDDU);
    core.mem2_heap_region_begin = 0x234U;
    core.mem2_heap_region_end = 0x345U;
    core.memory.fill_bytes(stack_sentinel_address, 0x80U, 0xB6U);
    std::array<std::uint8_t, 0x80U> stack_before{};
    core.memory.read_bytes(stack_sentinel_address, stack_before);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;

    const auto invoke = [&]() {
        core.state.cia = import;
        core.state.lr = return_address;
        const auto gprs_before = core.state.gpr;
        const auto fprs_before = core.state.fpr;
        const auto ps1_before = core.state.fpr_ps1;
        const std::uint32_t cr_before = core.state.cr;
        const std::uint32_t xer_before = core.state.xer;
        const std::uint32_t ctr_before = core.state.ctr;
        const std::uint32_t fpscr_before = core.state.fpscr;
        const std::uint32_t lr_before = core.state.lr;
        const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
        const std::uint32_t heap_limit_before = core.guest_heap_limit;
        const auto base_handles_before = core.base_heap_handles;
        const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
        const std::uint32_t mem2_end_before = core.mem2_heap_region_end;
        const std::uint32_t thread_before = core.current_thread_address;
        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(result.steps == 1U);
        assert(core.state.cia == return_address);
        assert(core.state.lr == lr_before);
        for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            if (reg != 3U && reg != 4U)
            {
                assert(core.state.gpr[reg] == gprs_before[reg]);
            }
        }
        assert(core.state.fpr == fprs_before);
        assert(core.state.fpr_ps1 == ps1_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.guest_heap_cursor == heap_cursor_before);
        assert(core.guest_heap_limit == heap_limit_before);
        assert(core.base_heap_handles == base_handles_before);
        assert(core.mem2_heap_region_begin == mem2_begin_before);
        assert(core.mem2_heap_region_end == mem2_end_before);
        assert(core.current_thread_address == thread_before);
        std::array<std::uint8_t, 0x80U> stack_after{};
        core.memory.read_bytes(stack_sentinel_address, stack_after);
        assert(stack_after == stack_before);
        return (static_cast<std::uint64_t>(core.state.gpr[3]) << 32U) |
               core.state.gpr[4];
    };

    // The time API lazily materializes the same persistent system-info object;
    // r3's stale caller value is not interpreted as an argument.
    core.state.gpr[3] = 0x100FEE58U;
    core.state.gpr[4] = 0xDEADBEEFU;
    const auto host_before_first = Clock::now();
    const std::uint64_t first = invoke();
    const auto host_after_first = Clock::now();
    const auto system_info_address = core.hle.data_address(
        "coreinit", "__affogato_internal_OSSystemInfo");
    assert(system_info_address);
    assert(core.state.gpr[3] != 0x100FEE58U);
    const std::uint64_t base_time = core.memory.read64_be(*system_info_address + 0x08U);
    assert(base_time != 0U);
    assert(base_time + first >= cafe_epoch_ticks_at(host_before_first));
    assert(base_time + first <= cafe_epoch_ticks_at(host_after_first) + 1U);

    std::array<std::uint8_t, 0x20U> info_before{};
    core.memory.read_bytes(*system_info_address, info_before);
    const std::uint64_t second = invoke();
    assert(second >= first);
    std::array<std::uint8_t, 0x20U> info_after{};
    core.memory.read_bytes(*system_info_address, info_after);
    assert(info_after == info_before);

    std::this_thread::sleep_for(std::chrono::milliseconds(8));
    const std::uint64_t third = invoke();
    assert(third > second);
    core.memory.read_bytes(*system_info_address, info_after);
    assert(info_after == info_before);

    // A zero baseTime makes elapsed ticks span more than 32 bits, validating
    // the PPC32 r3-high/r4-low return convention against host-clock bounds.
    core.memory.write64_be(*system_info_address + 0x08U, 0U);
    core.memory.read_bytes(*system_info_address, info_before);
    const auto host_before_wide = Clock::now();
    const std::uint64_t wide = invoke();
    const auto host_after_wide = Clock::now();
    assert(wide > UINT32_MAX);
    assert(wide >= cafe_epoch_ticks_at(host_before_wide));
    assert(wide <= cafe_epoch_ticks_at(host_after_wide) + 1U);
    assert(core.state.gpr[3] == static_cast<std::uint32_t>(wide >> 32U));
    assert(core.state.gpr[4] == static_cast<std::uint32_t>(wide));
    core.memory.read_bytes(*system_info_address, info_after);
    assert(info_after == info_before);

    // Each core gets its own guest baseTime and data object state.
    EspressoCore other(0x200U);
    register_coreinit_hle(other.hle);
    const std::uint32_t other_import =
        other.hle.bind_import("coreinit", "OSGetSystemTime");
    other.state.cia = other_import;
    other.state.lr = 0x88U;
    assert(other.step() == StepResult::executed);
    const auto other_info_address = other.hle.data_address(
        "coreinit", "__affogato_internal_OSSystemInfo");
    assert(other_info_address == system_info_address);
    assert(other.memory.read64_be(*other_info_address + 0x08U) != 0U);
    other.memory.write64_be(*other_info_address + 0x08U, 0U);
    assert(core.memory.read64_be(*system_info_address + 0x08U) == 0U);
}

void os_set_exception_callback_hle_tests()
{
    struct CallbackType
    {
        std::uint32_t exception_type;
        std::uint32_t callback_array_offset;
    };
    constexpr std::array<CallbackType, 5> supported_types{{
        {2U, 0x630U}, // DSI
        {3U, 0x63CU}, // ISI
        {6U, 0x648U}, // Program
        {11U, 0x654U}, // Performance monitor
        {5U, 0x680U}, // Alignment
    }};
    constexpr std::uint32_t thread_size = 0x6A0U;
    constexpr std::uint32_t return_address = 0x80U;

    EspressoCore core(0x5000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x1000U, 0x2000U);
    const std::uint32_t thread = initialize_default_guest_thread(
        core, 0x4800U, 0x4000U);
    const std::uint32_t import = core.hle.bind_import(
        "coreinit", "OSSetExceptionCallback");
    assert(core.current_thread_address == thread);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA0000000U + reg * 0x10101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.memory.write32_be(0x80U, 0xDEADBEEFU);

    // The callback fields are three-element arrays. This HLE must select only
    // the CPU1 member, matching the default thread's current affinity.
    core.memory.write32_be(thread + 0x630U, 0xAAAAAAAAU);
    core.memory.write32_be(thread + 0x638U, 0xCCCCCCCCU);

    const auto invoke = [&](std::uint32_t exception_type,
                            std::uint32_t callback,
                            std::uint32_t slot,
                            std::uint32_t expected_previous,
                            std::uint32_t return_cia) {
        core.state.gpr[3] = exception_type;
        core.state.gpr[4] = callback;
        core.state.lr = return_cia;
        core.state.cia = import;
        const auto gpr_before = core.state.gpr;
        const auto fpr_before = core.state.fpr;
        const auto fpr_ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto lr_before = core.state.lr;
        const auto ctr_before = core.state.ctr;
        const auto fpscr_before = core.state.fpscr;
        std::array<std::uint8_t, thread_size> thread_before{};
        for (std::uint32_t byte = 0; byte < thread_size; ++byte)
        {
            thread_before[byte] = core.memory.read8(thread + byte);
        }
        const std::uint32_t unrelated_memory_before = core.memory.read32_be(0x80U);

        assert(core.step() == StepResult::executed);
        assert(core.state.cia == return_cia);
        assert(core.state.gpr[3] == expected_previous);
        for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            if (reg != 3U)
            {
                assert(core.state.gpr[reg] == gpr_before[reg]);
            }
        }
        assert(core.state.fpr == fpr_before);
        assert(core.state.fpr_ps1 == fpr_ps1_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.lr == lr_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.memory.read32_be(slot) == callback);
        assert(core.memory.read32_be(0x80U) == unrelated_memory_before);
        for (std::uint32_t byte = 0; byte < thread_size; ++byte)
        {
            if (byte < slot - thread || byte >= slot - thread + 4U)
            {
                assert(core.memory.read8(thread + byte) == thread_before[byte]);
            }
        }
    };

    std::uint32_t return_cia = return_address;
    for (const CallbackType& type : supported_types)
    {
        const std::uint32_t slot = thread + type.callback_array_offset + 4U;
        assert(core.memory.read32_be(slot) == 0U);

        const std::uint32_t first_callback = type.exception_type == 2U
            ? 0x020347D0U
            : 0x02001000U + type.exception_type * 4U;
        invoke(type.exception_type, first_callback, slot, 0U, return_cia);
        assert(core.state.gpr[4] == first_callback);
        assert(core.memory.read32_be(slot) == first_callback);
        if (type.exception_type == 2U)
        {
            assert(slot == thread + 0x634U);
            assert(core.memory.read32_be(thread + 0x630U) == 0xAAAAAAAAU);
            assert(core.memory.read32_be(thread + 0x638U) == 0xCCCCCCCCU);
        }

        const std::uint32_t replacement = type.exception_type == 2U
            ? 0x02034820U
            : 0x02002000U + type.exception_type * 4U;
        return_cia += 4U;
        invoke(type.exception_type, replacement, slot, first_callback, return_cia);
        assert(core.memory.read32_be(slot) == replacement);

        // The HLE reads the guest-visible slot each time, with no hidden copy.
        core.memory.write32_be(slot, 0x12345678U);
        return_cia += 4U;
        invoke(type.exception_type, 0U, slot, 0x12345678U, return_cia);
        assert(core.memory.read32_be(slot) == 0U); // NULL unregisters the callback.
        return_cia += 4U;
    }

    // Unsupported exception kinds return NULL and leave every callback array
    // member untouched.
    std::array<std::uint32_t, 15> callback_slots{};
    std::size_t slot_index = 0;
    for (const CallbackType& type : supported_types)
    {
        for (std::uint32_t core_index = 0; core_index < 3U; ++core_index)
        {
            const std::uint32_t slot = thread + type.callback_array_offset +
                core_index * 4U;
            const std::uint32_t sentinel = 0xB0000000U +
                static_cast<std::uint32_t>(slot_index) * 0x101U;
            callback_slots[slot_index++] = slot;
            core.memory.write32_be(slot, sentinel);
        }
    }
    constexpr std::array<std::uint32_t, 10> unsupported_types{
        0U, 1U, 4U, 7U, 8U, 9U, 10U, 12U, 13U, 14U};
    for (const std::uint32_t exception_type : unsupported_types)
    {
        core.state.gpr[3] = exception_type;
        core.state.gpr[4] = 0x12345678U;
        core.state.lr = return_cia;
        core.state.cia = import;
        assert(core.step() == StepResult::executed);
        assert(core.state.gpr[3] == 0U);
        assert(core.state.gpr[4] == 0x12345678U);
        assert(core.state.cia == return_cia);
        for (std::size_t index = 0; index < callback_slots.size(); ++index)
        {
            const std::uint32_t expected = 0xB0000000U +
                static_cast<std::uint32_t>(index) * 0x101U;
            assert(core.memory.read32_be(callback_slots[index]) == expected);
        }
        return_cia += 4U;
    }

    // A supported call without a current guest thread is an explicit HLE error.
    EspressoCore no_thread_core(0x100U);
    register_coreinit_hle(no_thread_core.hle);
    no_thread_core.state.cia = no_thread_core.hle.bind_import(
        "coreinit", "OSSetExceptionCallback");
    no_thread_core.state.lr = 0x40U;
    no_thread_core.state.gpr[3] = 2U;
    no_thread_core.state.gpr[4] = 0x020347D0U;
    const RunResult no_thread = no_thread_core.run(1U);
    assert(no_thread.reason == StopReason::hle_error);
    assert(no_thread.detail.find("no current guest OSThread") != std::string::npos);
}

void os_get_thread_priority_hle_tests()
{
    constexpr std::uint32_t thread_size = 0x6A0U;
    constexpr std::uint32_t thread_tag = 0x74487244U;
    constexpr std::uint32_t return_address = 0x02760F04U;
    constexpr std::uint32_t arbitrary_thread = 0x4000U;
    constexpr std::uint32_t malformed_thread = 0x5000U;

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x1000U, 0x8000U);
    const std::uint32_t default_thread = initialize_default_guest_thread(
        core, 0xF000U, 0xE000U);
    assert(default_thread == core.current_thread_address);
    assert(core.memory.read32_be(default_thread + 0x320U) == thread_tag);
    assert(core.memory.read32_be(default_thread + 0x32CU) == 80U);
    assert(core.memory.read32_be(default_thread + 0x330U) == 80U);
    assert(core.memory.read32_be(default_thread + 0x5BCU) == 2U);

    core.memory.zero_fill(arbitrary_thread, thread_size);
    core.memory.write32_be(arbitrary_thread + 0x320U, thread_tag);
    core.memory.write32_be(arbitrary_thread + 0x32CU, 64U);
    core.memory.write32_be(arbitrary_thread + 0x330U, 71U);
    core.memory.write32_be(arbitrary_thread + 0x5BCU, 2U);
    const std::uint32_t import = core.hle.bind_import(
        "coreinit", "OSGetThreadPriority");

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;

    const auto invoke = [&](std::uint32_t thread, std::uint32_t expected) {
        core.state.gpr[3] = thread;
        core.state.lr = return_address;
        core.state.cia = import;
        const auto gprs_before = core.state.gpr;
        const auto fprs_before = core.state.fpr;
        const auto ps1_before = core.state.fpr_ps1;
        const std::uint32_t cr_before = core.state.cr;
        const std::uint32_t xer_before = core.state.xer;
        const std::uint32_t ctr_before = core.state.ctr;
        const std::uint32_t fpscr_before = core.state.fpscr;
        const std::uint32_t lr_before = core.state.lr;
        const std::uint32_t current_thread_before = core.current_thread_address;
        const std::uint32_t cursor_before = core.guest_heap_cursor;
        const std::uint32_t limit_before = core.guest_heap_limit;
        const auto base_handles_before = core.base_heap_handles;
        const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
        const std::uint32_t mem2_end_before = core.mem2_heap_region_end;
        std::vector<std::uint8_t> thread_before(thread_size);
        core.memory.read_bytes(thread, thread_before);

        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(result.steps == 1U);
        assert(core.state.gpr[3] == expected);
        assert(core.state.cia == return_address);
        assert(core.state.lr == lr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.fpr == fprs_before);
        assert(core.state.fpr_ps1 == ps1_before);
        for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            if (reg != 3U)
            {
                assert(core.state.gpr[reg] == gprs_before[reg]);
            }
        }
        std::vector<std::uint8_t> thread_after(thread_size);
        core.memory.read_bytes(thread, thread_after);
        assert(thread_after == thread_before);
        assert(core.current_thread_address == current_thread_before);
        assert(core.guest_heap_cursor == cursor_before);
        assert(core.guest_heap_limit == limit_before);
        assert(core.base_heap_handles == base_handles_before);
        assert(core.mem2_heap_region_begin == mem2_begin_before);
        assert(core.mem2_heap_region_end == mem2_end_before);
    };

    // The initialized APP thread's 80 internal priority maps to external 16.
    invoke(default_thread, 16U);

    // A distinct supplied thread wins over current_thread_address. The getter
    // reads basePriority (+0x330), not effective priority (+0x32C).
    assert(arbitrary_thread != core.current_thread_address);
    invoke(arbitrary_thread, 7U);
    core.memory.write32_be(arbitrary_thread + 0x32CU, 95U);
    invoke(arbitrary_thread, 7U);
    core.memory.write32_be(arbitrary_thread + 0x330U, 70U);
    invoke(arbitrary_thread, 6U);

    struct PriorityCase
    {
        std::uint32_t type;
        std::int32_t base_priority;
        std::uint32_t external_priority;
    };
    constexpr std::array<PriorityCase, 6> boundaries{{
        {0U, 0, 0U}, {0U, 31, 31U},
        {1U, 32, 0U}, {1U, 63, 31U},
        {2U, 64, 0U}, {2U, 95, 31U},
    }};
    for (const PriorityCase& test : boundaries)
    {
        core.memory.write32_be(arbitrary_thread + 0x5BCU, test.type);
        core.memory.write32_be(
            arbitrary_thread + 0x330U,
            static_cast<std::uint32_t>(test.base_priority));
        invoke(arbitrary_thread, test.external_priority);
    }

    const auto expect_hle_error = [&](std::uint32_t thread,
                                      const std::string& message) {
        core.state.gpr[3] = thread;
        core.state.lr = return_address;
        core.state.cia = import;
        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::hle_error);
        assert(result.detail.find(message) != std::string::npos);
        assert(result.steps == 0U);
    };

    core.memory.write32_be(arbitrary_thread + 0x5BCU, 3U);
    core.memory.write32_be(arbitrary_thread + 0x330U, 80U);
    expect_hle_error(arbitrary_thread, "unsupported thread type");

    struct InvalidPriorityCase
    {
        std::uint32_t type;
        std::int32_t base_priority;
    };
    constexpr std::array<InvalidPriorityCase, 5> invalid_priorities{{
        {0U, 32}, {1U, 31}, {1U, 64}, {2U, 63}, {2U, 96},
    }};
    for (const InvalidPriorityCase& test : invalid_priorities)
    {
        core.memory.write32_be(arbitrary_thread + 0x5BCU, test.type);
        core.memory.write32_be(
            arbitrary_thread + 0x330U,
            static_cast<std::uint32_t>(test.base_priority));
        expect_hle_error(
            arbitrary_thread,
            "base priority is outside the valid range");
    }

    // Invalid tags, null pointers, unmapped storage, and wrapping addresses
    // fail through the HLE error path without touching guest memory.
    core.memory.zero_fill(malformed_thread, thread_size);
    core.memory.write32_be(malformed_thread + 0x320U, 0xDEADBEEFU);
    core.memory.write32_be(malformed_thread + 0x330U, 80U);
    core.memory.write32_be(malformed_thread + 0x5BCU, 2U);
    std::vector<std::uint8_t> malformed_before(thread_size);
    core.memory.read_bytes(malformed_thread, malformed_before);
    expect_hle_error(malformed_thread, "invalid thread tag");
    std::vector<std::uint8_t> malformed_after(thread_size);
    core.memory.read_bytes(malformed_thread, malformed_after);
    assert(malformed_after == malformed_before);
    expect_hle_error(0U, "null OSThread pointer");
    expect_hle_error(0x20000U, "outside mapped guest memory");
    expect_hle_error(0xFFFFFF00U, "wraps guest address space");
}

void os_init_message_queue_hle_tests()
{
    constexpr std::uint32_t queue = 0x1000U;
    constexpr std::uint32_t messages = 0x2000U;
    constexpr std::uint32_t queue_size = 0x3CU;
    constexpr std::uint32_t message_array_size = 0x200U;
    constexpr std::uint32_t return_address = 0x02760BE0U;
    constexpr std::uint8_t sentinel = 0xA5U;

    EspressoCore core(0x4000U);
    register_coreinit_hle(core.hle);
    const std::uint32_t initialize =
        core.hle.bind_import("coreinit", "OSInitMessageQueue");
    core.memory.fill_bytes(queue - 1U, queue_size + 2U, sentinel);
    core.memory.fill_bytes(messages, message_array_size, 0x5AU);
    std::vector<std::uint8_t> messages_before(message_array_size);
    core.memory.read_bytes(messages, messages_before);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA5000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    core.state.gpr[3] = queue;
    core.state.gpr[4] = messages;
    core.state.gpr[5] = 0x20U;
    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const std::uint32_t cr_before = core.state.cr;
    const std::uint32_t xer_before = core.state.xer;
    const std::uint32_t ctr_before = core.state.ctr;
    const std::uint32_t fpscr_before = core.state.fpscr;
    const std::uint32_t lr_before = core.state.lr;
    const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
    const std::uint32_t heap_limit_before = core.guest_heap_limit;
    const auto base_heap_handles_before = core.base_heap_handles;
    const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
    const std::uint32_t mem2_end_before = core.mem2_heap_region_end;

    core.state.cia = initialize;
    const RunResult initialized = core.run(1U);
    assert(initialized.reason == StopReason::instruction_limit);
    assert(initialized.steps == 1U);
    assert(core.state.cia == return_address);
    assert(core.state.lr == lr_before);
    assert(core.state.gpr == gprs_before);
    assert(core.state.fpr == fprs_before);
    assert(core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before);
    assert(core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before);
    assert(core.state.fpscr == fpscr_before);
    assert(core.guest_heap_cursor == heap_cursor_before);
    assert(core.guest_heap_limit == heap_limit_before);
    assert(core.base_heap_handles == base_heap_handles_before);
    assert(core.mem2_heap_region_begin == mem2_begin_before);
    assert(core.mem2_heap_region_end == mem2_end_before);

    assert(core.memory.read32_be(queue + 0x00U) == 0x6D536751U);
    assert(core.memory.read32_be(queue + 0x04U) == 0U);
    assert(core.memory.read32_be(queue + 0x08U) == 0U);
    assert(core.memory.read32_be(queue + 0x0CU) == 0U);
    assert(core.memory.read32_be(queue + 0x10U) == 0U);
    assert(core.memory.read32_be(queue + 0x14U) == queue);
    assert(core.memory.read32_be(queue + 0x18U) == 0U);
    assert(core.memory.read32_be(queue + 0x1CU) == 0U);
    assert(core.memory.read32_be(queue + 0x20U) == 0U);
    assert(core.memory.read32_be(queue + 0x24U) == queue);
    assert(core.memory.read32_be(queue + 0x28U) == 0U);
    assert(core.memory.read32_be(queue + 0x2CU) == messages);
    assert(core.memory.read32_be(queue + 0x30U) == 0x20U);
    assert(core.memory.read32_be(queue + 0x34U) == 0U);
    assert(core.memory.read32_be(queue + 0x38U) == 0U);
    assert(core.memory.read8(queue - 1U) == sentinel);
    assert(core.memory.read8(queue + queue_size) == sentinel);
    std::vector<std::uint8_t> messages_after(message_array_size);
    core.memory.read_bytes(messages, messages_after);
    assert(messages_after == messages_before);

    // Reinitialization deterministically clears stale queue state and accepts
    // a guest message pointer without dereferencing it, including size zero.
    core.memory.write32_be(queue + 0x04U, 0x11111111U);
    core.memory.write32_be(queue + 0x08U, 0x22222222U);
    core.memory.write32_be(queue + 0x0CU, 0x33333333U);
    core.memory.write32_be(queue + 0x20U, 0x44444444U);
    core.memory.write32_be(queue + 0x34U, 7U);
    core.memory.write32_be(queue + 0x38U, 9U);
    core.state.gpr[3] = queue;
    core.state.gpr[4] = 0xDEADBEEFU;
    core.state.gpr[5] = 0U;
    core.state.cia = initialize;
    assert(core.step() == StepResult::executed);
    assert(core.state.cia == return_address);
    assert(core.state.gpr[3] == queue);
    assert(core.state.gpr[4] == 0xDEADBEEFU);
    assert(core.state.gpr[5] == 0U);
    assert(core.memory.read32_be(queue + 0x00U) == 0x6D536751U);
    assert(core.memory.read32_be(queue + 0x04U) == 0U);
    assert(core.memory.read32_be(queue + 0x08U) == 0U);
    assert(core.memory.read32_be(queue + 0x0CU) == 0U);
    assert(core.memory.read32_be(queue + 0x10U) == 0U);
    assert(core.memory.read32_be(queue + 0x14U) == queue);
    assert(core.memory.read32_be(queue + 0x18U) == 0U);
    assert(core.memory.read32_be(queue + 0x1CU) == 0U);
    assert(core.memory.read32_be(queue + 0x20U) == 0U);
    assert(core.memory.read32_be(queue + 0x24U) == queue);
    assert(core.memory.read32_be(queue + 0x28U) == 0U);
    assert(core.memory.read32_be(queue + 0x2CU) == 0xDEADBEEFU);
    assert(core.memory.read32_be(queue + 0x30U) == 0U);
    assert(core.memory.read32_be(queue + 0x34U) == 0U);
    assert(core.memory.read32_be(queue + 0x38U) == 0U);
    core.memory.read_bytes(messages, messages_after);
    assert(messages_after == messages_before);

    const auto expect_queue_error = [&](std::uint32_t invalid_queue,
                                        const std::string& expected_detail) {
        core.state.cia = initialize;
        core.state.lr = return_address;
        core.state.gpr[3] = invalid_queue;
        core.state.gpr[4] = messages;
        core.state.gpr[5] = 1U;
        const std::uint32_t gpr3_before = core.state.gpr[3];
        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::hle_error);
        assert(result.steps == 0U);
        assert(result.detail.find(expected_detail) != std::string::npos);
        assert(core.state.gpr[3] == gpr3_before);
    };
    expect_queue_error(0U, "null OSMessageQueue pointer");
    expect_queue_error(0x4000U, "outside writable guest memory");
    expect_queue_error(0xFFFFFFF0U, "wraps guest address space");

    EspressoCore partial_core(0x1000U);
    register_coreinit_hle(partial_core.hle);
    partial_core.memory.map_region(0x2000U, 0x10U);
    partial_core.memory.fill_bytes(0x2000U, 0x10U, sentinel);
    const std::uint32_t partial_initialize = partial_core.hle.bind_import(
        "coreinit", "OSInitMessageQueue");
    partial_core.state.cia = partial_initialize;
    partial_core.state.lr = return_address;
    partial_core.state.gpr[3] = 0x2000U;
    partial_core.state.gpr[4] = 0x3000U;
    partial_core.state.gpr[5] = 1U;
    const RunResult partial_fault = partial_core.run(1U);
    assert(partial_fault.reason == StopReason::hle_error);
    assert(partial_fault.detail.find("outside writable guest memory") !=
           std::string::npos);
    for (std::uint32_t address = 0x2000U; address < 0x2010U; ++address)
    {
        assert(partial_core.memory.read8(address) == sentinel);
    }
}

void mem_get_base_heap_handle_hle_tests()
{
    constexpr std::uint32_t heap_tag = 0x45585048U; // EXPH
    constexpr std::uint32_t header_size = 0x54U;
    constexpr std::uint32_t block_header_size = 0x14U;
    constexpr std::uint16_t free_block_tag = 0x4652U; // 'FR'
    constexpr std::uint32_t return_address = 0x02005EE8U;

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x1000U, 0x8000U);
    for (const std::uint32_t handle : core.base_heap_handles)
    {
        assert(handle == 0U);
    }

    const std::uint32_t thread = initialize_default_guest_thread(
        core, 0xF000U, 0xE000U);
    const std::uint32_t initial_cursor = core.guest_heap_cursor;
    const std::uint32_t initial_allocator_limit = core.guest_heap_limit;
    assert(initialize_default_guest_heaps(core));

    const std::uint32_t handle = core.base_heap_handles[1];
    const std::uint32_t foreground = core.base_heap_handles[8];
    assert(handle != 0U);
    assert(foreground != 0U);
    assert(core.base_heap_handles[0] == 0U);
    assert(core.mem2_heap_region_begin == handle);
    assert(core.mem2_heap_region_end == initial_allocator_limit);
    assert(core.guest_heap_cursor == initial_cursor);
    assert(core.guest_heap_limit == foreground);
    assert(handle > thread + 0x6A0U);
    assert((handle & 0xFU) == 0U);
    assert((foreground & 0xFU) == 0U);
    assert(foreground >= initial_cursor + 0x1000U);
    assert(foreground < handle);
    assert(core.memory.read32_be(foreground + 0x00U) == 0x46524D48U);
    assert(core.memory.read32_be(foreground + 0x18U) == foreground + 0x4CU);
    assert(core.memory.read32_be(foreground + 0x1CU) == handle);
    assert(core.memory.read32_be(foreground + 0x30U) == 0U);
    assert(core.memory.read32_be(foreground + 0x40U) == foreground + 0x4CU);
    assert(core.memory.read32_be(foreground + 0x44U) == handle);
    assert(core.memory.read32_be(foreground + 0x48U) == 0U);
    assert(core.memory.read32_be(handle + 0x00U) == heap_tag);
    assert(core.memory.read32_be(handle + 0x04U) == 0U);
    assert(core.memory.read32_be(handle + 0x08U) == 0U);
    assert(core.memory.read32_be(handle + 0x0CU) == 0U);
    assert(core.memory.read32_be(handle + 0x10U) == 0U);
    assert(core.memory.read32_be(handle + 0x14U) == 0U);
    const std::uint32_t free_block = handle + header_size;
    assert(core.memory.read32_be(handle + 0x18U) == free_block);
    assert(core.memory.read32_be(handle + 0x1CU) == core.mem2_heap_region_end);
    assert(core.memory.read32_be(handle + 0x30U) == 0U);
    assert(core.memory.read32_be(handle + 0x40U) == free_block);
    assert(core.memory.read32_be(handle + 0x44U) == free_block);
    assert(core.memory.read32_be(handle + 0x48U) == 0U);
    assert(core.memory.read32_be(handle + 0x4CU) == 0U);
    assert(core.memory.read16_be(handle + 0x50U) == 0U);
    assert(core.memory.read16_be(handle + 0x52U) == 0U);
    assert(core.memory.read32_be(free_block + 0x00U) == 0U);
    assert(core.memory.read32_be(free_block + 0x04U) ==
           core.mem2_heap_region_end - free_block - block_header_size);
    assert(core.memory.read32_be(free_block + 0x08U) == 0U);
    assert(core.memory.read32_be(free_block + 0x0CU) == 0U);
    assert(core.memory.read16_be(free_block + 0x10U) == free_block_tag);
    core.memory.validate_write_range(
        free_block, core.mem2_heap_region_end - free_block);
    core.memory.validate_write_range(
        core.mem2_heap_region_end - 1U, 1U);

    std::vector<std::uint8_t> initialized_heaps(
        core.mem2_heap_region_end - foreground);
    core.memory.read_bytes(foreground, initialized_heaps);
    const auto initialized_handles = core.base_heap_handles;
    const auto initialized_cursor = core.guest_heap_cursor;
    const auto initialized_limit = core.guest_heap_limit;
    assert(!initialize_default_guest_heaps(core));
    std::vector<std::uint8_t> duplicate_attempt_heaps(initialized_heaps.size());
    core.memory.read_bytes(foreground, duplicate_attempt_heaps);
    assert(duplicate_attempt_heaps == initialized_heaps);
    assert(core.base_heap_handles == initialized_handles);
    assert(core.guest_heap_cursor == initialized_cursor);
    assert(core.guest_heap_limit == initialized_limit);

    const std::uint32_t get_handle = core.hle.bind_import(
        "coreinit", "MEMGetBaseHeapHandle");
    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    std::vector<std::uint8_t> heap_snapshot(core.mem2_heap_region_end - handle);
    core.memory.read_bytes(handle, heap_snapshot);

    const auto lookup = [&](std::uint32_t type, std::uint32_t expected) {
        core.state.gpr[3] = type;
        core.state.lr = return_address;
        core.state.cia = get_handle;
        auto gpr_before = core.state.gpr;
        auto fpr_before = core.state.fpr;
        auto ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto ctr_before = core.state.ctr;
        const auto fpscr_before = core.state.fpscr;
        const auto lr_before = core.state.lr;
        const std::uint32_t cursor_before = core.guest_heap_cursor;
        const auto registry_before = core.base_heap_handles;

        assert(core.step() == StepResult::executed);
        assert(core.state.gpr[3] == expected);
        assert(core.state.cia == return_address);
        assert(core.state.lr == lr_before);
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.fpr == fpr_before);
        assert(core.state.fpr_ps1 == ps1_before);
        for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            if (reg != 3U)
            {
                assert(core.state.gpr[reg] == gpr_before[reg]);
            }
        }
        assert(core.guest_heap_cursor == cursor_before);
        assert(core.base_heap_handles == registry_before);
    };

    lookup(1U, handle);
    lookup(1U, handle);
    lookup(0U, 0U);
    lookup(8U, foreground);
    lookup(9U, 0U);
    lookup(0xFFFFFFFFU, 0U);
    std::vector<std::uint8_t> heap_after_lookups(heap_snapshot.size());
    core.memory.read_bytes(handle, heap_after_lookups);
    assert(heap_after_lookups == heap_snapshot);

    // Default bump allocations retain their established alignment and stay in
    // the low partition, never overlapping the MEM2 heap or its metadata.
    const std::uint32_t allocate_ex = core.hle.bind_import(
        "coreinit", "MEMAllocFromDefaultHeapEx");
    core.state.gpr[3] = 0x80U;
    core.state.gpr[4] = 64U;
    core.state.lr = return_address;
    core.state.cia = allocate_ex;
    assert(core.step() == StepResult::executed);
    const std::uint32_t first_allocation = core.state.gpr[3];
    core.state.gpr[3] = 0x100U;
    core.state.gpr[4] = 256U;
    core.state.lr = return_address;
    core.state.cia = allocate_ex;
    assert(core.step() == StepResult::executed);
    const std::uint32_t second_allocation = core.state.gpr[3];
    assert(first_allocation != 0U && (first_allocation & 63U) == 0U);
    assert(second_allocation != 0U && (second_allocation & 255U) == 0U);
    assert(first_allocation + 0x80U <= foreground);
    assert(second_allocation + 0x100U <= foreground);
    const std::uint32_t cursor_after_allocations = core.guest_heap_cursor;
    assert(core.allocate_guest_memory(foreground, 16U) == 0U);
    assert(core.guest_heap_cursor == cursor_after_allocations);
    assert(core.base_heap_handles[1] == handle);
    assert(core.base_heap_handles[8] == foreground);
    assert(core.memory.read32_be(handle) == heap_tag);

    // A small but viable selected gap initializes cleanly.
    EspressoCore small_success(0x4000U);
    small_success.configure_guest_heap(0x1000U, 0x2800U);
    assert(initialize_default_guest_heaps(small_success));
    assert(small_success.base_heap_handles[1] != 0U);
    assert(small_success.base_heap_handles[8] != 0U);
    assert(small_success.mem2_heap_region_end == 0x2800U);
    assert(small_success.memory.read32_be(small_success.base_heap_handles[1]) == heap_tag);
    assert(small_success.memory.read32_be(small_success.base_heap_handles[8]) ==
           0x46524D48U);

    EspressoCore unaligned_boundary(0x4000U);
    unaligned_boundary.configure_guest_heap(0x1001U, 0x280BU);
    assert(initialize_default_guest_heaps(unaligned_boundary));
    assert((unaligned_boundary.base_heap_handles[1] & 0xFU) == 0U);
    assert((unaligned_boundary.base_heap_handles[8] & 0xFU) == 0U);
    assert(unaligned_boundary.mem2_heap_region_end == 0x2800U);

    // Insufficient allocator room or an unmapped configured range fails
    // transactionally: cursor, bounds, registry, and guest bytes stay intact.
    EspressoCore too_small(0x4000U);
    too_small.configure_guest_heap(0x1000U, 0x2000U);
    too_small.memory.fill_bytes(0x1000U, 0x1000U, 0xA5U);
    const auto too_small_cursor = too_small.guest_heap_cursor;
    const auto too_small_limit = too_small.guest_heap_limit;
    assert(!initialize_default_guest_heaps(too_small));
    assert(too_small.guest_heap_cursor == too_small_cursor);
    assert(too_small.guest_heap_limit == too_small_limit);
    assert(too_small.base_heap_handles[1] == 0U);
    assert(too_small.base_heap_handles[8] == 0U);
    assert(too_small.memory.read8(0x1000U) == 0xA5U);
    assert(too_small.memory.read8(0x1FFFU) == 0xA5U);

    EspressoCore unmapped(0x4000U);
    unmapped.configure_guest_heap(0x1000U, 0x30000U);
    assert(!initialize_default_guest_heaps(unmapped));
    assert(unmapped.base_heap_handles[1] == 0U);
    assert(unmapped.base_heap_handles[8] == 0U);
    assert(unmapped.guest_heap_limit == 0x30000U);

    EspressoCore near_address_limit(0x4000U);
    near_address_limit.configure_guest_heap(0xFFFF0000U, 0xFFFFFFF0U);
    assert(!initialize_default_guest_heaps(near_address_limit));
    assert(near_address_limit.base_heap_handles[1] == 0U);
    assert(near_address_limit.base_heap_handles[8] == 0U);
    assert(near_address_limit.guest_heap_limit == 0xFFFFFFF0U);

    // A full Emulator initializes the heap after RPX placement and OSThread
    // allocation, before execution can reach the getter.
    affogato::Emulator emulator(0x40000U);
    const auto fixture_rpx = make_minimal_compressed_rpx(0x1000U);
    static_cast<void>(emulator.load_rpx(fixture_rpx));
    const EspressoCore& emulated_core = emulator.core();
    const std::uint32_t startup_handle = emulated_core.base_heap_handles[1];
    const std::uint32_t startup_foreground = emulated_core.base_heap_handles[8];
    assert(startup_handle != 0U);
    assert(startup_foreground != 0U);
    assert(emulated_core.current_thread_address != 0U);
    assert(emulated_core.memory.read32_be(startup_handle) == heap_tag);
    assert(emulated_core.guest_heap_limit == startup_foreground);

    EspressoCore independent(0x10000U);
    independent.configure_guest_heap(0x1000U, 0x8000U);
    assert(initialize_default_guest_heaps(independent));
    const std::uint32_t independent_handle = independent.base_heap_handles[1];
    const std::uint32_t independent_foreground = independent.base_heap_handles[8];
    assert(independent_handle != 0U);
    assert(independent_foreground != 0U);
    independent.memory.write32_be(independent_handle, 0xDEADBEEFU);
    assert(core.memory.read32_be(handle) == heap_tag);
    independent.reset();
    assert(independent.base_heap_handles[1] == 0U);
    assert(independent.base_heap_handles[8] == 0U);
    assert(independent.mem2_heap_region_begin == 0U);
    assert(independent.mem2_heap_region_end == 0U);
    assert(core.base_heap_handles[1] == handle);

    core.reset();
    assert(core.base_heap_handles[1] == 0U);
    assert(core.base_heap_handles[8] == 0U);
    assert(core.mem2_heap_region_begin == 0U);
    assert(core.mem2_heap_region_end == 0U);
}

void mem_get_total_free_size_for_exp_heap_hle_tests()
{
    constexpr std::uint32_t heap = 0x1000U;
    constexpr std::uint32_t data_start = 0x1100U;
    constexpr std::uint32_t data_end = 0x5000U;
    constexpr std::uint32_t block_a = 0x1200U;
    constexpr std::uint32_t block_b = 0x1500U;
    constexpr std::uint32_t block_c = 0x1900U;
    constexpr std::uint32_t return_address = 0x02005F08U;

    const auto initialize_heap = [](EspressoCore& core,
                                    std::uint32_t heap_address,
                                    std::uint32_t start,
                                    std::uint32_t end,
                                    std::uint32_t head,
                                    std::uint32_t tail) {
        core.memory.zero_fill(heap_address, 0x54U);
        core.memory.write32_be(heap_address, 0x45585048U);
        core.memory.write32_be(heap_address + 0x18U, start);
        core.memory.write32_be(heap_address + 0x1CU, end);
        core.memory.write32_be(heap_address + 0x40U, head);
        core.memory.write32_be(heap_address + 0x44U, tail);
    };
    const auto initialize_block = [](EspressoCore& core,
                                     std::uint32_t address,
                                     std::uint32_t size,
                                     std::uint32_t previous,
                                     std::uint32_t next) {
        core.memory.write32_be(address + 0x00U, 0U);
        core.memory.write32_be(address + 0x04U, size);
        core.memory.write32_be(address + 0x08U, previous);
        core.memory.write32_be(address + 0x0CU, next);
        core.memory.write16_be(address + 0x10U, 0x4652U);
    };

    EspressoCore fresh_heap(0x10000U);
    register_coreinit_hle(fresh_heap.hle);
    fresh_heap.configure_guest_heap(0x1000U, 0x8000U);
    assert(initialize_default_guest_heaps(fresh_heap));
    const std::uint32_t fresh_handle = fresh_heap.base_heap_handles[1];
    const std::uint32_t fresh_head =
        fresh_heap.memory.read32_be(fresh_handle + 0x40U);
    assert(fresh_head == fresh_heap.memory.read32_be(fresh_handle + 0x44U));
    fresh_heap.state.gpr[3] = fresh_handle;
    fresh_heap.state.lr = return_address;
    fresh_heap.state.cia = fresh_heap.hle.bind_import(
        "coreinit", "MEMGetTotalFreeSizeForExpHeap");
    const RunResult fresh_result = fresh_heap.run(1U);
    assert(fresh_result.reason == StopReason::instruction_limit);
    assert(fresh_heap.state.gpr[3] ==
           fresh_heap.memory.read32_be(fresh_head + 0x04U));
    assert(fresh_heap.state.cia == return_address);

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    initialize_heap(core, heap, data_start, data_end, block_a, block_a);
    initialize_block(core, block_a, 0x800U, 0U, 0U);
    const std::uint32_t import = core.hle.bind_import(
        "coreinit", "MEMGetTotalFreeSizeForExpHeap");

    const auto invoke = [&](std::uint32_t handle) {
        core.state.gpr[3] = handle;
        core.state.lr = return_address;
        core.state.cia = import;
        return core.run(1U);
    };

    const std::uint32_t guest_head = core.memory.read32_be(heap + 0x40U);
    const std::uint32_t guest_tail = core.memory.read32_be(heap + 0x44U);
    const std::uint32_t initial_size = core.memory.read32_be(guest_head + 4U);
    std::vector<std::uint8_t> heap_before(data_end - heap);
    core.memory.read_bytes(heap, heap_before);
    core.configure_guest_heap(0x8000U, 0x9000U);
    core.base_heap_handles[1] = heap;
    core.mem2_heap_region_begin = heap;
    core.mem2_heap_region_end = data_end;
    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    core.state.gpr[3] = heap;
    core.state.lr = return_address;
    const auto gprs_before = core.state.gpr;
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    const auto cr_before = core.state.cr;
    const auto xer_before = core.state.xer;
    const auto ctr_before = core.state.ctr;
    const auto fpscr_before = core.state.fpscr;
    const std::uint32_t cursor_before = core.guest_heap_cursor;
    const std::uint32_t limit_before = core.guest_heap_limit;
    const auto handles_before = core.base_heap_handles;
    const std::uint32_t region_begin_before = core.mem2_heap_region_begin;
    const std::uint32_t region_end_before = core.mem2_heap_region_end;

    RunResult result = invoke(heap);
    assert(result.steps == 1U);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == initial_size);
    assert(core.state.cia == return_address);
    assert(core.state.lr == return_address);
    assert(core.state.cr == cr_before);
    assert(core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before);
    assert(core.state.fpscr == fpscr_before);
    assert(core.state.fpr == fprs_before);
    assert(core.state.fpr_ps1 == ps1_before);
    for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        if (reg != 3U)
        {
            assert(core.state.gpr[reg] == gprs_before[reg]);
        }
    }
    assert(core.guest_heap_cursor == cursor_before);
    assert(core.guest_heap_limit == limit_before);
    assert(core.base_heap_handles == handles_before);
    assert(core.mem2_heap_region_begin == region_begin_before);
    assert(core.mem2_heap_region_end == region_end_before);
    std::vector<std::uint8_t> heap_after(data_end - heap);
    core.memory.read_bytes(heap, heap_after);
    assert(heap_after == heap_before);
    assert(guest_head == block_a && guest_tail == block_a);

    // The guest free-list block size, not host-side MEM2 bounds, is the
    // authoritative result on every query.
    core.memory.write32_be(block_a + 0x04U, 0x600U);
    result = invoke(heap);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x600U);

    // Sum all free blocks reached through the guest next pointers. The used
    // list is intentionally populated but must not contribute to the result.
    initialize_heap(core, heap, data_start, data_end, block_a, block_c);
    initialize_block(core, block_a, 0x100U, 0U, block_b);
    initialize_block(core, block_b, 0x200U, block_a, block_c);
    initialize_block(core, block_c, 0x40U, block_b, 0U);
    core.memory.write32_be(heap + 0x48U, 0x2200U);
    core.memory.write32_be(heap + 0x4CU, 0x2200U);
    initialize_block(core, 0x2200U, 0x777U, 0U, 0U);
    std::vector<std::uint8_t> fragmented_before(data_end - heap);
    core.memory.read_bytes(heap, fragmented_before);
    result = invoke(heap);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x340U);
    std::vector<std::uint8_t> fragmented_after(data_end - heap);
    core.memory.read_bytes(heap, fragmented_after);
    assert(fragmented_after == fragmented_before);

    // Empty list is a valid zero-free-space heap.
    initialize_heap(core, heap, data_start, data_end, 0U, 0U);
    result = invoke(heap);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0U);

    const auto expect_hle_error = [&](EspressoCore& target, std::uint32_t handle,
                                      const std::string& expected_text) {
        target.state.gpr[3] = handle;
        target.state.lr = return_address;
        target.state.cia = target.hle.bind_import(
            "coreinit", "MEMGetTotalFreeSizeForExpHeap");
        const RunResult error = target.run(1U);
        assert(error.reason == StopReason::hle_error);
        assert(error.detail.find(expected_text) != std::string::npos);
        assert(error.steps == 0U);
    };

    expect_hle_error(core, 0U, "null heap handle");

    EspressoCore malformed(0x10000U);
    register_coreinit_hle(malformed.hle);
    const auto prepare_malformed = [&](std::uint32_t head, std::uint32_t end) {
        initialize_heap(malformed, heap, data_start, end, head, head);
    };

    prepare_malformed(0U, data_end);
    malformed.memory.write32_be(heap, 0xDEADBEEFU);
    expect_hle_error(malformed, heap, "not an EXPH heap");

    prepare_malformed(data_start - 4U, data_end);
    expect_hle_error(malformed, heap, "outside heap bounds");

    prepare_malformed(data_end - 8U, data_end);
    expect_hle_error(malformed, heap, "outside heap bounds");

    prepare_malformed(block_a, data_end);
    initialize_block(malformed, block_a, data_end, 0U, 0U);
    expect_hle_error(malformed, heap, "exceeds heap bounds");

    prepare_malformed(block_a, data_end);
    initialize_block(malformed, block_a, 0x100U, 0U, 0U);
    malformed.memory.write16_be(block_a + 0x10U, 0x1234U);
    expect_hle_error(malformed, heap, "invalid tag");

    prepare_malformed(block_a, data_end);
    initialize_block(malformed, block_a, 0x100U, 0U, block_a);
    expect_hle_error(malformed, heap, "cycle detected");

    prepare_malformed(block_a, data_end);
    initialize_block(malformed, block_a, 0x100U, 0U, block_b);
    initialize_block(malformed, block_b, 0x100U, block_a, block_a);
    expect_hle_error(malformed, heap, "cycle detected");

    prepare_malformed(block_a, 0xFFFFFFFFU);
    initialize_block(malformed, block_a, 0xFFFFD000U, 0U, block_b);
    initialize_block(malformed, block_b, 0xFFFFD000U, block_a, 0U);
    expect_hle_error(malformed, heap, "free-size overflow");

    prepare_malformed(block_a, 0x30000U);
    initialize_block(malformed, block_a, 0x100U, 0U, block_b);
    // The heap claims a broad interval, but the next header itself is not
    // mapped; reject it cleanly instead of allowing a guest-memory fault.
    malformed.memory.write32_be(block_a + 0x0CU, 0x20000U);
    expect_hle_error(malformed, heap, "free block header is outside mapped guest memory");

    // An unmapped alleged heap handle is converted to a clear HLE diagnostic,
    // rather than exposing a host exception or dereferencing host memory.
    expect_hle_error(malformed, 0x20000U, "header is outside mapped guest memory");
}

void mem_get_allocatable_size_for_frm_heap_ex_hle_tests()
{
    constexpr std::uint32_t heap = 0x1000U;
    constexpr std::uint32_t header_size = 0x4CU;
    constexpr std::uint32_t data_start = heap + header_size;
    constexpr std::uint32_t data_end = 0x2000U;
    constexpr std::uint32_t return_address = 0x02005F18U;
    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    const std::uint32_t import = core.hle.bind_import(
        "coreinit", "MEMGetAllocatableSizeForFrmHeapEx");
    const auto initialize = [&](std::uint32_t address,
                                std::uint32_t start,
                                std::uint32_t end,
                                std::uint32_t head,
                                std::uint32_t tail,
                                std::uint32_t tag = 0x46524D48U) {
        core.memory.zero_fill(address, header_size);
        core.memory.write32_be(address, tag);
        core.memory.write32_be(address + 0x18U, start);
        core.memory.write32_be(address + 0x1CU, end);
        core.memory.write32_be(address + 0x30U, 0U);
        core.memory.write32_be(address + 0x40U, head);
        core.memory.write32_be(address + 0x44U, tail);
        core.memory.write32_be(address + 0x48U, 0xA5A5A5A5U);
    };
    const auto invoke = [&](std::uint32_t address, std::uint32_t alignment) {
        core.state.gpr[3] = address;
        core.state.gpr[4] = alignment;
        core.state.lr = return_address;
        core.state.cia = import;
        return core.run(1U);
    };
    const auto expect_error = [&](std::uint32_t address,
                                  std::uint32_t alignment,
                                  const std::string& message) {
        const RunResult result = invoke(address, alignment);
        assert(result.reason == StopReason::hle_error);
        assert(result.steps == 0U);
        assert(result.detail.find(message) != std::string::npos);
    };

    initialize(heap, data_start, data_end, 0x1234U, data_end);
    // This is an ordinary valid FRMH and need not be present in the base-heap
    // registry: the HLE reads its guest header on each invocation.
    assert(core.base_heap_handles[8] == 0U);
    const std::uint32_t get_handle = core.hle.bind_import(
        "coreinit", "MEMGetBaseHeapHandle");
    core.state.gpr[3] = 8U;
    core.state.lr = return_address;
    core.state.cia = get_handle;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0U);

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const auto cr_before = core.state.cr;
    const auto xer_before = core.state.xer;
    const auto ctr_before = core.state.ctr;
    const auto fpscr_before = core.state.fpscr;
    const auto cursor_before = core.guest_heap_cursor;
    const auto limit_before = core.guest_heap_limit;
    const auto handles_before = core.base_heap_handles;
    const auto region_begin_before = core.mem2_heap_region_begin;
    const auto region_end_before = core.mem2_heap_region_end;
    const auto query = [&](std::uint32_t alignment, std::uint32_t expected) {
        core.state.gpr[3] = heap;
        core.state.gpr[4] = alignment;
        const auto gprs_before = core.state.gpr;
        std::array<std::uint8_t, header_size> header_before{};
        core.memory.read_bytes(heap, header_before);
        const RunResult result = invoke(heap, alignment);
        assert(result.reason == StopReason::instruction_limit);
        assert(result.steps == 1U);
        assert(core.state.gpr[3] == expected);
        assert(core.state.gpr[4] == alignment);
        assert(core.state.cia == return_address);
        assert(core.state.lr == return_address);
        for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            if (reg != 3U)
            {
                assert(core.state.gpr[reg] == gprs_before[reg]);
            }
        }
        assert(core.state.cr == cr_before);
        assert(core.state.xer == xer_before);
        assert(core.state.ctr == ctr_before);
        assert(core.state.fpscr == fpscr_before);
        assert(core.state.fpr == fprs_before);
        assert(core.state.fpr_ps1 == ps1_before);
        assert(core.guest_heap_cursor == cursor_before);
        assert(core.guest_heap_limit == limit_before);
        assert(core.base_heap_handles == handles_before);
        assert(core.mem2_heap_region_begin == region_begin_before);
        assert(core.mem2_heap_region_end == region_end_before);
        std::array<std::uint8_t, header_size> header_after{};
        core.memory.read_bytes(heap, header_after);
        assert(header_after == header_before);
    };

    query(1U, data_end - 0x1234U);
    query(2U, data_end - 0x1234U);
    query(4U, data_end - 0x1234U);
    query(0x100U, 0xD00U); // align 0x1234 up to 0x1300
    core.memory.write32_be(heap + 0x40U, 0x1400U);
    query(0x100U, 0xC00U);
    core.memory.write32_be(heap + 0x40U, 0x1FF0U);
    query(0x100U, 0U);
    core.memory.write32_be(heap + 0x40U, 0x1234U);

    // Each call observes direct guest edits rather than a cached host copy.
    core.memory.write32_be(heap + 0x44U, 0x1800U);
    query(4U, 0x1800U - 0x1234U);
    core.memory.write32_be(heap + 0x44U, data_end);

    expect_error(heap, 0U, "zero alignment");
    for (const std::uint32_t alignment : {3U, 6U, 12U, 24U})
    {
        expect_error(heap, alignment, "power of two");
    }
    expect_error(heap, 0xFFFFFFFFU, "negative alignment");
    expect_error(heap, 0x80000000U, "negative alignment");
    expect_error(0U, 4U, "null heap handle");
    expect_error(0xFFFFFFC0U, 4U, "wraps guest address space");
    expect_error(0x20000U, 4U, "outside mapped guest memory");
    expect_error(0xFFF0U, 4U, "outside mapped guest memory");

    initialize(heap, data_start, data_end, 0x1234U, data_end, 0x45585048U);
    expect_error(heap, 4U, "not an FRMH");
    initialize(heap, data_start - 4U, data_end, data_start - 4U, data_end);
    expect_error(heap, 4U, "invalid frame-heap bounds");
    initialize(heap, data_start, data_end, data_end, data_end - 4U);
    expect_error(heap, 4U, "invalid frame-heap bounds");
    initialize(heap, data_start, data_end, data_start, data_end + 4U);
    expect_error(heap, 4U, "invalid frame-heap bounds");
    initialize(heap, data_end, data_start, data_start, data_end);
    expect_error(heap, 4U, "invalid frame-heap bounds");
}

void mem_alloc_from_frm_heap_ex_hle_tests()
{
    constexpr std::uint32_t heap = 0x1000U;
    constexpr std::uint32_t header_size = 0x4CU;
    constexpr std::uint32_t data_start = heap + header_size;
    constexpr std::uint32_t data_end = 0x4000U;
    constexpr std::uint32_t return_address = 0x02005F1CU;
    constexpr std::uint32_t zero_allocated = 1U;
    constexpr std::uint32_t debug_mode = 2U;
    constexpr std::uint32_t use_lock = 4U;

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x8000U, 0x9000U);
    const std::uint32_t allocate_import = core.hle.bind_import(
        "coreinit", "MEMAllocFromFrmHeapEx");
    const auto initialize = [&](std::uint32_t head,
                                std::uint32_t tail,
                                std::uint32_t flags = 0U,
                                std::uint32_t start = 0x104CU,
                                std::uint32_t end = 0x4000U,
                                std::uint32_t tag = 0x46524D48U) {
        core.memory.zero_fill(heap, header_size);
        core.memory.write32_be(heap, tag);
        core.memory.write32_be(heap + 0x18U, start);
        core.memory.write32_be(heap + 0x1CU, end);
        core.memory.write32_be(heap + 0x30U, flags);
        core.memory.write32_be(heap + 0x40U, head);
        core.memory.write32_be(heap + 0x44U, tail);
        core.memory.write32_be(heap + 0x48U, 0x12345678U);
    };
    const auto invoke = [&](std::uint32_t size,
                            std::int32_t alignment,
                            std::uint32_t address = 0x1000U) {
        for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
        {
            core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
        }
        core.state.gpr[3] = address;
        core.state.gpr[4] = size;
        core.state.gpr[5] = std::bit_cast<std::uint32_t>(alignment);
        for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
        {
            core.state.fpr[reg] = 0x1111000000000000ULL + reg;
            core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
        }
        core.state.cr = 0x12345678U;
        core.state.xer = 0xA00000A5U;
        core.state.ctr = 0xCAFEBABEU;
        core.state.fpscr = 0x5A5AA55AU;
        core.state.lr = return_address;
        core.state.cia = allocate_import;

        const auto gprs_before = core.state.gpr;
        const auto fprs_before = core.state.fpr;
        const auto ps1_before = core.state.fpr_ps1;
        const auto cr_before = core.state.cr;
        const auto xer_before = core.state.xer;
        const auto ctr_before = core.state.ctr;
        const auto fpscr_before = core.state.fpscr;
        const auto lr_before = core.state.lr;
        const auto cursor_before = core.guest_heap_cursor;
        const auto limit_before = core.guest_heap_limit;
        const auto handles_before = core.base_heap_handles;
        const auto mem2_begin_before = core.mem2_heap_region_begin;
        const auto mem2_end_before = core.mem2_heap_region_end;

        const RunResult result = core.run(1U);
        if (result.reason == StopReason::instruction_limit)
        {
            assert(result.steps == 1U);
            assert(core.state.cia == return_address);
            assert(core.state.lr == lr_before);
            for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
            {
                if (reg != 3U)
                {
                    assert(core.state.gpr[reg] == gprs_before[reg]);
                }
            }
            assert(core.state.fpr == fprs_before);
            assert(core.state.fpr_ps1 == ps1_before);
            assert(core.state.cr == cr_before);
            assert(core.state.xer == xer_before);
            assert(core.state.ctr == ctr_before);
            assert(core.state.fpscr == fpscr_before);
            assert(core.guest_heap_cursor == cursor_before);
            assert(core.guest_heap_limit == limit_before);
            assert(core.base_heap_handles == handles_before);
            assert(core.mem2_heap_region_begin == mem2_begin_before);
            assert(core.mem2_heap_region_end == mem2_end_before);
        }
        return result;
    };
    const auto frame_header = [&]() {
        std::array<std::uint8_t, header_size> bytes{};
        core.memory.read_bytes(heap, bytes);
        return bytes;
    };
    const auto expect_error = [&](std::uint32_t size,
                                  std::int32_t alignment,
                                  const std::string& text) {
        const auto before = frame_header();
        const RunResult result = invoke(size, alignment);
        assert(result.reason == StopReason::hle_error);
        assert(result.steps == 0U);
        assert(result.detail.find(text) != std::string::npos);
        assert(frame_header() == before);
    };

    // Positive allocation consumes alignment padding and changes only head.
    initialize(0x1234U, 0x2000U);
    core.memory.fill_bytes(0x1234U, 0x100U, 0xA5U);
    const auto positive_header_before = frame_header();
    RunResult result = invoke(0x20U, 0x100);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x1300U);
    assert(core.state.gpr[4] == 0x20U);
    assert(core.state.gpr[5] == 0x100U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1320U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x2000U);
    assert(core.memory.read32_be(heap + 0x18U) == data_start);
    assert(core.memory.read32_be(heap + 0x1CU) == data_end);
    assert(core.memory.read32_be(heap + 0x48U) == 0x12345678U);
    const auto positive_header_after = frame_header();
    for (std::size_t i = 0; i < positive_header_before.size(); ++i)
    {
        if (i < 0x40U || i >= 0x44U)
        {
            assert(positive_header_after[i] == positive_header_before[i]);
        }
    }
    assert(core.memory.read8(0x12FFU) == 0xA5U);
    assert(core.memory.read8(0x1300U) == 0xA5U); // flags are zero; no clearing

    // Requests are byte-sized (not rounded), and size zero consumes one byte.
    initialize(0x1100U, 0x2000U);
    result = invoke(3U, 4);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x1100U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1103U);
    initialize(0x1100U, 0x2000U);
    result = invoke(0U, 4);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x1100U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1101U);
    initialize(0x1101U, 0x3000U);
    result = invoke(3U, 1);
    assert(core.state.gpr[3] == 0x1101U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1104U);
    initialize(0x1101U, 0x3000U);
    result = invoke(3U, 2);
    assert(core.state.gpr[3] == 0x1102U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1105U);
    initialize(0x1101U, 0x3000U);
    result = invoke(3U, 0x1000);
    assert(core.state.gpr[3] == 0x2000U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x2003U);
    initialize(0x1100U, 0x3000U);
    result = invoke(0x10U, -0x1000);
    assert(core.state.gpr[3] == 0x2000U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x2000U);

    // Several positive allocations are governed solely by live guest head.
    initialize(0x1100U, 0x2000U);
    result = invoke(0x100U, 0x100);
    assert(core.state.gpr[3] == 0x1100U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1200U);
    result = invoke(0x80U, 0x200);
    assert(core.state.gpr[3] == 0x1200U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1280U);

    // Guest edits to both cursors are authoritative on the next allocation.
    initialize(0x1100U, 0x3000U);
    core.memory.write32_be(heap + 0x40U, 0x1203U);
    result = invoke(1U, 1);
    assert(core.state.gpr[3] == 0x1203U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1204U);
    core.memory.write32_be(heap + 0x44U, 0x2800U);
    result = invoke(8U, -4);
    assert(core.state.gpr[3] == 0x27F8U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x27F8U);

    // Head/tail allocations can be interleaved without host-side tracking.
    initialize(0x1100U, 0x3000U);
    result = invoke(0x100U, 0x100);
    assert(core.state.gpr[3] == 0x1100U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1200U);
    result = invoke(0x80U, -0x100);
    assert(core.state.gpr[3] == 0x2F00U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1200U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x2F00U);

    // Tail allocation aligns down and changes only tail; previousState survives.
    initialize(0x1200U, 0x2000U);
    const auto tail_header_before = frame_header();
    result = invoke(0x180U, -0x100);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x1E00U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1200U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x1E00U);
    assert(core.memory.read32_be(heap + 0x48U) == 0x12345678U);
    const auto tail_header_after = frame_header();
    for (std::size_t i = 0; i < tail_header_before.size(); ++i)
    {
        if (i < 0x44U || i >= 0x48U)
        {
            assert(tail_header_after[i] == tail_header_before[i]);
        }
    }

    // Exact-fit allocations from either direction are successful.
    initialize(0x1800U, 0x1900U);
    result = invoke(0x100U, 4);
    assert(core.state.gpr[3] == 0x1800U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1900U);
    initialize(0x1800U, 0x1900U);
    result = invoke(0x100U, -4);
    assert(core.state.gpr[3] == 0x1800U);
    assert(core.memory.read32_be(heap + 0x44U) == 0x1800U);

    // Alignment-induced exact fit succeeds; increasing the request by one
    // fails normally and leaves both metadata and payload untouched.
    initialize(0x18F1U, 0x1908U);
    result = invoke(8U, 0x20);
    assert(core.state.gpr[3] == 0x1900U);
    assert(core.memory.read32_be(heap + 0x40U) == 0x1908U);
    initialize(0x18F1U, 0x1908U);
    core.memory.fill_bytes(0x18F1U, 0x17U, 0xA5U);
    const auto oom_header = frame_header();
    result = invoke(9U, 0x20);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0U);
    assert(frame_header() == oom_header);
    assert(core.memory.read8(0x18F1U) == 0xA5U);

    initialize(0x1800U, 0x1900U);
    const auto head_oom_header = frame_header();
    result = invoke(0x200U, 4);
    assert(core.state.gpr[3] == 0U);
    assert(frame_header() == head_oom_header);

    // ZERO_ALLOCATED clears only the returned allocation, not alignment gap,
    // neighboring bytes, heap metadata, or an unsuccessful allocation.
    initialize(0x1234U, 0x2000U, zero_allocated);
    core.memory.fill_bytes(0x1200U, 0x200U, 0xA5U);
    result = invoke(3U, 0x100);
    assert(core.state.gpr[3] == 0x1300U);
    assert(core.memory.read8(0x12FFU) == 0xA5U);
    assert(core.memory.read8(0x1300U) == 0U);
    assert(core.memory.read8(0x1301U) == 0U);
    assert(core.memory.read8(0x1302U) == 0U);
    assert(core.memory.read8(0x1303U) == 0xA5U);
    initialize(0x1800U, 0x1900U, zero_allocated);
    core.memory.fill_bytes(0x1800U, 0x100U, 0xA5U);
    result = invoke(0x200U, 4);
    assert(core.state.gpr[3] == 0U);
    assert(core.memory.read8(0x1800U) == 0xA5U);

    // Lock use is a no-op in the single-lane model. Debug fill is explicitly
    // rejected until the heap fill-value APIs exist.
    initialize(0x1100U, 0x2000U, use_lock);
    core.memory.write32_be(heap + 0x20U, 0xAABBCCDDU);
    result = invoke(0x10U, 4);
    assert(core.state.gpr[3] == 0x1100U);
    assert(core.memory.read32_be(heap + 0x20U) == 0xAABBCCDDU);
    initialize(0x1100U, 0x2000U, debug_mode);
    core.memory.fill_bytes(0x1100U, 0x10U, 0xA5U);
    expect_error(0x10U, 4, "DEBUG_MODE");
    assert(core.memory.read8(0x1100U) == 0xA5U);

    // Invalid signed alignments and malformed heaps fail before any metadata
    // or payload changes.
    initialize(0x1100U, 0x2000U);
    expect_error(0x10U, 0, "zero alignment");
    expect_error(0x10U, std::numeric_limits<std::int32_t>::min(),
                 "magnitude is not representable");
    expect_error(0x10U, 6, "power of two");
    expect_error(0x10U, -6, "power of two");
    initialize(0x1100U, 0x2000U, 0U, data_start, data_end, 0x45585048U);
    expect_error(0x10U, 4, "not an FRMH");
    initialize(0x1100U, 0x2000U, 0U, data_start - 4U);
    expect_error(0x10U, 4, "invalid frame-heap bounds");
    initialize(0x20000U, 0x30000U, 0U, 0x20000U, 0x30000U);
    expect_error(0x10U, 4, "outside writable guest memory");

    EspressoCore boundary(0x10000U);
    register_coreinit_hle(boundary.hle);
    boundary.memory.map_region(0xFFFF0000U, 0x10000U);
    constexpr std::uint32_t boundary_heap = 0xFFFF0000U;
    boundary.memory.write32_be(boundary_heap, 0x46524D48U);
    boundary.memory.write32_be(boundary_heap + 0x18U, 0xFFFF004CU);
    boundary.memory.write32_be(boundary_heap + 0x1CU, 0xFFFFFFFFU);
    boundary.memory.write32_be(boundary_heap + 0x30U, 0U);
    boundary.memory.write32_be(boundary_heap + 0x40U, 0xFFFFFFF0U);
    boundary.memory.write32_be(boundary_heap + 0x44U, 0xFFFFFFFFU);
    const std::uint32_t boundary_import = boundary.hle.bind_import(
        "coreinit", "MEMAllocFromFrmHeapEx");
    const auto boundary_call = [&](std::uint32_t size, std::int32_t alignment) {
        boundary.state.gpr[3] = boundary_heap;
        boundary.state.gpr[4] = size;
        boundary.state.gpr[5] = std::bit_cast<std::uint32_t>(alignment);
        boundary.state.lr = return_address;
        boundary.state.cia = boundary_import;
        return boundary.run(1U);
    };
    auto boundary_result = boundary_call(8U, 0x1000);
    assert(boundary_result.reason == StopReason::instruction_limit);
    assert(boundary.state.gpr[3] == 0U); // align-up would cross 2^32
    assert(boundary.memory.read32_be(boundary_heap + 0x40U) == 0xFFFFFFF0U);
    assert(boundary.memory.read32_be(boundary_heap + 0x44U) == 0xFFFFFFFFU);
    boundary_result = boundary_call(0xFFFFFFFFU, -4);
    assert(boundary_result.reason == StopReason::instruction_limit);
    assert(boundary.state.gpr[3] == 0U); // tail subtraction cannot fit
    assert(boundary.memory.read32_be(boundary_heap + 0x40U) == 0xFFFFFFF0U);
    assert(boundary.memory.read32_be(boundary_heap + 0x44U) == 0xFFFFFFFFU);

    // The actual default foreground heap is accepted; allocator capacity and
    // all region bookkeeping remain coherent with the previous query HLE.
    EspressoCore default_heap(0x20000U);
    register_coreinit_hle(default_heap.hle);
    default_heap.configure_guest_heap(0x1000U, 0x1F000U);
    assert(initialize_default_guest_heaps(default_heap));
    const std::uint32_t foreground = default_heap.base_heap_handles[8];
    const std::uint32_t old_head = default_heap.memory.read32_be(foreground + 0x40U);
    const std::uint32_t old_tail = default_heap.memory.read32_be(foreground + 0x44U);
    const std::uint32_t old_cursor = default_heap.guest_heap_cursor;
    const std::uint32_t old_limit = default_heap.guest_heap_limit;
    const auto old_handles = default_heap.base_heap_handles;
    const std::uint32_t old_mem2_begin = default_heap.mem2_heap_region_begin;
    const std::uint32_t old_mem2_end = default_heap.mem2_heap_region_end;
    const std::uint32_t old_capacity =
        old_tail - ((old_head + 3U) & ~3U);
    const std::uint32_t mem2_tag_before =
        default_heap.memory.read32_be(old_mem2_begin);
    default_heap.state.gpr[3] = foreground;
    default_heap.state.gpr[4] = 0x40U;
    default_heap.state.gpr[5] = 16U;
    default_heap.state.lr = return_address;
    default_heap.state.cia = default_heap.hle.bind_import(
        "coreinit", "MEMAllocFromFrmHeapEx");
    result = default_heap.run(1U);
    assert(result.reason == StopReason::instruction_limit);
    const std::uint32_t allocated = default_heap.state.gpr[3];
    assert(allocated == ((old_head + 15U) & ~15U));
    assert(default_heap.memory.read32_be(foreground + 0x40U) == allocated + 0x40U);
    assert(default_heap.memory.read32_be(foreground + 0x44U) == old_tail);
    assert(default_heap.guest_heap_cursor == old_cursor);
    assert(default_heap.guest_heap_limit == old_limit);
    assert(default_heap.base_heap_handles == old_handles);
    assert(default_heap.mem2_heap_region_begin == old_mem2_begin);
    assert(default_heap.mem2_heap_region_end == old_mem2_end);
    assert(default_heap.memory.read32_be(old_mem2_begin) == mem2_tag_before);

    default_heap.state.gpr[3] = foreground;
    default_heap.state.gpr[4] = 4U;
    default_heap.state.lr = return_address;
    default_heap.state.cia = default_heap.hle.bind_import(
        "coreinit", "MEMGetAllocatableSizeForFrmHeapEx");
    result = default_heap.run(1U);
    assert(result.reason == StopReason::instruction_limit);
    const std::uint32_t expected_capacity =
        old_tail - (allocated + 0x40U);
    assert(default_heap.state.gpr[3] == expected_capacity);
    assert(old_capacity - expected_capacity ==
           allocated + 0x40U - ((old_head + 3U) & ~3U));
}

void mem_get_allocatable_size_for_exp_heap_ex_hle_tests()
{
    constexpr std::uint32_t heap = 0x1000U;
    constexpr std::uint32_t data_start = 0x1100U;
    constexpr std::uint32_t data_end = 0x4000U;
    constexpr std::uint32_t block_a = 0x1200U;
    constexpr std::uint32_t block_b = 0x2400U;
    constexpr std::uint32_t block_c = 0x3000U;
    constexpr std::uint32_t return_address = 0x02005F14U;

    const auto initialize_heap = [](EspressoCore& core,
                                    std::uint32_t head,
                                    std::uint32_t tail) {
        core.memory.zero_fill(heap, 0x54U);
        core.memory.write32_be(heap, 0x45585048U);
        core.memory.write32_be(heap + 0x18U, data_start);
        core.memory.write32_be(heap + 0x1CU, data_end);
        core.memory.write32_be(heap + 0x40U, head);
        core.memory.write32_be(heap + 0x44U, tail);
    };
    const auto initialize_block = [](EspressoCore& core,
                                     std::uint32_t address,
                                     std::uint32_t size,
                                     std::uint32_t next) {
        core.memory.write32_be(address + 0x04U, size);
        core.memory.write32_be(address + 0x0CU, next);
        core.memory.write16_be(address + 0x10U, 0x4652U);
    };

    EspressoCore core(0x10000U);
    register_coreinit_hle(core.hle);
    core.configure_guest_heap(0x8000U, 0x9000U);
    core.base_heap_handles[1] = heap;
    core.mem2_heap_region_begin = heap;
    core.mem2_heap_region_end = data_end;
    initialize_heap(core, block_a, block_c);
    initialize_block(core, block_a, 0x1000U, block_b);
    initialize_block(core, block_b, 0x900U, block_c);
    initialize_block(core, block_c, 0x500U, 0U);
    core.memory.write32_be(heap + 0x48U, 0x3800U); // unrelated used list
    core.memory.write32_be(heap + 0x4CU, 0x3800U);
    initialize_block(core, 0x3800U, 0x777U, 0U);

    const std::uint32_t total_import = core.hle.bind_import(
        "coreinit", "MEMGetTotalFreeSizeForExpHeap");
    const std::uint32_t allocatable_import = core.hle.bind_import(
        "coreinit", "MEMGetAllocatableSizeForExpHeapEx");
    const auto invoke = [&](std::uint32_t import, std::uint32_t handle,
                            std::uint32_t alignment_bits) {
        core.state.gpr[3] = handle;
        core.state.gpr[4] = alignment_bits;
        core.state.lr = return_address;
        core.state.cia = import;
        return core.run(1U);
    };

    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA1000000U + reg * 0x101U;
    }
    core.state.gpr[3] = heap;
    core.state.gpr[4] = 0x20U;
    core.state.lr = return_address;
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    const auto cr_before = core.state.cr;
    const auto xer_before = core.state.xer;
    const auto ctr_before = core.state.ctr;
    const auto fpscr_before = core.state.fpscr;
    const std::uint32_t cursor_before = core.guest_heap_cursor;
    const std::uint32_t limit_before = core.guest_heap_limit;
    const auto handles_before = core.base_heap_handles;
    const std::uint32_t region_begin_before = core.mem2_heap_region_begin;
    const std::uint32_t region_end_before = core.mem2_heap_region_end;
    std::vector<std::uint8_t> heap_before(data_end - heap);
    core.memory.read_bytes(heap, heap_before);

    RunResult result = invoke(allocatable_import, heap, 0x20U);
    assert(result.reason == StopReason::instruction_limit);
    // Each block loses 0xC bytes to 32-byte alignment:
    // 0xFF4, 0x8F4, 0x4F4. Return the largest single candidate.
    assert(core.state.gpr[3] == 0xFF4U);
    assert(core.state.gpr[4] == 0x20U);
    assert(core.state.cia == return_address);
    assert(core.state.lr == return_address);
    assert(core.state.cr == cr_before);
    assert(core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before);
    assert(core.state.fpscr == fpscr_before);
    assert(core.state.fpr == fprs_before);
    assert(core.state.fpr_ps1 == ps1_before);
    for (std::size_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        if (reg != 3U)
        {
            assert(core.state.gpr[reg] == gprs_before[reg]);
        }
    }
    std::vector<std::uint8_t> heap_after(data_end - heap);
    core.memory.read_bytes(heap, heap_after);
    assert(heap_after == heap_before);
    assert(core.guest_heap_cursor == cursor_before);
    assert(core.guest_heap_limit == limit_before);
    assert(core.base_heap_handles == handles_before);
    assert(core.mem2_heap_region_begin == region_begin_before);
    assert(core.mem2_heap_region_end == region_end_before);

    // Total free bytes exceed the largest contiguous aligned allocation.
    result = invoke(total_import, heap, 0U);
    assert(result.reason == StopReason::instruction_limit);
    const std::uint32_t total_free = core.state.gpr[3];
    assert(total_free == 0x1E00U);
    result = invoke(allocatable_import, heap, 4U);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x1000U);
    assert(total_free > core.state.gpr[3]);

    // Direct edits to guest block metadata are observed on the next call.
    core.memory.write32_be(block_b + 0x04U, 0x100U);
    core.memory.write32_be(block_c + 0x04U, 0x80U);
    core.memory.write32_be(block_a + 0x04U, 0x800U);
    result = invoke(allocatable_import, heap, 4U);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0x800U);

    // Empty free list is a valid heap and returns zero.
    initialize_heap(core, 0U, 0U);
    result = invoke(allocatable_import, heap, 4U);
    assert(result.reason == StopReason::instruction_limit);
    assert(core.state.gpr[3] == 0U);

    const auto expect_hle_error = [&](std::uint32_t handle,
                                      std::uint32_t alignment_bits,
                                      const std::string& text) {
        const RunResult error = invoke(allocatable_import, handle, alignment_bits);
        assert(error.reason == StopReason::hle_error);
        assert(error.detail.find(text) != std::string::npos);
        assert(error.steps == 0U);
    };
    expect_hle_error(0U, 4U, "null heap handle");
    expect_hle_error(heap, 0U, "zero alignment");
    expect_hle_error(heap, 0x80000000U, "magnitude is not representable");
    expect_hle_error(heap, 6U, "power of two and at least 4");

    initialize_heap(core, block_a, block_a);
    initialize_block(core, block_a, 0x100U, 0U);
    core.memory.write32_be(block_a + 0x0CU, block_a);
    expect_hle_error(heap, 4U, "cycle detected");

    EspressoCore invalid_heap(0x10000U);
    register_coreinit_hle(invalid_heap.hle);
    const std::uint32_t invalid_import = invalid_heap.hle.bind_import(
        "coreinit", "MEMGetAllocatableSizeForExpHeapEx");
    invalid_heap.memory.write32_be(heap, 0xDEADBEEFU);
    invalid_heap.state.gpr[3] = heap;
    invalid_heap.state.gpr[4] = 4U;
    invalid_heap.state.lr = return_address;
    invalid_heap.state.cia = invalid_import;
    const RunResult invalid_result = invalid_heap.run(1U);
    assert(invalid_result.reason == StopReason::hle_error);
    assert(invalid_result.detail.find("not an EXPH heap") != std::string::npos);

    // The real initializer's first free block is 4-byte aligned, and both
    // capacity queries agree for the initial single-block heap.
    EspressoCore fresh(0x10000U);
    register_coreinit_hle(fresh.hle);
    fresh.configure_guest_heap(0x1000U, 0x8000U);
    assert(initialize_default_guest_heaps(fresh));
    const std::uint32_t fresh_handle = fresh.base_heap_handles[1];
    const std::uint32_t fresh_block = fresh.memory.read32_be(fresh_handle + 0x40U);
    const std::uint32_t fresh_block_size = fresh.memory.read32_be(fresh_block + 4U);
    const std::uint32_t fresh_total_import = fresh.hle.bind_import(
        "coreinit", "MEMGetTotalFreeSizeForExpHeap");
    const std::uint32_t fresh_allocatable_import = fresh.hle.bind_import(
        "coreinit", "MEMGetAllocatableSizeForExpHeapEx");
    const auto invoke_fresh = [&](std::uint32_t import, std::uint32_t alignment) {
        fresh.state.gpr[3] = fresh_handle;
        fresh.state.gpr[4] = alignment;
        fresh.state.lr = return_address;
        fresh.state.cia = import;
        return fresh.run(1U);
    };
    result = invoke_fresh(fresh_total_import, 4U);
    assert(result.reason == StopReason::instruction_limit);
    const std::uint32_t fresh_total = fresh.state.gpr[3];
    assert(fresh_total == fresh_block_size);
    result = invoke_fresh(fresh_allocatable_import, 4U);
    assert(result.reason == StopReason::instruction_limit);
    assert(fresh.state.gpr[3] == fresh_block_size);
    assert(fresh.state.gpr[3] == fresh_total);

    const std::uint64_t raw_start = static_cast<std::uint64_t>(fresh_block) + 0x14U;
    const std::uint64_t aligned_start = (raw_start + 0x1FU) & ~std::uint64_t{0x1FU};
    const std::uint64_t raw_end = raw_start + fresh_block_size;
    const std::uint32_t expected_32 = aligned_start < raw_end
        ? static_cast<std::uint32_t>(raw_end - aligned_start)
        : 0U;
    result = invoke_fresh(fresh_allocatable_import, 0x20U);
    assert(result.reason == StopReason::instruction_limit);
    const std::uint32_t positive_32 = fresh.state.gpr[3];
    assert(positive_32 == expected_32);
    result = invoke_fresh(fresh_allocatable_import, 0xFFFFFFE0U);
    assert(result.reason == StopReason::instruction_limit);
    assert(fresh.state.gpr[3] == positive_32);
}

void gx2_init_hle_tests()
{
    constexpr std::uint32_t return_address = 0x0274BD9CU;
    constexpr std::uint32_t attributes_address = 0x1000U;
    constexpr std::uint32_t pool_address = 0x8000U;
    constexpr std::uint32_t pool_size = 0x8000U;
    constexpr std::uint32_t gx2_init_end = 0U;
    constexpr std::uint32_t gx2_init_command_buffer_base = 1U;
    constexpr std::uint32_t gx2_init_command_buffer_pool_size = 2U;
    constexpr std::uint32_t gx2_init_argc = 7U;
    constexpr std::uint32_t gx2_init_argv = 8U;
    constexpr std::uint32_t gx2_init_profile_mode = 9U;
    constexpr std::uint32_t gx2_init_toss_stage = 10U;
    constexpr std::uint32_t gx2_init_app_io_thread_stack_size = 11U;

    const auto write_stream = [](EspressoCore& core,
                                 std::uint32_t address,
                                 std::span<const std::uint32_t> words) {
        for (std::size_t index = 0; index < words.size(); ++index)
        {
            core.memory.write32_be(
                address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
                words[index]);
        }
    };
    const auto invoke = [&](EspressoCore& core,
                            std::uint32_t import,
                            std::uint32_t attributes) {
        core.state.gpr[3] = attributes;
        core.state.lr = return_address;
        core.state.cia = import;
        return core.run(1U);
    };

    EspressoCore core(0x20000U);
    register_gx2_hle(core.hle);
    const std::uint32_t gx2_init = core.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 9> wind_waker_attributes{
        gx2_init_command_buffer_base, pool_address,
        gx2_init_command_buffer_pool_size, pool_size,
        gx2_init_argc, 0U,
        gx2_init_argv, 0U,
        gx2_init_end};
    write_stream(core, attributes_address, wind_waker_attributes);
    const std::array<std::uint8_t, 36> attribute_bytes_before = [&] {
        std::array<std::uint8_t, 36> bytes{};
        core.memory.read_bytes(attributes_address, bytes);
        return bytes;
    }();
    core.memory.write8(pool_address, 0xA5U);
    core.memory.write8(pool_address + pool_size / 2U, 0x5AU);
    core.memory.write8(pool_address + pool_size - 1U, 0xC3U);
    const std::uint32_t initial_cursor = 0x18000U;
    core.configure_guest_heap(initial_cursor, 0x1F000U);
    core.current_thread_address = 0x1100U;
    core.fs_initialized = true;
    core.nn_save_initialized = true;
    core.base_heap_handles[1] = 0x1800U;
    core.mem2_heap_region_begin = 0x1900U;
    core.mem2_heap_region_end = 0x1A00U;
    for (std::uint32_t reg = 0; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = 0xA5000000U + reg * 0x101U;
    }
    for (std::uint32_t reg = 0; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = 0x1111000000000000ULL + reg;
        core.state.fpr_ps1[reg] = 0x2222000000000000ULL + reg;
    }
    core.state.cr = 0x12345678U;
    core.state.xer = 0xA00000A5U;
    core.state.ctr = 0xCAFEBABEU;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.gpr[3] = attributes_address;
    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const std::uint32_t cr_before = core.state.cr;
    const std::uint32_t xer_before = core.state.xer;
    const std::uint32_t ctr_before = core.state.ctr;
    const std::uint32_t fpscr_before = core.state.fpscr;
    const std::uint32_t lr_before = return_address;
    const std::uint32_t thread_before = core.current_thread_address;
    const auto handles_before = core.base_heap_handles;
    const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
    const std::uint32_t mem2_end_before = core.mem2_heap_region_end;
    assert(!core.gx2.initialized);
    const RunResult current_result = invoke(core, gx2_init, attributes_address);
    assert(current_result.reason == StopReason::instruction_limit);
    assert(current_result.steps == 1U);
    assert(core.gx2.initialized);
    assert(core.gx2.main_core_id == 1U);
    assert(core.gx2.command_buffer_pool_base == pool_address);
    assert(core.gx2.command_buffer_pool_size == pool_size);
    assert(!core.gx2.command_buffer_pool_owned);
    assert(core.gx2.argc == 0U && core.gx2.argv == 0U);
    assert(core.gx2.profile_mode == 0U && core.gx2.toss_stage == 0U);
    assert(core.gx2.app_io_thread_stack_size == 0x1000U);
    assert(core.gx2.gpu_timeout_ms == 10000U);
    assert(core.gx2.hang_state == 0U);
    assert(core.gx2.hang_response == 1U);
    assert(core.gx2.hang_reset_swap_timeout == 1000U);
    assert(core.gx2.hang_reset_swaps_outstanding == 3U);
    assert(core.gx2.swap_interval == 1U);
    assert(core.gx2.flip_request_count == 0U && core.gx2.flip_execute_count == 0U);
    assert(!core.gx2.db_stencilrefmask_valid && core.gx2.db_stencilrefmask == 0U);
    assert(!core.gx2.db_stencilrefmask_bf_valid && core.gx2.db_stencilrefmask_bf == 0U);
    assert(core.gx2.pending_commands.empty());
    assert(core.state.gpr == gprs_before);
    assert(core.state.fpr == fprs_before && core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before && core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before && core.state.fpscr == fpscr_before);
    assert(core.state.lr == lr_before && core.state.cia == return_address);
    assert(core.current_thread_address == thread_before);
    assert(core.guest_heap_cursor == initial_cursor);
    assert(core.base_heap_handles == handles_before);
    assert(core.mem2_heap_region_begin == mem2_begin_before);
    assert(core.mem2_heap_region_end == mem2_end_before);
    std::array<std::uint8_t, 36> attribute_bytes_after{};
    core.memory.read_bytes(attributes_address, attribute_bytes_after);
    assert(attribute_bytes_after == attribute_bytes_before);
    assert(core.memory.read8(pool_address) == 0xA5U);
    assert(core.memory.read8(pool_address + pool_size / 2U) == 0x5AU);
    assert(core.memory.read8(pool_address + pool_size - 1U) == 0xC3U);

    // Exact Wind Waker inputs live in sparse guest regions in this small
    // fixture, so the regression need not allocate the full default memory.
    EspressoCore wind_waker(0x1000U);
    constexpr std::uint32_t real_attributes_address = 0x100FEE20U;
    constexpr std::uint32_t real_pool_address = 0x06A23700U;
    constexpr std::uint32_t real_pool_size = 0x00400000U;
    wind_waker.memory.map_region(real_attributes_address, 9U * sizeof(std::uint32_t));
    wind_waker.memory.map_region(real_pool_address, real_pool_size);
    const std::array<std::uint32_t, 9> exact_wind_waker_attributes{
        gx2_init_command_buffer_base, real_pool_address,
        gx2_init_command_buffer_pool_size, real_pool_size,
        gx2_init_argc, 0U,
        gx2_init_argv, 0U,
        gx2_init_end};
    write_stream(wind_waker, real_attributes_address, exact_wind_waker_attributes);
    wind_waker.memory.write8(real_pool_address, 0xA1U);
    wind_waker.memory.write8(real_pool_address + real_pool_size / 2U, 0xB2U);
    wind_waker.memory.write8(real_pool_address + real_pool_size - 1U, 0xC3U);
    const auto exact_stream_before = [&] {
        std::array<std::uint8_t, 36> bytes{};
        wind_waker.memory.read_bytes(real_attributes_address, bytes);
        return bytes;
    }();
    register_gx2_hle(wind_waker.hle);
    const auto exact_import = wind_waker.hle.bind_import("gx2", "GX2Init");
    const auto coreinit_import = wind_waker.hle.bind_import("coreinit", "GX2Init");
    const std::uint32_t exact_cursor_before = wind_waker.guest_heap_cursor;
    const RunResult exact_result = invoke(
        wind_waker, exact_import, real_attributes_address);
    assert(exact_result.reason == StopReason::instruction_limit);
    assert(wind_waker.gx2.initialized);
    assert(wind_waker.gx2.main_core_id == 1U);
    assert(wind_waker.gx2.command_buffer_pool_base == 0x06A23700U);
    assert(wind_waker.gx2.command_buffer_pool_size == 0x00400000U);
    assert(static_cast<std::uint64_t>(wind_waker.gx2.command_buffer_pool_base) +
           wind_waker.gx2.command_buffer_pool_size == 0x06E23700ULL);
    assert(!wind_waker.gx2.command_buffer_pool_owned);
    assert(wind_waker.gx2.argc == 0U && wind_waker.gx2.argv == 0U);
    assert(wind_waker.gx2.profile_mode == 0U && wind_waker.gx2.toss_stage == 0U);
    assert(wind_waker.gx2.app_io_thread_stack_size == 0x1000U);
    assert(wind_waker.state.gpr[3] == real_attributes_address);
    assert(wind_waker.state.cia == return_address);
    assert(wind_waker.guest_heap_cursor == exact_cursor_before);
    std::array<std::uint8_t, 36> exact_stream_after{};
    wind_waker.memory.read_bytes(real_attributes_address, exact_stream_after);
    assert(exact_stream_after == exact_stream_before);
    assert(wind_waker.memory.read8(real_pool_address) == 0xA1U);
    assert(wind_waker.memory.read8(real_pool_address + real_pool_size / 2U) == 0xB2U);
    assert(wind_waker.memory.read8(real_pool_address + real_pool_size - 1U) == 0xC3U);
    wind_waker.state.cia = coreinit_import;
    wind_waker.state.lr = return_address + 4U;
    const RunResult wrong_library = wind_waker.run(1U);
    assert(wrong_library.reason == StopReason::unimplemented_hle_call);
    assert(wrong_library.hle_call == "coreinit::GX2Init");

    // An explicit base with no size attribute keeps the documented 4 MiB
    // default rather than interpreting absence as a zero-sized pool.
    EspressoCore default_size(0x410000U);
    register_gx2_hle(default_size.hle);
    const auto default_size_import = default_size.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 3> base_only_attributes{
        gx2_init_command_buffer_base, 0x8000U, gx2_init_end};
    write_stream(default_size, 0x100U, base_only_attributes);
    assert(invoke(default_size, default_size_import, 0x100U).reason ==
           StopReason::instruction_limit);
    assert(default_size.gx2.command_buffer_pool_base == 0x8000U);
    assert(default_size.gx2.command_buffer_pool_size == 0x00400000U);

    // Cemu's repeated-init policy is a successful no-op, even for a bad r3.
    const Gx2RuntimeState initialized_state = core.gx2;
    core.state.gpr[3] = 0xFFFFFFFFU;
    core.state.cia = gx2_init;
    core.state.lr = return_address + 4U;
    assert(core.step() == StepResult::executed);
    assert(core.gx2.initialized);
    assert(core.gx2.command_buffer_pool_base == initialized_state.command_buffer_pool_base);
    assert(core.gx2.command_buffer_pool_size == initialized_state.command_buffer_pool_size);
    assert(core.state.gpr[3] == 0xFFFFFFFFU);
    assert(core.state.cia == return_address + 4U);

    // Attributes are sequential, duplicate known IDs replace earlier values,
    // unknown pairs are skipped, and argv remains opaque.
    EspressoCore parsed(0x30000U);
    register_gx2_hle(parsed.hle);
    const auto parsed_import = parsed.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 19> richer_attributes{
        0x12345678U, 0xDEADBEEFU,
        gx2_init_command_buffer_base, 0x8000U,
        gx2_init_command_buffer_pool_size, 0x1000U,
        gx2_init_command_buffer_pool_size, 0x2000U,
        gx2_init_argc, 123U,
        gx2_init_argv, 0xDEADBEEFU,
        gx2_init_profile_mode, 4U,
        gx2_init_toss_stage, 9U,
        gx2_init_app_io_thread_stack_size, 0x8000U,
        gx2_init_end};
    write_stream(parsed, 0x100U, richer_attributes);
    const RunResult parsed_result = invoke(parsed, parsed_import, 0x100U);
    assert(parsed_result.reason == StopReason::instruction_limit);
    assert(parsed.gx2.command_buffer_pool_base == 0x8000U);
    assert(parsed.gx2.command_buffer_pool_size == 0x2000U); // Decaf minimum
    assert(!parsed.gx2.command_buffer_pool_owned);
    assert(parsed.gx2.argc == 123U && parsed.gx2.argv == 0xDEADBEEFU);
    assert(parsed.gx2.profile_mode == 4U && parsed.gx2.toss_stage == 9U);
    assert(parsed.gx2.app_io_thread_stack_size == 0x8000U);

    // A null stream uses defaults and the existing guest bump allocator, with
    // ownership recorded in the session state and no host allocation.
    EspressoCore fallback(0x500000U);
    register_gx2_hle(fallback.hle);
    const auto fallback_import = fallback.hle.bind_import("gx2", "GX2Init");
    fallback.configure_guest_heap(0x1000U, 0x480000U);
    fallback.memory.write8(0x1000U, 0xD1U);
    fallback.memory.write8(0x201000U, 0xD2U);
    fallback.memory.write8(0x400FFFU, 0xD3U);
    const RunResult fallback_result = invoke(fallback, fallback_import, 0U);
    assert(fallback_result.reason == StopReason::instruction_limit);
    assert(fallback.gx2.initialized);
    assert(fallback.gx2.command_buffer_pool_base == 0x1000U);
    assert(fallback.gx2.command_buffer_pool_size == 0x00400000U);
    assert(fallback.gx2.command_buffer_pool_owned);
    assert(fallback.guest_heap_cursor == 0x401000U);
    assert((fallback.gx2.command_buffer_pool_base & 0xFFU) == 0U);
    assert(fallback.gx2.argc == 0U && fallback.gx2.argv == 0U);
    assert(fallback.memory.read8(0x1000U) == 0xD1U);
    assert(fallback.memory.read8(0x201000U) == 0xD2U);
    assert(fallback.memory.read8(0x400FFFU) == 0xD3U);

    // A custom size without a base also uses the 0x100-aligned bump allocator.
    EspressoCore custom_fallback(0x10000U);
    register_gx2_hle(custom_fallback.hle);
    const auto custom_import = custom_fallback.hle.bind_import("gx2", "GX2Init");
    custom_fallback.configure_guest_heap(0x1001U, 0x9000U);
    const std::array<std::uint32_t, 3> custom_size_attributes{
        gx2_init_command_buffer_pool_size, 0x3000U, gx2_init_end};
    write_stream(custom_fallback, 0x100U, custom_size_attributes);
    const RunResult custom_result = invoke(custom_fallback, custom_import, 0x100U);
    assert(custom_result.reason == StopReason::instruction_limit);
    assert(custom_fallback.gx2.command_buffer_pool_base == 0x1100U);
    assert(custom_fallback.gx2.command_buffer_pool_size == 0x3000U);
    assert(custom_fallback.gx2.command_buffer_pool_owned);
    assert(custom_fallback.guest_heap_cursor == 0x4100U);

    // The END word is the final readable word: no value or following word is
    // accessed after termination.
    EspressoCore final_end(0x100U);
    final_end.memory.map_region(0x2000U, 4U);
    final_end.memory.map_region(0x3000U, 0x400000U);
    final_end.memory.write32_be(0x2000U, gx2_init_end);
    final_end.configure_guest_heap(0x3000U, 0x403000U);
    register_gx2_hle(final_end.hle);
    const auto final_end_import = final_end.hle.bind_import("gx2", "GX2Init");
    const RunResult final_end_result = invoke(final_end, final_end_import, 0x2000U);
    assert(final_end_result.reason == StopReason::instruction_limit);
    assert(final_end.gx2.initialized);

    const auto expect_init_error = [&](EspressoCore& failing,
                                       std::uint32_t import,
                                       std::uint32_t attributes) {
        const Gx2RuntimeState before = failing.gx2;
        const std::uint32_t cursor_before = failing.guest_heap_cursor;
        const RunResult result = invoke(failing, import, attributes);
        assert(result.reason == StopReason::hle_error);
        assert(result.detail.find("gx2::GX2Init") != std::string::npos);
        assert(!failing.gx2.initialized);
        assert(failing.gx2.command_buffer_pool_base == before.command_buffer_pool_base);
        assert(failing.gx2.command_buffer_pool_size == before.command_buffer_pool_size);
        assert(failing.guest_heap_cursor == cursor_before);
    };

    EspressoCore no_heap(0x10000U);
    register_gx2_hle(no_heap.hle);
    const auto no_heap_import = no_heap.hle.bind_import("gx2", "GX2Init");
    no_heap.configure_guest_heap(0x1000U, 0x2000U);
    expect_init_error(no_heap, no_heap_import, 0U);

    EspressoCore bad_pool(0x10000U);
    register_gx2_hle(bad_pool.hle);
    const auto bad_pool_import = bad_pool.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 5> bad_pool_attributes{
        gx2_init_command_buffer_base, 0xF000U,
        gx2_init_command_buffer_pool_size, 0x2000U,
        gx2_init_end};
    write_stream(bad_pool, 0x100U, bad_pool_attributes);
    expect_init_error(bad_pool, bad_pool_import, 0x100U);

    EspressoCore wrapping_pool(0x10000U);
    register_gx2_hle(wrapping_pool.hle);
    const auto wrapping_import = wrapping_pool.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 5> wrapping_pool_attributes{
        gx2_init_command_buffer_base, 0xFFFFF000U,
        gx2_init_command_buffer_pool_size, 0x2000U,
        gx2_init_end};
    write_stream(wrapping_pool, 0x100U, wrapping_pool_attributes);
    expect_init_error(wrapping_pool, wrapping_import, 0x100U);

    EspressoCore misaligned_pool(0x10000U);
    register_gx2_hle(misaligned_pool.hle);
    const auto misaligned_import = misaligned_pool.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 5> misaligned_attributes{
        gx2_init_command_buffer_base, 0x8004U,
        gx2_init_command_buffer_pool_size, 0x2000U,
        gx2_init_end};
    write_stream(misaligned_pool, 0x100U, misaligned_attributes);
    expect_init_error(misaligned_pool, misaligned_import, 0x100U);

    EspressoCore partial_pair(0x2000U);
    register_gx2_hle(partial_pair.hle);
    const auto partial_import = partial_pair.hle.bind_import("gx2", "GX2Init");
    partial_pair.memory.write32_be(0x1FFCU, gx2_init_command_buffer_base);
    expect_init_error(partial_pair, partial_import, 0x1FFCU);

    EspressoCore wrapping_attributes(0x100U);
    wrapping_attributes.memory.map_region(0xFFFFFFFCU, 4U);
    wrapping_attributes.memory.write32_be(0xFFFFFFFCU, gx2_init_command_buffer_base);
    register_gx2_hle(wrapping_attributes.hle);
    const auto wrapping_attributes_import =
        wrapping_attributes.hle.bind_import("gx2", "GX2Init");
    expect_init_error(wrapping_attributes, wrapping_attributes_import, 0xFFFFFFFCU);

    EspressoCore unterminated(0x1000U);
    register_gx2_hle(unterminated.hle);
    const auto unterminated_import = unterminated.hle.bind_import("gx2", "GX2Init");
    std::array<std::uint32_t, 2U * 65U> pairs{};
    for (std::size_t index = 0; index < pairs.size(); index += 2U)
    {
        pairs[index] = 0x12345678U;
        pairs[index + 1U] = static_cast<std::uint32_t>(index);
    }
    write_stream(unterminated, 0x100U, pairs);
    expect_init_error(unterminated, unterminated_import, 0x100U);

    // Explicit pool validation and failed fallback allocation are transactional.
    EspressoCore fallback_failure(0x8000U);
    register_gx2_hle(fallback_failure.hle);
    const auto fallback_failure_import = fallback_failure.hle.bind_import("gx2", "GX2Init");
    fallback_failure.configure_guest_heap(0x1000U, 0x2000U);
    expect_init_error(fallback_failure, fallback_failure_import, 0U);

    EspressoCore unmapped_fallback(0x1000U);
    register_gx2_hle(unmapped_fallback.hle);
    const auto unmapped_fallback_import = unmapped_fallback.hle.bind_import("gx2", "GX2Init");
    unmapped_fallback.configure_guest_heap(0x2000U, 0x500000U);
    expect_init_error(unmapped_fallback, unmapped_fallback_import, 0U);

    // Reset and independent cores own independent lifecycle state.
    core.reset();
    assert(!core.gx2.initialized);
    assert(core.gx2.command_buffer_pool_base == 0U);
    assert(core.gx2.command_buffer_pool_size == 0U);
    assert(!core.gx2.command_buffer_pool_owned);

    EspressoCore independent(0x20000U);
    register_gx2_hle(independent.hle);
    const auto independent_import = independent.hle.bind_import("gx2", "GX2Init");
    const std::array<std::uint32_t, 5> independent_attributes{
        gx2_init_command_buffer_base, 0x10000U,
        gx2_init_command_buffer_pool_size, 0x2000U,
        gx2_init_end};
    write_stream(independent, 0x100U, independent_attributes);
    assert(invoke(independent, independent_import, 0x100U).reason ==
           StopReason::instruction_limit);
    assert(independent.gx2.initialized);
    assert(!core.gx2.initialized);

    // Verify normal Emulator/RPX import registration, rather than relying only
    // on direct HLE registration in these focused unit tests.
    RpxImportOptions import_options;
    import_options.library = "gx2";
    RpxRelocationSymbol import_symbol{
        "GX2Init", import_options.section_address + import_options.slot_offset,
        1U, 2U, 8U, 0};
    RpxRelocationOptions relocation;
    relocation.type = 10U;
    relocation.offset = 4U;
    relocation.branch_instruction = 0x48000001U;
    const auto import_rpx = make_minimal_compressed_rpx(
        0x02000000U, 16U, import_symbol, relocation, import_options);
    affogato::Emulator emulator;
    static_cast<void>(emulator.load_rpx(import_rpx));
    const auto emulator_result = emulator.run(3U);
    assert(emulator_result.execution.steps == 3U);
    assert(emulator_result.execution.reason == StopReason::instruction_limit);
    assert(emulator.core().gx2.initialized);
    assert(emulator.core().gx2.command_buffer_pool_owned);
    assert(emulator.core().state.gpr[3] == 5U);
    assert(emulator.core().gx2.command_buffer_pool_size == 0x00400000U);
}

void gx2_set_depth_stencil_control_hle_tests()
{
    constexpr std::uint32_t return_address = 0x027507BCU;
    constexpr std::uint32_t stack_address = 0x3000U;
    constexpr std::uint32_t pool_address = 0x8000U;
    constexpr std::uint32_t db_depth_control_address = 0x00028800U;

    const auto invoke = [](EspressoCore& core, std::uint32_t import) {
        core.state.cia = import;
        core.state.lr = return_address;
        return core.run(1U);
    };
    const auto write_stack_arguments = [](EspressoCore& core,
                                          std::uint32_t address,
                                          const std::array<std::uint32_t, 5>& values) {
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            core.memory.write32_be(
                address + 0x08U + static_cast<std::uint32_t>(index * 4U),
                values[index]);
        }
    };
    const auto set_first_eight_arguments = [](EspressoCore& core,
                                               const std::array<std::uint32_t, 8>& values) {
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            core.state.gpr[index + 3U] = values[index];
        }
    };

    EspressoCore core(0x20000U);
    register_gx2_hle(core.hle);
    const std::uint32_t gx2_init = core.hle.bind_import("gx2", "GX2Init");
    const std::uint32_t setter = core.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    const std::uint32_t wrong_library = core.hle.bind_import(
        "coreinit", "GX2SetDepthStencilControl");

    // Initialize GX2 through its real HLE before testing the setter.
    core.memory.write32_be(0x100U, 1U);
    core.memory.write32_be(0x104U, pool_address);
    core.memory.write32_be(0x108U, 2U);
    core.memory.write32_be(0x10CU, 0x8000U);
    core.memory.write32_be(0x110U, 0U);
    core.state.gpr[3] = 0x100U;
    assert(invoke(core, gx2_init).reason == StopReason::instruction_limit);
    assert(core.gx2.initialized);

    // The setter may append a modeled command, but must not alter GX2
    // initialization, pool ownership, or any other lifecycle state.
    core.gx2.pending_commands.push_back({
        Gx2CommandType::set_context_register, 0x00028804U, 0xCAFEBABEU});
    core.memory.write8(pool_address, 0xA5U);
    core.memory.write8(pool_address + 0x4000U, 0x5AU);
    core.memory.write8(pool_address + 0x7FFFU, 0xC3U);
    core.state.gpr[1] = stack_address;
    write_stack_arguments(core, stack_address, {0U, 0U, 0U, 0U, 0U});
    for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
    {
        core.state.gpr[index] = 0xA5000000U + static_cast<std::uint32_t>(index) * 0x101U;
    }
    for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
    {
        core.state.fpr[index] = 0x1111000000000000ULL + index;
        core.state.fpr_ps1[index] = 0x2222000000000000ULL + index;
    }
    core.state.gpr[1] = stack_address;
    set_first_eight_arguments(core, {1U, 1U, 3U, 0U, 0U, 0U, 0U, 0U});
    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const std::uint32_t cr_before = core.state.cr = 0x12345678U;
    const std::uint32_t xer_before = core.state.xer = 0xA5A55A5AU;
    const std::uint32_t ctr_before = core.state.ctr = 0x87654321U;
    const std::uint32_t fpscr_before = core.state.fpscr = 0x5A5AA55AU;
    const std::uint32_t lr_before = return_address;
    const Gx2RuntimeState lifecycle_before = core.gx2;
    std::vector<std::uint8_t> guest_memory_before(core.memory.size());
    core.memory.read_bytes(0U, guest_memory_before);

    const RunResult wind_waker_result = invoke(core, setter);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert(wind_waker_result.steps == 1U);
    assert(core.gx2.db_depth_control_valid);
    assert(core.gx2.db_depth_control == 0x00000036U);
    assert(core.gx2.pending_commands.size() == 2U);
    assert(core.gx2.pending_commands[0].register_address == 0x00028804U);
    assert(core.gx2.pending_commands[0].value == 0xCAFEBABEU);
    assert(core.gx2.pending_commands[1].type == Gx2CommandType::set_context_register);
    assert(core.gx2.pending_commands[1].register_address == db_depth_control_address);
    assert(core.gx2.pending_commands[1].value == 0x00000036U);
    assert(core.state.gpr == gprs_before);
    assert(core.state.fpr == fprs_before && core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before && core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before && core.state.fpscr == fpscr_before);
    assert(core.state.lr == lr_before && core.state.cia == return_address);
    assert(core.gx2.initialized == lifecycle_before.initialized);
    assert(core.gx2.main_core_id == lifecycle_before.main_core_id);
    assert(core.gx2.command_buffer_pool_base == lifecycle_before.command_buffer_pool_base);
    assert(core.gx2.command_buffer_pool_size == lifecycle_before.command_buffer_pool_size);
    assert(core.gx2.command_buffer_pool_owned == lifecycle_before.command_buffer_pool_owned);
    assert(core.gx2.argc == lifecycle_before.argc && core.gx2.argv == lifecycle_before.argv);
    assert(core.gx2.profile_mode == lifecycle_before.profile_mode);
    assert(core.gx2.toss_stage == lifecycle_before.toss_stage);
    assert(core.gx2.app_io_thread_stack_size == lifecycle_before.app_io_thread_stack_size);
    assert(core.gx2.gpu_timeout_ms == lifecycle_before.gpu_timeout_ms);
    assert(core.gx2.hang_state == lifecycle_before.hang_state);
    assert(core.gx2.hang_response == lifecycle_before.hang_response);
    assert(core.gx2.hang_reset_swap_timeout == lifecycle_before.hang_reset_swap_timeout);
    assert(core.gx2.hang_reset_swaps_outstanding ==
           lifecycle_before.hang_reset_swaps_outstanding);
    assert(core.gx2.swap_interval == lifecycle_before.swap_interval);
    assert(core.gx2.flip_request_count == lifecycle_before.flip_request_count);
    assert(core.gx2.flip_execute_count == lifecycle_before.flip_execute_count);
    assert(core.memory.read8(pool_address) == 0xA5U);
    assert(core.memory.read8(pool_address + 0x4000U) == 0x5AU);
    assert(core.memory.read8(pool_address + 0x7FFFU) == 0xC3U);
    std::vector<std::uint8_t> guest_memory_after(core.memory.size());
    core.memory.read_bytes(0U, guest_memory_after);
    assert(guest_memory_after == guest_memory_before);

    // Distinct field values make the stack-argument order observable. In
    // particular, the API presents ZPass/ZFail/Fail while the packed register
    // stores Fail/ZPass/ZFail.
    set_first_eight_arguments(core, {
        1U, 0U, 0xFFFFFFFEU, 1U, 1U, 0xFFFFFFFDU, 6U, 7U});
    write_stack_arguments(core, stack_address, {
        0xFFFFFFFAU, 0xFFFFFFFCU, 0xFFFFFFFBU, 0xFFFFFFFCU, 0xFFFFFFFDU});
    const RunResult full_field_result = invoke(core, setter);
    assert(full_field_result.reason == StopReason::instruction_limit);
    assert(core.gx2.db_depth_control == 0x8ECF95E3U);
    assert(core.gx2.pending_commands.size() == 3U);
    assert(core.gx2.pending_commands[2].register_address == db_depth_control_address);
    assert(core.gx2.pending_commands[2].value == 0x8ECF95E3U);

    // Repeating an identical call still emits an ordered command; it is not
    // coalesced into the state mirror.
    const RunResult repeated_result = invoke(core, setter);
    assert(repeated_result.reason == StopReason::instruction_limit);
    assert(core.gx2.pending_commands.size() == 4U);
    assert(core.gx2.pending_commands[3].value == 0x8ECF95E3U);
    assert(core.gx2.db_depth_control == 0x8ECF95E3U);

    // BOOL arguments normalize any nonzero value to one.
    set_first_eight_arguments(core, {0xFFFFFFFFU, 2U, 0U, 3U, 4U, 0U, 0U, 0U});
    write_stack_arguments(core, stack_address, {0U, 0U, 0U, 0U, 0U});
    assert(invoke(core, setter).reason == StopReason::instruction_limit);
    assert(core.gx2.db_depth_control == 0x00000087U);

    // A second EspressoCore has its own state and command sequence.
    EspressoCore independent(0x10000U);
    register_gx2_hle(independent.hle);
    const std::uint32_t independent_setter = independent.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    independent.gx2.initialized = true;
    independent.state.gpr[1] = 0x1000U;
    write_stack_arguments(independent, 0x1000U, {0U, 0U, 0U, 0U, 0U});
    assert(invoke(independent, independent_setter).reason == StopReason::instruction_limit);
    assert(independent.gx2.pending_commands.size() == 1U);
    assert(independent.gx2.db_depth_control_valid);
    assert(core.gx2.pending_commands.size() == 5U);

    // Initialization is mandatory and checked before even consulting the
    // caller's stack pointer or changing the command/register state.
    EspressoCore uninitialized(0x1000U);
    register_gx2_hle(uninitialized.hle);
    const std::uint32_t uninitialized_setter = uninitialized.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    uninitialized.state.gpr[1] = 0xFFFFFFFFU;
    uninitialized.state.gpr[3] = 0xDEADBEEFU;
    const auto uninitialized_gprs = uninitialized.state.gpr;
    const RunResult uninitialized_result = invoke(uninitialized, uninitialized_setter);
    assert(uninitialized_result.reason == StopReason::hle_error);
    assert(uninitialized_result.detail.find("gx2::GX2SetDepthStencilControl") !=
           std::string::npos);
    assert(uninitialized.state.gpr == uninitialized_gprs);
    assert(!uninitialized.gx2.db_depth_control_valid);
    assert(uninitialized.gx2.pending_commands.empty());

    // Guest-memory faults while reading stack arguments happen before any GX2
    // mutation. The final word crosses out of this core's mapped flat memory.
    EspressoCore unmapped_stack(0x20000U);
    register_gx2_hle(unmapped_stack.hle);
    const std::uint32_t unmapped_setter = unmapped_stack.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    unmapped_stack.gx2.initialized = true;
    unmapped_stack.gx2.db_depth_control_valid = true;
    unmapped_stack.gx2.db_depth_control = 0x12345678U;
    unmapped_stack.gx2.pending_commands.push_back({
        Gx2CommandType::set_context_register, 0x28800U, 0x12345678U});
    const Gx2RuntimeState unmapped_gx2_before = unmapped_stack.gx2;
    unmapped_stack.state.gpr[1] = 0x1FFF0U;
    set_first_eight_arguments(unmapped_stack, {1U, 1U, 3U, 1U, 1U, 1U, 1U, 1U});
    const auto unmapped_gprs = unmapped_stack.state.gpr;
    const RunResult unmapped_result = invoke(unmapped_stack, unmapped_setter);
    assert(unmapped_result.reason == StopReason::memory_fault);
    assert(unmapped_stack.state.gpr == unmapped_gprs);
    assert(unmapped_stack.gx2.db_depth_control_valid ==
           unmapped_gx2_before.db_depth_control_valid);
    assert(unmapped_stack.gx2.db_depth_control == unmapped_gx2_before.db_depth_control);
    assert(unmapped_stack.gx2.pending_commands.size() ==
           unmapped_gx2_before.pending_commands.size());
    assert(unmapped_stack.gx2.pending_commands[0].value ==
           unmapped_gx2_before.pending_commands[0].value);

    EspressoCore wrapping_stack(0x1000U);
    register_gx2_hle(wrapping_stack.hle);
    const std::uint32_t wrapping_setter = wrapping_stack.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    wrapping_stack.gx2.initialized = true;
    wrapping_stack.gx2.db_depth_control_valid = true;
    wrapping_stack.gx2.db_depth_control = 0x87654321U;
    wrapping_stack.gx2.pending_commands.push_back({
        Gx2CommandType::set_context_register, 0x28800U, 0x87654321U});
    const Gx2RuntimeState wrapping_gx2_before = wrapping_stack.gx2;
    wrapping_stack.state.gpr[1] = 0xFFFFFFF0U;
    const auto wrapping_gprs = wrapping_stack.state.gpr;
    const RunResult wrapping_result = invoke(wrapping_stack, wrapping_setter);
    assert(wrapping_result.reason == StopReason::hle_error);
    assert(wrapping_result.detail.find("wrap guest address space") != std::string::npos);
    assert(wrapping_stack.state.gpr == wrapping_gprs);
    assert(wrapping_stack.gx2.db_depth_control_valid ==
           wrapping_gx2_before.db_depth_control_valid);
    assert(wrapping_stack.gx2.db_depth_control == wrapping_gx2_before.db_depth_control);
    assert(wrapping_stack.gx2.pending_commands.size() ==
           wrapping_gx2_before.pending_commands.size());
    assert(wrapping_stack.gx2.pending_commands[0].value ==
           wrapping_gx2_before.pending_commands[0].value);

    // Import names are exact: this symbol belongs to gx2, not coreinit.
    core.state.cia = wrong_library;
    core.state.lr = return_address;
    const RunResult wrong_library_result = core.run(1U);
    assert(wrong_library_result.reason == StopReason::unimplemented_hle_call);
    assert(wrong_library_result.hle_call == "coreinit::GX2SetDepthStencilControl");

    core.reset();
    assert(!core.gx2.db_depth_control_valid);
    assert(core.gx2.db_depth_control == 0U);
    assert(core.gx2.pending_commands.empty());
}

void gx2_set_stencil_mask_hle_tests()
{
    constexpr std::uint32_t return_address = 0x027507D8U;
    constexpr std::uint32_t pool_address = 0x8000U;
    constexpr std::uint32_t db_depth_control_address = 0x00028800U;
    constexpr std::uint32_t db_stencilrefmask_address = 0x00028430U;
    constexpr std::uint32_t db_stencilrefmask_bf_address = 0x00028434U;

    const auto invoke = [](EspressoCore& core, std::uint32_t import) {
        core.state.cia = import;
        core.state.lr = return_address;
        return core.run(1U);
    };
    const auto set_arguments = [](EspressoCore& core,
                                  const std::array<std::uint32_t, 6>& values) {
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            core.state.gpr[index + 3U] = values[index];
        }
    };

    EspressoCore core(0x20000U);
    register_gx2_hle(core.hle);
    const std::uint32_t gx2_init = core.hle.bind_import("gx2", "GX2Init");
    const std::uint32_t depth_import = core.hle.bind_import(
        "gx2", "GX2SetDepthStencilControl");
    const std::uint32_t stencil_import = core.hle.bind_import(
        "gx2", "GX2SetStencilMask");
    const std::uint32_t wrong_library = core.hle.bind_import(
        "coreinit", "GX2SetStencilMask");
    core.memory.write32_be(0x100U, 1U);
    core.memory.write32_be(0x104U, pool_address);
    core.memory.write32_be(0x108U, 2U);
    core.memory.write32_be(0x10CU, 0x8000U);
    core.memory.write32_be(0x110U, 0U);
    core.state.gpr[3] = 0x100U;
    assert(invoke(core, gx2_init).reason == StopReason::instruction_limit);
    assert(core.gx2.initialized);
    assert(!core.gx2.db_stencilrefmask_valid);
    assert(!core.gx2.db_stencilrefmask_bf_valid);
    assert(core.gx2.pending_commands.empty());

    // Reproduce the preceding depth-control write so the stencil setter must
    // preserve the already queued command and its register mirror.
    constexpr std::uint32_t stack_address = 0x3000U;
    core.state.gpr[1] = stack_address;
    set_arguments(core, {1U, 1U, 3U, 0U, 0U, 0U});
    core.state.gpr[9] = 0U;
    core.state.gpr[10] = 0U;
    core.memory.write32_be(stack_address + 0x08U, 0U);
    core.memory.write32_be(stack_address + 0x0CU, 0U);
    core.memory.write32_be(stack_address + 0x10U, 0U);
    core.memory.write32_be(stack_address + 0x14U, 0U);
    core.memory.write32_be(stack_address + 0x18U, 0U);
    assert(invoke(core, depth_import).reason == StopReason::instruction_limit);
    assert(core.gx2.db_depth_control_valid);
    assert(core.gx2.db_depth_control == 0x00000036U);
    assert(core.gx2.pending_commands.size() == 1U);
    assert(core.gx2.pending_commands[0].register_address == db_depth_control_address);
    assert(core.gx2.pending_commands[0].value == 0x00000036U);

    core.memory.write8(pool_address, 0xA5U);
    core.memory.write8(pool_address + 0x4000U, 0x5AU);
    core.memory.write8(pool_address + 0x7FFFU, 0xC3U);
    for (std::size_t index = 0; index < core.state.gpr.size(); ++index)
    {
        core.state.gpr[index] = 0xB6000000U + static_cast<std::uint32_t>(index) * 0x101U;
    }
    for (std::size_t index = 0; index < core.state.fpr.size(); ++index)
    {
        core.state.fpr[index] = 0x3333000000000000ULL + index;
        core.state.fpr_ps1[index] = 0x4444000000000000ULL + index;
    }
    core.state.cr = 0x89ABCDEFU;
    core.state.xer = 0xA5A55A5AU;
    core.state.ctr = 0x87654321U;
    core.state.fpscr = 0x5A5AA55AU;
    core.state.lr = return_address;
    core.state.gpr[1] = 0xFFFFFFFFU; // The six-register API must not touch the stack.
    set_arguments(core, {0xFFU, 0xFFU, 0U, 0xFFU, 0xFFU, 0U});

    const auto gprs_before = core.state.gpr;
    const auto fprs_before = core.state.fpr;
    const auto ps1_before = core.state.fpr_ps1;
    const std::uint32_t cr_before = core.state.cr;
    const std::uint32_t xer_before = core.state.xer;
    const std::uint32_t ctr_before = core.state.ctr;
    const std::uint32_t fpscr_before = core.state.fpscr;
    const std::uint32_t lr_before = core.state.lr;
    const Gx2RuntimeState gx2_before = core.gx2;
    const std::uint32_t heap_cursor_before = core.guest_heap_cursor;
    const std::uint32_t heap_limit_before = core.guest_heap_limit;
    const auto heap_handles_before = core.base_heap_handles;
    const std::uint32_t mem2_begin_before = core.mem2_heap_region_begin;
    const std::uint32_t mem2_end_before = core.mem2_heap_region_end;
    const bool fs_initialized_before = core.fs_initialized;
    const auto fs_clients_before = core.fs_clients;
    const bool save_initialized_before = core.nn_save_initialized;
    std::vector<std::uint8_t> memory_before(core.memory.size());
    core.memory.read_bytes(0U, memory_before);

    const RunResult wind_waker_result = invoke(core, stencil_import);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert(wind_waker_result.steps == 1U);
    assert(core.gx2.db_stencilrefmask_valid);
    assert(core.gx2.db_stencilrefmask == 0x00FFFF00U);
    assert(core.gx2.db_stencilrefmask_bf_valid);
    assert(core.gx2.db_stencilrefmask_bf == 0x00FFFF00U);
    assert(core.gx2.db_depth_control_valid == gx2_before.db_depth_control_valid);
    assert(core.gx2.db_depth_control == 0x00000036U);
    assert(core.gx2.pending_commands.size() == 3U);
    assert(core.gx2.pending_commands[0].register_address == db_depth_control_address);
    assert(core.gx2.pending_commands[0].value == 0x00000036U);
    assert(core.gx2.pending_commands[1].type == Gx2CommandType::set_context_register);
    assert(core.gx2.pending_commands[1].register_address == db_stencilrefmask_address);
    assert(core.gx2.pending_commands[1].value == 0x00FFFF00U);
    assert(core.gx2.pending_commands[2].type == Gx2CommandType::set_context_register);
    assert(core.gx2.pending_commands[2].register_address == db_stencilrefmask_bf_address);
    assert(core.gx2.pending_commands[2].value == 0x00FFFF00U);
    assert(core.state.gpr == gprs_before);
    assert(core.state.fpr == fprs_before && core.state.fpr_ps1 == ps1_before);
    assert(core.state.cr == cr_before && core.state.xer == xer_before);
    assert(core.state.ctr == ctr_before && core.state.fpscr == fpscr_before);
    assert(core.state.lr == lr_before && core.state.cia == return_address);
    assert(core.gx2.initialized == gx2_before.initialized);
    assert(core.gx2.main_core_id == gx2_before.main_core_id);
    assert(core.gx2.command_buffer_pool_base == gx2_before.command_buffer_pool_base);
    assert(core.gx2.command_buffer_pool_size == gx2_before.command_buffer_pool_size);
    assert(core.gx2.command_buffer_pool_owned == gx2_before.command_buffer_pool_owned);
    assert(core.gx2.argc == gx2_before.argc && core.gx2.argv == gx2_before.argv);
    assert(core.gx2.profile_mode == gx2_before.profile_mode);
    assert(core.gx2.toss_stage == gx2_before.toss_stage);
    assert(core.gx2.app_io_thread_stack_size == gx2_before.app_io_thread_stack_size);
    assert(core.gx2.gpu_timeout_ms == gx2_before.gpu_timeout_ms);
    assert(core.gx2.hang_state == gx2_before.hang_state);
    assert(core.gx2.hang_response == gx2_before.hang_response);
    assert(core.gx2.hang_reset_swap_timeout == gx2_before.hang_reset_swap_timeout);
    assert(core.gx2.hang_reset_swaps_outstanding ==
           gx2_before.hang_reset_swaps_outstanding);
    assert(core.gx2.swap_interval == gx2_before.swap_interval);
    assert(core.gx2.flip_request_count == gx2_before.flip_request_count);
    assert(core.gx2.flip_execute_count == gx2_before.flip_execute_count);
    assert(core.guest_heap_cursor == heap_cursor_before);
    assert(core.guest_heap_limit == heap_limit_before);
    assert(core.base_heap_handles == heap_handles_before);
    assert(core.mem2_heap_region_begin == mem2_begin_before);
    assert(core.mem2_heap_region_end == mem2_end_before);
    assert(core.fs_initialized == fs_initialized_before);
    assert(core.fs_clients.size() == fs_clients_before.size());
    for (std::size_t index = 0; index < fs_clients_before.size(); ++index)
    {
        assert(core.fs_clients[index].client_address ==
               fs_clients_before[index].client_address);
        assert(core.fs_clients[index].body_address == fs_clients_before[index].body_address);
        assert(core.fs_clients[index].synthetic_handle ==
               fs_clients_before[index].synthetic_handle);
    }
    assert(core.nn_save_initialized == save_initialized_before);
    assert(core.memory.read8(pool_address) == 0xA5U);
    assert(core.memory.read8(pool_address + 0x4000U) == 0x5AU);
    assert(core.memory.read8(pool_address + 0x7FFFU) == 0xC3U);
    std::vector<std::uint8_t> memory_after(core.memory.size());
    core.memory.read_bytes(0U, memory_after);
    assert(memory_after == memory_before);

    // Asymmetric inputs verify uint8 truncation, function-vs-register field
    // ordering, and that front/back do not get swapped.
    set_arguments(core, {
        0x12345612U, 0x87654334U, 0xABCDEF56U,
        0x111111ABU, 0x222222CDU, 0x333333EFU});
    assert(invoke(core, stencil_import).reason == StopReason::instruction_limit);
    assert(core.gx2.db_stencilrefmask == 0x00341256U);
    assert(core.gx2.db_stencilrefmask_bf == 0x00CDABEFU);
    assert(core.gx2.pending_commands.size() == 5U);
    assert(core.gx2.pending_commands[3].register_address == db_stencilrefmask_address);
    assert(core.gx2.pending_commands[3].value == 0x00341256U);
    assert(core.gx2.pending_commands[4].register_address == db_stencilrefmask_bf_address);
    assert(core.gx2.pending_commands[4].value == 0x00CDABEFU);

    // Zero and all-ones are valid register contents, not validity sentinels.
    set_arguments(core, {0U, 0U, 0U, 0U, 0U, 0U});
    assert(invoke(core, stencil_import).reason == StopReason::instruction_limit);
    assert(core.gx2.db_stencilrefmask_valid && core.gx2.db_stencilrefmask == 0U);
    assert(core.gx2.db_stencilrefmask_bf_valid && core.gx2.db_stencilrefmask_bf == 0U);
    assert(core.gx2.pending_commands.size() == 7U);
    set_arguments(core, {0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU, 0xFFU});
    assert(invoke(core, stencil_import).reason == StopReason::instruction_limit);
    assert(core.gx2.db_stencilrefmask == 0x00FFFFFFU);
    assert(core.gx2.db_stencilrefmask_bf == 0x00FFFFFFU);
    assert(core.gx2.pending_commands.size() == 9U);
    assert(invoke(core, stencil_import).reason == StopReason::instruction_limit);
    assert(core.gx2.pending_commands.size() == 11U);

    EspressoCore uninitialized(0x1000U);
    register_gx2_hle(uninitialized.hle);
    const std::uint32_t uninitialized_stencil = uninitialized.hle.bind_import(
        "gx2", "GX2SetStencilMask");
    for (std::size_t index = 0; index < 6U; ++index)
    {
        uninitialized.state.gpr[index + 3U] = 0xDEAD0000U + static_cast<std::uint32_t>(index);
    }
    uninitialized.state.gpr[1] = 0xFFFFFFFFU;
    const auto uninitialized_gprs = uninitialized.state.gpr;
    const RunResult uninitialized_result = invoke(uninitialized, uninitialized_stencil);
    assert(uninitialized_result.reason == StopReason::hle_error);
    assert(uninitialized_result.detail.find("gx2::GX2SetStencilMask") !=
           std::string::npos);
    assert(uninitialized.state.gpr == uninitialized_gprs);
    assert(!uninitialized.gx2.db_stencilrefmask_valid);
    assert(uninitialized.gx2.db_stencilrefmask == 0U);
    assert(!uninitialized.gx2.db_stencilrefmask_bf_valid);
    assert(uninitialized.gx2.db_stencilrefmask_bf == 0U);
    assert(uninitialized.gx2.pending_commands.empty());

    EspressoCore independent(0x1000U);
    register_gx2_hle(independent.hle);
    independent.gx2.initialized = true;
    independent.state.gpr[1] = 0xFFFFFFFFU;
    const std::uint32_t independent_stencil = independent.hle.bind_import(
        "gx2", "GX2SetStencilMask");
    set_arguments(independent, {1U, 2U, 3U, 4U, 5U, 6U});
    assert(invoke(independent, independent_stencil).reason == StopReason::instruction_limit);
    assert(independent.gx2.db_stencilrefmask_valid);
    assert(independent.gx2.db_stencilrefmask == 0x00020103U);
    assert(independent.gx2.db_stencilrefmask_bf_valid);
    assert(independent.gx2.db_stencilrefmask_bf == 0x00050406U);
    assert(independent.gx2.pending_commands.size() == 2U);
    assert(!core.gx2.db_stencilrefmask_valid ||
           core.gx2.db_stencilrefmask == 0x00FFFFFFU);

    core.state.cia = wrong_library;
    core.state.lr = return_address;
    const RunResult wrong_library_result = core.run(1U);
    assert(wrong_library_result.reason == StopReason::unimplemented_hle_call);
    assert(wrong_library_result.hle_call == "coreinit::GX2SetStencilMask");

    core.reset();
    assert(!core.gx2.db_stencilrefmask_valid);
    assert(core.gx2.db_stencilrefmask == 0U);
    assert(!core.gx2.db_stencilrefmask_bf_valid);
    assert(core.gx2.db_stencilrefmask_bf == 0U);
    assert(!core.gx2.db_depth_control_valid);
    assert(core.gx2.db_depth_control == 0U);
    assert(core.gx2.pending_commands.empty());
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

void store_halfword_update_tests()
{
    const auto encode_sthu = [](std::uint8_t source, std::uint8_t base,
                                std::int16_t displacement) {
        return 0xB4000000U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    constexpr std::uint32_t original_cr = 0xA5C36E91U;
    constexpr std::uint32_t original_xer = 0x800000A5U;
    constexpr std::uint32_t original_lr = 0x12345678U;
    constexpr std::uint32_t original_ctr = 0x87654321U;

    EspressoCore normal(0x200U);
    normal.state.gpr[8] = 0x80U;
    normal.state.gpr[31] = 0x1234ABCDU;
    normal.state.gpr[7] = 0xDEADBEEFU;
    normal.state.cr = original_cr;
    normal.state.xer = original_xer;
    normal.state.lr = original_lr;
    normal.state.ctr = original_ctr;
    normal.memory.write8(0x83U, 0x5AU);
    normal.memory.write8(0x86U, 0xA5U);
    normal.memory.write32_be(0U, encode_sthu(31U, 8U, 4));
    const RunResult normal_result = normal.run(1U);
    assert(normal_result.reason == StopReason::instruction_limit);
    assert(normal.state.gpr[8] == 0x84U);
    assert(normal.memory.read16_be(0x84U) == 0xABCDU);
    assert(normal.memory.read8(0x84U) == 0xABU);
    assert(normal.memory.read8(0x85U) == 0xCDU);
    assert(normal.memory.read8(0x83U) == 0x5AU);
    assert(normal.memory.read8(0x86U) == 0xA5U);
    assert(normal.state.gpr[31] == 0x1234ABCDU);
    assert(normal.state.gpr[7] == 0xDEADBEEFU);
    assert(normal.state.cr == original_cr);
    assert(normal.state.xer == original_xer);
    assert(normal.state.lr == original_lr);
    assert(normal.state.ctr == original_ctr);

    const InstructionHistoryEntry& history = normal_result.instruction_history.front();
    assert(history.opcode_name == "sthu");
    assert(history.source_count == 2U);
    assert(history.source_registers[0] == 8U && history.source_values[0] == 0x80U);
    assert(history.source_registers[1] == 31U && history.source_values[1] == 0x1234ABCDU);
    assert(history.has_effective_address && history.effective_address == 0x84U);
    assert(history.has_destination && history.destination_register == 8U);
    assert(history.destination_value == 0x84U);
    assert(history.has_stored_halfword_value && history.stored_halfword_value == 0xABCDU);
    assert(format_instruction_history(normal_result).find(
        "sthu r8=0x00000080 r31=0x1234ABCD [0x00000084] -> r8=0x00000084 -> mem16=0xABCD") !=
        std::string::npos);

    // The source/base alias must use the old GPR value for both the address and value.
    // Use a mapped low address for the observable aliasing store.
    EspressoCore mapped_alias(0x200U);
    mapped_alias.state.gpr[8] = 0x80U;
    mapped_alias.memory.write32_be(0U, encode_sthu(8U, 8U, 2));
    assert(mapped_alias.step() == StepResult::executed);
    assert(mapped_alias.memory.read16_be(0x82U) == 0x0080U);
    assert(mapped_alias.state.gpr[8] == 0x82U);

    // Signed negative displacement is applied before updating the base.
    EspressoCore negative(0x200U);
    negative.state.gpr[4] = 0x100U;
    negative.state.gpr[5] = 0xFEDCABCDU;
    negative.memory.write32_be(0U, encode_sthu(5U, 4U, -2));
    assert(negative.step() == StepResult::executed);
    assert(negative.memory.read16_be(0xFEU) == 0xABCDU);
    assert(negative.state.gpr[4] == 0xFEU);

    // Exact Wind Waker address is sparse-backed, not a multi-gigabyte flat allocation.
    EspressoCore wind_waker(0x100U);
    wind_waker.memory.map_region(0x1046C000U, 0x1000U);
    wind_waker.state.gpr[8] = 0x1046C22AU;
    wind_waker.state.gpr[31] = 0U;
    wind_waker.memory.write32_be(0U, 0xB7E80002U);
    assert(wind_waker.step() == StepResult::executed);
    assert(wind_waker.memory.read16_be(0x1046C22CU) == 0U);
    assert(wind_waker.state.gpr[8] == 0x1046C22CU);

    // A crossing-range failure validates both bytes before writing and leaves RA unchanged.
    EspressoCore fault(0x100U);
    fault.state.gpr[8] = 0xFDU;
    fault.state.gpr[31] = 0x1234ABCDU;
    fault.memory.write8(0xFFU, 0xA5U);
    fault.memory.write32_be(0U, 0xB7E80002U);
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("write 2 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("0x000000FF") != std::string::npos);
    assert(fault.state.gpr[8] == 0xFDU);
    assert(fault.memory.read8(0xFFU) == 0xA5U);
    assert(fault_result.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed = fault_result.instruction_history.front();
    assert(failed.opcode_name == "sthu");
    assert(failed.has_effective_address && failed.effective_address == 0xFFU);
    assert(!failed.completed);
    assert(!failed.has_stored_halfword_value);
    assert(failed.has_destination && failed.destination_register == 8U);
}

void store_halfword_indexed_tests()
{
    const auto encode_sthx = [](std::uint8_t source, std::uint8_t base,
                                std::uint8_t index, bool reserved_bit = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) |
            (407U << 1U) | static_cast<std::uint32_t>(reserved_bit);
    };

    constexpr std::uint32_t wind_waker_word = 0x7FA7DB2EU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::store_halfword_indexed);
    assert(decoded.destination == 29U);
    assert(decoded.base == 7U);
    assert(decoded.source == 27U);

    const auto run_store = [&](std::uint8_t source, std::uint8_t base,
                               std::uint8_t index, std::uint32_t source_value,
                               std::uint32_t base_value, std::uint32_t index_value,
                               std::size_t memory_size = 0x100U) {
        EspressoCore core(memory_size);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[source] = source_value;
        core.state.gpr[base] = base_value;
        core.state.gpr[index] = index_value;
        core.state.cr = 0x12345678U;
        core.state.xer = 0xA00000A5U;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const std::uint32_t address = (base == 0U ? 0U : base_value) + index_value;
        const std::uint32_t word = encode_sthx(source, base, index);
        core.memory.write32_be(0U, word);

        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            assert(core.state.gpr[reg] == original_gprs[reg]);
        }
        assert(core.state.cr == 0x12345678U);
        assert(core.state.xer == 0xA00000A5U);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read16_be(address) == static_cast<std::uint16_t>(source_value));
        return std::pair{address, result};
    };

    const auto basic = run_store(5U, 4U, 6U, 0x1234ABCDU, 0x40U, 0x08U);
    assert(basic.first == 0x48U);
    assert(basic.second.instruction_history.front().has_effective_address);
    assert(basic.second.instruction_history.front().effective_address == 0x48U);
    assert(basic.second.instruction_history.front().has_stored_halfword_value);
    assert(basic.second.instruction_history.front().stored_halfword_value == 0xABCDU);
    assert(format_instruction_history(basic.second).find(
        "sthx r4=0x00000040 r6=0x00000008 r5=0x1234ABCD [0x00000048] -> mem16=0xABCD") !=
        std::string::npos);
    assert(run_store(5U, 4U, 6U, 0xFFFF1234U, 0x30U, 0x12U).first == 0x42U);

    // RA=0 uses literal zero, not the deliberately nonzero GPR0 value.
    const auto zero_base = run_store(5U, 0U, 6U, 0xABCD9876U,
                                     0xDEADBEEFU, 0x40U, 0x100U);
    assert(zero_base.first == 0x40U);
    assert(zero_base.second.instruction_history.front().source_registers[0] == 6U);
    assert(zero_base.second.instruction_history.front().source_registers[1] == 5U);

    // Address addition wraps in 32-bit guest address space.
    assert(run_store(5U, 4U, 6U, 0xDEAD1234U, 0xFFFFFFFEU, 4U).first == 2U);

    // Aliases retain old source/base/index values and never update GPRs.
    assert(run_store(5U, 5U, 6U, 0x20U, 0x20U, 4U).first == 0x24U);
    assert(run_store(6U, 4U, 6U, 0x08U, 0x20U, 0x08U).first == 0x28U);
    assert(run_store(5U, 5U, 5U, 0x20U, 0x20U, 0x20U).first == 0x40U);

    // An adjacent sparse region makes a halfword crossing flat -> sparse valid.
    EspressoCore crossing(0x20U);
    crossing.memory.map_region(0x20U, 0x20U);
    crossing.state.gpr[6] = 0x1FU;
    crossing.state.gpr[5] = 0xCAFEU;
    crossing.memory.write32_be(0U, encode_sthx(5U, 0U, 6U));
    assert(crossing.step() == StepResult::executed);
    assert(crossing.memory.read8(0x1FU) == 0xCAU);
    assert(crossing.memory.read8(0x20U) == 0xFEU);

    // Wind Waker's address is sparse-backed; the preceding OSBlockMove bytes remain.
    EspressoCore wind_waker(0x100U);
    wind_waker.memory.map_region(0x10490800U, 0x1000U);
    wind_waker.state.gpr[29] = 0U;
    wind_waker.state.gpr[7] = 2U;
    wind_waker.state.gpr[27] = 0x104908D4U;
    wind_waker.memory.write8(0x104908D4U, 0x20U);
    wind_waker.memory.write8(0x104908D5U, 0x26U);
    wind_waker.memory.write8(0x104908D6U, 0xAAU);
    wind_waker.memory.write8(0x104908D7U, 0xBBU);
    wind_waker.memory.write32_be(0U, wind_waker_word);
    const auto wind_result = wind_waker.run(1U);
    assert(wind_result.reason == StopReason::instruction_limit);
    assert(wind_waker.memory.read32_be(0x104908D4U) == 0x20260000U);
    assert(wind_waker.state.gpr[29] == 0U);
    assert(wind_waker.state.gpr[7] == 2U);
    assert(wind_waker.state.gpr[27] == 0x104908D4U);
    assert(format_instruction_history(wind_result).find(
        "sthx r7=0x00000002 r27=0x104908D4 r29=0x00000000 "
        "[0x104908D6] -> mem16=0x0000") != std::string::npos);

    // Only the first target byte is mapped: write must be atomic and diagnosed as indexed.
    EspressoCore fault(0x20U);
    fault.state.gpr[6] = 0x1FU;
    fault.state.gpr[5] = 0x1234U;
    fault.memory.write8(0x1FU, 0xA5U);
    fault.memory.write32_be(0U, encode_sthx(5U, 0U, 6U));
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("write 2 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("rA=0 (0x00000000), rB=6 (0x0000001F)") !=
           std::string::npos);
    assert(fault.memory.read8(0x1FU) == 0xA5U);
    assert(fault_result.instruction_history.size() == 1U);
    assert(fault_result.instruction_history.front().opcode_name == "sthx");
    assert(!fault_result.instruction_history.front().completed);
    assert(!fault_result.instruction_history.front().has_stored_halfword_value);
}

void store_floating_as_integer_word_indexed_tests()
{
    const auto encode_stfiwx = [](std::uint8_t frs, std::uint8_t ra,
                                  std::uint8_t rb, bool reserved_bit = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(frs) << 21U) |
            (static_cast<std::uint32_t>(ra) << 16U) |
            (static_cast<std::uint32_t>(rb) << 11U) |
            (983U << 1U) | static_cast<std::uint32_t>(reserved_bit);
    };

    constexpr std::uint32_t wind_waker_word = 0x7C0007AEU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::store_floating_as_integer_word_indexed);
    assert(decoded.fp_register == 0U);
    assert(decoded.base == 0U);
    assert(decoded.source == 0U);
    assert(decoded.raw == wind_waker_word);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 983U);
    assert(encode_stfiwx(0U, 0U, 0U) == wind_waker_word);
    const DecodedInstruction general = decode(encode_stfiwx(31U, 29U, 30U));
    assert(general.opcode == Opcode::store_floating_as_integer_word_indexed);
    assert(general.fp_register == 31U && general.base == 29U && general.source == 30U);
    assert(decode(encode_stfiwx(3U, 4U, 5U, true)).opcode == Opcode::unsupported);

    // RA=0 is literal zero, while RB=0 still reads GPR0 as the index.
    EspressoCore zero_base(0x100U);
    zero_base.state.gpr[0] = 0x40U;
    zero_base.state.fpr[6] = 0x11223344A1B2C3D4ULL;
    zero_base.state.fpr_ps1[6] = 0xBBBBBBBBDEADBEEFULL;
    zero_base.memory.write8(0x3FU, 0xA5U);
    zero_base.memory.write8(0x44U, 0x5AU);
    zero_base.memory.write32_be(0U, encode_stfiwx(6U, 0U, 0U));
    const auto gprs_before = zero_base.state.gpr;
    const auto fprs_before = zero_base.state.fpr;
    const auto ps1_before = zero_base.state.fpr_ps1;
    const std::uint32_t cr_before = zero_base.state.cr;
    const std::uint32_t xer_before = zero_base.state.xer;
    const std::uint32_t fpscr_before = zero_base.state.fpscr;
    const std::uint32_t lr_before = zero_base.state.lr;
    const std::uint32_t ctr_before = zero_base.state.ctr;
    const RunResult raw_result = zero_base.run(1U);
    assert(raw_result.reason == StopReason::instruction_limit && raw_result.steps == 1U);
    assert(zero_base.state.cia == 4U);
    assert(zero_base.memory.read32_be(0x40U) == 0xA1B2C3D4U);
    assert(zero_base.memory.read8(0x3FU) == 0xA5U);
    assert(zero_base.memory.read8(0x44U) == 0x5AU);
    assert(zero_base.state.gpr == gprs_before);
    assert(zero_base.state.fpr == fprs_before);
    assert(zero_base.state.fpr_ps1 == ps1_before);
    assert(zero_base.state.cr == cr_before && zero_base.state.xer == xer_before);
    assert(zero_base.state.fpscr == fpscr_before);
    assert(zero_base.state.lr == lr_before && zero_base.state.ctr == ctr_before);
    const InstructionHistoryEntry& history = raw_result.instruction_history.front();
    assert(history.opcode_name == "stfiwx");
    assert(history.has_fp_source && history.fp_source_register == 6U);
    assert(history.fp_source_value == 0x11223344A1B2C3D4ULL);
    assert(history.source_count == 1U && history.source_registers[0] == 0U);
    assert(history.source_values[0] == 0x40U);
    assert(history.has_effective_address && history.effective_address == 0x40U);
    assert(history.has_stored_integer_word_value &&
           history.stored_integer_word_value == 0xA1B2C3D4U);
    const std::string trace = format_instruction_history(raw_result);
    assert(trace.find("stfiwx r0=0x00000040 f6=") != std::string::npos);
    assert(trace.find("0x11223344A1B2C3D4") != std::string::npos);
    assert(trace.find("[0x00000040] -> mem32=0xA1B2C3D4") != std::string::npos);

    // Nonzero RA contributes its GPR value; the FPR supplies only the low word.
    EspressoCore nonzero_base(0x100U);
    nonzero_base.state.gpr[4] = 0x80U;
    nonzero_base.state.gpr[5] = 0x20U;
    nonzero_base.state.fpr[3] = 0xFFF80000FFFFFFF9ULL;
    nonzero_base.memory.write32_be(0U, encode_stfiwx(3U, 4U, 5U));
    assert(nonzero_base.step() == StepResult::executed);
    assert(nonzero_base.memory.read32_be(0xA0U) == 0xFFFFFFF9U);
    assert(nonzero_base.state.gpr[4] == 0x80U && nonzero_base.state.gpr[5] == 0x20U);

    // Common fctiwz -> stfiwx producer/consumer sequence stores 1920 exactly.
    EspressoCore conversion_chain(0x100U);
    conversion_chain.state.gpr[0] = 0x80U;
    conversion_chain.state.fpr[13] = std::bit_cast<std::uint64_t>(1920.0);
    conversion_chain.memory.write32_be(0U, 0xFC00681EU); // fctiwz f0,f13
    conversion_chain.memory.write32_be(4U, wind_waker_word); // stfiwx f0,r0,r0
    const RunResult chain_result = conversion_chain.run(2U);
    assert(chain_result.steps == 2U);
    assert(chain_result.reason == StopReason::instruction_limit);
    assert(conversion_chain.state.cia == 8U);
    assert(conversion_chain.state.fpr[0] == 0xFFF8000000000780ULL);
    assert(conversion_chain.memory.read32_be(0x80U) == 0x00000780U);
    assert(conversion_chain.memory.read8(0x80U) == 0x00U);
    assert(conversion_chain.memory.read8(0x81U) == 0x00U);
    assert(conversion_chain.memory.read8(0x82U) == 0x07U);
    assert(conversion_chain.memory.read8(0x83U) == 0x80U);

    // Exact Wind Waker state/address, with a mapped synthetic stack target.
    EspressoCore wind_waker(0x100U);
    wind_waker.state.gpr[0] = 0x100FEEC0U;
    wind_waker.state.fpr[0] = 0xFFF8000000000780ULL;
    wind_waker.state.fpr_ps1[0] = 0xDEADBEEF01234567ULL;
    wind_waker.memory.map_region(0x100FEE80U, 0x100U);
    wind_waker.memory.write32_be(0U, wind_waker_word);
    const std::uint32_t wind_fpscr = wind_waker.state.fpscr;
    const RunResult wind_result = wind_waker.run(1U);
    assert(wind_result.reason == StopReason::instruction_limit);
    assert(wind_waker.memory.read32_be(0x100FEEC0U) == 0x00000780U);
    assert(wind_waker.state.cia == 4U);
    assert(wind_waker.state.gpr[0] == 0x100FEEC0U);
    assert(wind_waker.state.fpr[0] == 0xFFF8000000000780ULL);
    assert(wind_waker.state.fpr_ps1[0] == 0xDEADBEEF01234567ULL);
    assert(wind_waker.state.fpscr == wind_fpscr);
    const std::string wind_trace = format_instruction_history(wind_result);
    assert(wind_trace.find("stfiwx") != std::string::npos);
    assert(wind_trace.find("0xFFF8000000000780") != std::string::npos);
    assert(wind_trace.find("0x100FEEC0") != std::string::npos);
    assert(wind_trace.find("0x00000780") != std::string::npos);

    // A partial word mapping faults atomically and leaves the faulting CIA/registers intact.
    EspressoCore fault(0x20U);
    fault.state.gpr[5] = 0x1EU;
    fault.state.fpr[7] = 0x01234567CAFEBABEULL;
    fault.memory.write8(0x1EU, 0xAAU);
    fault.memory.write8(0x1FU, 0xBBU);
    fault.memory.write32_be(0U, encode_stfiwx(7U, 0U, 5U));
    const auto fault_gprs = fault.state.gpr;
    const auto fault_fprs = fault.state.fpr;
    const auto fault_ps1 = fault.state.fpr_ps1;
    const std::uint32_t fault_fpscr = fault.state.fpscr;
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("write 4 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("rA=0 (0x00000000), rB=5 (0x0000001E)") !=
           std::string::npos);
    assert(fault.state.cia == 0U);
    assert(fault.memory.read8(0x1EU) == 0xAAU && fault.memory.read8(0x1FU) == 0xBBU);
    assert(fault.state.gpr == fault_gprs && fault.state.fpr == fault_fprs);
    assert(fault.state.fpr_ps1 == fault_ps1 && fault.state.fpscr == fault_fpscr);
    assert(fault_result.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed = fault_result.instruction_history.front();
    assert(failed.opcode_name == "stfiwx" && !failed.completed);
    assert(failed.has_effective_address && failed.effective_address == 0x1EU);
    assert(failed.has_fp_source && failed.fp_source_value == 0x01234567CAFEBABEULL);
    assert(!failed.has_stored_integer_word_value);
    const std::string fault_trace = format_instruction_history(fault_result);
    assert(fault_trace.find("stfiwx") != std::string::npos);
    assert(fault_trace.find("[did not complete]") != std::string::npos);
}

void load_halfword_zero_indexed_tests()
{
    const auto encode_lhzx = [](std::uint8_t destination, std::uint8_t base,
                                std::uint8_t index, bool reserved_bit = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) |
            (279U << 1U) | static_cast<std::uint32_t>(reserved_bit);
    };
    const auto encode_sthx = [](std::uint8_t source, std::uint8_t base,
                                std::uint8_t index) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) | (407U << 1U);
    };
    const auto encode_lhax = [](std::uint8_t destination, std::uint8_t base,
                                std::uint8_t index) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) | (343U << 1U);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D7B422EU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::load_halfword_zero_indexed);
    assert(decoded.destination == 11U);
    assert(decoded.base == 27U);
    assert(decoded.source == 8U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 279U);
    assert(decode(encode_lhzx(11U, 27U, 8U, true)).opcode == Opcode::unsupported);
    constexpr std::uint32_t wind_waker_actual_word = 0x7D7B42AEU;
    assert(decode(wind_waker_actual_word).opcode ==
           Opcode::load_halfword_algebraic_indexed);

    const auto run_load = [&](std::uint8_t destination, std::uint8_t base,
                              std::uint8_t index, std::uint32_t base_value,
                              std::uint32_t index_value, std::uint32_t address,
                              std::uint16_t memory_value) {
        EspressoCore core(0x100U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[destination] = 0xDEADBEEFU;
        core.state.gpr[base] = base_value;
        core.state.gpr[index] = index_value;
        core.state.cr = 0x12345678U;
        core.state.xer = 0xA00000A5U;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        core.memory.write32_be(0U, encode_lhzx(destination, base, index));
        core.memory.write16_be(address, memory_value);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == static_cast<std::uint32_t>(memory_value));
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(core.state.cr == 0x12345678U);
        assert(core.state.xer == 0xA00000A5U);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read16_be(address) == memory_value);
        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "lhzx");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_destination && history.destination_register == destination);
        assert(history.destination_value == memory_value);
        assert(history.source_count == (base == 0U || base == index ? 1U : 2U));
        return result;
    };

    constexpr std::array<std::uint16_t, 5> values{
        0x0000U, 0x0001U, 0x7FFFU, 0x8000U, 0xFFFFU};
    for (const std::uint16_t value : values)
    {
        const RunResult result = run_load(11U, 4U, 8U, 0x30U, 0x10U,
                                          0x40U, value);
        assert(result.instruction_history.front().destination_value == value);
    }
    const RunResult zero_extend = run_load(11U, 4U, 8U, 0x30U, 0x10U,
                                           0x40U, 0xFFFFU);
    assert(zero_extend.instruction_history.front().destination_value == 0x0000FFFFU);
    assert(format_instruction_history(zero_extend).find(
        "lhzx r4=0x00000030 r8=0x00000010 [0x00000040] -> r11=0x0000FFFF") !=
        std::string::npos);

    // RA=0 contributes literal zero even though GPR0 is nonzero.
    const RunResult zero_base = run_load(11U, 0U, 4U, 0xDEADBEEFU, 0x40U,
                                         0x40U, 0x8000U);
    assert(zero_base.instruction_history.front().source_registers[0] == 4U);

    // The effective address uses old operand values before destination writes.
    (void)run_load(5U, 5U, 6U, 0x80U, 4U, 0x84U, 0x1234U); // rD == rA
    (void)run_load(5U, 4U, 5U, 0x80U, 4U, 0x84U, 0x1234U); // rD == rB
    (void)run_load(5U, 5U, 5U, 0x40U, 0x40U, 0x80U, 0xABCDU); // all equal

    // Flat/sparse adjacent backing supports a halfword read crossing the edge.
    EspressoCore crossing(0x20U);
    crossing.memory.map_region(0x20U, 0x20U);
    crossing.state.gpr[6] = 0x1FU;
    crossing.memory.write8(0x1FU, 0xCAU);
    crossing.memory.write8(0x20U, 0xFEU);
    crossing.memory.write32_be(0U, encode_lhzx(11U, 0U, 6U));
    assert(crossing.step() == StepResult::executed);
    assert(crossing.state.gpr[11] == 0x0000CAFEU);

    // The actual Wind Waker data flow: sthx writes 0x0001, then lhzx reads it.
    EspressoCore wind_waker(0x100U);
    wind_waker.memory.map_region(0x104A0000U, 0x2000U);
    wind_waker.state.gpr[27] = 0x104A147CU;
    wind_waker.state.gpr[0] = 2U;
    wind_waker.state.gpr[31] = 1U;
    wind_waker.state.gpr[8] = 2U;
    wind_waker.memory.write32_be(0U, encode_sthx(31U, 27U, 0U));
    wind_waker.memory.write32_be(4U, wind_waker_word);
    wind_waker.memory.write32_be(8U, encode_lhax(12U, 27U, 8U));
    const RunResult chain = wind_waker.run(3U);
    assert(chain.steps == 3U);
    assert(chain.reason == StopReason::instruction_limit);
    assert(wind_waker.memory.read16_be(0x104A147EU) == 0x0001U);
    assert(wind_waker.state.gpr[11] == 0x00000001U);
    assert(wind_waker.state.gpr[12] == 0x00000001U);
    assert(wind_waker.state.gpr[27] == 0x104A147CU);
    assert(wind_waker.state.gpr[8] == 2U);
    const std::string chain_trace = format_instruction_history(chain);
    assert(chain_trace.find(
        "lhzx r27=0x104A147C r8=0x00000002 [0x104A147E] -> r11=0x00000001") !=
        std::string::npos);
    assert(chain_trace.find(
        "lhax r27=0x104A147C r8=0x00000002 [0x104A147E] -> r12=0x00000001") !=
        std::string::npos);

    // The actual blocker is lhax, so verify its algebraic extension differs
    // from lhzx for a halfword with the sign bit set.
    EspressoCore signed_load(0x100U);
    signed_load.state.gpr[27] = 0x40U;
    signed_load.state.gpr[8] = 0U;
    signed_load.memory.write16_be(0x40U, 0x8000U);
    signed_load.memory.write32_be(0U, wind_waker_actual_word);
    assert(signed_load.step() == StepResult::executed);
    assert(signed_load.state.gpr[11] == 0xFFFF8000U);

    // A halfword whose second byte is unmapped faults without changing rD.
    EspressoCore fault(0x20U);
    fault.state.gpr.fill(0xA5A5A5A5U);
    fault.state.gpr[6] = 0x1FU;
    fault.state.gpr[11] = 0xDEADBEEFU;
    fault.memory.write8(0x1FU, 0x12U);
    fault.memory.write32_be(0U, encode_lhzx(11U, 0U, 6U));
    const auto before_fault_gprs = fault.state.gpr;
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("read 2 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("rA=0 (0x00000000), rB=6 (0x0000001F)") !=
           std::string::npos);
    assert(fault_result.instruction_history.size() == 1U);
    assert(fault_result.instruction_history.front().opcode_name == "lhzx");
    assert(fault_result.instruction_history.front().has_effective_address);
    assert(fault_result.instruction_history.front().effective_address == 0x1FU);
    assert(!fault_result.instruction_history.front().completed);
    assert(fault_result.instruction_history.front().has_destination);
    assert(fault_result.instruction_history.front().destination_register == 11U);
    assert(fault.state.gpr == before_fault_gprs);
    assert(fault.memory.read8(0x1FU) == 0x12U);
}

void load_byte_zero_indexed_tests()
{
    const auto encode_lbzx = [](std::uint8_t destination, std::uint8_t base,
                                std::uint8_t index, bool reserved_bit = false) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) | (87U << 1U) |
            static_cast<std::uint32_t>(reserved_bit);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D6900AEU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::load_byte_zero_indexed);
    assert(decoded.destination == 11U);
    assert(decoded.base == 9U);
    assert(decoded.source == 0U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 87U);
    assert(decode(encode_lbzx(11U, 9U, 0U, true)).opcode == Opcode::unsupported);

    const auto run_load = [&](std::uint8_t destination, std::uint8_t base,
                              std::uint8_t index, std::uint32_t base_value,
                              std::uint32_t index_value, std::uint32_t address,
                              std::uint8_t memory_value) {
        EspressoCore core(0x200U);
        core.state.gpr.fill(0xA5A5A5A5U);
        core.state.gpr[destination] = 0xDEADBEEFU;
        if (base != 0U)
        {
            core.state.gpr[base] = base_value;
        }
        core.state.gpr[index] = index_value;
        core.state.cr = 0x12345678U;
        core.state.xer = 0xA00000A5U;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.state.fpscr = 0xCAFEBABEU;
        core.state.fpr.fill(0x0123456789ABCDEFULL);
        core.state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
        core.memory.write32_be(0U, encode_lbzx(destination, base, index));
        core.memory.write8(address, memory_value);
        const auto original_gprs = core.state.gpr;
        const auto original_fpr = core.state.fpr;
        const auto original_fpr_ps1 = core.state.fpr_ps1;
        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        assert(core.state.gpr[destination] == memory_value);
        for (std::uint32_t reg = 0; reg < 32U; ++reg)
        {
            if (reg != destination)
            {
                assert(core.state.gpr[reg] == original_gprs[reg]);
            }
        }
        assert(core.state.cr == 0x12345678U);
        assert(core.state.xer == 0xA00000A5U);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);
        assert(core.state.fpscr == 0xCAFEBABEU);
        assert(core.state.fpr == original_fpr);
        assert(core.state.fpr_ps1 == original_fpr_ps1);
        assert(core.memory.read8(address) == memory_value);
        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "lbzx");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_destination && history.destination_register == destination);
        assert(history.destination_value == memory_value);
        return result;
    };

    constexpr std::array<std::uint8_t, 6> values{0x00U, 0x01U, 0x7FU,
                                                0x80U, 0xFEU, 0xFFU};
    for (const std::uint8_t value : values)
    {
        const RunResult result = run_load(11U, 4U, 8U, 0x40U, 8U, 0x48U, value);
        assert(result.instruction_history.front().destination_value == value);
    }
    const RunResult unsigned_ff = run_load(11U, 4U, 8U, 0x40U, 8U, 0x48U, 0xFFU);
    assert(unsigned_ff.instruction_history.front().destination_value == 0x000000FFU);

    // rA=0 is literal zero even though GPR0 holds a nonzero value.
    const RunResult zero_base = run_load(11U, 0U, 4U, 0xDEADBEEFU, 0x40U,
                                         0x40U, 0x80U);
    assert(zero_base.instruction_history.front().source_count == 1U);
    assert(zero_base.instruction_history.front().source_registers[0] == 4U);

    // rB=0 still reads GPR0 and participates in the effective address.
    EspressoCore index_zero(0x100U);
    index_zero.memory.map_region(0x100U, 0x100U);
    index_zero.state.gpr[9] = 0x100U;
    index_zero.state.gpr[0] = 0x20U;
    index_zero.memory.write8(0x120U, 0xABU);
    index_zero.memory.write32_be(0U, wind_waker_word);
    const RunResult index_zero_result = index_zero.run(1U);
    assert(index_zero_result.reason == StopReason::instruction_limit);
    assert(index_zero.state.gpr[11] == 0x000000ABU);
    assert(index_zero_result.instruction_history.front().effective_address == 0x120U);
    assert(index_zero_result.instruction_history.front().source_count == 2U);
    assert(index_zero_result.instruction_history.front().source_registers[1] == 0U);

    // Addition wraps in the 32-bit guest address space.
    EspressoCore wrapping(0x100U);
    wrapping.state.gpr[4] = 0xFFFFFFF0U;
    wrapping.state.gpr[5] = 0x20U;
    wrapping.memory.write8(0x10U, 0x6CU);
    wrapping.memory.write32_be(0U, encode_lbzx(6U, 4U, 5U));
    assert(wrapping.step() == StepResult::executed);
    assert(wrapping.state.gpr[6] == 0x6CU);

    // Destination aliasing uses the original base and/or index values.
    (void)run_load(5U, 5U, 6U, 0x40U, 3U, 0x43U, 0xA5U);
    (void)run_load(5U, 4U, 5U, 0x40U, 3U, 0x43U, 0x5AU);
    (void)run_load(5U, 5U, 5U, 0x20U, 0x20U, 0x40U, 0x77U);

    // Exact Wind Waker addressing regression; byte value is synthetic.
    EspressoCore wind_waker(0x100U);
    wind_waker.memory.map_region(0x1018A000U, 0x2000U);
    wind_waker.state.gpr[9] = 0x1018A5E8U;
    wind_waker.state.gpr[0] = 0x80U;
    wind_waker.state.gpr[11] = 0xDEADBEEFU;
    constexpr std::uint32_t wind_waker_address = 0x1018A668U;
    wind_waker.memory.write8(wind_waker_address, 0x5AU);
    wind_waker.memory.write32_be(0U, wind_waker_word);
    const RunResult wind_waker_result = wind_waker.run(1U);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert(wind_waker.state.gpr[11] == 0x0000005AU);
    assert(wind_waker.state.gpr[9] == 0x1018A5E8U);
    assert(wind_waker.state.gpr[0] == 0x80U);
    const InstructionHistoryEntry& wind_history =
        wind_waker_result.instruction_history.front();
    assert(wind_history.effective_address == wind_waker_address);
    assert(wind_history.destination_value == 0x5AU);
    const std::string trace = format_instruction_history(wind_waker_result);
    assert(trace.find("lbzx r9=0x1018A5E8 r0=0x00000080 [0x1018A668] -> r11=0x0000005A") !=
           std::string::npos);

    // Unmapped byte read faults before any destination-register update.
    EspressoCore fault(0x100U);
    fault.state.gpr.fill(0xA5A5A5A5U);
    fault.state.gpr[9] = 0x100U;
    fault.state.gpr[0] = 0x20U;
    fault.state.gpr[11] = 0xDEADBEEFU;
    fault.memory.write32_be(0U, wind_waker_word);
    const auto before_fault_gprs = fault.state.gpr;
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("read 1 byte(s) at guest address 0x00000120") !=
           std::string::npos);
    assert(fault_result.detail.find("rA=9 (0x00000100), rB=0 (0x00000020)") !=
           std::string::npos);
    assert(fault_result.instruction_history.size() == 1U);
    const InstructionHistoryEntry& fault_history = fault_result.instruction_history.front();
    assert(fault_history.opcode_name == "lbzx");
    assert(fault_history.has_effective_address && fault_history.effective_address == 0x120U);
    assert(fault_history.source_count == 2U);
    assert(fault_history.source_registers[1] == 0U);
    assert(!fault_history.completed);
    assert(fault_history.has_destination && fault_history.destination_register == 11U);
    assert(fault.state.gpr == before_fault_gprs);
}

void load_halfword_algebraic_tests()
{
    const auto run_lha = [](std::uint8_t destination, std::uint8_t base,
                            std::uint32_t base_value, std::int16_t displacement,
                            std::uint32_t address, std::uint16_t memory_value)
    {
        EspressoCore core(0x200U);
        constexpr std::uint32_t expected_cr = 0x12345678U;
        constexpr std::uint32_t expected_xer = 0xA00000A5U;
        core.state.gpr[base] = base_value;
        core.state.cr = expected_cr;
        core.state.xer = expected_xer;
        const std::uint32_t instruction = 0xA8000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
        core.memory.write32_be(0, instruction);
        core.memory.write16_be(address, memory_value);
        const RunResult result = core.run(1U);
        assert(result.steps == 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.cia == 4U);
        const std::int16_t signed_value = std::bit_cast<std::int16_t>(memory_value);
        const std::uint32_t expected = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(signed_value));
        assert(core.state.gpr[destination] == expected);
        assert(core.state.cr == expected_cr);
        assert(core.state.xer == expected_xer);
        assert(result.instruction_history.size() == 1U);
        assert(result.instruction_history[0].opcode_name == "lha");
        assert(result.instruction_history[0].has_effective_address);
        assert(result.instruction_history[0].effective_address == address);
        return std::pair{expected, result};
    };

    assert(run_lha(12, 10, 0x80U, 0, 0x80U, 0x0000U).first == 0x00000000U);
    assert(run_lha(12, 10, 0x80U, 0, 0x80U, 0x0001U).first == 0x00000001U);
    assert(run_lha(12, 10, 0x80U, 0, 0x80U, 0x7FFFU).first == 0x00007FFFU);
    assert(run_lha(12, 10, 0x80U, 0, 0x80U, 0x8000U).first == 0xFFFF8000U);
    const auto all_ones = run_lha(12, 10, 0x80U, 0, 0x80U, 0xFFFFU);
    assert(all_ones.first == 0xFFFFFFFFU);
    assert(format_instruction_history(all_ones.second).find(
        "lha r10=0x00000080 [0x00000080] -> r12=0xFFFFFFFF") != std::string::npos);

    assert(run_lha(12, 10, 0x70U, 0x10, 0x80U, 0x1234U).first == 0x1234U);
    assert(run_lha(12, 10, 0x90U, -0x10, 0x80U, 0xFFFFU).first == 0xFFFFFFFFU);
    assert(run_lha(12, 0, 0x40U, 0x80, 0x80U, 0x8000U).first == 0xFFFF8000U);

    // The EA is captured from the original base before writing an aliased rD.
    assert(run_lha(10, 10, 0x40U, 0, 0x40U, 0x8000U).first == 0xFFFF8000U);

    EspressoCore fault_core(0x100U);
    fault_core.state.gpr[10] = 0x100U;
    fault_core.memory.write32_be(0, 0xA98A0000U);
    const RunResult fault = fault_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 2 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x00000100") != std::string::npos);
    assert(fault.instruction_history.size() == 1U);
    assert(fault.instruction_history[0].has_effective_address);
    assert(fault.instruction_history[0].effective_address == 0x100U);
    assert(!fault.instruction_history[0].completed);
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

void load_string_word_immediate_tests()
{
    const auto encode_lswi = [](std::uint8_t destination, std::uint8_t base,
                                std::uint8_t byte_count) {
        return 0x7C0004AAU |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(byte_count & 0x1FU) << 11U);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D1F24AAU;
    const DecodedInstruction wind_waker = decode(wind_waker_word);
    assert(wind_waker.opcode == Opcode::load_string_word_immediate);
    assert(wind_waker.destination == 8U);
    assert(wind_waker.base == 31U);
    assert(wind_waker.string_byte_count == 4U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 597U);
    assert(decode(wind_waker_word | 1U).opcode == Opcode::unsupported);

    // Four ascending memory bytes become one conventional big-endian GPR.
    EspressoCore wind_waker_core(0x100U);
    wind_waker_core.state.gpr[31] = 0x80U;
    wind_waker_core.memory.write_bytes(0x80U,
        std::array<std::uint8_t, 4>{0x12U, 0x34U, 0x56U, 0x78U});
    wind_waker_core.memory.write32_be(0U, wind_waker_word);
    assert(wind_waker_core.step() == StepResult::executed);
    assert(wind_waker_core.state.gpr[8] == 0x12345678U);
    assert(wind_waker_core.state.gpr[31] == 0x80U);
    const RunResult wind_waker_result = wind_waker_core.run(0U);
    const std::string wind_waker_trace = format_instruction_history(wind_waker_result);
    assert(wind_waker_trace.find("lswi r8,r31,4 r31=0x00000080 [0x00000080] -> r8=0x12345678") !=
           std::string::npos);

    for (std::uint8_t count = 1U; count <= 3U; ++count)
    {
        EspressoCore partial(0x100U);
        partial.state.gpr[4] = 0x80U;
        partial.state.gpr[8] = 0xDEADBEEFU;
        partial.memory.write_bytes(0x80U,
            std::array<std::uint8_t, 4>{0xABU, 0xCDU, 0xEFU, 0x42U});
        partial.memory.write32_be(0U, encode_lswi(8U, 4U, count));
        assert(partial.step() == StepResult::executed);
        const std::array<std::uint32_t, 4> expected{
            0U, 0xAB000000U, 0xABCD0000U, 0xABCDEF00U};
        assert(partial.state.gpr[8] == expected[count]);
        assert(partial.state.gpr[4] == 0x80U);
    }

    EspressoCore five_bytes(0x100U);
    five_bytes.state.gpr[4] = 0x80U;
    five_bytes.memory.write_bytes(0x80U,
        std::array<std::uint8_t, 5>{0x11U, 0x22U, 0x33U, 0x44U, 0x55U});
    five_bytes.memory.write32_be(0U, encode_lswi(8U, 4U, 5U));
    assert(five_bytes.step() == StepResult::executed);
    assert(five_bytes.state.gpr[8] == 0x11223344U);
    assert(five_bytes.state.gpr[9] == 0x55000000U);

    // Register numbering wraps through r31 to r0.
    EspressoCore wraps(0x100U);
    wraps.state.gpr[4] = 0x80U;
    wraps.memory.write_bytes(0x80U,
        std::array<std::uint8_t, 8>{0x11U, 0x22U, 0x33U, 0x44U,
                                    0x55U, 0x66U, 0x77U, 0x88U});
    wraps.memory.write32_be(0U, encode_lswi(31U, 4U, 8U));
    assert(wraps.step() == StepResult::executed);
    assert(wraps.state.gpr[31] == 0x11223344U);
    assert(wraps.state.gpr[0] == 0x55667788U);

    // NB=0 means 32 bytes, filling exactly eight wrapping GPRs.
    EspressoCore zero_count(0x100U);
    zero_count.state.gpr[4] = 0x80U;
    std::array<std::uint8_t, 32> sequence{};
    for (std::size_t i = 0; i < sequence.size(); ++i)
    {
        sequence[i] = static_cast<std::uint8_t>(i + 1U);
    }
    zero_count.memory.write_bytes(0x80U, sequence);
    zero_count.memory.write32_be(0U, encode_lswi(28U, 4U, 0U));
    for (std::uint32_t reg = 0; reg < 32U; ++reg)
    {
        zero_count.state.gpr[reg] = 0xA5000000U + reg;
    }
    zero_count.state.gpr[4] = 0x80U;
    const auto zero_count_original_gprs = zero_count.state.gpr;
    assert(zero_count.step() == StepResult::executed);
    for (std::uint32_t reg = 0; reg < 32U; ++reg)
    {
        const bool loaded = ((reg - 28U) & 31U) < 8U;
        if (!loaded)
        {
            assert(zero_count.state.gpr[reg] == zero_count_original_gprs[reg]);
        }
    }
    for (std::uint32_t i = 0; i < 8U; ++i)
    {
        const std::uint32_t expected = ((i * 4U + 1U) << 24U) |
            ((i * 4U + 2U) << 16U) | ((i * 4U + 3U) << 8U) | (i * 4U + 4U);
        assert(zero_count.state.gpr[(28U + i) & 31U] == expected);
    }

    // RA=0 is a literal zero base, not GPR0. Instruction is placed elsewhere
    // so guest address zero can hold the string bytes.
    EspressoCore zero_base(0x100U);
    zero_base.state.cia = 0x40U;
    zero_base.state.gpr[0] = 0x80U;
    zero_base.memory.write_bytes(0U,
        std::array<std::uint8_t, 4>{0xCAU, 0xFEU, 0xBAU, 0xBEU});
    zero_base.memory.write32_be(0x40U, encode_lswi(8U, 0U, 4U));
    assert(zero_base.step() == StepResult::executed);
    assert(zero_base.state.gpr[8] == 0xCAFEBABEU);
    assert(zero_base.state.gpr[0] == 0x80U);

    // Destination may alias RA; the old RA remains the starting address for
    // the whole read, and all bytes are staged before the destination commits.
    EspressoCore alias(0x100U);
    alias.state.gpr[5] = 0x80U;
    alias.memory.write_bytes(0x80U,
        std::array<std::uint8_t, 5>{0xDEU, 0xADU, 0xBEU, 0xEFU, 0x12U});
    alias.memory.write32_be(0U, encode_lswi(5U, 5U, 5U));
    assert(alias.step() == StepResult::executed);
    assert(alias.state.gpr[5] == 0xDEADBEEFU);
    assert(alias.state.gpr[6] == 0x12000000U);

    // Reads preserve ascending order across a flat/sparse backing boundary.
    EspressoCore crossing(0x100U);
    crossing.memory.map_region(0x100U, 0x20U);
    crossing.state.gpr[4] = 0xFEU;
    crossing.memory.write_bytes(0xFEU,
        std::array<std::uint8_t, 4>{0x10U, 0x20U, 0x30U, 0x40U});
    crossing.memory.write32_be(0U, encode_lswi(8U, 4U, 4U));
    assert(crossing.step() == StepResult::executed);
    assert(crossing.state.gpr[8] == 0x10203040U);

    // Architectural state outside the destination GPR range is untouched.
    EspressoCore state_check(0x100U);
    state_check.state.gpr[4] = 0x80U;
    state_check.state.cr = 0x12345678U;
    state_check.state.xer = 0xA00000A5U;
    state_check.state.lr = 0x11223344U;
    state_check.state.ctr = 0x55667788U;
    state_check.state.fpscr = 0xCAFEBABEU;
    state_check.state.fpr[2] = 0x0123456789ABCDEFULL;
    state_check.state.fpr_ps1[2] = 0xFEDCBA9876543210ULL;
    state_check.memory.write_bytes(0x80U,
        std::array<std::uint8_t, 4>{1U, 2U, 3U, 4U});
    state_check.memory.write32_be(0U, encode_lswi(8U, 4U, 4U));
    assert(state_check.step() == StepResult::executed);
    assert(state_check.state.cr == 0x12345678U);
    assert(state_check.state.xer == 0xA00000A5U);
    assert(state_check.state.lr == 0x11223344U);
    assert(state_check.state.ctr == 0x55667788U);
    assert(state_check.state.fpscr == 0xCAFEBABEU);
    assert(state_check.state.fpr[2] == 0x0123456789ABCDEFULL);
    assert(state_check.state.fpr_ps1[2] == 0xFEDCBA9876543210ULL);
    assert(state_check.memory.read8(0x80U) == 1U);

    // Crossing into unmapped memory fails as one read and commits no GPRs.
    EspressoCore fault(0x100U);
    fault.state.cia = 0x40U;
    fault.state.gpr[4] = 0xFEU;
    fault.state.gpr[8] = 0xAAAAAAAAU;
    fault.state.gpr[9] = 0xBBBBBBBBU;
    fault.memory.write8(0xFEU, 0x11U);
    fault.memory.write8(0xFFU, 0x22U);
    fault.memory.write32_be(0x40U, encode_lswi(8U, 4U, 5U));
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.cia == 0x40U);
    assert(fault_result.detail.find("read 5 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("0x000000FE") != std::string::npos);
    assert(fault.state.gpr[8] == 0xAAAAAAAAU);
    assert(fault.state.gpr[9] == 0xBBBBBBBBU);
    assert(fault.memory.read8(0xFEU) == 0x11U);
    assert(fault.memory.read8(0xFFU) == 0x22U);
    assert(fault_result.instruction_history.size() == 1U);
    assert(fault_result.instruction_history[0].opcode_name == "lswi");
    assert(fault_result.instruction_history[0].has_effective_address);
    assert(fault_result.instruction_history[0].effective_address == 0xFEU);
    assert(!fault_result.instruction_history[0].completed);
    const std::string fault_trace = format_instruction_history(fault_result);
    assert(fault_trace.find("lswi r8,r4,5 r4=0x000000FE [0x000000FE]") !=
           std::string::npos);
}

void store_string_word_immediate_tests()
{
    const auto encode_stswi = [](std::uint8_t source, std::uint8_t base,
                                 std::uint8_t byte_count) {
        return 0x7C0005AAU |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(byte_count & 0x1FU) << 11U);
    };

    constexpr std::uint32_t wind_waker_word = 0x7D0725AAU;
    const DecodedInstruction decoded = decode(wind_waker_word);
    assert(decoded.opcode == Opcode::store_string_word_immediate);
    assert(decoded.source == 8U);
    assert(decoded.base == 7U);
    assert(decoded.string_byte_count == 4U);
    assert(((wind_waker_word >> 1U) & 0x3FFU) == 725U);
    assert(decode(wind_waker_word | 1U).opcode == Opcode::unsupported);

    // Wind Waker stores r8's most-significant byte first, preserving both GPRs.
    EspressoCore wind_waker(0x100U);
    wind_waker.state.gpr[7] = 0x80U;
    wind_waker.state.gpr[8] = 0x80U;
    const auto initial_gprs = wind_waker.state.gpr;
    wind_waker.memory.write32_be(0U, wind_waker_word);
    assert(wind_waker.step() == StepResult::executed);
    assert(wind_waker.memory.read32_be(0x80U) == 0x00000080U);
    assert(wind_waker.state.gpr == initial_gprs);
    const RunResult wind_waker_result = wind_waker.run(0U);
    const std::string wind_waker_trace = format_instruction_history(wind_waker_result);
    assert(wind_waker_trace.find(
        "stswi r8,r7,4 r8=0x00000080 r7=0x00000080 [0x00000080] -> mem=00 00 00 80") !=
        std::string::npos);

    const std::array<std::uint8_t, 4> expected_prefix{0xAAU, 0xBBU, 0xCCU, 0xDDU};
    for (std::uint8_t count = 1U; count <= 4U; ++count)
    {
        EspressoCore partial(0x100U);
        partial.state.gpr[8] = 0xAABBCCDDU;
        partial.state.gpr[4] = 0x80U;
        partial.memory.write32_be(0U, encode_stswi(8U, 4U, count));
        assert(partial.step() == StepResult::executed);
        for (std::uint8_t i = 0; i < count; ++i)
        {
            assert(partial.memory.read8(0x80U + i) == expected_prefix[i]);
        }
        assert(partial.state.gpr[8] == 0xAABBCCDDU);
        assert(partial.state.gpr[4] == 0x80U);
    }

    EspressoCore multiple(0x100U);
    multiple.state.gpr[4] = 0x80U;
    multiple.state.gpr[8] = 0x11223344U;
    multiple.state.gpr[9] = 0x55667788U;
    multiple.memory.write32_be(0U, encode_stswi(8U, 4U, 5U));
    assert(multiple.step() == StepResult::executed);
    assert(multiple.memory.read8(0x80U) == 0x11U);
    assert(multiple.memory.read8(0x81U) == 0x22U);
    assert(multiple.memory.read8(0x82U) == 0x33U);
    assert(multiple.memory.read8(0x83U) == 0x44U);
    assert(multiple.memory.read8(0x84U) == 0x55U);

    EspressoCore wraps(0x100U);
    wraps.state.gpr[31] = 0x11223344U;
    wraps.state.gpr[0] = 0x55667788U;
    wraps.state.gpr[4] = 0x80U;
    wraps.memory.write32_be(0U, encode_stswi(31U, 4U, 8U));
    assert(wraps.step() == StepResult::executed);
    assert(wraps.memory.read64_be(0x80U) == 0x1122334455667788ULL);

    // NB=0 stores exactly 32 bytes from eight GPRs, wrapping at r31.
    EspressoCore zero_count(0x100U);
    zero_count.state.gpr[4] = 0x80U;
    for (std::uint32_t reg = 0; reg < 32U; ++reg)
    {
        zero_count.state.gpr[reg] = 0x10203040U + reg;
    }
    zero_count.state.gpr[4] = 0x80U;
    const auto zero_count_gprs = zero_count.state.gpr;
    zero_count.memory.write32_be(0U, encode_stswi(28U, 4U, 0U));
    assert(zero_count.step() == StepResult::executed);
    for (std::uint32_t i = 0; i < 8U; ++i)
    {
        const std::uint32_t reg = (28U + i) & 31U;
        assert(zero_count.memory.read32_be(0x80U + i * 4U) == zero_count_gprs[reg]);
    }
    assert(zero_count.state.gpr == zero_count_gprs);

    // rA=0 uses address zero even when GPR0 is nonzero.
    EspressoCore zero_base(0x100U);
    zero_base.state.cia = 0x40U;
    zero_base.state.gpr[0] = 0x80U;
    zero_base.state.gpr[8] = 0xCAFEBABEU;
    zero_base.memory.write32_be(0x40U, encode_stswi(8U, 0U, 4U));
    assert(zero_base.step() == StepResult::executed);
    assert(zero_base.memory.read32_be(0U) == 0xCAFEBABEU);
    assert(zero_base.state.gpr[0] == 0x80U);

    // Source/base aliasing uses the original GPR value for both purposes.
    EspressoCore alias(0x100U);
    alias.state.gpr[7] = 0x80U;
    alias.memory.write32_be(0U, encode_stswi(7U, 7U, 4U));
    assert(alias.step() == StepResult::executed);
    assert(alias.memory.read32_be(0x80U) == 0x00000080U);
    assert(alias.state.gpr[7] == 0x80U);

    // A valid write spans the last two flat bytes and the adjacent sparse map.
    EspressoCore crossing(0x100U);
    crossing.memory.map_region(0x100U, 0x20U);
    crossing.state.gpr[4] = 0xFEU;
    crossing.state.gpr[8] = 0x10203040U;
    crossing.memory.write32_be(0U, encode_stswi(8U, 4U, 4U));
    assert(crossing.step() == StepResult::executed);
    assert(crossing.memory.read32_be(0xFEU) == 0x10203040U);

    // No partial write occurs when a requested byte range reaches unmapped memory.
    EspressoCore fault(0x100U);
    fault.state.cia = 0x40U;
    fault.state.gpr[4] = 0xFEU;
    fault.state.gpr[8] = 0xAABBCCDDU;
    fault.state.gpr[9] = 0x11223344U;
    fault.state.cr = 0x12345678U;
    fault.state.xer = 0xA00000A5U;
    fault.state.lr = 0x55667788U;
    fault.state.ctr = 0x99AABBCCU;
    fault.state.fpscr = 0xCAFEBABEU;
    fault.state.fpr[2] = 0x0123456789ABCDEFULL;
    fault.state.fpr_ps1[2] = 0xFEDCBA9876543210ULL;
    fault.memory.write8(0xFEU, 0x5AU);
    fault.memory.write8(0xFFU, 0xA5U);
    const auto fault_gprs = fault.state.gpr;
    fault.memory.write32_be(0x40U, encode_stswi(8U, 4U, 5U));
    const RunResult fault_result = fault.run(1U);
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.cia == 0x40U);
    assert(fault_result.detail.find("write 5 byte(s)") != std::string::npos);
    assert(fault_result.detail.find("0x000000FE") != std::string::npos);
    assert(fault.state.gpr == fault_gprs);
    assert(fault.state.cr == 0x12345678U);
    assert(fault.state.xer == 0xA00000A5U);
    assert(fault.state.lr == 0x55667788U);
    assert(fault.state.ctr == 0x99AABBCCU);
    assert(fault.state.fpscr == 0xCAFEBABEU);
    assert(fault.state.fpr[2] == 0x0123456789ABCDEFULL);
    assert(fault.state.fpr_ps1[2] == 0xFEDCBA9876543210ULL);
    assert(fault.memory.read8(0xFEU) == 0x5AU);
    assert(fault.memory.read8(0xFFU) == 0xA5U);
    assert(fault_result.instruction_history.size() == 1U);
    assert(fault_result.instruction_history[0].opcode_name == "stswi");
    assert(fault_result.instruction_history[0].has_effective_address);
    assert(fault_result.instruction_history[0].effective_address == 0xFEU);
    assert(!fault_result.instruction_history[0].completed);
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
    assert(core.state.fpr_ps1[13] == 0x3FF8000000000000ULL);
    assert(core.state.gpr[12] == 0x100U); // lfs does not update its base.
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
    assert(core.state.fpr_ps1[7] == 0xC002000000000000ULL);

    // rA == 0 uses address zero as the base, and +0 remains exactly zero.
    core.state.cia = 8U;
    core.memory.write32_be(0x80U, 0U);
    core.memory.write32_be(8, encode_lfs(2, 0, 0x80));
    assert(core.step() == StepResult::executed);
    assert(core.state.fpr[2] == 0U);
    assert(core.state.fpr_ps1[2] == 0U);
    assert(core.state.fpr[13] == 0x3FF8000000000000ULL);

    // Reset clears all architectural FPR bits deterministically.
    core.state.fpr[31] = 0xFFFFFFFFFFFFFFFFULL;
    core.state.fpr_ps1[31] = 0x123456789ABCDEF0ULL;
    core.state.reset();
    for (const std::uint64_t value : core.state.fpr)
    {
        assert(value == 0U);
    }
    for (const std::uint64_t value : core.state.fpr_ps1)
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

void floating_point_single_update_tests()
{
    const auto encode_lfsu = [](std::uint8_t destination, std::uint8_t base,
                                std::int16_t displacement) {
        return 0xC4000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    constexpr std::uint32_t initial_cr = 0x12345678U;
    constexpr std::uint32_t initial_xer = 0xA00000A5U;
    constexpr std::uint32_t initial_lr = 0x10203040U;
    constexpr std::uint32_t initial_ctr = 0x50607080U;
    constexpr std::uint32_t initial_base = 0x80U;
    constexpr std::uint32_t effective = 0x84U;
    const std::array<std::uint32_t, 4> values{
        0x3FC00000U, // +1.5
        0xC0100000U, // -2.25
        0x00000000U, // +0
        0x80000000U, // -0
    };

    for (const std::uint32_t single_bits : values)
    {
        EspressoCore core(0x200U);
        core.state.gpr[9] = initial_base;
        core.state.fpr[0] = 0x1111222233334444ULL;
        core.state.fpr_ps1[0] = 0xAAAABBBBCCCCDDDDULL;
        core.state.cr = initial_cr;
        core.state.xer = initial_xer;
        core.state.lr = initial_lr;
        core.state.ctr = initial_ctr;
        core.memory.write32_be(effective, single_bits);
        core.memory.write32_be(0U, encode_lfsu(0U, 9U, 4));

        const RunResult result = core.run(1U);
        const float single = std::bit_cast<float>(single_bits);
        const std::uint64_t expected = std::bit_cast<std::uint64_t>(
            static_cast<double>(single));
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.fpr[0] == expected);
        assert(core.state.fpr_ps1[0] == expected);
        assert(core.state.gpr[9] == effective);
        assert(core.state.cr == initial_cr);
        assert(core.state.xer == initial_xer);
        assert(core.state.lr == initial_lr);
        assert(core.state.ctr == initial_ctr);

        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "lfsu");
        assert(history.has_effective_address && history.effective_address == effective);
        assert(history.has_fp_destination && history.fp_destination_register == 0U);
        assert(history.fp_destination_value == expected);
        assert(history.has_destination && history.destination_register == 9U);
        assert(history.destination_value == effective);
        const std::string trace = format_instruction_history(result);
        assert(trace.find("lfsu r9=0x00000080 [0x00000084]") != std::string::npos);
        assert(trace.find("-> r9=0x00000084 -> f0=") != std::string::npos);
    }

    // Signed negative displacement uses the original base, then updates it.
    EspressoCore negative_displacement(0x200U);
    negative_displacement.state.gpr[9] = 0x120U;
    negative_displacement.memory.write32_be(0x11CU, 0xC0490FDBU); // -pi single
    negative_displacement.memory.write32_be(0U, encode_lfsu(0U, 9U, -4));
    assert(negative_displacement.step() == StepResult::executed);
    assert(negative_displacement.state.gpr[9] == 0x11CU);
    assert(negative_displacement.state.fpr[0] == 0xC00921FB60000000ULL);
    assert(negative_displacement.state.fpr_ps1[0] == 0xC00921FB60000000ULL);

    // A failed read changes neither the base nor either FPR lane.
    EspressoCore fault_core(0x100U);
    constexpr std::uint32_t fault_base = 0xFFFFFFFCU;
    fault_core.state.gpr[9] = fault_base;
    fault_core.state.fpr[0] = 0x1111222233334444ULL;
    fault_core.state.fpr_ps1[0] = 0xAAAABBBBCCCCDDDDULL;
    fault_core.memory.write32_be(0U, encode_lfsu(0U, 9U, 0));
    const RunResult fault = fault_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s)") != std::string::npos);
    assert(fault.detail.find("0xFFFFFFFC") != std::string::npos);
    assert(fault_core.state.gpr[9] == fault_base);
    assert(fault_core.state.fpr[0] == 0x1111222233334444ULL);
    assert(fault_core.state.fpr_ps1[0] == 0xAAAABBBBCCCCDDDDULL);
    assert(fault.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed = fault.instruction_history.front();
    assert(failed.opcode_name == "lfsu");
    assert(failed.has_effective_address && failed.effective_address == fault_base);
    assert(failed.has_destination && failed.destination_register == 9U);
    assert(!failed.completed);
}

void floating_point_double_load_tests()
{
    const auto encode_lfd = [](std::uint8_t destination, std::uint8_t base,
                               std::int16_t displacement) {
        return 0xC8000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    constexpr std::uint32_t expected_cr = 0x12345678U;
    constexpr std::uint32_t expected_xer = 0xA00000A5U;
    constexpr std::uint32_t base = 0x80U;
    constexpr std::uint32_t address = 0x90U;
    const std::array<std::uint64_t, 7> raw_values{
        0x0000000000000000ULL, // +0.0
        0x8000000000000000ULL, // -0.0
        0x3FF4000000000000ULL, // +1.25
        0xBFFE000000000000ULL, // -1.875
        0x7FF0000000000000ULL, // +infinity
        0xFFF0000000000000ULL, // -infinity
        0x7FF8123456789ABCULL, // NaN payload
    };

    const DecodedInstruction wind_waker_lfd = decode(0xCBE10010U);
    assert(wind_waker_lfd.opcode == Opcode::load_double);
    assert(wind_waker_lfd.fp_register == 31U);
    assert(wind_waker_lfd.base == 1U);
    assert(wind_waker_lfd.immediate == 16);

    for (const std::uint64_t raw : raw_values)
    {
        EspressoCore core(0x200U);
        core.state.gpr[1] = base;
        core.state.fpr[31] = 0xDEADBEEF01234567ULL;
        core.state.fpr_ps1[31] = 0xCAFEBABE76543210ULL;
        core.state.cr = expected_cr;
        core.state.xer = expected_xer;
        core.memory.write64_be(address, raw);
        core.memory.write32_be(0U, encode_lfd(31U, 1U, 16));

        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.fpr[31] == raw);
        assert(core.state.fpr_ps1[31] == 0xCAFEBABE76543210ULL);
        assert(core.state.gpr[1] == base);
        assert(core.state.cr == expected_cr);
        assert(core.state.xer == expected_xer);

        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "lfd");
        assert(history.has_fp_destination && history.fp_destination_register == 31U);
        assert(history.fp_destination_value == raw);
        assert(history.has_effective_address && history.effective_address == address);
        assert(format_instruction_history(result).find(
            "lfd r1=0x00000080 [0x00000090] -> f31=") != std::string::npos);
    }

    EspressoCore negative_displacement(0x200U);
    negative_displacement.state.gpr[3] = 0xA0U;
    negative_displacement.memory.write64_be(0x90U, 0x0123456789ABCDEFULL);
    negative_displacement.memory.write32_be(0U, encode_lfd(5U, 3U, -16));
    assert(negative_displacement.step() == StepResult::executed);
    assert(negative_displacement.state.fpr[5] == 0x0123456789ABCDEFULL);
    assert(negative_displacement.state.gpr[3] == 0xA0U);

    EspressoCore zero_base(0x100U);
    zero_base.state.gpr[0] = 0x40U;
    zero_base.memory.write64_be(0x40U, 0xFEDCBA9876543210ULL);
    zero_base.memory.write32_be(0U, encode_lfd(2U, 0U, 0x40));
    assert(zero_base.step() == StepResult::executed);
    assert(zero_base.state.fpr[2] == 0xFEDCBA9876543210ULL);
    assert(zero_base.state.gpr[0] == 0x40U);

    EspressoCore fault_core(0x100U);
    constexpr std::uint32_t fault_address = 0xFCU;
    fault_core.state.gpr[1] = fault_address;
    fault_core.state.fpr[31] = 0x1111222233334444ULL;
    fault_core.state.fpr_ps1[31] = 0xAAAABBBBCCCCDDDDULL;
    fault_core.memory.write32_be(0U, encode_lfd(31U, 1U, 0));
    const RunResult fault = fault_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 8 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x000000FC") != std::string::npos);
    assert(fault_core.state.fpr[31] == 0x1111222233334444ULL);
    assert(fault_core.state.fpr_ps1[31] == 0xAAAABBBBCCCCDDDDULL);
    assert(fault.instruction_history.size() == 1U);
    assert(fault.instruction_history.front().has_effective_address);
    assert(fault.instruction_history.front().effective_address == fault_address);
    assert(!fault.instruction_history.front().completed);
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
    const auto encode_stfsu = [](std::uint8_t source, std::uint8_t base,
                                 std::int16_t displacement) {
        return 0xD4000000U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    EspressoCore positive_core(0x200U);
    positive_core.state.gpr[4] = 0x100U;
    positive_core.state.fpr[13] = 0x400A000000000000ULL; // 3.25
    positive_core.state.fpr_ps1[13] = 0x7FF8123456789ABCULL;
    positive_core.memory.write32_be(0, encode_stfs(13, 4, 4));
    const RunResult positive_result = positive_core.run(1);
    assert(positive_result.reason == StopReason::instruction_limit);
    assert(positive_core.memory.read32_be(0x104U) == 0x40500000U);
    assert(positive_core.memory.read8(0x104U) == 0x40U);
    assert(positive_core.memory.read8(0x105U) == 0x50U);
    assert(positive_core.memory.read8(0x106U) == 0x00U);
    assert(positive_core.memory.read8(0x107U) == 0x00U);
    assert(positive_core.state.fpr[13] == 0x400A000000000000ULL);
    assert(positive_core.state.fpr_ps1[13] == 0x7FF8123456789ABCULL);
    const std::string positive_trace = format_instruction_history(positive_result);
    assert(positive_trace.find(
        "stfs r4=0x00000100 f13=3.25 [0x400A000000000000] "
        "[0x00000104] -> mem32=0x40500000") != std::string::npos);

    // Negative D-form displacement and negative finite source.
    EspressoCore negative_core(0x200U);
    negative_core.state.gpr[12] = 0x120U;
    negative_core.state.fpr[7] = 0xC002000000000000ULL; // -2.25
    negative_core.state.fpr_ps1[7] = 0xFEDCBA9876543210ULL;
    negative_core.memory.write32_be(0, encode_stfs(7, 12, -4));
    assert(negative_core.step() == StepResult::executed);
    assert(negative_core.memory.read32_be(0x11CU) == 0xC0100000U);
    assert(negative_core.state.fpr[7] == 0xC002000000000000ULL);
    assert(negative_core.state.fpr_ps1[7] == 0xFEDCBA9876543210ULL);

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
    assert(round_trip_core.state.fpr_ps1[13] == 0xC00921FB60000000ULL);

    // A failed write reports the address/width and retains source FPR details.
    EspressoCore fault_core(0x100U);
    fault_core.state.gpr[12] = 0x1000U;
    fault_core.state.fpr[13] = 0xC00921FB60000000ULL;
    fault_core.state.fpr_ps1[13] = 0x0123456789ABCDEFULL;
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
    assert(fault_core.state.fpr_ps1[13] == 0x0123456789ABCDEFULL);

    // stfsu shares stfs's binary64-to-binary32 conversion and updates RA only
    // after the guest-memory store completes.
    EspressoCore update_positive(0x200U);
    update_positive.state.gpr[11] = 0x100U;
    update_positive.state.fpr[13] = 0x400A000000000000ULL; // 3.25
    update_positive.state.fpr_ps1[13] = 0x7FF8123456789ABCULL;
    update_positive.state.cr = 0x12345678U;
    update_positive.state.xer = 0xA00000A5U;
    update_positive.memory.write32_be(0, encode_stfsu(13, 11, 4));
    const RunResult update_positive_result = update_positive.run(1U);
    assert(update_positive_result.reason == StopReason::instruction_limit);
    assert(update_positive.memory.read32_be(0x104U) == 0x40500000U);
    assert(update_positive.memory.read8(0x104U) == 0x40U);
    assert(update_positive.memory.read8(0x105U) == 0x50U);
    assert(update_positive.state.gpr[11] == 0x104U);
    assert(update_positive.state.fpr[13] == 0x400A000000000000ULL);
    assert(update_positive.state.fpr_ps1[13] == 0x7FF8123456789ABCULL);
    assert(update_positive.state.cr == 0x12345678U);
    assert(update_positive.state.xer == 0xA00000A5U);
    const InstructionHistoryEntry& update_history =
        update_positive_result.instruction_history.front();
    assert(update_history.opcode_name == "stfsu");
    assert(update_history.has_effective_address && update_history.effective_address == 0x104U);
    assert(update_history.has_fp_source && update_history.fp_source_register == 13U);
    assert(update_history.fp_source_value == 0x400A000000000000ULL);
    assert(update_history.has_stored_single_value &&
           update_history.stored_single_value == 0x40500000U);
    assert(update_history.has_destination && update_history.destination_register == 11U);
    assert(update_history.destination_value == 0x104U);
    assert(format_instruction_history(update_positive_result).find(
        "stfsu r11=0x00000100 f13=3.25 [0x400A000000000000] "
        "[0x00000104] -> r11=0x00000104 -> mem32=0x40500000") != std::string::npos);

    EspressoCore update_negative(0x200U);
    update_negative.state.gpr[12] = 0x120U;
    update_negative.state.fpr[7] = 0xC002000000000000ULL; // -2.25
    update_negative.state.fpr_ps1[7] = 0xFEDCBA9876543210ULL;
    update_negative.memory.write32_be(0, encode_stfsu(7, 12, -4));
    assert(update_negative.step() == StepResult::executed);
    assert(update_negative.memory.read32_be(0x11CU) == 0xC0100000U);
    assert(update_negative.state.gpr[12] == 0x11CU);
    assert(update_negative.state.fpr[7] == 0xC002000000000000ULL);
    assert(update_negative.state.fpr_ps1[7] == 0xFEDCBA9876543210ULL);

    EspressoCore update_zero(0x100U);
    update_zero.state.gpr[4] = 0x40U;
    update_zero.state.fpr[2] = 0U;
    update_zero.state.fpr_ps1[2] = 0x8000000000000000ULL;
    update_zero.memory.write32_be(0, encode_stfsu(2, 4, 8));
    assert(update_zero.step() == StepResult::executed);
    assert(update_zero.memory.read32_be(0x48U) == 0U);
    assert(update_zero.state.gpr[4] == 0x48U);
    assert(update_zero.state.fpr_ps1[2] == 0x8000000000000000ULL);

    EspressoCore update_zero_base(0x100U);
    update_zero_base.state.fpr[13] = 0x4000000000000000ULL;
    update_zero_base.memory.write32_be(0, encode_stfsu(13, 0, 0x40));
    const RunResult rejected_zero_base = update_zero_base.run(1U);
    assert(rejected_zero_base.reason == StopReason::unsupported_instruction);
    assert(update_zero_base.memory.read32_be(0x40U) == 0U);
    assert(update_zero_base.state.gpr[0] == 0U);

    EspressoCore update_fault(0x100U);
    update_fault.state.gpr[11] = 0x100U;
    update_fault.state.fpr[13] = 0U;
    update_fault.memory.write32_be(0, encode_stfsu(13, 11, 0));
    const RunResult update_fault_result = update_fault.run(1U);
    assert(update_fault_result.reason == StopReason::memory_fault);
    assert(update_fault_result.detail.find("write 4 byte(s)") != std::string::npos);
    assert(update_fault.state.gpr[11] == 0x100U); // failed store did not update RA
    const InstructionHistoryEntry& failed_update =
        update_fault_result.instruction_history.front();
    assert(failed_update.opcode_name == "stfsu");
    assert(failed_update.has_effective_address && failed_update.effective_address == 0x100U);
    assert(failed_update.has_fp_source && failed_update.fp_source_register == 13U);
    assert(!failed_update.completed && !failed_update.has_stored_single_value);
}

void floating_point_double_store_tests()
{
    const auto encode_stfd = [](std::uint8_t source, std::uint8_t base,
                                std::int16_t displacement) {
        return 0xD8000000U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    const std::array<std::uint64_t, 7> raw_values{
        0x0000000000000000ULL, // +0.0
        0x8000000000000000ULL, // -0.0
        0x3FF4000000000000ULL, // +1.25
        0xBFFE000000000000ULL, // -1.875
        0x7FF0000000000000ULL, // +infinity
        0xFFF0000000000000ULL, // -infinity
        0x7FF8123456789ABCULL, // NaN with payload
    };
    for (const std::uint64_t raw : raw_values)
    {
        EspressoCore core(0x200U);
        constexpr std::uint32_t base_before = 0x80U;
        constexpr std::uint32_t address = 0x88U;
        constexpr std::uint32_t expected_cr = 0x12345678U;
        constexpr std::uint32_t expected_xer = 0xA00000A5U;
        core.state.gpr[1] = base_before;
        core.state.fpr[31] = raw;
        core.state.fpr_ps1[31] = 0xDEADBEEF01234567ULL;
        core.state.cr = expected_cr;
        core.state.xer = expected_xer;
        core.memory.write32_be(0U, encode_stfd(31U, 1U, 8));

        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.state.gpr[1] == base_before);
        assert(core.state.fpr[31] == raw);
        assert(core.state.fpr_ps1[31] == 0xDEADBEEF01234567ULL);
        assert(core.state.cr == expected_cr);
        assert(core.state.xer == expected_xer);
        for (std::uint32_t byte = 0; byte < 8U; ++byte)
        {
            const unsigned shift = static_cast<unsigned>((7U - byte) * 8U);
            assert(core.memory.read8(address + byte) ==
                   static_cast<std::uint8_t>(raw >> shift));
        }

        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "stfd");
        assert(history.has_fp_source && history.fp_source_register == 31U);
        assert(history.fp_source_value == raw);
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_stored_double_value && history.stored_double_value == raw);
        assert(!history.has_stored_single_value);
        const std::string trace = format_instruction_history(result);
        assert(trace.find("stfd r1=0x00000080 f31=") != std::string::npos);
        assert(trace.find("[0x00000088] -> mem64=0x") != std::string::npos);
    }

    // Verify exact architectural decode of Wind Waker's store.
    const DecodedInstruction wind_waker_stfd = decode(0xDBE10010U);
    assert(wind_waker_stfd.opcode == Opcode::store_double);
    assert(wind_waker_stfd.fp_register == 31U);
    assert(wind_waker_stfd.base == 1U);
    assert(wind_waker_stfd.immediate == 16);

    // Signed negative displacement.
    EspressoCore negative_displacement(0x200U);
    negative_displacement.state.gpr[3] = 0xA0U;
    negative_displacement.state.fpr[5] = 0x0123456789ABCDEFULL;
    negative_displacement.memory.write32_be(0U, encode_stfd(5U, 3U, -16));
    assert(negative_displacement.step() == StepResult::executed);
    assert(negative_displacement.memory.read8(0x90U) == 0x01U);
    assert(negative_displacement.memory.read8(0x97U) == 0xEFU);
    assert(negative_displacement.state.gpr[3] == 0xA0U);

    // rA == 0 supplies a zero base even when GPR0 contains a nonzero value.
    EspressoCore zero_base(0x100U);
    zero_base.state.gpr[0] = 0x40U;
    zero_base.state.fpr[2] = 0xFEDCBA9876543210ULL;
    zero_base.memory.write32_be(0U, encode_stfd(2U, 0U, 0x40));
    assert(zero_base.step() == StepResult::executed);
    assert(zero_base.memory.read8(0x40U) == 0xFEU);
    assert(zero_base.memory.read8(0x47U) == 0x10U);

    // The helper and instruction both support stores crossing flat/sparse memory.
    EspressoCore boundary_core(0x20U);
    boundary_core.memory.map_region(0x20U, 0x10U);
    boundary_core.state.gpr[2] = 0x1CU;
    boundary_core.state.fpr[4] = 0x0123456789ABCDEFULL;
    boundary_core.memory.write32_be(0U, encode_stfd(4U, 2U, 0));
    assert(boundary_core.step() == StepResult::executed);
    for (std::uint32_t byte = 0; byte < 8U; ++byte)
    {
        const unsigned shift = static_cast<unsigned>((7U - byte) * 8U);
        assert(boundary_core.memory.read8(0x1CU + byte) ==
               static_cast<std::uint8_t>(0x0123456789ABCDEFULL >> shift));
    }

    // The complete write is validated before any byte is changed.
    EspressoCore fault_core(0x100U);
    constexpr std::uint32_t fault_address = 0xFCU;
    fault_core.state.gpr[1] = fault_address;
    fault_core.state.fpr[31] = 0x7FF8123456789ABCULL;
    fault_core.memory.fill_bytes(fault_address, 4U, 0xA5U);
    fault_core.memory.write32_be(0U, encode_stfd(31U, 1U, 0));
    const RunResult fault = fault_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 8 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x000000FC") != std::string::npos);
    for (std::uint32_t byte = 0; byte < 4U; ++byte)
    {
        assert(fault_core.memory.read8(fault_address + byte) == 0xA5U);
    }
    assert(fault.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed_history = fault.instruction_history.front();
    assert(failed_history.opcode_name == "stfd");
    assert(failed_history.has_fp_source && failed_history.fp_source_register == 31U);
    assert(failed_history.fp_source_value == 0x7FF8123456789ABCULL);
    assert(failed_history.has_effective_address &&
           failed_history.effective_address == fault_address);
    assert(!failed_history.completed && !failed_history.has_stored_double_value);
    assert(fault_core.state.gpr[1] == fault_address);
    assert(fault_core.state.fpr[31] == 0x7FF8123456789ABCULL);

}

void floating_point_double_update_store_tests()
{
    const auto encode_stfdu = [](std::uint8_t source, std::uint8_t base,
                                 std::int16_t displacement) {
        return 0xDC000000U |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(displacement);
    };

    constexpr std::uint32_t expected_cr = 0x12345678U;
    constexpr std::uint32_t expected_xer = 0xA00000A5U;
    constexpr std::uint32_t expected_fpscr = 0x5A5AA55AU;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    const std::array<std::uint64_t, 9> raw_values{
        0x0000000000000000ULL, // +0.0
        0x8000000000000000ULL, // -0.0
        0x3FF0000000000000ULL, // +1.0
        0xBFF8000000000000ULL, // -1.5
        0x7FF0000000000000ULL, // +infinity
        0xFFF0000000000000ULL, // -infinity
        0x7FF8123456789ABCULL, // quiet NaN payload
        0x7FF0123456789ABCULL, // signaling NaN payload
        0x0123456789ABCDEFULL, // arbitrary raw bits
    };

    for (const std::uint64_t raw : raw_values)
    {
        EspressoCore core(0x200U);
        constexpr std::uint32_t base_before = 0x80U;
        constexpr std::uint32_t address = 0x88U;
        core.state.gpr[9] = base_before;
        core.state.fpr[13] = raw;
        core.state.fpr_ps1[13] = ps1_sentinel;
        core.state.cr = expected_cr;
        core.state.xer = expected_xer;
        core.state.fpscr = expected_fpscr;
        core.state.lr = 0x11223344U;
        core.state.ctr = 0x55667788U;
        core.memory.write32_be(0U, encode_stfdu(13U, 9U, 8));

        const RunResult result = core.run(1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(core.memory.read64_be(address) == raw);
        assert(core.state.gpr[9] == address);
        assert(core.state.fpr[13] == raw);
        assert(core.state.fpr_ps1[13] == ps1_sentinel);
        assert(core.state.cr == expected_cr);
        assert(core.state.xer == expected_xer);
        assert(core.state.fpscr == expected_fpscr);
        assert(core.state.lr == 0x11223344U);
        assert(core.state.ctr == 0x55667788U);

        const InstructionHistoryEntry& history = result.instruction_history.front();
        assert(history.opcode_name == "stfdu");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_fp_source && history.fp_source_register == 13U);
        assert(history.fp_source_value == raw);
        assert(history.has_stored_double_value && history.stored_double_value == raw);
        assert(history.has_destination && history.destination_register == 9U);
        assert(history.destination_value == address);
        assert(history.completed);
        const std::string trace = format_instruction_history(result);
        assert(trace.find("stfdu r9=0x00000080 f13=") != std::string::npos);
        assert(trace.find("[0x00000088] -> r9=0x00000088 -> mem64=0x") !=
               std::string::npos);
    }

    // Exact Wind Waker instruction and behavior: raw PS0 is stored, then r9 updates.
    const DecodedInstruction wind_waker = decode(0xDDA904E8U);
    assert(wind_waker.opcode == Opcode::store_double_update);
    assert(wind_waker.fp_register == 13U && wind_waker.base == 9U);
    assert(wind_waker.immediate == 1256);
    EspressoCore wind_waker_core(0x200U);
    wind_waker_core.state.gpr[9] = 0x104A0000U;
    wind_waker_core.state.fpr[13] = 0U;
    wind_waker_core.state.fpr_ps1[13] = ps1_sentinel;
    wind_waker_core.memory.map_region(0x104A0000U, 0x1000U);
    wind_waker_core.memory.write32_be(0U, 0xDDA904E8U);
    const RunResult wind_waker_result = wind_waker_core.run(1U);
    assert(wind_waker_result.reason == StopReason::instruction_limit);
    assert(wind_waker_core.memory.read64_be(0x104A04E8U) == 0U);
    assert(wind_waker_core.state.gpr[9] == 0x104A04E8U);
    assert(wind_waker_core.state.fpr[13] == 0U);
    assert(wind_waker_core.state.fpr_ps1[13] == ps1_sentinel);

    // Signed displacement is applied with 32-bit guest-address arithmetic.
    EspressoCore negative(0x200U);
    negative.state.gpr[3] = 0x100U;
    negative.state.fpr[5] = 0x0123456789ABCDEFULL;
    negative.memory.write32_be(0U, encode_stfdu(5U, 3U, -16));
    assert(negative.step() == StepResult::executed);
    assert(negative.memory.read64_be(0xF0U) == 0x0123456789ABCDEFULL);
    assert(negative.state.gpr[3] == 0xF0U);

    // Eight-byte store crosses from flat to adjacent sparse backing.
    EspressoCore boundary(0x20U);
    boundary.memory.map_region(0x20U, 0x20U);
    boundary.state.gpr[2] = 0x1CU;
    boundary.state.fpr[4] = 0x0123456789ABCDEFULL;
    boundary.memory.write32_be(0U, encode_stfdu(4U, 2U, 0));
    assert(boundary.step() == StepResult::executed);
    assert(boundary.memory.read64_be(0x1CU) == 0x0123456789ABCDEFULL);
    assert(boundary.state.gpr[2] == 0x1CU);

    // The full store validates before writing, and rA changes only on success.
    EspressoCore fault_core(0x100U);
    constexpr std::uint32_t fault_address = 0xFCU;
    constexpr std::uint32_t old_base = fault_address - 4U;
    constexpr std::uint64_t fault_source = 0x7FF0123456789ABCULL;
    fault_core.state.gpr[6] = old_base;
    fault_core.state.fpr[7] = fault_source;
    fault_core.state.fpr_ps1[7] = ps1_sentinel;
    fault_core.memory.fill_bytes(fault_address, 4U, 0xA5U);
    fault_core.memory.write32_be(0U, encode_stfdu(7U, 6U, 4));
    const RunResult fault = fault_core.run(1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 8 byte(s)") != std::string::npos);
    assert(fault.detail.find("0x000000FC") != std::string::npos);
    assert(fault.detail.find("rA=6 (0x000000F8)") != std::string::npos);
    assert(fault_core.state.gpr[6] == old_base);
    assert(fault_core.state.fpr[7] == fault_source);
    assert(fault_core.state.fpr_ps1[7] == ps1_sentinel);
    for (std::uint32_t i = 0; i < 4U; ++i)
    {
        assert(fault_core.memory.read8(fault_address + i) == 0xA5U);
    }
    assert(fault.instruction_history.size() == 1U);
    const InstructionHistoryEntry& failed = fault.instruction_history.front();
    assert(failed.opcode_name == "stfdu");
    assert(failed.has_effective_address && failed.effective_address == fault_address);
    assert(failed.has_fp_source && failed.fp_source_register == 7U);
    assert(failed.fp_source_value == fault_source);
    assert(!failed.completed && !failed.has_stored_double_value);
    assert(failed.has_destination && failed.destination_register == 6U);
}

void paired_single_merge10_tests()
{
    const auto encode_ps_merge10 = [](std::uint8_t destination, std::uint8_t source_a,
                                      std::uint8_t source_b, bool record = false) {
        return 0x10000000U |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (592U << 1U) | static_cast<std::uint32_t>(record);
    };

    constexpr std::uint32_t expected_cr = 0x12345678U;
    constexpr std::uint32_t expected_xer = 0xA00000A5U;

    // Distinct source and destination registers prove the lane selection.
    EspressoCore distinct(4U);
    distinct.state.fpr[5] = 0x5000000000000000ULL;
    distinct.state.fpr_ps1[5] = 0x5100000000000000ULL;
    distinct.state.fpr[9] = 0x9000000000000000ULL;
    distinct.state.fpr_ps1[9] = 0x9100000000000000ULL;
    distinct.state.fpr[3] = 0x3000000000000000ULL;
    distinct.state.fpr_ps1[3] = 0x3100000000000000ULL;
    distinct.state.cr = expected_cr;
    distinct.state.xer = expected_xer;
    distinct.memory.write32_be(0U, encode_ps_merge10(3U, 5U, 9U));
    const RunResult distinct_result = distinct.run(1U);
    assert(distinct_result.reason == StopReason::instruction_limit);
    assert(distinct.state.fpr[3] == 0x5100000000000000ULL);
    assert(distinct.state.fpr_ps1[3] == 0x9000000000000000ULL);
    assert(distinct.state.fpr[5] == 0x5000000000000000ULL);
    assert(distinct.state.fpr_ps1[5] == 0x5100000000000000ULL);
    assert(distinct.state.fpr[9] == 0x9000000000000000ULL);
    assert(distinct.state.fpr_ps1[9] == 0x9100000000000000ULL);
    assert(distinct.state.cr == expected_cr);
    assert(distinct.state.xer == expected_xer);
    const InstructionHistoryEntry& distinct_history =
        distinct_result.instruction_history.front();
    assert(distinct_history.opcode_name == "ps_merge10");
    assert(distinct_history.has_paired_fp_state);
    assert(distinct_history.paired_fp_destination_register == 3U);
    assert(distinct_history.paired_fp_source_a_register == 5U);
    assert(distinct_history.paired_fp_source_b_register == 9U);
    assert(distinct_history.paired_fp_source_a_ps0 == 0x5000000000000000ULL);
    assert(distinct_history.paired_fp_source_a_ps1 == 0x5100000000000000ULL);
    assert(distinct_history.paired_fp_source_b_ps0 == 0x9000000000000000ULL);
    assert(distinct_history.paired_fp_source_b_ps1 == 0x9100000000000000ULL);
    assert(distinct_history.paired_fp_destination_ps0 == 0x5100000000000000ULL);
    assert(distinct_history.paired_fp_destination_ps1 == 0x9000000000000000ULL);
    const std::string trace = format_instruction_history(distinct_result);
    assert(trace.find(
        "ps_merge10 fA=f5{ps0=0x5000000000000000,ps1=0x5100000000000000} "
        "fB=f9{ps0=0x9000000000000000,ps1=0x9100000000000000} "
        "-> f3{ps0=0x5100000000000000,ps1=0x9000000000000000}") !=
        std::string::npos);

    // Destination aliases frA: source lane values must be read before writes.
    EspressoCore aliases_a(4U);
    aliases_a.state.fpr[5] = 0x5000000000000000ULL;
    aliases_a.state.fpr_ps1[5] = 0x5100000000000000ULL;
    aliases_a.state.fpr[9] = 0x9000000000000000ULL;
    aliases_a.memory.write32_be(0U, encode_ps_merge10(5U, 5U, 9U));
    assert(aliases_a.step() == StepResult::executed);
    assert(aliases_a.state.fpr[5] == 0x5100000000000000ULL);
    assert(aliases_a.state.fpr_ps1[5] == 0x9000000000000000ULL);

    // Destination aliases frB.
    EspressoCore aliases_b(4U);
    aliases_b.state.fpr[5] = 0x5000000000000000ULL;
    aliases_b.state.fpr_ps1[5] = 0x5100000000000000ULL;
    aliases_b.state.fpr[9] = 0x9000000000000000ULL;
    aliases_b.state.fpr_ps1[9] = 0x9100000000000000ULL;
    aliases_b.memory.write32_be(0U, encode_ps_merge10(9U, 5U, 9U));
    assert(aliases_b.step() == StepResult::executed);
    assert(aliases_b.state.fpr[9] == 0x5100000000000000ULL);
    assert(aliases_b.state.fpr_ps1[9] == 0x9000000000000000ULL);

    // Same source register, distinct destination.
    EspressoCore same_sources(4U);
    same_sources.state.fpr[5] = 0x5000000000000000ULL;
    same_sources.state.fpr_ps1[5] = 0x5100000000000000ULL;
    same_sources.memory.write32_be(0U, encode_ps_merge10(3U, 5U, 5U));
    assert(same_sources.step() == StepResult::executed);
    assert(same_sources.state.fpr[3] == 0x5100000000000000ULL);
    assert(same_sources.state.fpr_ps1[3] == 0x5000000000000000ULL);

    // Wind Waker's all-aliased form swaps the two raw lanes.
    EspressoCore all_alias(4U);
    all_alias.state.fpr[31] = 0x1111111111111111ULL;
    all_alias.state.fpr_ps1[31] = 0x2222222222222222ULL;
    all_alias.state.cr = expected_cr;
    all_alias.state.xer = expected_xer;
    all_alias.memory.write32_be(0U, 0x13FFFCA0U);
    const RunResult all_alias_result = all_alias.run(1U);
    assert(all_alias_result.reason == StopReason::instruction_limit);
    assert(all_alias.state.fpr[31] == 0x2222222222222222ULL);
    assert(all_alias.state.fpr_ps1[31] == 0x1111111111111111ULL);
    assert(all_alias.state.cr == expected_cr);
    assert(all_alias.state.xer == expected_xer);
    assert(format_instruction_history(all_alias_result).find(
        "ps_merge10 fA=f31{ps0=0x1111111111111111,ps1=0x2222222222222222} "
        "fB=f31{ps0=0x1111111111111111,ps1=0x2222222222222222} "
        "-> f31{ps0=0x2222222222222222,ps1=0x1111111111111111}") !=
        std::string::npos);

    // Rc=1 requires FPSCR/CR1 behavior that is not implemented, so reject it.
    EspressoCore record_form(4U);
    record_form.state.fpr[31] = 0x1111111111111111ULL;
    record_form.state.fpr_ps1[31] = 0x2222222222222222ULL;
    record_form.memory.write32_be(0U, encode_ps_merge10(31U, 31U, 31U, true));
    const RunResult record_result = record_form.run(1U);
    assert(record_result.reason == StopReason::unsupported_instruction);
    assert(record_result.instruction_word == 0x13FFFCA1U);
    assert(record_form.state.fpr[31] == 0x1111111111111111ULL);
    assert(record_form.state.fpr_ps1[31] == 0x2222222222222222ULL);
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
            constexpr std::size_t rpx_instruction_limit = 10'000'000U;
            affogato::Emulator emulator;
            const auto image = emulator.load_rpx(file);
            std::cout << "Loaded RPX entry point 0x" << std::hex << image.entry_point
                      << " (" << std::dec << image.loaded_sections << " sections)\n";
            const auto& core = emulator.core();
            const std::uint32_t foreground_heap = core.base_heap_handles[8];
            const std::uint32_t mem2_heap = core.base_heap_handles[1];
            std::uint32_t foreground_head_before_run = 0U;
            std::uint32_t foreground_tail_before_run = 0U;
            if (foreground_heap != 0U && mem2_heap != 0U)
            {
                foreground_head_before_run =
                    core.memory.read32_be(foreground_heap + 0x40U);
                foreground_tail_before_run =
                    core.memory.read32_be(foreground_heap + 0x44U);
                const std::uint32_t aligned_head =
                    (foreground_head_before_run + 3U) & ~3U;
                const std::uint32_t alignment4_capacity =
                    aligned_head < foreground_tail_before_run
                    ? foreground_tail_before_run - aligned_head
                    : 0U;
                std::cout << "Heap layout: bump=[0x" << std::hex
                          << core.guest_heap_cursor << ",0x" << core.guest_heap_limit
                          << ") FRMH=0x" << foreground_heap << " tag=0x"
                          << core.memory.read32_be(foreground_heap)
                          << " data=[0x"
                          << core.memory.read32_be(foreground_heap + 0x18U)
                          << ",0x"
                          << core.memory.read32_be(foreground_heap + 0x1CU)
                          << ") head=0x" << foreground_head_before_run << " tail=0x"
                          << foreground_tail_before_run << " MEM2=[0x" << mem2_heap << ",0x"
                          << core.mem2_heap_region_end << ")\n"
                          << "MEMGetAllocatableSizeForFrmHeapEx(FRMH,+4) initial size=0x"
                          << alignment4_capacity << std::dec << '\n';
            }
            const bool nn_save_initialized_before_run = core.nn_save_initialized;
            const auto session_result = emulator.run(rpx_instruction_limit);
            const auto& execution = session_result.execution;
            if (foreground_heap != 0U && mem2_heap != 0U)
            {
                const std::uint32_t foreground_head_after =
                    core.memory.read32_be(foreground_heap + 0x40U);
                const std::uint32_t aligned_head_candidate =
                    (foreground_head_before_run + 3U) & ~3U;
                std::cout << "FRMH post-run: heap=0x" << std::hex
                          << foreground_heap << " head before=0x"
                          << foreground_head_before_run << " +4 start=0x"
                          << aligned_head_candidate << " head after=0x"
                          << foreground_head_after << " tail before=0x"
                          << foreground_tail_before_run
                          << " tail after=0x"
                          << core.memory.read32_be(foreground_heap + 0x44U)
                          << " MEM2 tag@0x" << mem2_heap << "=0x"
                          << core.memory.read32_be(mem2_heap) << std::dec << '\n';
            }
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
            std::cout << "Guest GPR3: 0x" << std::hex << core.state.gpr[3]
                      << "  GPR4: 0x" << core.state.gpr[4]
                      << "  GPR5: 0x" << core.state.gpr[5] << std::dec << '\n';
            std::cout << "nn_save initialized before execution: "
                      << (nn_save_initialized_before_run ? "true" : "false")
                      << "  after: "
                      << (core.nn_save_initialized ? "true" : "false") << '\n';
            std::cout << "FS initialized: "
                      << (core.fs_initialized ? "true" : "false")
                      << "  registered clients: " << core.fs_clients.size()
                      << "  next client handle: " << core.next_fs_client_handle << '\n';
            if (!core.fs_clients.empty())
            {
                constexpr std::uint32_t fs_client_body_fsm_offset = 0x1448U;
                constexpr std::uint32_t fs_client_body_queue_offset = 0x1480U;
                constexpr std::uint32_t fs_client_body_mutex_offset = 0x1560U;
                constexpr std::uint32_t fs_client_body_alarm_offset = 0x1590U;
                constexpr std::uint32_t fs_client_body_link_offset = 0x1614U;
                constexpr std::uint32_t fs_client_body_state_notification_fsm_offset =
                    0x1448U;
                constexpr std::uint32_t fs_state_notification_enabled_offset = 0x10U;
                constexpr std::uint32_t fs_state_notification_info_offset = 0x14U;
                constexpr std::uint32_t fs_state_notification_message_offset = 0x0CU;
                const auto& registration = core.fs_clients.front();
                const std::uint32_t body = registration.body_address;
                std::cout << "FSAddClient: client=0x" << std::hex
                          << registration.client_address << " body=0x" << body
                          << " handle=0x" << registration.synthetic_handle
                          << " status=0x00000000" << std::dec << '\n';
                std::cout << "  FSM=" << core.memory.read32_be(body + fs_client_body_fsm_offset)
                          << " volume=" << core.memory.read32_be(body + fs_client_body_fsm_offset + 4U)
                          << " queue(head,tail,max)=(0x" << std::hex
                          << core.memory.read32_be(body + fs_client_body_queue_offset)
                          << ",0x" << core.memory.read32_be(body + fs_client_body_queue_offset + 4U)
                          << ",0x" << core.memory.read32_be(body + fs_client_body_queue_offset + 0x3CU)
                          << ") queueMutex=0x"
                          << core.memory.read32_be(body + fs_client_body_queue_offset + 8U)
                          << " clientMutex=0x"
                          << core.memory.read32_be(body + fs_client_body_mutex_offset)
                          << " alarm=0x"
                          << core.memory.read32_be(body + fs_client_body_alarm_offset)
                          << " link(next,prev)=(0x"
                          << core.memory.read32_be(body + fs_client_body_link_offset)
                          << ",0x"
                          << core.memory.read32_be(body + fs_client_body_link_offset + 4U)
                          << ")" << std::dec << '\n';
                const std::uint32_t fsm =
                    body + fs_client_body_state_notification_fsm_offset;
                const std::uint32_t info =
                    fsm + fs_state_notification_info_offset;
                std::cout << "FSSetStateChangeNotification: enabled="
                          << core.memory.read32_be(
                                 fsm + fs_state_notification_enabled_offset)
                          << " callback=0x" << std::hex
                          << core.memory.read32_be(info)
                          << " param=0x" << core.memory.read32_be(info + 4U)
                          << " queue=0x" << core.memory.read32_be(info + 8U)
                          << " message.data=0x"
                          << core.memory.read32_be(
                                 info + fs_state_notification_message_offset)
                          << " message.type=0x"
                          << core.memory.read32_be(
                                 info + fs_state_notification_message_offset + 0x0CU)
                          << " client=0x" << core.memory.read32_be(info + 0x1CU)
                          << " state=0x" << core.memory.read32_be(info + 0x20U)
                          << std::dec << '\n';
            }
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
    add_carrying_tests();
    add_to_zero_extended_tests();
    subtract_from_extended_tests();
    multiply_high_word_unsigned_tests();
    divide_word_unsigned_tests();
    shift_right_word_tests();
    or_immediate_shifted_tests();
    extend_sign_byte_tests();
    extend_sign_halfword_tests();
    leaf_function_abi_tests();
    floating_point_load_tests();
    floating_point_single_update_tests();
    floating_point_double_load_tests();
    floating_point_store_tests();
    floating_point_double_store_tests();
    floating_point_double_update_store_tests();
    paired_single_merge10_tests();
    elf_loader_tests();
    rpx_loader_tests();
    hle_dispatch_tests();
    nn_olv_hle_tests();
    guest_mutex_tests();
    memset_hle_tests();
    memcpy_hle_tests();
    os_block_move_hle_tests();
    os_get_system_info_hle_tests();
    nn_save_init_hle_tests();
    fs_init_hle_tests();
    fs_add_client_hle_tests();
    fs_set_state_change_notification_hle_tests();
    os_get_system_time_hle_tests();
    os_set_exception_callback_hle_tests();
    os_get_thread_priority_hle_tests();
    os_init_message_queue_hle_tests();
    mem_get_base_heap_handle_hle_tests();
    mem_get_allocatable_size_for_frm_heap_ex_hle_tests();
    mem_alloc_from_frm_heap_ex_hle_tests();
    gx2_init_hle_tests();
    gx2_set_depth_stencil_control_hle_tests();
    gx2_set_stencil_mask_hle_tests();
    mem_get_total_free_size_for_exp_heap_hle_tests();
    mem_get_allocatable_size_for_exp_heap_ex_hle_tests();
    compare_and_conditional_branch_tests();
    instruction_sync_tests();
    floating_compare_unordered_tests();
    floating_divide_single_tests();
    floating_divide_double_tests();
    floating_add_single_tests();
    floating_multiply_add_single_tests();
    floating_multiply_single_tests();
    floating_move_register_tests();
    floating_negate_tests();
    floating_absolute_value_tests();
    floating_subtract_double_tests();
    floating_add_double_tests();
    floating_multiply_double_tests();
    floating_multiply_add_double_tests();
    floating_negative_multiply_subtract_double_tests();
    floating_round_to_single_tests();
    floating_convert_to_integer_word_zero_tests();
    load_store_tests();
    store_halfword_update_tests();
    store_halfword_indexed_tests();
    store_floating_as_integer_word_indexed_tests();
    load_halfword_zero_indexed_tests();
    load_byte_zero_indexed_tests();
    load_halfword_algebraic_tests();
    multiple_word_load_store_tests();
    load_string_word_immediate_tests();
    store_string_word_immediate_tests();
    function_call_and_stack_tests();
    return 0;
}
