#include "cpu/espresso/decoder.hpp"
#include "cpu/espresso/cafe_os_hle.hpp"
#include "cpu/espresso/elf_loader.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/rpx_loader.hpp"
#include "cpu_state_test.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
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

[[nodiscard]] std::vector<std::uint8_t> make_minimal_compressed_rpx()
{
    constexpr std::uint32_t code_address = 0x02000000U;
    constexpr std::size_t header_size = 52;
    constexpr std::size_t section_header_size = 40;
    constexpr std::size_t section_count = 8;
    constexpr std::size_t section_table_offset = header_size;
    constexpr std::size_t text_offset = 384;
    const std::array<std::uint8_t, 12> code{
        0x38, 0x63, 0x00, 0x05, // addi r3, r3, 5
        0x4E, 0x80, 0x00, 0x20, // blr
        0, 0, 0, 0, // GHS relocation test slot
    };

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
    const std::size_t strtab_offset = symtab_offset + 16;
    const std::size_t rela_offset = (strtab_offset + 1U + 3U) & ~std::size_t{3U};
    std::vector<std::uint8_t> file(rela_offset + 24, 0);

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
    set_section(5, 41, 2, 0, 0, static_cast<std::uint32_t>(symtab_offset), 16, 4, 16, 6, 1);
    set_section(6, 49, 3, 0, 0, static_cast<std::uint32_t>(strtab_offset), 1, 1, 0);
    set_section(7, 57, 4, 0, 0, static_cast<std::uint32_t>(rela_offset), 24, 4, 12, 5, 1);

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
    std::copy(names.begin(), names.end(), file.begin() + names_offset);
    set_be32(file, fileinfo_offset, 0xCAFE0402U);
    return file;
}

void decoder_tests()
{
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

    const DecodedInstruction bit_and = decode(0x7C882838U); // and r8, r4, r5
    assert(bit_and.opcode == Opcode::bitwise_and);
    assert(bit_and.source == 4);
    assert(bit_and.destination == 8);
    assert(bit_and.base == 5);

    const DecodedInstruction bit_xor = decode(0x7C892A78U); // xor r9, r4, r5
    assert(bit_xor.opcode == Opcode::bitwise_xor);
    assert(bit_xor.source == 4);
    assert(bit_xor.destination == 9);
    assert(bit_xor.base == 5);

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

    const DecodedInstruction cmpw = decode(0x7E832000U); // cmpw cr5, r3, r4
    assert(cmpw.opcode == Opcode::compare_signed_register);
    assert(cmpw.cr_field == 5);
    assert(cmpw.base == 3);
    assert(cmpw.source == 4);

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

    const DecodedInstruction stw = decode(0x90610000U); // stw r3, 0(r1)
    assert(stw.opcode == Opcode::store_word);
    assert(stw.destination == 3);

    const DecodedInstruction lbz = decode(0x88810002U); // lbz r4, 2(r1)
    assert(lbz.opcode == Opcode::load_byte_zero);
    assert(lbz.destination == 4);
    assert(lbz.base == 1);
    assert(lbz.immediate == 2);

    const DecodedInstruction stb = decode(0x98610004U); // stb r3, 4(r1)
    assert(stb.opcode == Opcode::store_byte);
    assert(stb.destination == 3);

    const DecodedInstruction lhz = decode(0xA0A10002U); // lhz r5, 2(r1)
    assert(lhz.opcode == Opcode::load_halfword_zero);
    assert(lhz.destination == 5);

    const DecodedInstruction sth = decode(0xB0610006U); // sth r3, 6(r1)
    assert(sth.opcode == Opcode::store_halfword);
    assert(sth.destination == 3);

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
}

void interpreter_tests()
{
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
    assert(core.state.gpr[3] == 7);
    assert(core.state.gpr[4] == 8);
    assert(core.state.cia == 0x14);

    EspressoCore link_core(0x10);
    // bl +8
    link_core.memory.write32_be(0, 0x48000009U);

    assert(link_core.step() == StepResult::executed);
    assert(link_core.state.cia == 8);
    assert(link_core.state.lr == 4);
}

void integer_alu_tests()
{
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

    // rlwinm masks may wrap across the word boundary: MB=28 through ME=3.
    EspressoCore wrapping_rotate_core(8);
    wrapping_rotate_core.state.gpr[4] = 0xF000000FU;
    // rlwinm r3, r4, 0, 28, 3
    wrapping_rotate_core.memory.write32_be(0, 0x54830706U);
    assert(wrapping_rotate_core.step() == StepResult::executed);
    assert(wrapping_rotate_core.state.gpr[3] == 0xF000000FU);
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
    assert(core.state.cia == entry_point);
    assert(core.memory.read32_be(entry_point) == 0x38630005U);
    assert(core.memory.read32_be(entry_point + 4) == 0x4E800020U);
    assert(core.memory.read32_be(entry_point + 8) == 0xFDFFFFF6U);

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
}

void hle_dispatch_tests()
{
    EspressoCore core(0x100);
    register_coreinit_hle(core.hle);

    const std::uint32_t debugger_check =
        core.hle.bind_import("coreinit", "OSIsDebuggerInitialized");
    core.state.cia = debugger_check;
    core.state.lr = 0x40;
    core.state.gpr[3] = 1;
    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0);
    assert(core.state.cia == 0x40);

    const std::uint32_t missing_import = core.hle.bind_import("coreinit", "OSFatal");
    core.state.cia = missing_import;
    core.state.lr = 0x44;
    assert(core.step() == StepResult::unimplemented_hle_call);
    assert(core.state.cia == missing_import);
    assert(core.hle.last_unimplemented_call() == "coreinit::OSFatal");
}

void compare_and_conditional_branch_tests()
{
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
            EspressoCore core(0x10004000U);
            register_coreinit_hle(core.hle);
            const RpxLoadResult result = load_rpx32_powerpc(core, file);
            std::cout << "Loaded RPX entry point 0x" << std::hex << result.entry_point
                      << " (" << std::dec << result.loaded_sections << " sections)\n";
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
    elf_loader_tests();
    rpx_loader_tests();
    hle_dispatch_tests();
    compare_and_conditional_branch_tests();
    load_store_tests();
    function_call_and_stack_tests();
    return 0;
}
