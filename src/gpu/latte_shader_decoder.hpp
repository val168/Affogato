#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace affogato::gpu
{

enum class LatteControlFlowOpcode : std::uint8_t
{
    call_fetch_shader = 0x13U,
    export_data = 0x27U,
    export_done = 0x28U,
    alu_clause = 0x48U
};

struct LatteControlFlowRecord
{
    std::size_t word_offset{}; // Index of this record's first captured DWORD.
    std::uint32_t word0{};
    std::uint32_t word1{};
    LatteControlFlowOpcode opcode{};
    bool end_of_program{};
    std::uint32_t clause_address_8byte_units{};
    std::uint32_t alu_slot_count{};

    bool operator==(const LatteControlFlowRecord&) const = default;
};

struct LatteAluClauseSlot
{
    std::size_t word_offset{}; // Index of the slot's first captured DWORD.
    std::uint32_t word0{};
    std::uint32_t word1{};

    bool operator==(const LatteAluClauseSlot&) const = default;
};

// Decode the verified non-ALU control-flow records in a shader program.
// GX2 shader memory is captured as big-endian CPU words; Latte instruction
// DWORDs use the little-endian word values emitted in GFD shader programs.
[[nodiscard]] inline std::vector<LatteControlFlowRecord>
decode_latte_control_flow(
    std::span<const std::uint32_t> guest_big_endian_words)
{
    constexpr std::size_t words_per_record = 2U;
    constexpr std::uint32_t opcode_shift = 23U;
    constexpr std::uint32_t opcode_mask = 0x7FU;
    constexpr std::uint32_t end_of_program_shift = 21U;

    if ((guest_big_endian_words.size() % words_per_record) != 0U)
    {
        throw std::runtime_error(
            "Latte shader control-flow program has an incomplete DWORD pair");
    }

    std::vector<LatteControlFlowRecord> records;

    for (std::size_t word_offset = 0U;
         word_offset < guest_big_endian_words.size();
         word_offset += words_per_record)
    {
        // The GFD fixture stores each Latte DWORD little-endian, while the
        // guest CPU helper captured it as a big-endian integer.
        const std::uint32_t word0 =
            std::byteswap(guest_big_endian_words[word_offset]);
        const std::uint32_t word1 =
            std::byteswap(guest_big_endian_words[word_offset + 1U]);
        const std::uint32_t raw_cf_opcode =
            (word1 >> opcode_shift) & opcode_mask;
        const std::uint32_t raw_opcode = raw_cf_opcode < 0x40U
            ? raw_cf_opcode
            : (((word1 >> 26U) & 0x0FU) | 0x40U);

        LatteControlFlowOpcode opcode{};
        switch (raw_opcode)
        {
        case static_cast<std::uint32_t>(
            LatteControlFlowOpcode::call_fetch_shader):
            opcode = LatteControlFlowOpcode::call_fetch_shader;
            break;
        case static_cast<std::uint32_t>(LatteControlFlowOpcode::export_data):
            opcode = LatteControlFlowOpcode::export_data;
            break;
        case static_cast<std::uint32_t>(LatteControlFlowOpcode::export_done):
            opcode = LatteControlFlowOpcode::export_done;
            break;
        case static_cast<std::uint32_t>(LatteControlFlowOpcode::alu_clause):
            opcode = LatteControlFlowOpcode::alu_clause;
            break;
        default:
            throw std::runtime_error(
                "Unsupported Latte shader control-flow opcode " +
                std::to_string(raw_opcode) + " at DWORD " +
                std::to_string(word_offset));
        }

        const bool is_alu_clause = opcode == LatteControlFlowOpcode::alu_clause;
        const bool end_of_program = !is_alu_clause &&
            ((word1 >> end_of_program_shift) & 1U) != 0U;
        std::uint32_t clause_address_8byte_units = 0U;
        std::uint32_t alu_slot_count = 0U;

        if (is_alu_clause)
        {
            constexpr std::uint32_t clause_address_mask = 0x003FFFFFU;
            constexpr std::uint32_t alu_slot_count_shift = 18U;
            constexpr std::uint32_t alu_slot_count_mask = 0x7FU;

            clause_address_8byte_units = word0 & clause_address_mask;
            alu_slot_count =
                ((word1 >> alu_slot_count_shift) & alu_slot_count_mask) + 1U;

            const std::uint64_t clause_end_8byte_units =
                static_cast<std::uint64_t>(clause_address_8byte_units) +
                alu_slot_count;
            const std::uint64_t program_size_8byte_units =
                guest_big_endian_words.size() / words_per_record;
            if (clause_end_8byte_units > program_size_8byte_units)
            {
                throw std::runtime_error(
                    "Latte shader ALU clause exceeds the captured program");
            }
        }

        records.push_back({
            word_offset,
            word0,
            word1,
            opcode,
            end_of_program,
            clause_address_8byte_units,
            alu_slot_count});

        if (end_of_program)
        {
            return records;
        }
    }

    throw std::runtime_error(
        "Latte shader control-flow program is missing its end marker");
}

// Extract the bounded raw 64-bit slots referenced by a decoded ALU clause.
// Opcode fields and VLIW execution semantics are intentionally not interpreted.
[[nodiscard]] inline std::vector<LatteAluClauseSlot>
decode_latte_alu_clause_slots(
    std::span<const std::uint32_t> guest_big_endian_words,
    const LatteControlFlowRecord& clause_record)
{
    constexpr std::size_t words_per_slot = 2U;

    if (clause_record.opcode != LatteControlFlowOpcode::alu_clause)
    {
        throw std::runtime_error(
            "Latte ALU slot extraction requires an ALU clause record");
    }

    if ((guest_big_endian_words.size() % words_per_slot) != 0U)
    {
        throw std::runtime_error(
            "Latte shader ALU clause has an incomplete DWORD pair");
    }

    const std::uint64_t clause_end_8byte_units =
        static_cast<std::uint64_t>(clause_record.clause_address_8byte_units) +
        clause_record.alu_slot_count;
    const std::uint64_t program_size_8byte_units =
        guest_big_endian_words.size() / words_per_slot;
    if (clause_record.alu_slot_count == 0U ||
        clause_end_8byte_units > program_size_8byte_units)
    {
        throw std::runtime_error(
            "Latte shader ALU clause exceeds the captured program");
    }

    const std::size_t clause_start_word_offset =
        static_cast<std::size_t>(clause_record.clause_address_8byte_units) *
        words_per_slot;
    std::vector<LatteAluClauseSlot> slots;
    slots.reserve(clause_record.alu_slot_count);

    for (std::size_t slot_index = 0U;
         slot_index < clause_record.alu_slot_count;
         ++slot_index)
    {
        const std::size_t word_offset =
            clause_start_word_offset + slot_index * words_per_slot;
        slots.push_back({
            word_offset,
            std::byteswap(guest_big_endian_words[word_offset]),
            std::byteswap(guest_big_endian_words[word_offset + 1U])});
    }

    return slots;
}

} // namespace affogato::gpu
