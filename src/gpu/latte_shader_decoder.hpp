#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
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

enum class LatteExportType : std::uint8_t
{
    pixel,
    position,
    parameter,
    unused
};

enum class LatteControlFlowCondition : std::uint8_t
{
    active = 0U,
    false_predicate = 1U,
    boolean_constant = 2U,
    inverse_boolean_constant = 3U,
};

struct LatteCallFetchFields
{
    std::uint32_t address_8byte_units{};
    std::uint8_t pop_count{};
    LatteControlFlowCondition condition{};
    std::uint8_t call_count{};

    bool operator==(const LatteCallFetchFields&) const = default;
};

struct LatteExportFields
{
    LatteExportType type{};
    std::uint32_t array_base{};
    std::uint8_t read_write_gpr{};
    std::uint8_t index_gpr{};
    std::uint8_t burst_count{};
    std::array<std::uint8_t, 4U> component_selectors{};
    bool read_write_relative{};
    bool uses_buffer_encoding{};
    bool valid_pixel_mode{};
    bool whole_quad_mode{};

    bool operator==(const LatteExportFields&) const = default;
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
    std::optional<LatteCallFetchFields> call_fetch_fields;
    std::optional<LatteExportFields> export_fields;

    bool operator==(const LatteControlFlowRecord&) const = default;
};

[[nodiscard]] inline LatteExportFields decode_latte_export_fields(
    std::uint32_t word0,
    std::uint32_t word1)
{
    const std::uint32_t raw_type = (word0 >> 13U) & 0x03U;
    const auto type = static_cast<LatteExportType>(raw_type);
    const bool uses_buffer_encoding = ((word1 >> 12U) & 0x0FU) != 0U;

    std::array<std::uint8_t, 4U> component_selectors{};
    if (!uses_buffer_encoding)
    {
        component_selectors = {
            static_cast<std::uint8_t>(word1 & 0x07U),
            static_cast<std::uint8_t>((word1 >> 3U) & 0x07U),
            static_cast<std::uint8_t>((word1 >> 6U) & 0x07U),
            static_cast<std::uint8_t>((word1 >> 9U) & 0x07U)};
    }

    return {
        type,
        word0 & 0x1FFFU,
        static_cast<std::uint8_t>((word0 >> 15U) & 0x7FU),
        static_cast<std::uint8_t>((word0 >> 23U) & 0x7FU),
        static_cast<std::uint8_t>(((word1 >> 17U) & 0x0FU) + 1U),
        component_selectors,
        ((word0 >> 22U) & 1U) != 0U,
        uses_buffer_encoding,
        ((word1 >> 22U) & 1U) != 0U,
        ((word1 >> 30U) & 1U) != 0U};
}

[[nodiscard]] inline LatteCallFetchFields decode_latte_call_fetch_fields(
    std::uint32_t word0,
    std::uint32_t word1)
{
    return {
        word0,
        static_cast<std::uint8_t>(word1 & 0x07U),
        static_cast<LatteControlFlowCondition>((word1 >> 8U) & 0x03U),
        static_cast<std::uint8_t>((word1 >> 13U) & 0x3FU)};
}

struct LatteAluClauseSlot
{
    std::size_t word_offset{}; // Index of the slot's first captured DWORD.
    std::uint32_t word0{};
    std::uint32_t word1{};

    bool operator==(const LatteAluClauseSlot&) const = default;
};

struct LatteAluOp2Instruction
{
    std::size_t word_offset{};
    std::uint16_t opcode{};
    std::uint16_t source0_selector{};
    std::uint16_t source1_selector{};
    std::uint8_t source0_channel{};
    std::uint8_t source1_channel{};
    std::uint8_t index_mode{};
    std::uint8_t output_modifier{};
    std::uint8_t destination_gpr{};
    std::uint8_t destination_component{};
    bool source0_relative{};
    bool source1_relative{};
    bool source0_negate{};
    bool source1_negate{};
    bool source0_absolute{};
    bool source1_absolute{};
    bool update_execute_mask{};
    bool update_predicate{};
    std::uint8_t bank_swizzle{};
    std::uint8_t predicate_selection{};
    bool destination_relative{};
    bool destination_clamp{};
    bool write_mask{};
    bool last_in_group{};

    bool operator==(const LatteAluOp2Instruction&) const = default;
};

struct LatteAluOp2InstructionGroup
{
    std::size_t word_offset{};
    std::vector<LatteAluOp2Instruction> instructions;

    bool operator==(const LatteAluOp2InstructionGroup&) const = default;
};

// Decode verified control-flow records and the supported ALU-clause header.
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

        LatteControlFlowRecord record{
            word_offset,
            word0,
            word1,
            opcode,
            end_of_program,
            clause_address_8byte_units,
            alu_slot_count,
            std::nullopt,
            std::nullopt};

        if (opcode == LatteControlFlowOpcode::call_fetch_shader)
        {
            record.call_fetch_fields = decode_latte_call_fetch_fields(
                word0,
                word1);
        }
        if (opcode == LatteControlFlowOpcode::export_data ||
            opcode == LatteControlFlowOpcode::export_done)
        {
            record.export_fields = decode_latte_export_fields(word0, word1);
        }

        records.push_back(std::move(record));

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

// Decode the common OP2 field layout without assigning semantics to its opcode.
// Cemu's Latte ISA definitions distinguish OP2/OP3 using word1 bits 17:13.
[[nodiscard]] inline LatteAluOp2Instruction decode_latte_alu_op2_instruction(
    const LatteAluClauseSlot& slot)
{
    constexpr std::uint32_t op3_encoding_shift = 13U;
    constexpr std::uint32_t op3_encoding_mask = 0x1FU;
    constexpr std::uint32_t first_op3_encoding = 0x08U;

    const std::uint32_t op3_encoding =
        (slot.word1 >> op3_encoding_shift) & op3_encoding_mask;
    if (op3_encoding >= first_op3_encoding)
    {
        throw std::runtime_error(
            "Latte ALU slot uses OP3 encoding, not the OP2 layout");
    }

    constexpr std::uint32_t source_selector_mask = 0x1FFU;
    constexpr std::uint32_t source_channel_mask = 0x03U;
    constexpr std::uint32_t destination_gpr_mask = 0x7FU;
    constexpr std::uint32_t destination_component_mask = 0x03U;

    return {
        slot.word_offset,
        static_cast<std::uint16_t>((slot.word1 >> 7U) & 0x7FFU),
        static_cast<std::uint16_t>(slot.word0 & source_selector_mask),
        static_cast<std::uint16_t>((slot.word0 >> 13U) & source_selector_mask),
        static_cast<std::uint8_t>((slot.word0 >> 10U) & source_channel_mask),
        static_cast<std::uint8_t>((slot.word0 >> 23U) & source_channel_mask),
        static_cast<std::uint8_t>((slot.word0 >> 26U) & 0x07U),
        static_cast<std::uint8_t>((slot.word1 >> 5U) & 0x03U),
        static_cast<std::uint8_t>((slot.word1 >> 21U) & destination_gpr_mask),
        static_cast<std::uint8_t>(
            (slot.word1 >> 29U) & destination_component_mask),
        ((slot.word0 >> 9U) & 1U) != 0U,
        ((slot.word0 >> 22U) & 1U) != 0U,
        ((slot.word0 >> 12U) & 1U) != 0U,
        ((slot.word0 >> 25U) & 1U) != 0U,
        (slot.word1 & 1U) != 0U,
        ((slot.word1 >> 1U) & 1U) != 0U,
        ((slot.word1 >> 2U) & 1U) != 0U,
        ((slot.word1 >> 3U) & 1U) != 0U,
        static_cast<std::uint8_t>((slot.word1 >> 18U) & 0x07U),
        static_cast<std::uint8_t>((slot.word0 >> 29U) & 0x03U),
        ((slot.word1 >> 28U) & 1U) != 0U,
        ((slot.word1 >> 31U) & 1U) != 0U,
        ((slot.word1 >> 4U) & 1U) != 0U,
        ((slot.word0 >> 31U) & 1U) != 0U};
}

// Split OP2 slots into LAST-delimited groups when no literal payload is used.
[[nodiscard]] inline std::vector<LatteAluOp2InstructionGroup>
decode_latte_alu_op2_groups(std::span<const LatteAluClauseSlot> slots)
{
    constexpr std::uint16_t literal_source_selector = 253U;

    std::vector<LatteAluOp2InstructionGroup> groups;
    LatteAluOp2InstructionGroup current_group;

    for (const auto& slot : slots)
    {
        const auto instruction = decode_latte_alu_op2_instruction(slot);
        if (instruction.source0_selector == literal_source_selector ||
            instruction.source1_selector == literal_source_selector)
        {
            throw std::runtime_error(
                "Latte ALU group decoding does not support literal operands");
        }

        if (current_group.instructions.empty())
        {
            current_group.word_offset = slot.word_offset;
        }
        current_group.instructions.push_back(instruction);

        if (instruction.last_in_group)
        {
            groups.push_back(std::move(current_group));
            current_group = {};
        }
    }

    if (!current_group.instructions.empty())
    {
        throw std::runtime_error(
            "Latte ALU clause ends before the final instruction-group marker");
    }

    return groups;
}

} // namespace affogato::gpu
