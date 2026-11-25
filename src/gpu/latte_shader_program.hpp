#pragma once

#include "latte_shader_interpreter.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace affogato::gpu
{

struct LatteShaderExportBatch
{
    bool completes_type{};
    std::vector<LatteShaderExportValue> values;

    bool operator==(const LatteShaderExportBatch&) const = default;
};

struct LatteShaderExecutionResult
{
    std::vector<LatteShaderRegister> registers;
    std::vector<LatteShaderExportBatch> exports;

    bool operator==(const LatteShaderExecutionResult&) const = default;
};

struct LatteShaderIrAluClause
{
    std::vector<LatteAluOp2InstructionGroup> groups;
    std::vector<LatteAluOp3InstructionGroup> op3_groups;

    bool operator==(const LatteShaderIrAluClause&) const = default;
};

struct LatteShaderIrExport
{
    bool completes_type{};
    LatteExportFields fields;

    bool operator==(const LatteShaderIrExport&) const = default;
};

using LatteShaderIrInstruction =
    std::variant<LatteShaderIrAluClause, LatteShaderIrExport>;

struct LatteShaderIrProgram
{
    std::vector<LatteShaderIrInstruction> instructions;

    bool operator==(const LatteShaderIrProgram&) const = default;
};

// Decode only the straight-line CPU subset: OP2 or OP3 ALU clauses and normal
// exports. A vertex caller may declare its leading canonical CALL_FS handled
// separately by the captured fetch-shader subset.
[[nodiscard]] inline LatteShaderIrProgram decode_latte_shader_ir_subset(
    std::span<const std::uint32_t> guest_big_endian_words,
    bool fetch_subroutine_already_executed = false)
{
    const std::vector<LatteControlFlowRecord> control_flow =
        decode_latte_control_flow(guest_big_endian_words);
    if (control_flow.empty() || !control_flow.back().end_of_program)
    {
        throw std::runtime_error(
            "Latte shader subset requires a bounded end-of-program record");
    }

    std::size_t first_record_index = 0U;
    if (fetch_subroutine_already_executed)
    {
        if (control_flow.front().opcode !=
                LatteControlFlowOpcode::call_fetch_shader ||
            !control_flow.front().call_fetch_fields.has_value())
        {
            if (std::ranges::any_of(
                    control_flow,
                    [](const LatteControlFlowRecord& record)
                    {
                        return record.opcode ==
                            LatteControlFlowOpcode::call_fetch_shader;
                    }))
            {
                throw std::runtime_error(
                    "Latte vertex shader fetch call must be the first control-flow record");
            }
        }
        else
        {
            const LatteCallFetchFields& call_fields =
                *control_flow.front().call_fetch_fields;
            if (call_fields.address_8byte_units != 0U ||
                call_fields.pop_count != 0U ||
                call_fields.condition != LatteControlFlowCondition::active ||
                call_fields.call_count != 0U ||
                control_flow.front().end_of_program)
            {
                throw std::runtime_error(
                    "Latte vertex shader fetch call uses unsupported address, condition, or stack fields");
            }
            if (std::ranges::any_of(
                    std::span<const LatteControlFlowRecord>(control_flow).subspan(1U),
                    [](const LatteControlFlowRecord& record)
                    {
                        return record.opcode ==
                            LatteControlFlowOpcode::call_fetch_shader;
                    }))
            {
                throw std::runtime_error(
                    "Latte vertex shader subset supports only one leading fetch call");
            }

            first_record_index = 1U;
        }
    }

    LatteShaderIrProgram program{};
    program.instructions.reserve(control_flow.size() - first_record_index);
    for (std::size_t record_index = first_record_index;
         record_index < control_flow.size();
         ++record_index)
    {
        const LatteControlFlowRecord& record = control_flow[record_index];
        if (record.opcode == LatteControlFlowOpcode::alu_clause)
        {
            const std::vector<LatteAluClauseSlot> slots =
                decode_latte_alu_clause_slots(guest_big_endian_words, record);
            bool contains_op2 = false;
            bool contains_op3 = false;
            for (const LatteAluClauseSlot& slot : slots)
            {
                const std::uint32_t opcode_encoding =
                    (slot.word1 >> latte_alu_opcode_encoding_shift) &
                    latte_alu_opcode_encoding_mask;
                contains_op3 = contains_op3 ||
                    opcode_encoding >= latte_alu_first_op3_encoding;
                contains_op2 = contains_op2 ||
                    opcode_encoding < latte_alu_first_op3_encoding;
            }

            if (contains_op2 && contains_op3)
            {
                throw std::runtime_error(
                    "Latte shader subset does not support mixed OP2/OP3 ALU clauses");
            }

            std::vector<LatteAluOp2InstructionGroup> groups;
            std::vector<LatteAluOp3InstructionGroup> op3_groups;
            if (contains_op3)
            {
                op3_groups = decode_latte_alu_op3_groups(slots);
            }
            else
            {
                groups = decode_latte_alu_op2_groups(slots);
            }

            if (groups.empty() && op3_groups.empty())
            {
                throw std::runtime_error(
                    "Latte shader subset requires at least one supported ALU group per clause");
            }
            program.instructions.emplace_back(
                LatteShaderIrAluClause{
                    std::move(groups),
                    std::move(op3_groups)});
            continue;
        }

        if (record.opcode == LatteControlFlowOpcode::export_data ||
            record.opcode == LatteControlFlowOpcode::export_done)
        {
            if (!record.export_fields.has_value())
            {
                throw std::runtime_error(
                    "Latte shader export record has no decoded fields");
            }
            program.instructions.emplace_back(LatteShaderIrExport{
                record.opcode == LatteControlFlowOpcode::export_done,
                *record.export_fields});
            continue;
        }

        throw std::runtime_error(
            "Latte shader subset does not support control-flow opcode " +
            std::to_string(static_cast<std::uint32_t>(record.opcode)) +
            " at DWORD " + std::to_string(record.word_offset));
    }

    return program;
}

namespace detail
{

[[nodiscard]] inline LatteShaderExecutionResult execute_latte_shader_subset_impl(
    std::span<const std::uint32_t> guest_big_endian_words,
    std::span<const LatteShaderRegister> initial_registers,
    bool fetch_subroutine_already_executed,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    const LatteShaderIrProgram program = decode_latte_shader_ir_subset(
        guest_big_endian_words,
        fetch_subroutine_already_executed);

    LatteShaderExecutionResult result{};
    result.registers.assign(
        initial_registers.begin(),
        initial_registers.end());

    for (const LatteShaderIrInstruction& instruction : program.instructions)
    {
        if (const auto* alu_clause =
                std::get_if<LatteShaderIrAluClause>(&instruction))
        {
            if (!alu_clause->groups.empty() &&
                !alu_clause->op3_groups.empty())
            {
                throw std::runtime_error(
                    "Latte shader IR clause contains both OP2 and OP3 groups");
            }

            if (!alu_clause->op3_groups.empty())
            {
                static_cast<void>(execute_latte_alu_op3_clause(
                    alu_clause->op3_groups,
                    result.registers,
                    constant_file));
            }
            else
            {
                static_cast<void>(execute_latte_alu_op2_clause(
                    alu_clause->groups,
                    result.registers,
                    constant_file));
            }
            continue;
        }

        const auto& export_instruction =
            std::get<LatteShaderIrExport>(instruction);
        result.exports.push_back({
            export_instruction.completes_type,
            evaluate_latte_normal_export(
                export_instruction.fields,
                result.registers)});
    }

    return result;
}

} // namespace detail

// Execute a straight-line shader made only of supported OP2 ALU clauses and
// normal exports. Inputs supplied in initial_registers stand in for stage
// inputs; constant_registers supplies absolute DX9 constant-file vectors.
// Fetch-shader calls, branch/loop control flow, and texture clauses are
// rejected.
[[nodiscard]] inline LatteShaderExecutionResult
execute_latte_shader_subset(
    std::span<const std::uint32_t> guest_big_endian_words,
    std::span<const LatteShaderRegister> initial_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    return detail::execute_latte_shader_subset_impl(
        guest_big_endian_words,
        initial_registers,
        false,
        constant_file);
}

// Execute the straight-line vertex-shader subset after its bound fetch program
// was handled separately. Only one leading unconditional CALL_FS with a zero
// relative address and no stack change is elided; other call forms are rejected.
[[nodiscard]] inline LatteShaderExecutionResult
execute_latte_vertex_shader_program_subset(
    std::span<const std::uint32_t> guest_big_endian_words,
    std::span<const LatteShaderRegister> initial_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    return detail::execute_latte_shader_subset_impl(
        guest_big_endian_words,
        initial_registers,
        true,
        constant_file);
}

} // namespace affogato::gpu
