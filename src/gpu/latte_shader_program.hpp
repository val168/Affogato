#pragma once

#include "latte_shader_interpreter.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
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

// Execute a straight-line shader made only of supported OP2 ALU clauses and
// normal exports. Inputs supplied in initial_registers stand in for stage
// inputs; fetch-shader calls, branch/loop control flow, and texture clauses
// are rejected.
[[nodiscard]] inline LatteShaderExecutionResult
execute_latte_shader_subset(
    std::span<const std::uint32_t> guest_big_endian_words,
    std::span<const LatteShaderRegister> initial_registers)
{
    const std::vector<LatteControlFlowRecord> control_flow =
        decode_latte_control_flow(guest_big_endian_words);
    if (control_flow.empty() || !control_flow.back().end_of_program)
    {
        throw std::runtime_error(
            "Latte shader subset requires a bounded end-of-program record");
    }

    LatteShaderExecutionResult result{};
    result.registers.assign(
        initial_registers.begin(),
        initial_registers.end());

    for (const LatteControlFlowRecord& record : control_flow)
    {
        if (record.opcode == LatteControlFlowOpcode::alu_clause)
        {
            const std::vector<LatteAluClauseSlot> slots =
                decode_latte_alu_clause_slots(guest_big_endian_words, record);
            static_cast<void>(execute_latte_alu_op2_clause(
                slots,
                result.registers));
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

            result.exports.push_back({
                record.opcode == LatteControlFlowOpcode::export_done,
                evaluate_latte_normal_export(
                    *record.export_fields,
                    result.registers)});
            continue;
        }

        throw std::runtime_error(
            "Latte shader subset does not support control-flow opcode " +
            std::to_string(static_cast<std::uint32_t>(record.opcode)) +
            " at DWORD " + std::to_string(record.word_offset));
    }

    return result;
}

} // namespace affogato::gpu
