#pragma once

#include "latte_shader_interpreter.hpp"

#include <algorithm>
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

namespace detail
{

[[nodiscard]] inline LatteShaderExecutionResult execute_latte_shader_subset_impl(
    std::span<const std::uint32_t> guest_big_endian_words,
    std::span<const LatteShaderRegister> initial_registers,
    bool fetch_subroutine_already_executed,
    const LatteShaderConstantFile* constant_file = nullptr)
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

    LatteShaderExecutionResult result{};
    result.registers.assign(
        initial_registers.begin(),
        initial_registers.end());

    for (std::size_t record_index = first_record_index;
         record_index < control_flow.size();
         ++record_index)
    {
        const LatteControlFlowRecord& record = control_flow[record_index];
        if (record.opcode == LatteControlFlowOpcode::alu_clause)
        {
            const std::vector<LatteAluClauseSlot> slots =
                decode_latte_alu_clause_slots(guest_big_endian_words, record);
            static_cast<void>(execute_latte_alu_op2_clause(
                slots,
                result.registers,
                constant_file));
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
