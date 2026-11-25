#pragma once

#include "latte_shader_program.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace affogato::gpu
{

namespace detail
{

class LatteSpirvFragmentSubsetBuilder
{
public:
    using RegisterValues =
        std::array<std::array<std::optional<std::uint32_t>, 4U>, 128U>;

    [[nodiscard]] std::array<std::uint32_t, 4U> evaluate(
        const LatteShaderIrProgram& program)
    {
        std::optional<std::array<std::uint32_t, 4U>> pixel_color;
        for (const LatteShaderIrInstruction& instruction : program.instructions)
        {
            if (const auto* clause =
                    std::get_if<LatteShaderIrAluClause>(&instruction))
            {
                execute_clause(*clause);
                continue;
            }

            const LatteShaderIrExport& shader_export =
                std::get<LatteShaderIrExport>(instruction);
            if (pixel_color.has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V fragment subset supports one pixel export");
            }
            pixel_color = read_pixel_export(shader_export);
        }

        if (!pixel_color.has_value())
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment subset requires a completed pixel export");
        }
        return *pixel_color;
    }

private:
    static constexpr std::uint16_t source_zero_selector = 248U;
    static constexpr std::uint16_t source_one_float_selector = 249U;
    static constexpr std::uint16_t source_half_float_selector = 252U;

    void execute_clause(const LatteShaderIrAluClause& clause)
    {
        if (!clause.op3_groups.empty() || !clause.vector_groups.empty() ||
            clause.groups.empty())
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment subset requires OP2-only ALU clauses");
        }

        for (const LatteAluOp2InstructionGroup& group : clause.groups)
        {
            if (group.instructions.size() != 4U)
            {
                throw std::runtime_error(
                    "Latte SPIR-V fragment subset requires four-lane ALU groups");
            }

            std::array<std::uint32_t, 4U> lane_values{};
            std::array<std::uint8_t, 4U> destination_gprs{};
            for (std::size_t lane = 0U; lane < group.instructions.size(); ++lane)
            {
                const LatteAluOp2Instruction& operation =
                    group.instructions[lane];
                if (operation.opcode != latte_alu_op2_move_opcode ||
                    operation.destination_component != lane ||
                    operation.last_in_group != (lane == 3U) ||
                    operation.destination_gpr >= registers_.size() ||
                    operation.source0_relative || operation.source0_absolute ||
                    operation.source0_negate || operation.source1_relative ||
                    operation.source1_absolute || operation.source1_negate ||
                    operation.update_execute_mask || operation.update_predicate ||
                    operation.bank_swizzle != 0U ||
                    operation.predicate_selection != 0U ||
                    operation.destination_relative ||
                    operation.destination_clamp || operation.output_modifier != 0U ||
                    operation.index_mode != 0U || !operation.write_mask)
                {
                    throw std::runtime_error(
                        "Latte SPIR-V fragment subset received an unsupported MOV operation or modifier");
                }

                lane_values[lane] = read_inline_float_constant(
                    operation.source0_selector);
                destination_gprs[lane] = operation.destination_gpr;
            }

            for (std::size_t lane = 0U; lane < lane_values.size(); ++lane)
            {
                registers_[destination_gprs[lane]][lane] = lane_values[lane];
            }
        }
    }

    [[nodiscard]] static std::uint32_t read_inline_float_constant(
        std::uint16_t selector)
    {
        switch (selector)
        {
        case source_zero_selector:
            return 0x00000000U;
        case source_one_float_selector:
            return 0x3F800000U;
        case source_half_float_selector:
            return 0x3F000000U;
        default:
            throw std::runtime_error(
                "Latte SPIR-V fragment subset supports only inline float constants");
        }
    }

    [[nodiscard]] std::array<std::uint32_t, 4U> read_pixel_export(
        const LatteShaderIrExport& shader_export) const
    {
        constexpr std::array<std::uint8_t, 4U> identity_selectors{
            0U, 1U, 2U, 3U};
        const LatteExportFields& fields = shader_export.fields;
        if (!shader_export.completes_type ||
            fields.type != LatteExportType::pixel || fields.array_base != 0U ||
            fields.read_write_gpr >= registers_.size() ||
            fields.index_gpr != 0U || fields.burst_count != 1U ||
            fields.component_selectors != identity_selectors ||
            fields.read_write_relative || fields.uses_buffer_encoding ||
            fields.valid_pixel_mode || fields.whole_quad_mode)
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment subset requires one completed identity export to pixel target zero");
        }

        std::array<std::uint32_t, 4U> color{};
        for (std::size_t component = 0U; component < color.size(); ++component)
        {
            const std::optional<std::uint32_t>& value =
                registers_[fields.read_write_gpr][component];
            if (!value.has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V fragment export reads an unwritten color component");
            }
            color[component] = *value;
        }
        return color;
    }

    RegisterValues registers_{};
};

class LatteSpirvFragmentModuleBuilder
{
public:
    [[nodiscard]] std::vector<std::uint32_t> build(
        const std::array<std::uint32_t, 4U>& color)
    {
        constexpr std::uint32_t void_type_id = 1U;
        constexpr std::uint32_t function_type_id = 2U;
        constexpr std::uint32_t float_type_id = 3U;
        constexpr std::uint32_t vector_type_id = 4U;
        constexpr std::uint32_t output_pointer_type_id = 5U;
        constexpr std::uint32_t function_id = 6U;
        constexpr std::uint32_t label_id = 7U;
        constexpr std::uint32_t output_variable_id = 8U;
        constexpr std::uint32_t first_color_constant_id = 9U;
        constexpr std::uint32_t color_vector_id = 13U;
        constexpr std::uint32_t module_bound = 14U;

        std::vector<std::uint32_t> capabilities;
        std::vector<std::uint32_t> memory_model;
        std::vector<std::uint32_t> entry_point;
        std::vector<std::uint32_t> execution_modes;
        std::vector<std::uint32_t> annotations;
        std::vector<std::uint32_t> types;
        std::vector<std::uint32_t> constants;
        std::vector<std::uint32_t> globals;
        std::vector<std::uint32_t> function_body;

        append_instruction(capabilities, 17U, {1U});
        append_instruction(memory_model, 14U, {0U, 1U});
        append_instruction(entry_point, 15U, {
            4U, function_id, 0x6E69616DU, 0U, output_variable_id});
        append_instruction(execution_modes, 16U, {function_id, 7U});
        append_instruction(annotations, 71U, {output_variable_id, 30U, 0U});

        append_instruction(types, 19U, {void_type_id});
        append_instruction(types, 33U, {function_type_id, void_type_id});
        append_instruction(types, 22U, {float_type_id, 32U});
        append_instruction(types, 23U, {vector_type_id, float_type_id, 4U});
        append_instruction(types, 32U, {
            output_pointer_type_id, 3U, vector_type_id});
        for (std::size_t component = 0U; component < color.size(); ++component)
        {
            append_instruction(constants, 43U, {
                float_type_id,
                first_color_constant_id + static_cast<std::uint32_t>(component),
                color[component]});
        }
        append_instruction(constants, 44U, {
            vector_type_id,
            color_vector_id,
            first_color_constant_id,
            first_color_constant_id + 1U,
            first_color_constant_id + 2U,
            first_color_constant_id + 3U});
        append_instruction(globals, 59U, {
            output_pointer_type_id, output_variable_id, 3U});

        append_instruction(function_body, 54U, {
            void_type_id, function_id, 0U, function_type_id});
        append_instruction(function_body, 248U, {label_id});
        append_instruction(function_body, 62U, {
            output_variable_id, color_vector_id});
        append_instruction(function_body, 253U, {});
        append_instruction(function_body, 56U, {});

        std::vector<std::uint32_t> module{
            0x07230203U, 0x00010000U, 0U, module_bound, 0U};
        append_section(module, capabilities);
        append_section(module, memory_model);
        append_section(module, entry_point);
        append_section(module, execution_modes);
        append_section(module, annotations);
        append_section(module, types);
        append_section(module, constants);
        append_section(module, globals);
        append_section(module, function_body);
        return module;
    }

private:
    static void append_instruction(
        std::vector<std::uint32_t>& words,
        std::uint16_t opcode,
        std::initializer_list<std::uint32_t> operands)
    {
        const std::size_t word_count = operands.size() + 1U;
        if (word_count > 0xFFFFU)
        {
            throw std::runtime_error(
                "SPIR-V instruction exceeds the word-count limit");
        }
        words.push_back(
            (static_cast<std::uint32_t>(word_count) << 16U) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }

    static void append_section(
        std::vector<std::uint32_t>& destination,
        const std::vector<std::uint32_t>& section)
    {
        destination.insert(destination.end(), section.begin(), section.end());
    }
};

} // namespace detail

// Translate only constant-only OP2 MOV groups and a completed MRT0 pixel
// export. Guest fragment inputs and all other shader operations are rejected.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_constant_color_subset(
    const LatteShaderIrProgram& program)
{
    detail::LatteSpirvFragmentSubsetBuilder subset_builder;
    const std::array<std::uint32_t, 4U> color =
        subset_builder.evaluate(program);

    detail::LatteSpirvFragmentModuleBuilder module_builder;
    return module_builder.build(color);
}

} // namespace affogato::gpu
