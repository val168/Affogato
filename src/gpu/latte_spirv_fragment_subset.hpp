#pragma once

#include "latte_shader_interface.hpp"
#include "latte_shader_program.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <optional>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace affogato::gpu
{

namespace detail
{

class LatteSpirvFragmentVaryingPassthroughBuilder
{
public:
    [[nodiscard]] std::vector<std::uint32_t> build(
        const LatteLinkedPixelShaderInput& linked_input) const
    {
        if (linked_input.source != LattePixelInputSource::vertex_parameter ||
            !linked_input.vertex_export_index.has_value() ||
            linked_input.pixel_input_index >= 32U ||
            linked_input.flat_shade || linked_input.centroid ||
            linked_input.linear || linked_input.point_sprite_texture ||
            linked_input.sample)
        {
            throw std::invalid_argument(
                "Latte fragment varying passthrough requires one matched perspective input");
        }

        constexpr std::uint32_t void_type_id = 1U;
        constexpr std::uint32_t function_type_id = 2U;
        constexpr std::uint32_t float_type_id = 3U;
        constexpr std::uint32_t vector_type_id = 4U;
        constexpr std::uint32_t input_pointer_type_id = 5U;
        constexpr std::uint32_t output_pointer_type_id = 6U;
        constexpr std::uint32_t function_id = 7U;
        constexpr std::uint32_t label_id = 8U;
        constexpr std::uint32_t input_variable_id = 9U;
        constexpr std::uint32_t output_variable_id = 10U;
        constexpr std::uint32_t input_value_id = 11U;
        constexpr std::uint32_t module_bound = 12U;

        std::vector<std::uint32_t> module{
            0x07230203U, 0x00010000U, 0U, module_bound, 0U};
        append_instruction(module, 17U, {1U});
        append_instruction(module, 14U, {0U, 1U});
        append_instruction(module, 15U, {
            4U, function_id, 0x6E69616DU, 0U,
            input_variable_id, output_variable_id});
        append_instruction(module, 16U, {function_id, 7U});
        append_instruction(module, 71U, {
            input_variable_id, 30U,
            static_cast<std::uint32_t>(linked_input.pixel_input_index)});
        append_instruction(module, 71U, {output_variable_id, 30U, 0U});
        append_instruction(module, 19U, {void_type_id});
        append_instruction(module, 33U, {
            function_type_id, void_type_id});
        append_instruction(module, 22U, {float_type_id, 32U});
        append_instruction(module, 23U, {
            vector_type_id, float_type_id, 4U});
        append_instruction(module, 32U, {
            input_pointer_type_id, 1U, vector_type_id});
        append_instruction(module, 32U, {
            output_pointer_type_id, 3U, vector_type_id});
        append_instruction(module, 59U, {
            input_pointer_type_id, input_variable_id, 1U});
        append_instruction(module, 59U, {
            output_pointer_type_id, output_variable_id, 3U});
        append_instruction(module, 54U, {
            void_type_id, function_id, 0U, function_type_id});
        append_instruction(module, 248U, {label_id});
        append_instruction(module, 61U, {
            vector_type_id, input_value_id, input_variable_id});
        append_instruction(module, 62U, {
            output_variable_id, input_value_id});
        append_instruction(module, 253U, {});
        append_instruction(module, 56U, {});
        return module;
    }

private:
    static void append_instruction(
        std::vector<std::uint32_t>& words,
        std::uint16_t opcode,
        std::initializer_list<std::uint32_t> operands)
    {
        const std::size_t word_count = operands.size() + 1U;
        words.push_back(
            (static_cast<std::uint32_t>(word_count) << 16U) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }
};

} // namespace detail

// Diagnostic-only module that reads one linked perspective varying and
// writes it to MRT0. It does not execute the guest fragment program.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_varying_passthrough_subset(
    const LatteLinkedPixelShaderInput& linked_input)
{
    return detail::LatteSpirvFragmentVaryingPassthroughBuilder{}.build(
        linked_input);
}

namespace detail
{

class LatteSpirvFragmentSubsetBuilder
{
public:
    using RegisterValues =
        std::array<std::array<std::optional<std::uint32_t>, 4U>, 128U>;

    explicit LatteSpirvFragmentSubsetBuilder(
        const LattePixelShaderInputRegisters& input_registers)
    {
        for (std::size_t input_index = 0U;
             input_index < input_registers.size();
             ++input_index)
        {
            if (!input_registers[input_index].has_value())
            {
                continue;
            }

            for (std::size_t component = 0U;
                 component < registers_[input_index].size();
                 ++component)
            {
                registers_[input_index][component] =
                    (*input_registers[input_index])[component];
            }
        }
    }

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

                lane_values[lane] = read_move_source(
                    operation.source0_selector,
                    operation.source0_channel);
                destination_gprs[lane] = operation.destination_gpr;
            }

            for (std::size_t lane = 0U; lane < lane_values.size(); ++lane)
            {
                registers_[destination_gprs[lane]][lane] = lane_values[lane];
            }
        }
    }

    [[nodiscard]] std::uint32_t read_move_source(
        std::uint16_t selector,
        std::uint8_t channel) const
    {
        if (channel >= 4U)
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment MOV source channel is out of range");
        }

        switch (selector)
        {
        case source_zero_selector:
            return 0U;
        case source_one_float_selector:
            return 0x3F800000U;
        case source_half_float_selector:
            return 0x3F000000U;
        default:
            break;
        }

        if (selector >= registers_.size() ||
            !registers_[selector][channel].has_value())
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment MOV reads an unsupported or uninitialized GPR");
        }

        return *registers_[selector][channel];
    }

    [[nodiscard]] std::array<std::uint32_t, 4U> read_pixel_export(
        const LatteShaderIrExport& shader_export) const
    {
        constexpr std::uint8_t zero_float_selector = 4U;
        constexpr std::uint8_t one_float_selector = 5U;
        constexpr std::uint8_t reserved_selector = 6U;
        constexpr std::uint8_t masked_selector = 7U;
        const LatteExportFields& fields = shader_export.fields;
        if (!shader_export.completes_type ||
            fields.type != LatteExportType::pixel || fields.array_base != 0U ||
            fields.read_write_gpr >= registers_.size() ||
            fields.index_gpr != 0U || fields.burst_count != 1U ||
            fields.read_write_relative || fields.uses_buffer_encoding ||
            fields.valid_pixel_mode || fields.whole_quad_mode)
        {
            throw std::runtime_error(
                "Latte SPIR-V fragment subset requires one completed export to pixel target zero");
        }

        std::array<std::uint32_t, 4U> color{};
        for (std::size_t component = 0U; component < color.size(); ++component)
        {
            const std::uint8_t selector = fields.component_selectors[component];
            if (selector < color.size())
            {
                const std::optional<std::uint32_t>& value =
                    registers_[fields.read_write_gpr][selector];
                if (!value.has_value())
                {
                    throw std::runtime_error(
                        "Latte SPIR-V fragment export reads an unwritten color component");
                }
                color[component] = *value;
                continue;
            }

            if (selector == zero_float_selector)
            {
                color[component] = 0U;
            }
            else if (selector == one_float_selector)
            {
                color[component] = 0x3F800000U;
            }
            else if (selector >= reserved_selector &&
                     selector <= masked_selector)
            {
                throw std::runtime_error(
                    "Latte SPIR-V fragment export uses a reserved or masked component selector");
            }
            else
            {
                throw std::runtime_error(
                    "Latte SPIR-V fragment export component selector is out of range");
            }
        }
        return color;
    }

    RegisterValues registers_{};
};

class LatteSpirvFragmentMatchedAluBuilder
{
public:
    using RegisterValues =
        std::array<std::array<std::optional<std::uint32_t>, 4U>, 128U>;

    [[nodiscard]] std::vector<std::uint32_t> emit(
        const LatteShaderIrProgram& program,
        std::span<const LatteLinkedPixelShaderInput> linked_inputs)
    {
        validate_linked_inputs(linked_inputs);
        create_module_declarations(linked_inputs);
        append_instruction(function_body_, 54U, {
            void_type_id_, function_id_, 0U, function_type_id_});
        append_instruction(function_body_, 248U, {label_id_});
        for (const LatteLinkedPixelShaderInput& linked_input : linked_inputs)
        {
            if (linked_input.source == LattePixelInputSource::vertex_parameter)
            {
                load_input_register(linked_input.pixel_input_index);
            }
            else
            {
                load_default_input_register(linked_input);
            }
        }

        std::optional<LatteShaderIrExport> pixel_export;
        for (const LatteShaderIrInstruction& instruction : program.instructions)
        {
            if (const auto* clause =
                    std::get_if<LatteShaderIrAluClause>(&instruction))
            {
                execute_clause(*clause);
                continue;
            }

            if (pixel_export.has_value())
            {
                throw std::runtime_error(
                "Latte fragment SPIR-V subset supports one pixel export");
            }
            pixel_export = std::get<LatteShaderIrExport>(instruction);
        }

        if (!pixel_export.has_value())
        {
            throw std::runtime_error(
                "Latte fragment SPIR-V subset requires an MRT0 export");
        }
        store_pixel_export(*pixel_export);
        append_instruction(function_body_, 253U, {});
        append_instruction(function_body_, 56U, {});

        std::vector<std::uint32_t> module{
            0x07230203U, 0x00010000U, 0U, next_id_, 0U};
        append_section(module, capabilities_);
        append_section(module, memory_model_);
        append_section(module, entry_point_);
        append_section(module, execution_modes_);
        append_section(module, annotations_);
        append_section(module, types_);
        append_section(module, constants_);
        append_section(module, globals_);
        append_section(module, function_body_);
        return module;
    }

private:
    static constexpr std::uint16_t input_gpr_count = 32U;

    static void validate_linked_inputs(
        std::span<const LatteLinkedPixelShaderInput> linked_inputs)
    {
        std::array<bool, input_gpr_count> assigned_registers{};
        bool has_vertex_varying = false;
        for (const LatteLinkedPixelShaderInput& linked_input : linked_inputs)
        {
            if (linked_input.pixel_input_index >= input_gpr_count ||
                assigned_registers[linked_input.pixel_input_index])
            {
                throw std::invalid_argument(
                    "Latte fragment SPIR-V subset received an invalid or duplicate input GPR");
            }
            assigned_registers[linked_input.pixel_input_index] = true;

            if (linked_input.source == LattePixelInputSource::vertex_parameter)
            {
                if (!linked_input.vertex_export_index.has_value() ||
                    linked_input.flat_shade ||
                    linked_input.point_sprite_texture ||
                    linked_input.sample)
                {
                    throw std::invalid_argument(
                        "Latte fragment SPIR-V subset does not support this vertex-input mode");
                }
                has_vertex_varying = true;
                continue;
            }
            if (linked_input.source != LattePixelInputSource::default_value ||
                linked_input.vertex_export_index.has_value())
            {
                throw std::invalid_argument(
                    "Latte fragment SPIR-V subset received an unsupported input source");
            }
        }

        if (!has_vertex_varying)
        {
            throw std::invalid_argument(
                "Latte fragment SPIR-V matched-input subset requires a vertex varying");
        }
    }

    [[nodiscard]] std::uint32_t allocate_id()
    {
        return next_id_++;
    }

    static void append_instruction(
        std::vector<std::uint32_t>& words,
        std::uint16_t opcode,
        std::initializer_list<std::uint32_t> operands)
    {
        const std::size_t word_count = operands.size() + 1U;
        words.push_back(
            (static_cast<std::uint32_t>(word_count) << 16U) | opcode);
        words.insert(words.end(), operands.begin(), operands.end());
    }

    static void append_instruction(
        std::vector<std::uint32_t>& words,
        std::uint16_t opcode,
        const std::vector<std::uint32_t>& operands)
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

    [[nodiscard]] std::uint32_t emit_integer_comparison(
        const LatteAluOp2Instruction& operation,
        std::uint32_t source0_id,
        std::uint32_t source1_id)
    {
        std::uint32_t comparison_source0_id = source0_id;
        std::uint32_t comparison_source1_id = source1_id;
        std::uint16_t comparison_opcode = 0U;
        bool returns_selected_operand = false;
        switch (operation.opcode)
        {
        case latte_alu_op2_maximum_signed_integer_opcode:
        case latte_alu_op2_minimum_signed_integer_opcode:
        case latte_alu_op2_set_greater_integer_opcode:
        case latte_alu_op2_set_greater_equal_integer_opcode:
            comparison_source0_id = allocate_id();
            append_instruction(function_body_, 124U, {
                int_type_id_, comparison_source0_id, source0_id});
            comparison_source1_id = allocate_id();
            append_instruction(function_body_, 124U, {
                int_type_id_, comparison_source1_id, source1_id});
            break;
        default:
            break;
        }

        switch (operation.opcode)
        {
        case latte_alu_op2_maximum_signed_integer_opcode:
        case latte_alu_op2_maximum_unsigned_integer_opcode:
            comparison_opcode = operation.opcode ==
                    latte_alu_op2_maximum_signed_integer_opcode
                ? 175U
                : 174U;
            returns_selected_operand = true;
            break;
        case latte_alu_op2_minimum_signed_integer_opcode:
        case latte_alu_op2_minimum_unsigned_integer_opcode:
            comparison_opcode = operation.opcode ==
                    latte_alu_op2_minimum_signed_integer_opcode
                ? 179U
                : 178U;
            returns_selected_operand = true;
            break;
        case latte_alu_op2_set_equal_integer_opcode:
            comparison_opcode = 170U;
            break;
        case latte_alu_op2_set_not_equal_integer_opcode:
            comparison_opcode = 171U;
            break;
        case latte_alu_op2_set_greater_integer_opcode:
            comparison_opcode = 173U;
            break;
        case latte_alu_op2_set_greater_equal_integer_opcode:
            comparison_opcode = 175U;
            break;
        case latte_alu_op2_set_greater_unsigned_opcode:
            comparison_opcode = 172U;
            break;
        case latte_alu_op2_set_greater_equal_unsigned_opcode:
            comparison_opcode = 174U;
            break;
        default:
            throw std::runtime_error(
                "Latte fragment SPIR-V subset received an unsupported integer comparison");
        }

        const std::uint32_t comparison_id = allocate_id();
        append_instruction(function_body_, comparison_opcode, {
            bool_type_id_, comparison_id,
            comparison_source0_id, comparison_source1_id});
        const std::uint32_t result_id = allocate_id();
        if (returns_selected_operand)
        {
            append_instruction(function_body_, 169U, {
                uint_type_id_, result_id, comparison_id,
                source0_id, source1_id});
            return result_id;
        }

        append_instruction(function_body_, 169U, {
            uint_type_id_, result_id, comparison_id,
            constant(0xFFFFFFFFU), constant(0U)});
        return result_id;
    }

    void create_module_declarations(
        std::span<const LatteLinkedPixelShaderInput> linked_inputs)
    {
        void_type_id_ = allocate_id();
        function_type_id_ = allocate_id();
        float_type_id_ = allocate_id();
        uint_type_id_ = allocate_id();
        bool_type_id_ = allocate_id();
        int_type_id_ = allocate_id();
        vector_type_id_ = allocate_id();
        input_pointer_type_id_ = allocate_id();
        output_pointer_type_id_ = allocate_id();
        function_id_ = allocate_id();
        label_id_ = allocate_id();
        output_variable_id_ = allocate_id();

        append_instruction(capabilities_, 17U, {1U});
        append_instruction(memory_model_, 14U, {0U, 1U});
        std::vector<std::uint32_t> entry_point_operands{
            4U, function_id_, 0x6E69616DU, 0U};
        for (const LatteLinkedPixelShaderInput& linked_input : linked_inputs)
        {
            if (linked_input.source != LattePixelInputSource::vertex_parameter)
            {
                continue;
            }

            const std::uint32_t input_variable_id = allocate_id();
            input_variable_ids_[linked_input.pixel_input_index] =
                input_variable_id;
            entry_point_operands.push_back(input_variable_id);
            append_instruction(annotations_, 71U, {
                input_variable_id, 30U,
                static_cast<std::uint32_t>(linked_input.pixel_input_index)});
            if (linked_input.linear)
            {
                append_instruction(annotations_, 71U, {
                    input_variable_id, 13U});
            }
            if (linked_input.centroid)
            {
                append_instruction(annotations_, 71U, {
                    input_variable_id, 16U});
            }
            append_instruction(globals_, 59U, {
                input_pointer_type_id_, input_variable_id, 1U});
        }
        entry_point_operands.push_back(output_variable_id_);
        append_instruction(entry_point_, 15U, entry_point_operands);
        append_instruction(execution_modes_, 16U, {function_id_, 7U});
        append_instruction(annotations_, 71U, {
            output_variable_id_, 30U, 0U});

        append_instruction(types_, 19U, {void_type_id_});
        append_instruction(types_, 33U, {
            function_type_id_, void_type_id_});
        append_instruction(types_, 22U, {float_type_id_, 32U});
        append_instruction(types_, 21U, {uint_type_id_, 32U, 0U});
        append_instruction(types_, 21U, {int_type_id_, 32U, 1U});
        append_instruction(types_, 20U, {bool_type_id_});
        append_instruction(types_, 23U, {
            vector_type_id_, float_type_id_, 4U});
        append_instruction(types_, 32U, {
            input_pointer_type_id_, 1U, vector_type_id_});
        append_instruction(types_, 32U, {
            output_pointer_type_id_, 3U, vector_type_id_});
        append_instruction(globals_, 59U, {
            output_pointer_type_id_, output_variable_id_, 3U});
    }

    void load_input_register(std::size_t input_index)
    {
        const std::uint32_t input_variable_id =
            input_variable_ids_[input_index];
        const std::uint32_t input_vector_id = allocate_id();
        append_instruction(function_body_, 61U, {
            vector_type_id_, input_vector_id, input_variable_id});
        for (std::uint32_t component = 0U; component < 4U; ++component)
        {
            const std::uint32_t float_component_id = allocate_id();
            append_instruction(function_body_, 81U, {
                float_type_id_, float_component_id,
                input_vector_id, component});
            const std::uint32_t register_value_id = allocate_id();
            append_instruction(function_body_, 124U, {
                uint_type_id_, register_value_id, float_component_id});
            registers_[input_index][component] = register_value_id;
        }
    }

    void load_default_input_register(
        const LatteLinkedPixelShaderInput& linked_input)
    {
        for (std::size_t component = 0U;
             component < linked_input.default_component_bits.size();
             ++component)
        {
            registers_[linked_input.pixel_input_index][component] =
                constant(linked_input.default_component_bits[component]);
        }
    }

    [[nodiscard]] std::uint32_t constant(std::uint32_t bits)
    {
        const auto existing = constant_ids_.find(bits);
        if (existing != constant_ids_.end())
        {
            return existing->second;
        }
        const std::uint32_t constant_id = allocate_id();
        append_instruction(constants_, 43U, {
            uint_type_id_, constant_id, bits});
        constant_ids_.emplace(bits, constant_id);
        return constant_id;
    }

    [[nodiscard]] std::uint32_t source_value(
        std::uint16_t selector,
        std::uint8_t channel)
    {
        if (channel >= 4U)
        {
            throw std::runtime_error(
                "Latte fragment SPIR-V source channel is out of range");
        }
        switch (selector)
        {
        case latte_alu_source_zero_selector:
            return constant(0U);
        case latte_alu_source_one_float_selector:
            return constant(0x3F800000U);
        case latte_alu_source_one_integer_selector:
            return constant(1U);
        case latte_alu_source_negative_one_integer_selector:
            return constant(0xFFFFFFFFU);
        case latte_alu_source_half_float_selector:
            return constant(0x3F000000U);
        default:
            break;
        }

        if (selector >= registers_.size() ||
            !registers_[selector][channel].has_value())
        {
            throw std::runtime_error(
                "Latte fragment SPIR-V instruction reads an unsupported or uninitialized GPR");
        }
        return *registers_[selector][channel];
    }

    [[nodiscard]] std::uint32_t apply_source_modifiers(
        std::uint32_t value,
        bool absolute,
        bool negate)
    {
        // Match the CPU reference's bit-preserving binary32 modifiers.
        if (absolute)
        {
            const std::uint32_t absolute_id = allocate_id();
            append_instruction(function_body_, 199U, {
                uint_type_id_, absolute_id, value,
                constant(latte_binary32_magnitude_mask)});
            value = absolute_id;
        }
        if (negate)
        {
            const std::uint32_t negate_id = allocate_id();
            append_instruction(function_body_, 198U, {
                uint_type_id_, negate_id, value,
                constant(latte_binary32_sign_mask)});
            value = negate_id;
        }
        return value;
    }

    void execute_clause(const LatteShaderIrAluClause& clause)
    {
        if (!clause.op3_groups.empty() || !clause.vector_groups.empty() ||
            clause.groups.empty())
        {
            throw std::runtime_error(
                "Latte fragment SPIR-V subset requires OP2-only clauses");
        }

        for (const LatteAluOp2InstructionGroup& group : clause.groups)
        {
            if (group.instructions.size() != 4U)
            {
                throw std::runtime_error(
                    "Latte fragment SPIR-V subset requires four-lane groups");
            }
            std::array<std::uint32_t, 4U> lane_values{};
            std::array<std::uint8_t, 4U> destination_gprs{};
            for (std::size_t lane = 0U; lane < group.instructions.size(); ++lane)
            {
                const LatteAluOp2Instruction& operation =
                    group.instructions[lane];
                const bool is_integer_operation =
                    operation.opcode == latte_alu_op2_and_integer_opcode ||
                    operation.opcode == latte_alu_op2_or_integer_opcode ||
                    operation.opcode == latte_alu_op2_xor_integer_opcode ||
                    operation.opcode == latte_alu_op2_not_integer_opcode ||
                    operation.opcode == latte_alu_op2_add_integer_opcode ||
                    operation.opcode == latte_alu_op2_subtract_integer_opcode ||
                    operation.opcode == latte_alu_op2_maximum_signed_integer_opcode ||
                    operation.opcode == latte_alu_op2_minimum_signed_integer_opcode ||
                    operation.opcode == latte_alu_op2_maximum_unsigned_integer_opcode ||
                    operation.opcode == latte_alu_op2_minimum_unsigned_integer_opcode ||
                    operation.opcode == latte_alu_op2_set_equal_integer_opcode ||
                    operation.opcode == latte_alu_op2_set_not_equal_integer_opcode ||
                    operation.opcode == latte_alu_op2_set_greater_integer_opcode ||
                    operation.opcode == latte_alu_op2_set_greater_equal_integer_opcode ||
                    operation.opcode == latte_alu_op2_set_greater_unsigned_opcode ||
                    operation.opcode == latte_alu_op2_set_greater_equal_unsigned_opcode;
                const bool supported_opcode =
                    operation.opcode == latte_alu_op2_move_opcode ||
                    operation.opcode == latte_alu_op2_add_float_opcode ||
                    operation.opcode == latte_alu_op2_multiply_ieee_opcode ||
                    operation.opcode == latte_alu_op2_maximum_dx10_opcode ||
                    operation.opcode == latte_alu_op2_minimum_dx10_opcode ||
                    is_integer_operation;
                if (!supported_opcode ||
                    operation.destination_component != lane ||
                    operation.last_in_group != (lane == 3U) ||
                    operation.destination_gpr >= registers_.size() ||
                    operation.source0_relative || operation.source1_relative ||
                    operation.update_execute_mask || operation.update_predicate ||
                    operation.bank_swizzle != 0U ||
                    operation.predicate_selection != 0U ||
                    operation.destination_relative ||
                    operation.destination_clamp ||
                    operation.output_modifier != 0U ||
                    operation.index_mode != 0U || !operation.write_mask ||
                    (is_integer_operation &&
                     (operation.source0_absolute ||
                      operation.source0_negate ||
                      operation.source1_absolute ||
                      operation.source1_negate)) ||
                    (operation.opcode == latte_alu_op2_move_opcode &&
                     (operation.source1_absolute ||
                      operation.source1_negate)))
                {
                    throw std::runtime_error(
                        "Latte fragment SPIR-V subset received an unsupported operation or modifier");
                }
                const std::uint32_t source0_value = source_value(
                    operation.source0_selector,
                    operation.source0_channel);
                if (is_integer_operation)
                {
                    if (operation.opcode == latte_alu_op2_not_integer_opcode)
                    {
                        lane_values[lane] = allocate_id();
                        append_instruction(function_body_, 200U, {
                            uint_type_id_, lane_values[lane], source0_value});
                    }
                    else
                    {
                        const std::uint32_t source1_value = source_value(
                            operation.source1_selector,
                            operation.source1_channel);
                        std::uint16_t spirv_opcode = 0U;
                        switch (operation.opcode)
                        {
                        case latte_alu_op2_add_integer_opcode:
                            spirv_opcode = 128U;
                            break;
                        case latte_alu_op2_subtract_integer_opcode:
                            spirv_opcode = 130U;
                            break;
                        case latte_alu_op2_and_integer_opcode:
                            spirv_opcode = 199U;
                            break;
                        case latte_alu_op2_or_integer_opcode:
                            spirv_opcode = 197U;
                            break;
                        case latte_alu_op2_xor_integer_opcode:
                            spirv_opcode = 198U;
                            break;
                        case latte_alu_op2_maximum_signed_integer_opcode:
                        case latte_alu_op2_minimum_signed_integer_opcode:
                        case latte_alu_op2_maximum_unsigned_integer_opcode:
                        case latte_alu_op2_minimum_unsigned_integer_opcode:
                        case latte_alu_op2_set_equal_integer_opcode:
                        case latte_alu_op2_set_not_equal_integer_opcode:
                        case latte_alu_op2_set_greater_integer_opcode:
                        case latte_alu_op2_set_greater_equal_integer_opcode:
                        case latte_alu_op2_set_greater_unsigned_opcode:
                        case latte_alu_op2_set_greater_equal_unsigned_opcode:
                            lane_values[lane] = emit_integer_comparison(
                                operation, source0_value, source1_value);
                            destination_gprs[lane] = operation.destination_gpr;
                            continue;
                        default:
                            throw std::runtime_error(
                                "Latte fragment SPIR-V subset received an unsupported integer operation");
                        }
                        lane_values[lane] = allocate_id();
                        append_instruction(function_body_, spirv_opcode, {
                            uint_type_id_, lane_values[lane],
                            source0_value, source1_value});
                    }
                    destination_gprs[lane] = operation.destination_gpr;
                    continue;
                }

                const std::uint32_t source0_id = apply_source_modifiers(
                    source0_value,
                    operation.source0_absolute,
                    operation.source0_negate);
                if (operation.opcode == latte_alu_op2_move_opcode)
                {
                    lane_values[lane] = source0_id;
                }
                else
                {
                    const std::uint32_t source1_id = apply_source_modifiers(
                        source_value(
                            operation.source1_selector,
                            operation.source1_channel),
                        operation.source1_absolute,
                        operation.source1_negate);
                    const std::uint32_t source0_float_id = allocate_id();
                    append_instruction(function_body_, 124U, {
                        float_type_id_, source0_float_id, source0_id});
                    const std::uint32_t source1_float_id = allocate_id();
                    append_instruction(function_body_, 124U, {
                        float_type_id_, source1_float_id, source1_id});
                    const std::uint32_t result_float_id = allocate_id();
                    if (operation.opcode == latte_alu_op2_add_float_opcode ||
                        operation.opcode == latte_alu_op2_multiply_ieee_opcode)
                    {
                        const std::uint16_t spirv_opcode =
                            operation.opcode == latte_alu_op2_add_float_opcode
                                ? 129U
                                : 133U;
                        append_instruction(function_body_, spirv_opcode, {
                            float_type_id_, result_float_id,
                            source0_float_id, source1_float_id});
                    }
                    else
                    {
                        const std::uint32_t comparison_id = allocate_id();
                        const std::uint16_t comparison_opcode =
                            operation.opcode ==
                                    latte_alu_op2_maximum_dx10_opcode
                                ? 190U
                                : 184U;
                        append_instruction(function_body_, comparison_opcode, {
                            bool_type_id_, comparison_id,
                            source0_float_id, source1_float_id});
                        append_instruction(function_body_, 169U, {
                            float_type_id_, result_float_id, comparison_id,
                            source0_float_id, source1_float_id});
                    }
                    lane_values[lane] = allocate_id();
                    append_instruction(function_body_, 124U, {
                        uint_type_id_, lane_values[lane], result_float_id});
                }
                destination_gprs[lane] = operation.destination_gpr;
            }

            for (std::size_t lane = 0U; lane < lane_values.size(); ++lane)
            {
                registers_[destination_gprs[lane]][lane] = lane_values[lane];
            }
        }
    }

    void store_pixel_export(const LatteShaderIrExport& shader_export)
    {
        constexpr std::uint8_t zero_float_selector = 4U;
        constexpr std::uint8_t one_float_selector = 5U;
        const LatteExportFields& fields = shader_export.fields;
        if (!shader_export.completes_type ||
            fields.type != LatteExportType::pixel || fields.array_base != 0U ||
            fields.read_write_gpr >= registers_.size() ||
            fields.index_gpr != 0U || fields.burst_count != 1U ||
            fields.read_write_relative || fields.uses_buffer_encoding ||
            fields.valid_pixel_mode || fields.whole_quad_mode)
        {
            throw std::runtime_error(
                "Latte fragment SPIR-V subset requires a completed MRT0 export");
        }

        std::array<std::uint32_t, 4U> color_components{};
        for (std::size_t component = 0U;
             component < color_components.size();
             ++component)
        {
            const std::uint8_t selector = fields.component_selectors[component];
            if (selector < color_components.size())
            {
                const auto& register_value =
                    registers_[fields.read_write_gpr][selector];
                if (!register_value.has_value())
                {
                    throw std::runtime_error(
                        "Latte fragment SPIR-V export reads an unwritten GPR component");
                }
                color_components[component] = *register_value;
            }
            else if (selector == zero_float_selector)
            {
                color_components[component] = constant(0U);
            }
            else if (selector == one_float_selector)
            {
                color_components[component] = constant(0x3F800000U);
            }
            else
            {
                throw std::runtime_error(
                    "Latte fragment SPIR-V export uses a reserved component selector");
            }
        }

        std::array<std::uint32_t, 4U> float_components{};
        for (std::size_t component = 0U;
             component < float_components.size();
             ++component)
        {
            float_components[component] = allocate_id();
            append_instruction(function_body_, 124U, {
                float_type_id_, float_components[component],
                color_components[component]});
        }
        const std::uint32_t color_vector_id = allocate_id();
        append_instruction(function_body_, 80U, {
            vector_type_id_, color_vector_id,
            float_components[0U], float_components[1U],
            float_components[2U], float_components[3U]});
        append_instruction(function_body_, 62U, {
            output_variable_id_, color_vector_id});
    }

    std::uint32_t next_id_{1U};
    std::uint32_t void_type_id_{};
    std::uint32_t function_type_id_{};
    std::uint32_t float_type_id_{};
    std::uint32_t uint_type_id_{};
    std::uint32_t bool_type_id_{};
    std::uint32_t int_type_id_{};
    std::uint32_t vector_type_id_{};
    std::uint32_t input_pointer_type_id_{};
    std::uint32_t output_pointer_type_id_{};
    std::uint32_t function_id_{};
    std::uint32_t label_id_{};
    std::uint32_t output_variable_id_{};
    std::array<std::uint32_t, input_gpr_count> input_variable_ids_{};
    RegisterValues registers_{};
    std::unordered_map<std::uint32_t, std::uint32_t> constant_ids_;
    std::vector<std::uint32_t> capabilities_;
    std::vector<std::uint32_t> memory_model_;
    std::vector<std::uint32_t> entry_point_;
    std::vector<std::uint32_t> execution_modes_;
    std::vector<std::uint32_t> annotations_;
    std::vector<std::uint32_t> types_;
    std::vector<std::uint32_t> constants_;
    std::vector<std::uint32_t> globals_;
    std::vector<std::uint32_t> function_body_;
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

// Translate matched-input MOV/ADD/MUL guest pixel subsets with one perspective
// input, preserving its SPIR-V location and guest MRT0 swizzle.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_matched_alu_subset(
    const LatteShaderIrProgram& program,
    const LatteLinkedPixelShaderInput& linked_input)
{
    return detail::LatteSpirvFragmentMatchedAluBuilder{}.emit(
        program,
        std::span<const LatteLinkedPixelShaderInput>(&linked_input, 1U));
}

[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_matched_alu_subset(
    const LatteShaderIrProgram& program,
    std::span<const LatteLinkedPixelShaderInput> linked_inputs)
{
    return detail::LatteSpirvFragmentMatchedAluBuilder{}.emit(
        program,
        linked_inputs);
}

// Translate only constant-only OP2 MOV groups and a completed MRT0 pixel
// export. Guest fragment inputs and all other shader operations are rejected.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_constant_color_subset(
    const LatteShaderIrProgram& program)
{
    detail::LatteSpirvFragmentSubsetBuilder subset_builder(
        LattePixelShaderInputRegisters{});
    const std::array<std::uint32_t, 4U> color =
        subset_builder.evaluate(program);

    detail::LatteSpirvFragmentModuleBuilder module_builder;
    return module_builder.build(color);
}

// Resolve a fragment program whose active inputs all come from SPI defaults.
// These values are constant across fragments, so the bounded MOV subset can
// fold them into the generated module without inventing interpolated inputs.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_fragment_default_input_subset(
    const LatteShaderIrProgram& program,
    std::span<const LatteLinkedPixelShaderInput> linked_inputs)
{
    LattePixelShaderInputRegisters input_registers{};
    for (const LatteLinkedPixelShaderInput& linked_input : linked_inputs)
    {
        if (linked_input.source != LattePixelInputSource::default_value)
        {
            throw std::runtime_error(
                "Latte SPIR-V default-input subset cannot compile matched varyings");
        }
        if (linked_input.pixel_input_index >= input_registers.size() ||
            input_registers[linked_input.pixel_input_index].has_value())
        {
            throw std::runtime_error(
                "Latte SPIR-V default-input subset has an invalid or duplicate GPR assignment");
        }

        input_registers[linked_input.pixel_input_index] =
            linked_input.default_component_bits;
    }

    detail::LatteSpirvFragmentSubsetBuilder subset_builder(input_registers);
    const std::array<std::uint32_t, 4U> color =
        subset_builder.evaluate(program);

    detail::LatteSpirvFragmentModuleBuilder module_builder;
    return module_builder.build(color);
}

} // namespace affogato::gpu
