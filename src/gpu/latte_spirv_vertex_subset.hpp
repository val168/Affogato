#pragma once

#include "latte_shader_program.hpp"

#include <algorithm>
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

struct LatteSpirvVertexInput
{
    std::uint8_t guest_gpr{};
    std::uint32_t location{};
};

struct LatteSpirvVertexParameterOutput
{
    std::uint32_t export_index{};
    std::uint32_t location{};
    bool no_perspective{};
};

namespace detail
{

class LatteSpirvVertexSubsetBuilder
{
public:
    using RegisterValues =
        std::array<std::array<std::optional<std::uint32_t>, 4U>, 128U>;
    using VectorValue = std::array<std::optional<std::uint32_t>, 4U>;

    explicit LatteSpirvVertexSubsetBuilder(
        LatteSpirvVertexInput vertex_input,
        std::span<const LatteSpirvVertexParameterOutput> parameter_outputs)
        : vertex_input_(vertex_input),
          parameter_outputs_(parameter_outputs.begin(), parameter_outputs.end())
    {
        void_type_id_ = allocate_id();
        function_type_id_ = allocate_id();
        float_type_id_ = allocate_id();
        int_type_id_ = allocate_id();
        uint_type_id_ = allocate_id();
        vector_type_id_ = allocate_id();
        input_pointer_type_id_ = allocate_id();
        output_struct_type_id_ = allocate_id();
        output_struct_pointer_type_id_ = allocate_id();
        output_vector_pointer_type_id_ = allocate_id();
        function_id_ = allocate_id();
        label_id_ = allocate_id();
        input_variable_id_ = allocate_id();
        output_variable_id_ = allocate_id();

        append_instruction(capabilities_, 17U, {1U});
        append_instruction(memory_model_, 14U, {0U, 1U});
        std::vector<std::uint32_t> entry_point_operands{
            0U, function_id_, 0x6E69616DU, 0U,
            input_variable_id_, output_variable_id_};
        parameter_output_variable_ids_.reserve(parameter_outputs_.size());
        for (std::size_t output_index = 0U;
             output_index < parameter_outputs_.size();
             ++output_index)
        {
            const LatteSpirvVertexParameterOutput& parameter_output =
                parameter_outputs_[output_index];
            if (parameter_output.export_index >= 32U)
            {
                throw std::invalid_argument(
                    "Latte SPIR-V vertex parameter export index exceeds the supported interface");
            }
            for (std::size_t previous_index = 0U;
                 previous_index < output_index;
                 ++previous_index)
            {
                if (parameter_outputs_[previous_index].export_index ==
                        parameter_output.export_index ||
                    parameter_outputs_[previous_index].location ==
                        parameter_output.location)
                {
                    throw std::invalid_argument(
                        "Latte SPIR-V vertex parameter outputs contain a duplicate export or location");
                }
            }
            const std::uint32_t variable_id = allocate_id();
            parameter_output_variable_ids_.push_back(variable_id);
            entry_point_operands.push_back(variable_id);
            append_instruction(annotations_, 71U, {
                variable_id, 30U, parameter_output.location});
            if (parameter_output.no_perspective)
            {
                append_instruction(annotations_, 71U, {
                    variable_id, 13U});
            }
            append_instruction(globals_, 59U, {
                output_vector_pointer_type_id_, variable_id, 3U});
        }
        append_instruction(entry_point_, 15U, entry_point_operands);

        append_instruction(annotations_, 71U, {
            input_variable_id_, 30U, vertex_input_.location});
        append_instruction(annotations_, 71U, {
            output_struct_type_id_, 2U});
        append_instruction(annotations_, 72U, {
            output_struct_type_id_, 0U, 11U, 0U});

        append_instruction(types_, 19U, {void_type_id_});
        append_instruction(types_, 33U, {
            function_type_id_, void_type_id_});
        append_instruction(types_, 22U, {float_type_id_, 32U});
        append_instruction(types_, 21U, {int_type_id_, 32U, 1U});
        append_instruction(types_, 21U, {uint_type_id_, 32U, 0U});
        unsigned_zero_id_ = allocate_id();
        append_instruction(constants_, 43U, {
            uint_type_id_, unsigned_zero_id_, 0U});
        unsigned_all_bits_set_id_ = allocate_id();
        append_instruction(constants_, 43U, {
            uint_type_id_, unsigned_all_bits_set_id_, 0xFFFFFFFFU});
        signed_zero_id_ = allocate_id();
        append_instruction(constants_, 43U, {
            int_type_id_, signed_zero_id_, 0U});
        bool_type_id_ = allocate_id();
        append_instruction(types_, 20U, {bool_type_id_});
        append_instruction(types_, 23U, {vector_type_id_, float_type_id_, 4U});
        append_instruction(types_, 30U, {
            output_struct_type_id_, vector_type_id_});
        append_instruction(types_, 32U, {
            input_pointer_type_id_, 1U, vector_type_id_});
        append_instruction(types_, 32U, {
            output_struct_pointer_type_id_, 3U, output_struct_type_id_});
        append_instruction(types_, 32U, {
            output_vector_pointer_type_id_, 3U, vector_type_id_});
        append_instruction(globals_, 59U, {
            input_pointer_type_id_, input_variable_id_, 1U});
        append_instruction(globals_, 59U, {
            output_struct_pointer_type_id_, output_variable_id_, 3U});

        append_instruction(function_body_, 248U, {label_id_});
        const std::uint32_t input_vector = allocate_id();
        append_instruction(function_body_, 61U, {
            vector_type_id_, input_vector, input_variable_id_});
        for (std::uint32_t component = 0U; component < 4U; ++component)
        {
            const std::uint32_t float_component = allocate_id();
            append_instruction(function_body_, 81U, {
                float_type_id_, float_component, input_vector, component});
            const std::uint32_t raw_component = allocate_id();
            append_instruction(function_body_, 124U, {
                uint_type_id_, raw_component, float_component});
            registers_[vertex_input_.guest_gpr][component] = raw_component;
        }
    }

    [[nodiscard]] std::vector<std::uint32_t> emit(
        const LatteShaderIrProgram& program)
    {
        std::optional<LatteShaderIrExport> position_export;
        std::vector<LatteShaderIrExport> parameter_exports;
        for (const LatteShaderIrInstruction& instruction : program.instructions)
        {
            if (const auto* clause =
                    std::get_if<LatteShaderIrAluClause>(&instruction))
            {
                execute_clause(*clause);
                continue;
            }

            const auto& shader_export =
                std::get<LatteShaderIrExport>(instruction);
            if (shader_export.fields.type == LatteExportType::parameter)
            {
                parameter_exports.push_back(shader_export);
                continue;
            }
            if (shader_export.fields.type != LatteExportType::position ||
                position_export.has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V vertex subset supports one position export and parameter outputs");
            }
            validate_position_export(shader_export);
            position_export = shader_export;
        }

        if (!position_export.has_value())
        {
            throw std::runtime_error(
                "Latte SPIR-V vertex subset requires a completed position export");
        }

        for (std::size_t output_index = 0U;
             output_index < parameter_outputs_.size();
             ++output_index)
        {
            const LatteSpirvVertexParameterOutput& parameter_output =
                parameter_outputs_[output_index];
            const auto is_matching_export =
                [&parameter_output](const LatteShaderIrExport& shader_export)
                {
                    const std::uint64_t export_end =
                        static_cast<std::uint64_t>(
                            shader_export.fields.array_base) +
                        shader_export.fields.burst_count;
                    return parameter_output.export_index >=
                            shader_export.fields.array_base &&
                        parameter_output.export_index < export_end;
                };
            const auto export_iterator = std::find_if(
                parameter_exports.begin(),
                parameter_exports.end(),
                is_matching_export);
            const std::size_t matching_export_count = std::count_if(
                parameter_exports.begin(),
                parameter_exports.end(),
                is_matching_export);
            if (export_iterator == parameter_exports.end() ||
                matching_export_count != 1U)
            {
                throw std::runtime_error(
                    "Latte SPIR-V vertex parameter output has no unique matching guest export");
            }
            store_parameter(
                *export_iterator,
                parameter_output.export_index,
                parameter_output_variable_ids_[output_index]);
        }

        store_position(position_export->fields);
        append_instruction(function_body_, 253U, {});
        append_instruction(function_body_, 56U, {});

        std::vector<std::uint32_t> module{
            0x07230203U, 0x00010000U, 0U, next_id_, 0U};
        append_section(module, capabilities_);
        append_section(module, memory_model_);
        append_section(module, entry_point_);
        append_section(module, annotations_);
        append_section(module, types_);
        append_section(module, constants_);
        append_section(module, globals_);
        append_instruction(module, 54U, {
            void_type_id_, function_id_, 0U, function_type_id_});
        append_section(module, function_body_);
        return module;
    }

private:
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
        if (word_count > 0xFFFFU)
        {
            throw std::runtime_error(
                "SPIR-V instruction exceeds the word-count limit");
        }
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

    [[nodiscard]] std::uint32_t constant(std::uint32_t value)
    {
        const auto existing = constant_ids_.find(value);
        if (existing != constant_ids_.end())
        {
            return existing->second;
        }
        const std::uint32_t result_id = allocate_id();
        append_instruction(constants_, 43U, {
            uint_type_id_, result_id, value});
        constant_ids_.emplace(value, result_id);
        return result_id;
    }

    [[nodiscard]] std::uint32_t source_value(
        std::uint16_t selector,
        std::uint8_t channel,
        const RegisterValues& source_registers,
        const std::optional<VectorValue>& previous_vector)
    {
        if (channel >= 4U)
        {
            throw std::runtime_error(
                "Latte SPIR-V source channel is outside a four-component register");
        }
        if (selector == latte_alu_source_previous_vector_selector)
        {
            if (!previous_vector.has_value() ||
                !(*previous_vector)[channel].has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V PV source has no preceding vector result");
            }
            return *(*previous_vector)[channel];
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

        if (selector >= source_registers.size() ||
            !source_registers[selector][channel].has_value())
        {
            throw std::runtime_error(
                "Latte SPIR-V source reads an uninitialized or unsupported GPR");
        }
        return *source_registers[selector][channel];
    }

    [[nodiscard]] std::uint32_t apply_source_modifiers(
        std::uint32_t value,
        bool absolute,
        bool negate)
    {
        if (absolute)
        {
            const std::uint32_t absolute_id = allocate_id();
            append_instruction(function_body_, 199U, {
                uint_type_id_, absolute_id, value,
                constant(0x7FFFFFFFU)});
            value = absolute_id;
        }
        if (negate)
        {
            const std::uint32_t negate_id = allocate_id();
            append_instruction(function_body_, 198U, {
                uint_type_id_, negate_id, value,
                constant(0x80000000U)});
            value = negate_id;
        }
        return value;
    }

    [[nodiscard]] std::uint32_t emit_op3_value(
        const LatteAluOp3Instruction& operation,
        const RegisterValues& source_registers,
        const std::optional<VectorValue>& previous_vector)
    {
        if ((operation.opcode != latte_alu_op3_conditional_move_equal_opcode &&
             operation.opcode != latte_alu_op3_conditional_move_greater_opcode &&
             operation.opcode != latte_alu_op3_conditional_move_greater_equal_opcode &&
             operation.opcode != latte_alu_op3_conditional_move_equal_integer_opcode &&
             operation.opcode != latte_alu_op3_conditional_move_greater_integer_opcode &&
             operation.opcode != latte_alu_op3_conditional_move_greater_equal_integer_opcode) ||
            operation.destination_gpr >= registers_.size() ||
            operation.source0_relative || operation.source1_relative ||
            operation.source2_relative || operation.index_mode != 0U ||
            operation.bank_swizzle != 0U ||
            operation.predicate_selection != 0U ||
            operation.destination_relative || operation.destination_clamp)
        {
            throw std::runtime_error(
                "Latte SPIR-V vertex subset received an unsupported OP3 operation or modifier");
        }

        const bool is_integer_operation =
            operation.opcode ==
                latte_alu_op3_conditional_move_equal_integer_opcode ||
            operation.opcode ==
                latte_alu_op3_conditional_move_greater_integer_opcode ||
            operation.opcode ==
                latte_alu_op3_conditional_move_greater_equal_integer_opcode;
        if (is_integer_operation &&
            (operation.source0_negate || operation.source1_negate ||
             operation.source2_negate))
        {
            throw std::runtime_error(
                "Latte SPIR-V integer OP3 conditional move does not support source negation");
        }

        std::uint32_t source0_id = source_value(
            operation.source0_selector,
            operation.source0_channel,
            source_registers,
            previous_vector);
        source0_id = apply_source_modifiers(
            source0_id,
            false,
            operation.source0_negate);
        const std::uint32_t source1_id = apply_source_modifiers(
            source_value(
                operation.source1_selector,
                operation.source1_channel,
                source_registers,
                previous_vector),
            false,
            operation.source1_negate);
        const std::uint32_t source2_id = apply_source_modifiers(
            source_value(
                operation.source2_selector,
                operation.source2_channel,
                source_registers,
                previous_vector),
            false,
            operation.source2_negate);

        std::uint32_t comparison_id = allocate_id();
        std::uint16_t comparison_opcode = 0U;
        std::uint32_t comparison_source_id = allocate_id();
        std::uint32_t zero_id = 0U;
        if (is_integer_operation)
        {
            append_instruction(function_body_, 124U, {
                int_type_id_, comparison_source_id, source0_id});
            zero_id = signed_zero_id_;
            switch (operation.opcode)
            {
            case latte_alu_op3_conditional_move_equal_integer_opcode:
                comparison_opcode = 170U;
                break;
            case latte_alu_op3_conditional_move_greater_integer_opcode:
                comparison_opcode = 173U;
                break;
            case latte_alu_op3_conditional_move_greater_equal_integer_opcode:
                comparison_opcode = 175U;
                break;
            default:
                throw std::runtime_error(
                    "Latte SPIR-V vertex subset received an unsupported integer OP3 opcode");
            }
        }
        else
        {
            append_instruction(function_body_, 124U, {
                float_type_id_, comparison_source_id, source0_id});
            zero_id = allocate_id();
            append_instruction(function_body_, 124U, {
                float_type_id_, zero_id, constant(0U)});
            switch (operation.opcode)
            {
            case latte_alu_op3_conditional_move_equal_opcode:
                comparison_opcode = 180U;
                break;
            case latte_alu_op3_conditional_move_greater_opcode:
                comparison_opcode = 186U;
                break;
            case latte_alu_op3_conditional_move_greater_equal_opcode:
                comparison_opcode = 190U;
                break;
            default:
                throw std::runtime_error(
                    "Latte SPIR-V vertex subset received an unsupported floating-point OP3 opcode");
            }
        }
        append_instruction(function_body_, comparison_opcode, {
            bool_type_id_, comparison_id, comparison_source_id, zero_id});

        const std::uint32_t result_id = allocate_id();
        append_instruction(function_body_, 169U, {
            uint_type_id_, result_id, comparison_id, source1_id, source2_id});
        return result_id;
    }

    [[nodiscard]] std::uint32_t emit_op2_value(
        const LatteAluOp2Instruction& operation,
        const RegisterValues& source_registers,
        const std::optional<VectorValue>& previous_vector)
    {
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
        if ((operation.opcode != latte_alu_op2_move_opcode &&
             operation.opcode != latte_alu_op2_add_float_opcode &&
             operation.opcode != latte_alu_op2_multiply_ieee_opcode &&
             operation.opcode != latte_alu_op2_maximum_dx10_opcode &&
             operation.opcode != latte_alu_op2_minimum_dx10_opcode &&
             !is_integer_operation) ||
            operation.destination_gpr >= registers_.size() ||
            !operation.write_mask || operation.source0_relative ||
            operation.update_execute_mask || operation.update_predicate ||
            operation.bank_swizzle != 0U ||
            operation.predicate_selection != 0U ||
            operation.destination_relative || operation.destination_clamp ||
            operation.output_modifier != 0U || operation.index_mode != 0U ||
            (operation.opcode == latte_alu_op2_move_opcode &&
             (operation.source1_relative || operation.source1_absolute ||
              operation.source1_negate)) ||
            (operation.opcode != latte_alu_op2_move_opcode &&
             operation.source1_relative) ||
            (is_integer_operation &&
             (operation.source0_absolute || operation.source0_negate ||
              operation.source1_absolute || operation.source1_negate)))
        {
            throw std::runtime_error(
                "Latte SPIR-V vertex subset received an unsupported ALU operation or modifier");
        }

        std::uint32_t source0_id = source_value(
            operation.source0_selector,
            operation.source0_channel,
            source_registers,
            previous_vector);
        source0_id = apply_source_modifiers(
            source0_id,
            operation.source0_absolute,
            operation.source0_negate);
        if (operation.opcode == latte_alu_op2_move_opcode)
        {
            return source0_id;
        }

        if (is_integer_operation)
        {
            if (operation.opcode == latte_alu_op2_not_integer_opcode)
            {
                const std::uint32_t result_id = allocate_id();
                append_instruction(function_body_, 200U, {
                    uint_type_id_, result_id, source0_id});
                return result_id;
            }
        }

        std::uint32_t source1_id = source_value(
            operation.source1_selector,
            operation.source1_channel,
            source_registers,
            previous_vector);
        if (is_integer_operation)
        {
            if (operation.opcode == latte_alu_op2_add_integer_opcode ||
                operation.opcode == latte_alu_op2_subtract_integer_opcode ||
                operation.opcode == latte_alu_op2_and_integer_opcode ||
                operation.opcode == latte_alu_op2_or_integer_opcode ||
                operation.opcode == latte_alu_op2_xor_integer_opcode)
            {
                std::uint16_t arithmetic_opcode = 0U;
                switch (operation.opcode)
                {
                case latte_alu_op2_add_integer_opcode:
                    arithmetic_opcode = 128U;
                    break;
                case latte_alu_op2_subtract_integer_opcode:
                    arithmetic_opcode = 130U;
                    break;
                case latte_alu_op2_and_integer_opcode:
                    arithmetic_opcode = 199U;
                    break;
                case latte_alu_op2_or_integer_opcode:
                    arithmetic_opcode = 197U;
                    break;
                case latte_alu_op2_xor_integer_opcode:
                    arithmetic_opcode = 198U;
                    break;
                default:
                    throw std::runtime_error(
                        "Latte SPIR-V vertex subset received an unsupported integer operation");
                }
                const std::uint32_t result_id = allocate_id();
                append_instruction(function_body_, arithmetic_opcode, {
                    uint_type_id_, result_id, source0_id, source1_id});
                return result_id;
            }

            std::uint32_t comparison_source0_id = source0_id;
            std::uint32_t comparison_source1_id = source1_id;
            std::uint16_t comparison_opcode = 0U;
            bool return_comparison_mask = true;
            switch (operation.opcode)
            {
            case latte_alu_op2_maximum_signed_integer_opcode:
                comparison_opcode = 175U;
                return_comparison_mask = false;
                comparison_source0_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source0_id, source0_id});
                comparison_source1_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source1_id, source1_id});
                break;
            case latte_alu_op2_minimum_signed_integer_opcode:
                comparison_opcode = 179U;
                return_comparison_mask = false;
                comparison_source0_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source0_id, source0_id});
                comparison_source1_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source1_id, source1_id});
                break;
            case latte_alu_op2_maximum_unsigned_integer_opcode:
                comparison_opcode = 174U;
                return_comparison_mask = false;
                break;
            case latte_alu_op2_minimum_unsigned_integer_opcode:
                comparison_opcode = 178U;
                return_comparison_mask = false;
                break;
            case latte_alu_op2_set_equal_integer_opcode:
                comparison_opcode = 170U;
                break;
            case latte_alu_op2_set_not_equal_integer_opcode:
                comparison_opcode = 171U;
                break;
            case latte_alu_op2_set_greater_integer_opcode:
                comparison_opcode = 173U;
                comparison_source0_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source0_id, source0_id});
                comparison_source1_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source1_id, source1_id});
                break;
            case latte_alu_op2_set_greater_equal_integer_opcode:
                comparison_opcode = 175U;
                comparison_source0_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source0_id, source0_id});
                comparison_source1_id = allocate_id();
                append_instruction(function_body_, 124U, {
                    int_type_id_, comparison_source1_id, source1_id});
                break;
            case latte_alu_op2_set_greater_unsigned_opcode:
                comparison_opcode = 172U;
                break;
            case latte_alu_op2_set_greater_equal_unsigned_opcode:
                comparison_opcode = 174U;
                break;
            default:
                throw std::runtime_error(
                    "Latte SPIR-V vertex subset received an unsupported integer comparison");
            }

            const std::uint32_t comparison_id = allocate_id();
            append_instruction(function_body_, comparison_opcode, {
                bool_type_id_, comparison_id,
                comparison_source0_id, comparison_source1_id});
            const std::uint32_t result_id = allocate_id();
            if (return_comparison_mask)
            {
                append_instruction(function_body_, 169U, {
                    uint_type_id_, result_id, comparison_id,
                    unsigned_all_bits_set_id_, unsigned_zero_id_});
            }
            else
            {
                append_instruction(function_body_, 169U, {
                    uint_type_id_, result_id, comparison_id,
                    source0_id, source1_id});
            }
            return result_id;
        }

        source1_id = apply_source_modifiers(
            source1_id,
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
                operation.opcode == latte_alu_op2_maximum_dx10_opcode
                    ? 190U
                    : 184U;
            append_instruction(function_body_, comparison_opcode, {
                bool_type_id_, comparison_id,
                source0_float_id, source1_float_id});
            append_instruction(function_body_, 169U, {
                float_type_id_, result_float_id,
                comparison_id, source0_float_id, source1_float_id});
        }
        const std::uint32_t result_id = allocate_id();
        append_instruction(function_body_, 124U, {
            uint_type_id_, result_id, result_float_id});
        return result_id;
    }

    void execute_vector_clause(
        const std::vector<LatteAluVectorInstructionGroup>& groups)
    {
        if (groups.empty())
        {
            throw std::runtime_error(
                "Latte SPIR-V ALU clause must contain an instruction group");
        }

        std::optional<VectorValue> previous_vector;
        for (const LatteAluVectorInstructionGroup& group : groups)
        {
            if (group.instructions.size() != 4U)
            {
                throw std::runtime_error(
                    "Latte SPIR-V ALU group has an invalid lane count");
            }
            const RegisterValues source_registers = registers_;
            VectorValue group_result{};
            for (std::size_t lane = 0U;
                 lane < group.instructions.size();
                 ++lane)
            {
                const LatteAluVectorInstruction& vector_instruction =
                    group.instructions[lane];
                const std::uint8_t destination_component = std::visit(
                    [](const auto& operation)
                    {
                        return operation.destination_component;
                    },
                    vector_instruction);
                const bool last_in_group = std::visit(
                    [](const auto& operation)
                    {
                        return operation.last_in_group;
                    },
                    vector_instruction);
                const std::uint8_t destination_gpr = std::visit(
                    [](const auto& operation)
                    {
                        return operation.destination_gpr;
                    },
                    vector_instruction);
                if (destination_component != lane ||
                    last_in_group != (lane == 3U) ||
                    destination_gpr >= registers_.size())
                {
                    throw std::runtime_error(
                        "Latte SPIR-V ALU group has an invalid destination lane");
                }

                if (const auto* op2_operation = std::get_if<
                        LatteAluOp2Instruction>(&vector_instruction))
                {
                    group_result[lane] = emit_op2_value(
                        *op2_operation,
                        source_registers,
                        previous_vector);
                }
                else
                {
                    group_result[lane] = emit_op3_value(
                        std::get<LatteAluOp3Instruction>(vector_instruction),
                        source_registers,
                        previous_vector);
                }
            }

            for (std::size_t lane = 0U;
                 lane < group.instructions.size();
                 ++lane)
            {
                const std::uint8_t destination_gpr = std::visit(
                    [](const auto& operation)
                    {
                        return operation.destination_gpr;
                    },
                    group.instructions[lane]);
                registers_[destination_gpr][lane] = group_result[lane];
            }
            previous_vector = group_result;
        }
    }

    void execute_clause(const LatteShaderIrAluClause& clause)
    {
        const std::size_t active_group_representations =
            static_cast<std::size_t>(!clause.groups.empty()) +
            static_cast<std::size_t>(!clause.op3_groups.empty()) +
            static_cast<std::size_t>(!clause.vector_groups.empty());
        if (active_group_representations != 1U)
        {
            throw std::runtime_error(
                "Latte SPIR-V ALU clause has incompatible or empty group representations");
        }

        std::vector<LatteAluVectorInstructionGroup> vector_groups;
        if (!clause.vector_groups.empty())
        {
            execute_vector_clause(clause.vector_groups);
            return;
        }

        if (!clause.groups.empty())
        {
            vector_groups.reserve(clause.groups.size());
            for (const LatteAluOp2InstructionGroup& group : clause.groups)
            {
                LatteAluVectorInstructionGroup vector_group{};
                vector_group.word_offset = group.word_offset;
                vector_group.instructions.reserve(group.instructions.size());
                for (const LatteAluOp2Instruction& operation : group.instructions)
                {
                    vector_group.instructions.emplace_back(operation);
                }
                vector_groups.push_back(std::move(vector_group));
            }
        }
        else
        {
            vector_groups.reserve(clause.op3_groups.size());
            for (const LatteAluOp3InstructionGroup& group : clause.op3_groups)
            {
                LatteAluVectorInstructionGroup vector_group{};
                vector_group.word_offset = group.word_offset;
                vector_group.instructions.reserve(group.instructions.size());
                for (const LatteAluOp3Instruction& operation : group.instructions)
                {
                    vector_group.instructions.emplace_back(operation);
                }
                vector_groups.push_back(std::move(vector_group));
            }
        }
        execute_vector_clause(vector_groups);
    }

    static void validate_position_export(
        const LatteShaderIrExport& shader_export)
    {
        constexpr std::array<std::uint8_t, 4U> identity_selectors{
            0U, 1U, 2U, 3U};
        const LatteExportFields& fields = shader_export.fields;
        if (!shader_export.completes_type ||
            fields.type != LatteExportType::position ||
            fields.array_base != 0x3CU || fields.burst_count != 1U ||
            fields.read_write_relative || fields.index_gpr != 0U ||
            fields.uses_buffer_encoding ||
            fields.component_selectors != identity_selectors)
        {
            throw std::runtime_error(
                "Latte SPIR-V vertex subset requires one completed identity position export");
        }
    }

    void store_parameter(
        const LatteShaderIrExport& shader_export,
        std::uint32_t export_index,
        std::uint32_t output_variable_id)
    {
        constexpr std::array<std::uint8_t, 4U> identity_selectors{
            0U, 1U, 2U, 3U};
        const LatteExportFields& fields = shader_export.fields;
        if (fields.type != LatteExportType::parameter ||
            fields.burst_count != 1U || fields.read_write_relative ||
            fields.index_gpr != 0U || fields.uses_buffer_encoding ||
            fields.component_selectors != identity_selectors ||
            export_index < fields.array_base ||
            static_cast<std::uint64_t>(export_index) >=
                static_cast<std::uint64_t>(fields.array_base) +
                    fields.burst_count)
        {
            throw std::runtime_error(
                "Latte SPIR-V vertex subset requires completed identity parameter exports");
        }

        std::array<std::uint32_t, 4U> float_components{};
        const std::size_t source_gpr =
            fields.read_write_gpr + (export_index - fields.array_base);
        if (source_gpr >= registers_.size())
        {
            throw std::runtime_error(
                "Latte SPIR-V parameter export exceeds the shader GPR file");
        }
        for (std::size_t component = 0U; component < float_components.size(); ++component)
        {
            const auto& register_component =
                registers_[source_gpr][component];
            if (!register_component.has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V parameter export reads an unwritten GPR component");
            }
            float_components[component] = allocate_id();
            append_instruction(function_body_, 124U, {
                float_type_id_, float_components[component],
                *register_component});
        }

        const std::uint32_t parameter_id = allocate_id();
        append_instruction(function_body_, 80U, {
            vector_type_id_, parameter_id,
            float_components[0U], float_components[1U],
            float_components[2U], float_components[3U]});
        append_instruction(function_body_, 62U, {
            output_variable_id, parameter_id});
    }

    void store_position(const LatteExportFields& position_export)
    {
        std::array<std::uint32_t, 4U> float_components{};
        for (std::size_t component = 0U; component < 4U; ++component)
        {
            const std::size_t source_gpr = position_export.read_write_gpr;
            if (!registers_[source_gpr][component].has_value())
            {
                throw std::runtime_error(
                    "Latte SPIR-V position export reads an unwritten GPR component");
            }
            float_components[component] = allocate_id();
            append_instruction(function_body_, 124U, {
                float_type_id_, float_components[component],
                *registers_[source_gpr][component]});
        }

        const std::uint32_t position_id = allocate_id();
        append_instruction(function_body_, 80U, {
            vector_type_id_, position_id,
            float_components[0U], float_components[1U],
            float_components[2U], float_components[3U]});
        const std::uint32_t zero_index = constant(0U);
        const std::uint32_t position_pointer = allocate_id();
        append_instruction(function_body_, 65U, {
            output_vector_pointer_type_id_, position_pointer,
            output_variable_id_, zero_index});
        append_instruction(function_body_, 62U, {
            position_pointer, position_id});
    }

    LatteSpirvVertexInput vertex_input_{};
    std::vector<LatteSpirvVertexParameterOutput> parameter_outputs_;
    std::vector<std::uint32_t> parameter_output_variable_ids_;
    std::uint32_t next_id_{1U};
    std::uint32_t void_type_id_{};
    std::uint32_t function_type_id_{};
    std::uint32_t float_type_id_{};
    std::uint32_t int_type_id_{};
    std::uint32_t uint_type_id_{};
    std::uint32_t signed_zero_id_{};
    std::uint32_t unsigned_zero_id_{};
    std::uint32_t unsigned_all_bits_set_id_{};
    std::uint32_t bool_type_id_{};
    std::uint32_t vector_type_id_{};
    std::uint32_t input_pointer_type_id_{};
    std::uint32_t output_struct_type_id_{};
    std::uint32_t output_struct_pointer_type_id_{};
    std::uint32_t output_vector_pointer_type_id_{};
    std::uint32_t function_id_{};
    std::uint32_t label_id_{};
    std::uint32_t input_variable_id_{};
    std::uint32_t output_variable_id_{};
    RegisterValues registers_{};
    std::unordered_map<std::uint32_t, std::uint32_t> constant_ids_;
    std::vector<std::uint32_t> capabilities_;
    std::vector<std::uint32_t> memory_model_;
    std::vector<std::uint32_t> entry_point_;
    std::vector<std::uint32_t> annotations_;
    std::vector<std::uint32_t> types_;
    std::vector<std::uint32_t> constants_;
    std::vector<std::uint32_t> globals_;
    std::vector<std::uint32_t> function_body_;
};

} // namespace detail

// Emit one float32x4 input, no-literal MOV/ADD/MUL/MAX_DX10/MIN_DX10 groups,
// and one completed identity position export. Unsupported behavior is rejected.
[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_vertex_position_subset(
    const LatteShaderIrProgram& program,
    LatteSpirvVertexInput vertex_input,
    std::span<const LatteSpirvVertexParameterOutput> parameter_outputs = {})
{
    if (vertex_input.guest_gpr >= 128U)
    {
        throw std::runtime_error(
            "Latte SPIR-V vertex input GPR is outside the guest register file");
    }
    detail::LatteSpirvVertexSubsetBuilder builder(
        vertex_input, parameter_outputs);
    return builder.emit(program);
}

[[nodiscard]] inline std::vector<std::uint32_t>
emit_latte_spirv_vertex_position_mov_subset(
    const LatteShaderIrProgram& program,
    LatteSpirvVertexInput vertex_input)
{
    return emit_latte_spirv_vertex_position_subset(program, vertex_input);
}

} // namespace affogato::gpu
