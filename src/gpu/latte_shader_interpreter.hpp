#pragma once

#include "latte_shader_decoder.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace affogato::gpu
{

inline constexpr std::uint16_t latte_alu_op2_move_opcode = 0x019U; // OP2_INST_MOV
inline constexpr std::uint16_t latte_alu_op3_conditional_move_equal_opcode =
    0x018U;
inline constexpr std::uint16_t latte_alu_op3_conditional_move_greater_opcode =
    0x019U;
inline constexpr std::uint16_t
    latte_alu_op3_conditional_move_greater_equal_opcode = 0x01AU;
inline constexpr std::uint16_t
    latte_alu_op3_conditional_move_equal_integer_opcode = 0x01CU;
inline constexpr std::uint16_t
    latte_alu_op3_conditional_move_greater_integer_opcode = 0x01DU;
inline constexpr std::uint16_t
    latte_alu_op3_conditional_move_greater_equal_integer_opcode = 0x01EU;
inline constexpr std::uint16_t latte_alu_op2_add_float_opcode = 0x000U;
inline constexpr std::uint16_t latte_alu_op2_multiply_ieee_opcode = 0x002U;
inline constexpr std::uint16_t latte_alu_op2_maximum_dx10_opcode = 0x005U;
inline constexpr std::uint16_t latte_alu_op2_minimum_dx10_opcode = 0x006U;
inline constexpr std::uint32_t latte_binary32_sign_mask = 0x80000000U;
inline constexpr std::uint32_t latte_binary32_magnitude_mask = 0x7FFFFFFFU;
inline constexpr std::uint16_t latte_alu_op2_and_integer_opcode = 0x030U;
inline constexpr std::uint16_t latte_alu_op2_or_integer_opcode = 0x031U;
inline constexpr std::uint16_t latte_alu_op2_xor_integer_opcode = 0x032U;
inline constexpr std::uint16_t latte_alu_op2_not_integer_opcode = 0x033U;
inline constexpr std::uint16_t latte_alu_op2_add_integer_opcode = 0x034U;
inline constexpr std::uint16_t latte_alu_op2_subtract_integer_opcode = 0x035U;
inline constexpr std::uint16_t latte_alu_op2_maximum_signed_integer_opcode =
    0x036U;
inline constexpr std::uint16_t latte_alu_op2_minimum_signed_integer_opcode =
    0x037U;
inline constexpr std::uint16_t latte_alu_op2_maximum_unsigned_integer_opcode =
    0x038U;
inline constexpr std::uint16_t latte_alu_op2_minimum_unsigned_integer_opcode =
    0x039U;
inline constexpr std::uint16_t latte_alu_op2_set_equal_integer_opcode =
    0x03AU;
inline constexpr std::uint16_t latte_alu_op2_set_greater_integer_opcode =
    0x03BU;
inline constexpr std::uint16_t
    latte_alu_op2_set_greater_equal_integer_opcode = 0x03CU;
inline constexpr std::uint16_t latte_alu_op2_set_not_equal_integer_opcode =
    0x03DU;
inline constexpr std::uint16_t latte_alu_op2_set_greater_unsigned_opcode =
    0x03EU;
inline constexpr std::uint16_t
    latte_alu_op2_set_greater_equal_unsigned_opcode = 0x03FU;
inline constexpr std::uint16_t latte_alu_source_zero_selector = 248U;
inline constexpr std::uint16_t latte_alu_source_one_float_selector = 249U;
inline constexpr std::uint16_t latte_alu_source_one_integer_selector = 250U;
inline constexpr std::uint16_t latte_alu_source_negative_one_integer_selector =
    251U;
inline constexpr std::uint16_t latte_alu_source_half_float_selector = 252U;
inline constexpr std::uint16_t latte_alu_source_previous_vector_selector = 254U;
// R700 DX9 constant-file selectors address c[0, 255] in 256-511.
inline constexpr std::uint16_t latte_alu_source_constant_file_base = 256U;
inline constexpr std::uint16_t latte_alu_source_constant_file_last = 511U;
inline constexpr std::size_t latte_shader_register_component_count = 4U;

using LatteShaderRegister =
    std::array<std::uint32_t, latte_shader_register_component_count>;

struct LatteShaderConstantFile
{
    std::array<std::array<std::optional<std::uint32_t>,
        latte_shader_register_component_count>, 256U> entries{};
};

struct LatteShaderExportValue
{
    LatteExportType type{};
    std::uint32_t array_index{};
    std::array<std::optional<std::uint32_t>, 4U> components{};

    bool operator==(const LatteShaderExportValue&) const = default;
};

// Materialize the bounded normal-export swizzle form. This produces typed
// guest output values only; it does not implement stage linking or Vulkan IO.
[[nodiscard]] inline std::vector<LatteShaderExportValue>
evaluate_latte_normal_export(
    const LatteExportFields& export_fields,
    std::span<const LatteShaderRegister> guest_registers)
{
    constexpr std::uint8_t maximum_burst_count = 0x0FU + 1U;
    constexpr std::uint8_t zero_float_selector = 4U;
    constexpr std::uint8_t one_float_selector = 5U;
    constexpr std::uint8_t reserved_selector = 6U;
    constexpr std::uint8_t masked_selector = 7U;
    constexpr std::uint32_t one_float_bits = 0x3F800000U;

    if (export_fields.type == LatteExportType::unused ||
        export_fields.read_write_relative ||
        export_fields.index_gpr != 0U ||
        export_fields.uses_buffer_encoding ||
        export_fields.burst_count == 0U ||
        export_fields.burst_count > maximum_burst_count)
    {
        throw std::runtime_error(
            "Latte normal export uses unsupported type or addressing");
    }

    if (export_fields.read_write_gpr >= guest_registers.size() ||
        export_fields.burst_count >
            guest_registers.size() - export_fields.read_write_gpr)
    {
        throw std::runtime_error(
            "Latte normal export exceeds the shader GPR file");
    }

    for (const std::uint8_t selector : export_fields.component_selectors)
    {
        if (selector == reserved_selector || selector > masked_selector)
        {
            throw std::runtime_error(
                "Latte normal export contains a reserved component selector");
        }
    }

    std::vector<LatteShaderExportValue> values;
    values.reserve(export_fields.burst_count);

    for (std::uint8_t export_index = 0U;
         export_index < export_fields.burst_count;
         ++export_index)
    {
        const LatteShaderRegister& source =
            guest_registers[export_fields.read_write_gpr + export_index];
        LatteShaderExportValue value{};
        value.type = export_fields.type;
        value.array_index = export_fields.array_base + export_index;

        for (std::size_t component = 0U;
             component < export_fields.component_selectors.size();
             ++component)
        {
            const std::uint8_t selector =
                export_fields.component_selectors[component];
            if (selector < latte_shader_register_component_count)
            {
                value.components[component] = source[selector];
            }
            else if (selector == zero_float_selector)
            {
                value.components[component] = 0U;
            }
            else if (selector == one_float_selector)
            {
                value.components[component] = one_float_bits;
            }
            // masked_selector leaves the destination component unset.
        }

        values.push_back(value);
    }

    return values;
}

[[nodiscard]] inline std::uint32_t read_latte_alu_source(
    std::uint16_t source_selector,
    std::uint8_t source_channel,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (source_channel >= latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte ALU source channel is outside a four-component register");
    }

    if (source_selector == latte_alu_source_previous_vector_selector)
    {
        if (!incoming_previous_vector.has_value())
        {
            throw std::runtime_error(
                "Latte ALU PV source has no preceding vector result");
        }

        return (*incoming_previous_vector)[source_channel];
    }

    switch (source_selector)
    {
    case latte_alu_source_zero_selector:
        return 0x00000000U;
    case latte_alu_source_one_float_selector:
        return 0x3F800000U;
    case latte_alu_source_one_integer_selector:
        return 0x00000001U;
    case latte_alu_source_negative_one_integer_selector:
        return 0xFFFFFFFFU;
    case latte_alu_source_half_float_selector:
        return 0x3F000000U;
    default:
        break;
    }

    if (source_selector >= latte_alu_source_constant_file_base &&
        source_selector <= latte_alu_source_constant_file_last)
    {
        const std::size_t constant_index =
            source_selector - latte_alu_source_constant_file_base;
        if (constant_file == nullptr)
        {
            throw std::runtime_error(
                "Latte ALU constant-file source has no supplied constant file");
        }

        const std::optional<std::uint32_t>& constant_value =
            constant_file->entries[constant_index][source_channel];
        if (!constant_value.has_value())
        {
            throw std::runtime_error(
                "Latte ALU constant-file source reads an uninitialized component");
        }

        return *constant_value;
    }

    constexpr std::uint16_t maximum_gpr_selector = 127U;
    if (source_selector > maximum_gpr_selector ||
        source_selector >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte ALU source selector is not a supported GPR, constant-file "
            "entry, inline constant, or PV");
    }

    return guest_registers[source_selector][source_channel];
}

// Evaluate the documented floating-point OP3 conditional-move family.
// The selected source is returned as raw bits so NaN payloads and signed zero
// are preserved. Predicate state and relative addressing are not modeled.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_op3_conditional_move_value(
    const LatteAluOp3Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (instruction.opcode !=
            latte_alu_op3_conditional_move_equal_opcode &&
        instruction.opcode !=
            latte_alu_op3_conditional_move_greater_opcode &&
        instruction.opcode !=
            latte_alu_op3_conditional_move_greater_equal_opcode &&
        instruction.opcode !=
            latte_alu_op3_conditional_move_equal_integer_opcode &&
        instruction.opcode !=
            latte_alu_op3_conditional_move_greater_integer_opcode &&
        instruction.opcode !=
            latte_alu_op3_conditional_move_greater_equal_integer_opcode)
    {
        throw std::runtime_error(
            "Latte OP3 conditional move received an unsupported opcode");
    }

    if (instruction.source0_relative || instruction.source1_relative ||
        instruction.source2_relative || instruction.index_mode != 0U ||
        instruction.bank_swizzle != 0U ||
        instruction.predicate_selection != 0U ||
        instruction.destination_relative || instruction.destination_clamp)
    {
        throw std::runtime_error(
            "Latte OP3 conditional move received unsupported modifiers");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte OP3 conditional move destination is outside the register file");
    }

    if (instruction.destination_component >=
        latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte OP3 conditional move destination is outside a four-component GPR");
    }

    const bool is_integer_operation =
        instruction.opcode == latte_alu_op3_conditional_move_equal_integer_opcode ||
        instruction.opcode == latte_alu_op3_conditional_move_greater_integer_opcode ||
        instruction.opcode ==
            latte_alu_op3_conditional_move_greater_equal_integer_opcode;
    if (is_integer_operation &&
        (instruction.source0_negate || instruction.source1_negate ||
         instruction.source2_negate))
    {
        throw std::runtime_error(
            "Latte integer OP3 conditional move does not support source negation");
    }

    const auto read_source = [&](std::uint16_t selector,
                                 std::uint8_t channel,
                                 bool negate)
    {
        std::uint32_t value = read_latte_alu_source(
            selector,
            channel,
            guest_registers,
            incoming_previous_vector,
            constant_file);
        if (negate)
        {
            value ^= latte_binary32_sign_mask;
        }
        return value;
    };

    const std::uint32_t condition_bits = read_source(
        instruction.source0_selector,
        instruction.source0_channel,
        instruction.source0_negate);
    bool take_source1 = false;

    switch (instruction.opcode)
    {
    case latte_alu_op3_conditional_move_equal_opcode:
        take_source1 = std::bit_cast<float>(condition_bits) == 0.0F;
        break;
    case latte_alu_op3_conditional_move_greater_opcode:
        take_source1 = std::bit_cast<float>(condition_bits) > 0.0F;
        break;
    case latte_alu_op3_conditional_move_greater_equal_opcode:
        take_source1 = std::bit_cast<float>(condition_bits) >= 0.0F;
        break;
    case latte_alu_op3_conditional_move_equal_integer_opcode:
        take_source1 = condition_bits == 0U;
        break;
    case latte_alu_op3_conditional_move_greater_integer_opcode:
        take_source1 = std::bit_cast<std::int32_t>(condition_bits) > 0;
        break;
    case latte_alu_op3_conditional_move_greater_equal_integer_opcode:
        take_source1 = std::bit_cast<std::int32_t>(condition_bits) >= 0;
        break;
    default:
        throw std::runtime_error(
            "Latte OP3 conditional move received an unsupported opcode");
    }

    if (take_source1)
    {
        return read_source(
            instruction.source1_selector,
            instruction.source1_channel,
            instruction.source1_negate);
    }

    return read_source(
        instruction.source2_selector,
        instruction.source2_channel,
        instruction.source2_negate);
}

// Execute four ordered OP3 vector slots. Values are read before any writes,
// matching the group-level source snapshot used by the OP2 evaluator.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_op3_vector_group(
    const LatteAluOp3InstructionGroup& group,
    std::span<LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (group.instructions.size() != latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte OP3 group must contain exactly four vector instructions");
    }

    LatteShaderRegister group_vector{};
    const std::span<const LatteShaderRegister> read_only_registers(
        guest_registers);

    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const LatteAluOp3Instruction& instruction =
            group.instructions[component];
        if (instruction.destination_component != component ||
            instruction.last_in_group !=
                (component + 1U == group.instructions.size()))
        {
            throw std::runtime_error(
                "Latte OP3 group does not match the supported X/Y/Z/W layout");
        }

        group_vector[component] =
            evaluate_latte_alu_op3_conditional_move_value(
                instruction,
                read_only_registers,
                incoming_previous_vector,
                constant_file);
    }

    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const LatteAluOp3Instruction& instruction =
            group.instructions[component];
        guest_registers[instruction.destination_gpr][component] =
            group_vector[component];
    }

    return group_vector;
}

// Execute an OP3-only clause transactionally. PV is forwarded between its
// groups; the caller starts a new clause with no incoming PV value.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_op3_clause(
    std::span<const LatteAluOp3InstructionGroup> groups,
    std::span<LatteShaderRegister> guest_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (groups.empty())
    {
        throw std::runtime_error("Latte OP3 clause must contain a group");
    }

    std::vector<LatteShaderRegister> working_registers(
        guest_registers.begin(),
        guest_registers.end());
    std::optional<LatteShaderRegister> previous_vector;

    for (const LatteAluOp3InstructionGroup& group : groups)
    {
        previous_vector = execute_latte_alu_op3_vector_group(
            group,
            working_registers,
            previous_vector,
            constant_file);
    }

    std::copy(
        working_registers.begin(),
        working_registers.end(),
        guest_registers.begin());
    return *previous_vector;
}

// Evaluate supported GPR, inline-constant, constant-file, and prior-PV MOVs.
// The group executor applies destination writes and returns the vector result;
// previous-scalar state and other source modes remain unsupported.
// Shader registers hold raw 32-bit values so MOV preserves all source bits.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_move_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (instruction.opcode != latte_alu_op2_move_opcode)
    {
        throw std::runtime_error(
            "Latte ALU MOV evaluation received a different OP2 opcode");
    }

    if (instruction.source0_relative || instruction.update_execute_mask ||
        instruction.update_predicate || instruction.bank_swizzle != 0U ||
        instruction.predicate_selection != 0U ||
        instruction.destination_relative || instruction.destination_clamp ||
        instruction.output_modifier != 0U || instruction.index_mode != 0U ||
        !instruction.write_mask)
    {
        throw std::runtime_error(
            "Latte ALU MOV evaluation received unsupported modifiers");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte ALU MOV destination register is outside the register file");
    }

    if (instruction.destination_component >=
        latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte ALU MOV destination is outside a four-component GPR");
    }

    std::uint32_t source_value = read_latte_alu_source(
        instruction.source0_selector,
        instruction.source0_channel,
        guest_registers,
        incoming_previous_vector,
        constant_file);

    // R700 applies floating-point absolute value before source negation.
    // Sign-bit operations preserve every other binary32 bit, including NaN
    // payloads and the distinction between positive and negative zero.
    if (instruction.source0_absolute)
    {
        source_value &= latte_binary32_magnitude_mask;
    }

    if (instruction.source0_negate)
    {
        source_value ^= latte_binary32_sign_mask;
    }

    return source_value;
}

// Evaluate the documented two-source binary32 OP2 subset. The DX10 min/max
// forms select source 1 when the ordered comparison with source 0 is false,
// including when either operand is NaN.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_float_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (instruction.opcode != latte_alu_op2_add_float_opcode &&
        instruction.opcode != latte_alu_op2_multiply_ieee_opcode &&
        instruction.opcode != latte_alu_op2_maximum_dx10_opcode &&
        instruction.opcode != latte_alu_op2_minimum_dx10_opcode)
    {
        throw std::runtime_error(
            "Latte floating-point evaluation received an unsupported opcode");
    }

    if (instruction.source0_relative || instruction.source1_relative ||
        instruction.update_execute_mask || instruction.update_predicate ||
        instruction.bank_swizzle != 0U ||
        instruction.predicate_selection != 0U ||
        instruction.destination_relative || instruction.destination_clamp ||
        instruction.index_mode != 0U || !instruction.write_mask)
    {
        throw std::runtime_error(
            "Latte floating-point evaluation received unsupported modifiers");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte floating-point destination is outside the register file");
    }

    if (instruction.destination_component >=
        latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte floating-point destination is outside a four-component GPR");
    }

    const auto apply_source_modifiers = [](std::uint32_t value,
                                           bool absolute,
                                           bool negate)
    {
        if (absolute)
        {
            value &= latte_binary32_magnitude_mask;
        }
        if (negate)
        {
            value ^= latte_binary32_sign_mask;
        }
        return std::bit_cast<float>(value);
    };

    const float source0 = apply_source_modifiers(
        read_latte_alu_source(
            instruction.source0_selector,
            instruction.source0_channel,
            guest_registers,
            incoming_previous_vector,
            constant_file),
        instruction.source0_absolute,
        instruction.source0_negate);
    const float source1 = apply_source_modifiers(
        read_latte_alu_source(
            instruction.source1_selector,
            instruction.source1_channel,
            guest_registers,
            incoming_previous_vector,
            constant_file),
        instruction.source1_absolute,
        instruction.source1_negate);

    float result = 0.0F;
    switch (instruction.opcode)
    {
    case latte_alu_op2_add_float_opcode:
        result = source0 + source1;
        break;
    case latte_alu_op2_multiply_ieee_opcode:
        result = source0 * source1;
        break;
    case latte_alu_op2_maximum_dx10_opcode:
        result = source0 >= source1 ? source0 : source1;
        break;
    case latte_alu_op2_minimum_dx10_opcode:
        result = source0 < source1 ? source0 : source1;
        break;
    default:
        throw std::runtime_error(
            "Latte floating-point evaluation received an unsupported opcode");
    }

    switch (instruction.output_modifier)
    {
    case 0U:
        break;
    case 1U:
        result *= 2.0F;
        break;
    case 2U:
        result *= 4.0F;
        break;
    case 3U:
        result *= 0.5F;
        break;
    default:
        throw std::runtime_error(
            "Latte floating-point output modifier is invalid");
    }

    return std::bit_cast<std::uint32_t>(result);
}

// Evaluate supported GPR, inline-constant, constant-file, and prior-PV
// integer operands.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_integer_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (instruction.opcode != latte_alu_op2_and_integer_opcode &&
        instruction.opcode != latte_alu_op2_or_integer_opcode &&
        instruction.opcode != latte_alu_op2_xor_integer_opcode &&
        instruction.opcode != latte_alu_op2_not_integer_opcode &&
        instruction.opcode != latte_alu_op2_add_integer_opcode &&
        instruction.opcode != latte_alu_op2_subtract_integer_opcode &&
        instruction.opcode != latte_alu_op2_maximum_signed_integer_opcode &&
        instruction.opcode != latte_alu_op2_minimum_signed_integer_opcode &&
        instruction.opcode != latte_alu_op2_maximum_unsigned_integer_opcode &&
        instruction.opcode != latte_alu_op2_minimum_unsigned_integer_opcode &&
        instruction.opcode != latte_alu_op2_set_equal_integer_opcode &&
        instruction.opcode != latte_alu_op2_set_greater_integer_opcode &&
        instruction.opcode != latte_alu_op2_set_greater_equal_integer_opcode &&
        instruction.opcode != latte_alu_op2_set_not_equal_integer_opcode &&
        instruction.opcode != latte_alu_op2_set_greater_unsigned_opcode &&
        instruction.opcode != latte_alu_op2_set_greater_equal_unsigned_opcode)
    {
        throw std::runtime_error(
            "Latte integer evaluation received an unsupported opcode");
    }

    if (instruction.source0_relative || instruction.source1_relative ||
        instruction.source0_negate || instruction.source1_negate ||
        instruction.source0_absolute || instruction.source1_absolute ||
        instruction.update_execute_mask || instruction.update_predicate ||
        instruction.bank_swizzle != 0U ||
        instruction.predicate_selection != 0U ||
        instruction.destination_relative || instruction.destination_clamp ||
        instruction.output_modifier != 0U || instruction.index_mode != 0U ||
        !instruction.write_mask)
    {
        throw std::runtime_error(
            "Latte integer evaluation received unsupported modifiers");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte integer destination is outside the register file");
    }

    if (instruction.destination_component >=
        latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte integer destination is outside a four-component GPR");
    }

    const std::uint32_t source0 = read_latte_alu_source(
        instruction.source0_selector,
        instruction.source0_channel,
        guest_registers,
        incoming_previous_vector,
        constant_file);
    if (instruction.opcode == latte_alu_op2_not_integer_opcode)
    {
        return ~source0;
    }

    const std::uint32_t source1 = read_latte_alu_source(
        instruction.source1_selector,
        instruction.source1_channel,
        guest_registers,
        incoming_previous_vector,
        constant_file);
    constexpr std::uint32_t signed_order_bias = 0x80000000U;
    const std::uint32_t signed_order_source0 = source0 ^ signed_order_bias;
    const std::uint32_t signed_order_source1 = source1 ^ signed_order_bias;
    bool comparison_result = false;

    switch (instruction.opcode)
    {
    case latte_alu_op2_add_integer_opcode:
        return source0 + source1;
    case latte_alu_op2_subtract_integer_opcode:
        return source0 - source1;
    case latte_alu_op2_maximum_signed_integer_opcode:
        return signed_order_source0 >= signed_order_source1 ? source0 : source1;
    case latte_alu_op2_minimum_signed_integer_opcode:
        return signed_order_source0 <= signed_order_source1 ? source0 : source1;
    case latte_alu_op2_maximum_unsigned_integer_opcode:
        return source0 >= source1 ? source0 : source1;
    case latte_alu_op2_minimum_unsigned_integer_opcode:
        return source0 <= source1 ? source0 : source1;
    case latte_alu_op2_set_equal_integer_opcode:
        comparison_result = source0 == source1;
        break;
    case latte_alu_op2_set_not_equal_integer_opcode:
        comparison_result = source0 != source1;
        break;
    case latte_alu_op2_set_greater_integer_opcode:
        comparison_result = signed_order_source0 > signed_order_source1;
        break;
    case latte_alu_op2_set_greater_equal_integer_opcode:
        comparison_result = signed_order_source0 >= signed_order_source1;
        break;
    case latte_alu_op2_set_greater_unsigned_opcode:
        comparison_result = source0 > source1;
        break;
    case latte_alu_op2_set_greater_equal_unsigned_opcode:
        comparison_result = source0 >= source1;
        break;
    case latte_alu_op2_and_integer_opcode:
        return source0 & source1;
    case latte_alu_op2_or_integer_opcode:
        return source0 | source1;
    case latte_alu_op2_xor_integer_opcode:
        return source0 ^ source1;
    default:
        throw std::runtime_error(
            "Latte integer evaluation received an unsupported opcode");
    }

    return comparison_result ? 0xFFFFFFFFU : 0x00000000U;
}

// Execute a supported four-slot X/Y/Z/W OP2 group. Source values are gathered
// before destination writes, preserving same-group parallel read semantics.
// The returned vector is the group's ALU.[X,Y,Z,W] result (PV).
[[nodiscard]] inline LatteShaderRegister
execute_latte_alu_op2_vector_group(
    const LatteAluOp2InstructionGroup& group,
    std::span<LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (group.instructions.size() != latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte OP2 group must contain exactly four vector instructions");
    }

    LatteShaderRegister group_vector{};
    const std::span<const LatteShaderRegister> read_only_registers(
        guest_registers);

    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const auto& instruction = group.instructions[component];
        if (instruction.destination_component != component ||
            instruction.last_in_group !=
                (component + 1U == group.instructions.size()))
        {
            throw std::runtime_error(
                "Latte OP2 group does not match the supported X/Y/Z/W layout");
        }

        switch (instruction.opcode)
        {
        case latte_alu_op2_move_opcode:
            group_vector[component] = evaluate_latte_alu_move_value(
                instruction,
                read_only_registers,
                incoming_previous_vector,
                constant_file);
            break;
        case latte_alu_op2_add_float_opcode:
        case latte_alu_op2_multiply_ieee_opcode:
        case latte_alu_op2_maximum_dx10_opcode:
        case latte_alu_op2_minimum_dx10_opcode:
            group_vector[component] = evaluate_latte_alu_float_value(
                instruction,
                read_only_registers,
                incoming_previous_vector,
                constant_file);
            break;
        case latte_alu_op2_and_integer_opcode:
        case latte_alu_op2_or_integer_opcode:
        case latte_alu_op2_xor_integer_opcode:
        case latte_alu_op2_not_integer_opcode:
        case latte_alu_op2_add_integer_opcode:
        case latte_alu_op2_subtract_integer_opcode:
        case latte_alu_op2_maximum_signed_integer_opcode:
        case latte_alu_op2_minimum_signed_integer_opcode:
        case latte_alu_op2_maximum_unsigned_integer_opcode:
        case latte_alu_op2_minimum_unsigned_integer_opcode:
        case latte_alu_op2_set_equal_integer_opcode:
        case latte_alu_op2_set_greater_integer_opcode:
        case latte_alu_op2_set_greater_equal_integer_opcode:
        case latte_alu_op2_set_not_equal_integer_opcode:
        case latte_alu_op2_set_greater_unsigned_opcode:
        case latte_alu_op2_set_greater_equal_unsigned_opcode:
            group_vector[component] =
                evaluate_latte_alu_integer_value(
                    instruction,
                    read_only_registers,
                    incoming_previous_vector,
                    constant_file);
            break;
        default:
            throw std::runtime_error(
                "Latte OP2 group contains an unsupported instruction");
        }
    }

    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const auto& instruction = group.instructions[component];
        guest_registers[instruction.destination_gpr][component] =
            group_vector[component];
    }

    return group_vector;
}

// Execute a four-slot vector group that may mix supported OP2 and OP3
// operations. All reads use the pre-group register snapshot, then writes are
// committed together so lane ordering does not become observable.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_vector_group(
    const LatteAluVectorInstructionGroup& group,
    std::span<LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (group.instructions.size() != latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte mixed ALU group must contain exactly four vector instructions");
    }

    const std::span<const LatteShaderRegister> read_only_registers(
        guest_registers);
    LatteShaderRegister group_vector{};
    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const LatteAluVectorInstruction& vector_instruction =
            group.instructions[component];
        if (const auto* op2_instruction =
                std::get_if<LatteAluOp2Instruction>(&vector_instruction))
        {
            if (op2_instruction->destination_component != component ||
                op2_instruction->last_in_group != (component == 3U))
            {
                throw std::runtime_error(
                    "Latte mixed OP2/OP3 group has an invalid OP2 lane layout");
            }

            switch (op2_instruction->opcode)
            {
            case latte_alu_op2_move_opcode:
                group_vector[component] = evaluate_latte_alu_move_value(
                    *op2_instruction,
                    read_only_registers,
                    incoming_previous_vector,
                    constant_file);
                break;
            case latte_alu_op2_add_float_opcode:
            case latte_alu_op2_multiply_ieee_opcode:
            case latte_alu_op2_maximum_dx10_opcode:
            case latte_alu_op2_minimum_dx10_opcode:
                group_vector[component] = evaluate_latte_alu_float_value(
                    *op2_instruction,
                    read_only_registers,
                    incoming_previous_vector,
                    constant_file);
                break;
            default:
                group_vector[component] = evaluate_latte_alu_integer_value(
                    *op2_instruction,
                    read_only_registers,
                    incoming_previous_vector,
                    constant_file);
                break;
            }
            continue;
        }

        const LatteAluOp3Instruction& op3_instruction =
            std::get<LatteAluOp3Instruction>(vector_instruction);
        if (op3_instruction.destination_component != component ||
            op3_instruction.last_in_group != (component == 3U))
        {
            throw std::runtime_error(
                "Latte mixed OP2/OP3 group has an invalid OP3 lane layout");
        }
        group_vector[component] =
            evaluate_latte_alu_op3_conditional_move_value(
                op3_instruction,
                read_only_registers,
                incoming_previous_vector,
                constant_file);
    }

    for (std::size_t component = 0U;
         component < group.instructions.size();
         ++component)
    {
        const std::uint8_t destination_gpr = std::visit(
            [](const auto& instruction)
            {
                return instruction.destination_gpr;
            },
            group.instructions[component]);
        if (destination_gpr >= guest_registers.size())
        {
            throw std::runtime_error(
                "Latte mixed ALU destination is outside the register file");
        }
        guest_registers[destination_gpr][component] = group_vector[component];
    }

    return group_vector;
}

// Execute mixed OP2/OP3 vector groups transactionally; PV is forwarded only
// between groups in this clause.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_vector_clause(
    std::span<const LatteAluVectorInstructionGroup> groups,
    std::span<LatteShaderRegister> guest_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (groups.empty())
    {
        throw std::runtime_error("Latte mixed ALU clause must contain a group");
    }

    std::vector<LatteShaderRegister> working_registers(
        guest_registers.begin(),
        guest_registers.end());
    std::optional<LatteShaderRegister> previous_vector;
    for (const LatteAluVectorInstructionGroup& group : groups)
    {
        previous_vector = execute_latte_alu_vector_group(
            group,
            working_registers,
            previous_vector,
            constant_file);
    }

    std::copy(
        working_registers.begin(),
        working_registers.end(),
        guest_registers.begin());
    return *previous_vector;
}

// Execute a clause containing only four-slot OP2 vector groups. PV is visible
// to later groups in the same clause and is reset when the caller starts a
// separate clause. Work on a temporary register file so an unsupported later
// group cannot leave partially updated guest shader state.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_op2_clause(
    std::span<const LatteAluOp2InstructionGroup> groups,
    std::span<LatteShaderRegister> guest_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    if (groups.empty())
    {
        throw std::runtime_error("Latte OP2 clause must contain a group");
    }

    std::vector<LatteShaderRegister> working_registers(
        guest_registers.begin(),
        guest_registers.end());
    std::optional<LatteShaderRegister> previous_vector;

    for (const LatteAluOp2InstructionGroup& group : groups)
    {
        previous_vector = execute_latte_alu_op2_vector_group(
            group,
            working_registers,
            previous_vector,
            constant_file);
    }

    std::copy(
        working_registers.begin(),
        working_registers.end(),
        guest_registers.begin());
    return *previous_vector;
}

// Decode and execute a raw clause for callers that do not retain decoded IR.
[[nodiscard]] inline LatteShaderRegister execute_latte_alu_op2_clause(
    std::span<const LatteAluClauseSlot> slots,
    std::span<LatteShaderRegister> guest_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
{
    const std::vector<LatteAluOp2InstructionGroup> groups =
        decode_latte_alu_op2_groups(slots);
    return execute_latte_alu_op2_clause(
        groups,
        guest_registers,
        constant_file);
}

} // namespace affogato::gpu
