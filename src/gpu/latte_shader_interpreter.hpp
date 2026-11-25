#pragma once

#include "latte_shader_decoder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>

namespace affogato::gpu
{

inline constexpr std::uint16_t latte_alu_op2_move_opcode = 0x019U; // OP2_INST_MOV
inline constexpr std::uint16_t latte_alu_op2_and_integer_opcode = 0x030U;
inline constexpr std::uint16_t latte_alu_op2_or_integer_opcode = 0x031U;
inline constexpr std::uint16_t latte_alu_op2_xor_integer_opcode = 0x032U;
inline constexpr std::uint16_t latte_alu_op2_not_integer_opcode = 0x033U;
inline constexpr std::uint16_t latte_alu_source_zero_selector = 248U;
inline constexpr std::uint16_t latte_alu_source_one_float_selector = 249U;
inline constexpr std::uint16_t latte_alu_source_one_integer_selector = 250U;
inline constexpr std::uint16_t latte_alu_source_negative_one_integer_selector =
    251U;
inline constexpr std::uint16_t latte_alu_source_half_float_selector = 252U;
inline constexpr std::uint16_t latte_alu_source_previous_vector_selector = 254U;
inline constexpr std::size_t latte_shader_register_component_count = 4U;

using LatteShaderRegister =
    std::array<std::uint32_t, latte_shader_register_component_count>;

[[nodiscard]] inline std::uint32_t read_latte_alu_source(
    std::uint16_t source_selector,
    std::uint8_t source_channel,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector)
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

    constexpr std::uint16_t maximum_gpr_selector = 127U;
    if (source_selector > maximum_gpr_selector ||
        source_selector >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte ALU source selector is not a supported GPR, "
            "inline constant, or PV");
    }

    return guest_registers[source_selector][source_channel];
}

// Evaluate supported GPR, inline-constant, and prior-PV forms of MOV.
// The group executor applies destination writes and returns the vector result;
// previous-scalar state and other source modes remain unsupported.
// Shader registers hold raw 32-bit values so MOV preserves all source bits.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_move_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt)
{
    if (instruction.opcode != latte_alu_op2_move_opcode)
    {
        throw std::runtime_error(
            "Latte ALU MOV evaluation received a different OP2 opcode");
    }

    if (instruction.source0_relative || instruction.source0_negate ||
        instruction.source0_absolute ||
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

    return read_latte_alu_source(
        instruction.source0_selector,
        instruction.source0_channel,
        guest_registers,
        incoming_previous_vector);
}

// Evaluate supported GPR, inline-constant, and prior-PV integer logic operands.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_integer_logic_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt)
{
    if (instruction.opcode != latte_alu_op2_and_integer_opcode &&
        instruction.opcode != latte_alu_op2_or_integer_opcode &&
        instruction.opcode != latte_alu_op2_xor_integer_opcode &&
        instruction.opcode != latte_alu_op2_not_integer_opcode)
    {
        throw std::runtime_error(
            "Latte integer logic evaluation received an unsupported opcode");
    }

    if (instruction.source0_relative || instruction.source1_relative ||
        instruction.source0_negate || instruction.source1_negate ||
        instruction.source0_absolute || instruction.source1_absolute ||
        instruction.destination_relative || instruction.destination_clamp ||
        instruction.output_modifier != 0U || instruction.index_mode != 0U ||
        !instruction.write_mask)
    {
        throw std::runtime_error(
            "Latte integer logic evaluation received unsupported modifiers");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte integer logic destination is outside the register file");
    }

    if (instruction.destination_component >=
        latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte integer logic destination is outside a four-component GPR");
    }

    const std::uint32_t source0 = read_latte_alu_source(
        instruction.source0_selector,
        instruction.source0_channel,
        guest_registers,
        incoming_previous_vector);
    if (instruction.opcode == latte_alu_op2_not_integer_opcode)
    {
        return ~source0;
    }

    const std::uint32_t source1 = read_latte_alu_source(
        instruction.source1_selector,
        instruction.source1_channel,
        guest_registers,
        incoming_previous_vector);

    switch (instruction.opcode)
    {
    case latte_alu_op2_and_integer_opcode:
        return source0 & source1;
    case latte_alu_op2_or_integer_opcode:
        return source0 | source1;
    case latte_alu_op2_xor_integer_opcode:
        return source0 ^ source1;
    default:
        throw std::runtime_error(
            "Latte integer logic evaluation received an unsupported opcode");
    }
}

// Execute a supported four-slot X/Y/Z/W OP2 group. Source values are gathered
// before destination writes, preserving same-group parallel read semantics.
// The returned vector is the group's ALU.[X,Y,Z,W] result (PV).
[[nodiscard]] inline LatteShaderRegister
execute_latte_alu_op2_vector_group(
    const LatteAluOp2InstructionGroup& group,
    std::span<LatteShaderRegister> guest_registers,
    const std::optional<LatteShaderRegister>& incoming_previous_vector =
        std::nullopt)
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
                incoming_previous_vector);
            break;
        case latte_alu_op2_and_integer_opcode:
        case latte_alu_op2_or_integer_opcode:
        case latte_alu_op2_xor_integer_opcode:
        case latte_alu_op2_not_integer_opcode:
            group_vector[component] =
                evaluate_latte_alu_integer_logic_value(
                    instruction,
                    read_only_registers,
                    incoming_previous_vector);
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

} // namespace affogato::gpu
