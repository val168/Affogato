#pragma once

#include "latte_shader_decoder.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>

namespace affogato::gpu
{

inline constexpr std::uint16_t latte_alu_op2_move_opcode = 0x019U; // OP2_INST_MOV
inline constexpr std::size_t latte_shader_register_component_count = 4U;

using LatteShaderRegister =
    std::array<std::uint32_t, latte_shader_register_component_count>;

// Evaluate the fixture-verified absolute GPR component-copy form of MOV.
// This returns the source value only; destination, PV, and PS updates belong
// to the instruction-group executor and are not implemented here.
// Shader registers hold raw 32-bit values so MOV preserves all source bits.
[[nodiscard]] inline std::uint32_t evaluate_latte_alu_move_value(
    const LatteAluOp2Instruction& instruction,
    std::span<const LatteShaderRegister> guest_registers)
{
    if (instruction.opcode != latte_alu_op2_move_opcode)
    {
        throw std::runtime_error(
            "Latte ALU MOV evaluation received a different OP2 opcode");
    }

    if (instruction.source0_relative || instruction.source0_negate ||
        instruction.destination_relative || instruction.destination_clamp ||
        instruction.output_modifier != 0U || instruction.index_mode != 0U ||
        !instruction.write_mask)
    {
        throw std::runtime_error(
            "Latte ALU MOV evaluation received unsupported modifiers");
    }

    // Source selectors 128 and above encode constants or previous results.
    constexpr std::uint16_t maximum_gpr_selector = 127U;
    if (instruction.source0_selector > maximum_gpr_selector ||
        instruction.source0_selector >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte ALU MOV source selector is not a supported GPR");
    }

    if (instruction.destination_gpr >= guest_registers.size())
    {
        throw std::runtime_error(
            "Latte ALU MOV destination register is outside the register file");
    }

    if (instruction.source0_channel >= latte_shader_register_component_count ||
        instruction.destination_component >=
            latte_shader_register_component_count)
    {
        throw std::runtime_error(
            "Latte ALU MOV component index is outside a four-component GPR");
    }

    return guest_registers[instruction.source0_selector]
                         [instruction.source0_channel];
}

} // namespace affogato::gpu
