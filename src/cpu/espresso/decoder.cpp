#include "cpu/espresso/decoder.hpp"

namespace affogato::cpu::espresso
{
namespace
{

[[nodiscard]] constexpr std::uint32_t field(
    std::uint32_t value,
    unsigned shift,
    std::uint32_t mask) noexcept
{
    return (value >> shift) & mask;
}

[[nodiscard]] constexpr std::int32_t sign_extend(
    std::uint32_t value,
    unsigned bits) noexcept
{
    const std::uint32_t mask = (1U << bits) - 1U;
    const std::uint32_t sign_bit = 1U << (bits - 1U);
    value &= mask;

    return static_cast<std::int32_t>((value ^ sign_bit) - sign_bit);
}

}

DecodedInstruction decode(std::uint32_t raw) noexcept
{
    DecodedInstruction instruction{};
    instruction.raw = raw;

    switch (field(raw, 26, 0x3FU))
    {
    case 14: // addi
        instruction.opcode = Opcode::addi;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 15: // addis
        instruction.opcode = Opcode::addis;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 24: // ori
        instruction.opcode = Opcode::ori;
        instruction.source = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.destination = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = static_cast<std::int32_t>(field(raw, 0, 0xFFFFU));
        break;

    case 18: // b / bl
        instruction.opcode = Opcode::branch;
        instruction.immediate = sign_extend(raw & 0x03FFFFFCU, 26);
        instruction.absolute = (raw & 0x2U) != 0;
        instruction.link = (raw & 0x1U) != 0;
        break;

    case 11: // cmpwi (L=0)
        if ((raw & (1U << 22U)) == 0)
        {
            instruction.opcode = Opcode::compare_signed_immediate;
            instruction.cr_field = static_cast<std::uint8_t>(field(raw, 23, 0x7U));
            instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.immediate = sign_extend(raw, 16);
        }
        break;

    case 16: // bc / bcl / bca / bcla
        instruction.opcode = Opcode::conditional_branch;
        instruction.branch_options = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.condition_bit = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw & 0xFFFCU, 16);
        instruction.absolute = (raw & 0x2U) != 0;
        instruction.link = (raw & 0x1U) != 0;
        break;

    case 31: // cmpw (cmp with L=0)
        if (field(raw, 1, 0x3FFU) == 0 && (raw & (1U << 22U)) == 0)
        {
            instruction.opcode = Opcode::compare_signed_register;
            instruction.cr_field = static_cast<std::uint8_t>(field(raw, 23, 0x7U));
            instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.source = static_cast<std::uint8_t>(field(raw, 11, 0x1FU));
        }
        break;

    default:
        break;
    }

    return instruction;
}

}
