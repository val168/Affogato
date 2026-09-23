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

    case 21: // rlwinm
        instruction.opcode = Opcode::rotate_left_word_and_mask;
        instruction.source = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.destination = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.shift = static_cast<std::uint8_t>(field(raw, 11, 0x1FU));
        instruction.mask_begin = static_cast<std::uint8_t>(field(raw, 6, 0x1FU));
        instruction.mask_end = static_cast<std::uint8_t>(field(raw, 1, 0x1FU));
        instruction.record = (raw & 1U) != 0;
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

    case 28: // andi.
        instruction.opcode = Opcode::and_immediate_record;
        instruction.source = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.destination = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = static_cast<std::int32_t>(field(raw, 0, 0xFFFFU));
        instruction.record = true;
        break;

    case 19: // bclr / blrl
        if (field(raw, 1, 0x3FFU) == 16)
        {
            instruction.opcode = Opcode::conditional_branch_to_link_register;
            instruction.branch_options = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
            instruction.condition_bit = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.link = (raw & 0x1U) != 0;
        }
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

    case 31: // X-form arithmetic, logical, and compare instructions
    {
        const std::uint32_t extended_opcode = field(raw, 1, 0x3FFU);
        if (extended_opcode == 339 || extended_opcode == 467)
        {
            // The SPR number is split into two five-bit fields in reverse
            // order: the encoded rA field contains SPR[0:4].
            const std::uint32_t spr =
                (field(raw, 11, 0x1FU) << 5U) | field(raw, 16, 0x1FU);
            if (spr == 8)
            {
                instruction.destination =
                    static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
                instruction.opcode = extended_opcode == 339
                    ? Opcode::move_from_link_register
                    : Opcode::move_to_link_register;
            }
        }
        else if (extended_opcode == 0 && (raw & (1U << 22U)) == 0)
        {
            instruction.opcode = Opcode::compare_signed_register;
            instruction.cr_field = static_cast<std::uint8_t>(field(raw, 23, 0x7U));
            instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.source = static_cast<std::uint8_t>(field(raw, 11, 0x1FU));
        }
        else if (extended_opcode == 266 || extended_opcode == 40)
        {
            instruction.opcode = extended_opcode == 266 ? Opcode::add : Opcode::subtract_from;
            instruction.destination =
                static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
            instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.source = static_cast<std::uint8_t>(field(raw, 11, 0x1FU));
            instruction.record = (raw & 1U) != 0;
        }
        else if (extended_opcode == 444 || extended_opcode == 28 || extended_opcode == 316)
        {
            instruction.opcode = extended_opcode == 444 ? Opcode::bitwise_or
                : extended_opcode == 28 ? Opcode::bitwise_and
                                        : Opcode::bitwise_xor;
            instruction.source = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
            instruction.destination =
                static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
            instruction.base = static_cast<std::uint8_t>(field(raw, 11, 0x1FU));
            instruction.record = (raw & 1U) != 0;
        }
        break;
    }

    case 32: // lwz
        instruction.opcode = Opcode::load_word_zero;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 34: // lbz
        instruction.opcode = Opcode::load_byte_zero;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 36: // stw
        instruction.opcode = Opcode::store_word;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 37: // stwu (rA must not be zero)
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        if (instruction.base != 0)
        {
            instruction.opcode = Opcode::store_word_update;
            instruction.destination =
                static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
            instruction.immediate = sign_extend(raw, 16);
        }
        break;

    case 38: // stb
        instruction.opcode = Opcode::store_byte;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 40: // lhz
        instruction.opcode = Opcode::load_halfword_zero;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    case 44: // sth
        instruction.opcode = Opcode::store_halfword;
        instruction.destination = static_cast<std::uint8_t>(field(raw, 21, 0x1FU));
        instruction.base = static_cast<std::uint8_t>(field(raw, 16, 0x1FU));
        instruction.immediate = sign_extend(raw, 16);
        break;

    default:
        break;
    }

    return instruction;
}

}
