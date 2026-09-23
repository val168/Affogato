#include "cpu/espresso/interpreter.hpp"

#include "cpu/espresso/decoder.hpp"

#include <bit>
#include <cstdint>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t xer_summary_overflow_mask = 0x80000000U;
constexpr std::uint8_t cr_less_than = 0x8U;
constexpr std::uint8_t cr_greater_than = 0x4U;
constexpr std::uint8_t cr_equal = 0x2U;

void set_compare_result(CpuState& state, std::uint8_t field, std::int32_t lhs, std::int32_t rhs)
{
    std::uint8_t result = 0;

    if (lhs < rhs)
    {
        result |= cr_less_than;
    }
    else if (lhs > rhs)
    {
        result |= cr_greater_than;
    }
    else
    {
        result |= cr_equal;
    }

    if ((state.xer & xer_summary_overflow_mask) != 0)
    {
        result |= 1U;
    }

    const unsigned shift = (7U - field) * 4U;
    const std::uint32_t mask = 0xFU << shift;
    state.cr = (state.cr & ~mask) | (static_cast<std::uint32_t>(result) << shift);
}

void set_record_result(CpuState& state, std::uint32_t value)
{
    set_compare_result(state, 0, std::bit_cast<std::int32_t>(value), 0);
}

[[nodiscard]] std::uint32_t rotate_mask(std::uint8_t begin, std::uint8_t end) noexcept
{
    std::uint32_t mask = 0;
    for (std::uint8_t bit = 0; bit < 32; ++bit)
    {
        const bool selected = begin <= end
            ? bit >= begin && bit <= end
            : bit >= begin || bit <= end;
        if (selected)
        {
            // PowerPC numbers mask bits from the most-significant end.
            mask |= 1U << (31U - bit);
        }
    }
    return mask;
}

[[nodiscard]] bool read_cr_bit(const CpuState& state, std::uint8_t bit) noexcept
{
    return ((state.cr >> (31U - bit)) & 1U) != 0;
}

[[nodiscard]] bool conditional_branch_taken(
    CpuState& state,
    std::uint8_t branch_options,
    std::uint8_t condition_bit) noexcept
{
    const bool ignore_condition = (branch_options & 0x10U) != 0;
    const bool condition_sense = (branch_options & 0x08U) != 0;
    const bool ignore_count = (branch_options & 0x04U) != 0;
    const bool count_sense = (branch_options & 0x02U) != 0;

    const bool condition_ok =
        ignore_condition || (read_cr_bit(state, condition_bit) == condition_sense);

    bool count_ok = true;
    if (!ignore_count)
    {
        --state.ctr;
        count_ok = ((state.ctr != 0) != count_sense);
    }

    return condition_ok && count_ok;
}

[[nodiscard]] std::uint32_t effective_address(
    const CpuState& state,
    std::uint8_t base_register,
    std::int32_t displacement) noexcept
{
    const std::uint32_t base =
        base_register == 0 ? 0U : state.gpr[base_register];
    return base + static_cast<std::uint32_t>(displacement);
}

}

StepResult EspressoCore::step()
{
    const std::uint32_t cia = state.cia;
    const std::uint32_t instruction_word = memory.read32_be(cia);
    const DecodedInstruction instruction = decode(instruction_word);

    if (instruction.opcode == Opcode::unsupported)
    {
        return StepResult::unsupported_instruction;
    }

    const std::uint32_t fallthrough = cia + 4U;
    std::uint32_t next_cia = fallthrough;

    switch (instruction.opcode)
    {
    case Opcode::addi:
    {
        const std::uint32_t base =
            instruction.base == 0 ? 0U : state.gpr[instruction.base];
        state.gpr[instruction.destination] =
            base + static_cast<std::uint32_t>(instruction.immediate);
        break;
    }

    case Opcode::addis:
    {
        const std::uint32_t base =
            instruction.base == 0 ? 0U : state.gpr[instruction.base];
        const std::uint32_t shifted_immediate =
            static_cast<std::uint32_t>(instruction.immediate) << 16U;
        state.gpr[instruction.destination] = base + shifted_immediate;
        break;
    }

    case Opcode::add:
    {
        const std::uint32_t result =
            state.gpr[instruction.base] + state.gpr[instruction.source];
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from:
    {
        const std::uint32_t result =
            state.gpr[instruction.source] - state.gpr[instruction.base];
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::ori:
        state.gpr[instruction.destination] =
            state.gpr[instruction.source] |
            static_cast<std::uint32_t>(instruction.immediate);
        break;

    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_xor:
    {
        const std::uint32_t lhs = state.gpr[instruction.source];
        const std::uint32_t rhs = state.gpr[instruction.base];
        const std::uint32_t result = instruction.opcode == Opcode::bitwise_or
            ? lhs | rhs
            : instruction.opcode == Opcode::bitwise_and ? lhs & rhs : lhs ^ rhs;
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::and_immediate_record:
    {
        const std::uint32_t result = state.gpr[instruction.source] &
            static_cast<std::uint32_t>(instruction.immediate);
        state.gpr[instruction.destination] = result;
        set_record_result(state, result);
        break;
    }

    case Opcode::rotate_left_word_and_mask:
    {
        const std::uint32_t rotated = std::rotl(
            state.gpr[instruction.source], static_cast<int>(instruction.shift));
        const std::uint32_t result =
            rotated & rotate_mask(instruction.mask_begin, instruction.mask_end);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::branch:
        if (instruction.link)
        {
            state.lr = fallthrough;
        }

        if (instruction.absolute)
        {
            next_cia = static_cast<std::uint32_t>(instruction.immediate);
        }
        else
        {
            next_cia = cia + static_cast<std::uint32_t>(instruction.immediate);
        }
        break;

    case Opcode::compare_signed_immediate:
        set_compare_result(
            state,
            instruction.cr_field,
            std::bit_cast<std::int32_t>(state.gpr[instruction.base]),
            instruction.immediate);
        break;

    case Opcode::compare_signed_register:
        set_compare_result(
            state,
            instruction.cr_field,
            std::bit_cast<std::int32_t>(state.gpr[instruction.base]),
            std::bit_cast<std::int32_t>(state.gpr[instruction.source]));
        break;

    case Opcode::conditional_branch:
        if (instruction.link)
        {
            state.lr = fallthrough;
        }

        if (conditional_branch_taken(
                state,
                instruction.branch_options,
                instruction.condition_bit))
        {
            next_cia = instruction.absolute
                ? static_cast<std::uint32_t>(instruction.immediate)
                : cia + static_cast<std::uint32_t>(instruction.immediate);
        }
        break;

    case Opcode::conditional_branch_to_link_register:
    {
        // Capture the target first: blrl writes the fall-through address to LR,
        // but must still branch to the LR value from before that write.
        const std::uint32_t target = state.lr & ~0x3U;
        if (instruction.link)
        {
            state.lr = fallthrough;
        }

        if (conditional_branch_taken(
                state,
                instruction.branch_options,
                instruction.condition_bit))
        {
            next_cia = target;
        }
        break;
    }

    case Opcode::move_from_link_register:
        state.gpr[instruction.destination] = state.lr;
        break;

    case Opcode::move_to_link_register:
        state.lr = state.gpr[instruction.destination];
        break;

    case Opcode::load_word_zero:
        state.gpr[instruction.destination] =
            memory.read32_be(effective_address(state, instruction.base, instruction.immediate));
        break;

    case Opcode::store_word:
        memory.write32_be(
            effective_address(state, instruction.base, instruction.immediate),
            state.gpr[instruction.destination]);
        break;

    case Opcode::store_word_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint32_t value = state.gpr[instruction.destination];
        memory.write32_be(address, value);
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::load_byte_zero:
        state.gpr[instruction.destination] =
            memory.read8(effective_address(state, instruction.base, instruction.immediate));
        break;

    case Opcode::store_byte:
        memory.write8(
            effective_address(state, instruction.base, instruction.immediate),
            static_cast<std::uint8_t>(state.gpr[instruction.destination]));
        break;

    case Opcode::load_halfword_zero:
        state.gpr[instruction.destination] =
            memory.read16_be(effective_address(state, instruction.base, instruction.immediate));
        break;

    case Opcode::store_halfword:
        memory.write16_be(
            effective_address(state, instruction.base, instruction.immediate),
            static_cast<std::uint16_t>(state.gpr[instruction.destination]));
        break;

    case Opcode::unsupported:
        return StepResult::unsupported_instruction;
    }

    state.cia = next_cia;
    return StepResult::executed;
}

RunResult EspressoCore::run(std::size_t max_steps)
{
    RunResult result{};

    while (result.steps < max_steps)
    {
        if (step() == StepResult::unsupported_instruction)
        {
            result.reason = StopReason::unsupported_instruction;
            return result;
        }

        ++result.steps;
    }

    result.reason = StopReason::instruction_limit;
    return result;
}

}
