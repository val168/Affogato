#include "cpu/espresso/interpreter.hpp"

#include "cpu/espresso/decoder.hpp"

#include <bit>
#include <cstdint>
#include <exception>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t xer_summary_overflow_mask = 0x80000000U;
constexpr std::uint32_t xer_carry_mask = 0x20000000U;
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

void set_compare_result_unsigned(
    CpuState& state,
    std::uint8_t field,
    std::uint32_t lhs,
    std::uint32_t rhs)
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
    switch (hle.dispatch(*this, cia))
    {
    case HleDispatchResult::executed:
        return StepResult::executed;
    case HleDispatchResult::unimplemented:
        return StepResult::unimplemented_hle_call;
    case HleDispatchResult::not_hle:
        break;
    }

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

    case Opcode::add_immediate_carry:
    {
        const std::uint32_t lhs = instruction.base == 0 ? 0U : state.gpr[instruction.base];
        const std::uint32_t rhs = static_cast<std::uint32_t>(instruction.immediate);
        const std::uint64_t sum = static_cast<std::uint64_t>(lhs) + rhs;
        const std::uint32_t value = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = value;
        if ((sum >> 32U) != 0)
        {
            state.xer |= xer_carry_mask;
        }
        else
        {
            state.xer &= ~xer_carry_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, value);
        }
        break;
    }

    case Opcode::subtract_from_immediate_carry:
    {
        const std::uint32_t immediate = static_cast<std::uint32_t>(instruction.immediate);
        const std::uint32_t source = state.gpr[instruction.base];
        state.gpr[instruction.destination] = immediate - source;
        if (immediate >= source)
        {
            state.xer |= xer_carry_mask;
        }
        else
        {
            state.xer &= ~xer_carry_mask;
        }
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
    case Opcode::bitwise_and_complement:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_equivalence:
    {
        const std::uint32_t lhs = state.gpr[instruction.source];
        const std::uint32_t rhs = state.gpr[instruction.base];
        const std::uint32_t result = instruction.opcode == Opcode::bitwise_or
            ? lhs | rhs
            : instruction.opcode == Opcode::bitwise_and ? lhs & rhs
            : instruction.opcode == Opcode::bitwise_and_complement ? lhs & ~rhs
            : instruction.opcode == Opcode::bitwise_equivalence ? ~(lhs ^ rhs)
                                                                  : lhs ^ rhs;
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

    case Opcode::arithmetic_shift_right_immediate:
    {
        const std::int32_t source = std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        const std::uint32_t mask = instruction.shift == 0
            ? 0U
            : (std::uint32_t{1} << instruction.shift) - 1U;
        const bool carry = source < 0 && (state.gpr[instruction.source] & mask) != 0;
        const std::uint32_t result = static_cast<std::uint32_t>(source >> instruction.shift);
        state.gpr[instruction.destination] = result;
        if (carry)
        {
            state.xer |= xer_carry_mask;
        }
        else
        {
            state.xer &= ~xer_carry_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::shift_left_word:
    {
        const std::uint32_t shift = state.gpr[instruction.base] & 0x3FU;
        const std::uint32_t result = shift >= 32U
            ? 0U
            : state.gpr[instruction.source] << shift;
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

    case Opcode::compare_unsigned_immediate:
        set_compare_result_unsigned(
            state,
            instruction.cr_field,
            state.gpr[instruction.base],
            static_cast<std::uint32_t>(instruction.immediate));
        break;

    case Opcode::compare_unsigned_register:
        set_compare_result_unsigned(
            state,
            instruction.cr_field,
            state.gpr[instruction.base],
            state.gpr[instruction.source]);
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

    case Opcode::conditional_branch_to_count_register:
    {
        const std::uint32_t target = state.ctr & ~0x3U;
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

    case Opcode::move_from_count_register:
        state.gpr[instruction.destination] = state.ctr;
        break;

    case Opcode::move_to_count_register:
        state.ctr = state.gpr[instruction.destination];
        break;

    case Opcode::load_word_zero:
        state.gpr[instruction.destination] =
            memory.read32_be(effective_address(state, instruction.base, instruction.immediate));
        break;

    case Opcode::load_word_indexed:
    {
        const std::uint32_t base = instruction.base == 0 ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        state.gpr[instruction.destination] = memory.read32_be(address);
        break;
    }

    case Opcode::load_word_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint32_t value = memory.read32_be(address);
        state.gpr[instruction.destination] = value;
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::store_word:
        memory.write32_be(
            effective_address(state, instruction.base, instruction.immediate),
            state.gpr[instruction.destination]);
        break;

    case Opcode::store_word_indexed:
    {
        const std::uint32_t base = instruction.base == 0 ? 0 : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        memory.write32_be(address, state.gpr[instruction.destination]);
        break;
    }

    case Opcode::store_byte_indexed:
    {
        const std::uint32_t base = instruction.base == 0 ? 0 : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        memory.write8(address, static_cast<std::uint8_t>(state.gpr[instruction.destination]));
        break;
    }

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

    case Opcode::load_byte_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint8_t value = memory.read8(address);
        state.gpr[instruction.destination] = value;
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::store_byte:
        memory.write8(
            effective_address(state, instruction.base, instruction.immediate),
            static_cast<std::uint8_t>(state.gpr[instruction.destination]));
        break;

    case Opcode::store_byte_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        memory.write8(address, static_cast<std::uint8_t>(state.gpr[instruction.destination]));
        state.gpr[instruction.base] = address;
        break;
    }

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
    result.cia = state.cia;

    while (result.steps < max_steps)
    {
        StepResult step_result{};
        try
        {
            step_result = step();
        }
        catch (const std::exception& error)
        {
            result.reason = StopReason::memory_fault;
            result.cia = state.cia;
            result.detail = error.what();
            return result;
        }
        if (step_result == StepResult::unsupported_instruction)
        {
            result.reason = StopReason::unsupported_instruction;
            result.cia = state.cia;
            result.instruction_word = memory.read32_be(state.cia);
            return result;
        }
        if (step_result == StepResult::unimplemented_hle_call)
        {
            result.reason = StopReason::unimplemented_hle_call;
            result.cia = state.cia;
            result.hle_call = hle.last_unimplemented_call();
            return result;
        }

        ++result.steps;
    }

    result.reason = StopReason::instruction_limit;
    result.cia = state.cia;
    return result;
}

}
