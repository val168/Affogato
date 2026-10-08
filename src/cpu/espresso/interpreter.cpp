#include "cpu/espresso/interpreter.hpp"

#include "cpu/espresso/decoder.hpp"

#include <bit>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <sstream>
#include <utility>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t xer_summary_overflow_mask = 0x80000000U;
constexpr std::uint32_t xer_carry_mask = 0x20000000U;
constexpr std::uint8_t cr_less_than = 0x8U;
constexpr std::uint8_t cr_greater_than = 0x4U;
constexpr std::uint8_t cr_equal = 0x2U;
constexpr std::uint8_t fp_unordered = 0x1U;
constexpr std::uint64_t binary64_exponent_mask = 0x7FF0000000000000ULL;
constexpr std::uint64_t binary64_fraction_mask = 0x000FFFFFFFFFFFFFULL;
constexpr std::uint64_t binary64_quiet_nan_bit = 0x0008000000000000ULL;

[[nodiscard]] bool is_binary64_nan(std::uint64_t raw) noexcept
{
    return (raw & binary64_exponent_mask) == binary64_exponent_mask &&
           (raw & binary64_fraction_mask) != 0U;
}

[[nodiscard]] bool is_binary64_signaling_nan(std::uint64_t raw) noexcept
{
    return is_binary64_nan(raw) && (raw & binary64_quiet_nan_bit) == 0U;
}

constexpr std::uint64_t binary64_sign_mask = 0x8000000000000000ULL;
constexpr std::uint64_t binary64_infinity_bits = 0x7FF0000000000000ULL;

[[nodiscard]] bool is_binary64_zero(std::uint64_t raw) noexcept
{
    return (raw & ~binary64_sign_mask) == 0U;
}

[[nodiscard]] bool is_binary64_infinity(std::uint64_t raw) noexcept
{
    return (raw & ~binary64_sign_mask) == binary64_infinity_bits;
}

// Espresso rounds the frC operand of single-precision multiply-family
// instructions to a 25-bit significand before performing the operation.
// Subnormals use the same rounding point after normalizing their significand.
[[nodiscard]] std::uint64_t round_multiply_operand_25bit(std::uint64_t raw) noexcept
{
    const std::uint64_t exponent = raw & binary64_exponent_mask;
    const std::uint64_t fraction = raw & binary64_fraction_mask;
    if (exponent == 0U && fraction != 0U)
    {
        const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
        const std::int64_t signed_keep_mask =
            std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
        const std::uint64_t keep_mask = static_cast<std::uint64_t>(signed_keep_mask >> shift);
        const std::uint64_t round_bit = 0x0000000008000000ULL >> shift;
        return (raw & keep_mask) + (raw & round_bit);
    }

    return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x0000000008000000ULL);
}

constexpr std::uint32_t implemented_invalid_masks =
    fpscr::vxsnan_mask | fpscr::vxzdz_mask | fpscr::vximz_mask |
    fpscr::vxidi_mask | fpscr::vxisi_mask;

void update_fpscr_summaries(CpuState& state) noexcept
{
    state.fpscr &= ~(fpscr::vx_mask | fpscr::fex_mask);
    if ((state.fpscr & implemented_invalid_masks) != 0U)
    {
        state.fpscr |= fpscr::vx_mask;
    }

    const bool enabled_exception =
        ((state.fpscr & fpscr::vx_mask) != 0U &&
         (state.fpscr & fpscr::ve_mask) != 0U) ||
        ((state.fpscr & fpscr::zx_mask) != 0U &&
         (state.fpscr & fpscr::ze_mask) != 0U) ||
        ((state.fpscr & fpscr::xx_mask) != 0U &&
         (state.fpscr & fpscr::xe_mask) != 0U);
    if (enabled_exception)
    {
        state.fpscr |= fpscr::fex_mask;
    }
}

void raise_fpscr_exception(CpuState& state, std::uint32_t subexception) noexcept
{
    state.fpscr |= subexception | fpscr::fx_mask;
    update_fpscr_summaries(state);
}

[[nodiscard]] std::uint8_t single_result_fprf(std::uint32_t bits) noexcept
{
    constexpr std::uint32_t exponent_mask = 0x7F800000U;
    constexpr std::uint32_t fraction_mask = 0x007FFFFFU;
    constexpr std::uint32_t sign_mask = 0x80000000U;
    const bool negative = (bits & sign_mask) != 0U;
    const std::uint32_t exponent = bits & exponent_mask;
    const std::uint32_t fraction = bits & fraction_mask;

    if (exponent == exponent_mask)
    {
        return fraction != 0U ? 0x11U : negative ? 0x09U : 0x05U;
    }
    if (exponent == 0U)
    {
        if (fraction != 0U)
        {
            return negative ? 0x18U : 0x14U;
        }
        return negative ? 0x12U : 0x02U;
    }
    return negative ? 0x08U : 0x04U;
}

[[nodiscard]] std::uint32_t quiet_nan_single_bits(std::uint64_t raw) noexcept
{
    if (!is_binary64_nan(raw))
    {
        raw = 0x7FF8000000000000ULL;
    }
    constexpr std::uint32_t single_sign_mask = 0x80000000U;
    constexpr std::uint32_t single_exponent_mask = 0x7F800000U;
    constexpr std::uint32_t single_quiet_nan_bit = 0x00400000U;
    const std::uint32_t sign = (raw & binary64_sign_mask) != 0U
        ? single_sign_mask
        : 0U;
    const std::uint32_t payload = static_cast<std::uint32_t>(
        (raw & binary64_fraction_mask) >> 29U);
    return sign | single_exponent_mask | payload | single_quiet_nan_bit;
}

[[nodiscard]] std::uint64_t quiet_nan_binary64_bits(std::uint64_t raw) noexcept
{
    return is_binary64_nan(raw) ? raw | binary64_quiet_nan_bit
                                : 0x7FF8000000000000ULL;
}

[[nodiscard]] std::uint8_t binary64_result_fprf(std::uint64_t raw) noexcept
{
    const std::uint64_t exponent = raw & binary64_exponent_mask;
    const std::uint64_t fraction = raw & binary64_fraction_mask;
    const bool negative = (raw & binary64_sign_mask) != 0U;

    if (exponent == binary64_exponent_mask)
    {
        return fraction != 0U ? 0x11U : negative ? 0x09U : 0x05U;
    }
    if (exponent == 0U)
    {
        if (fraction != 0U)
        {
            return negative ? 0x18U : 0x14U;
        }
        return negative ? 0x12U : 0x02U;
    }
    return negative ? 0x08U : 0x04U;
}

void set_double_arithmetic_result_status(
    CpuState& state,
    std::uint64_t raw_result,
    bool inexact,
    bool rounded_up) noexcept
{
    state.fpscr &= ~(fpscr::fprf_mask | fpscr::fi_mask | fpscr::fr_mask);
    state.fpscr |= static_cast<std::uint32_t>(binary64_result_fprf(raw_result)) << 12U;
    if (inexact)
    {
        state.fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
        if (rounded_up)
        {
            state.fpscr |= fpscr::fr_mask;
        }
    }
    update_fpscr_summaries(state);
}

void set_double_arithmetic_fprf(CpuState& state, std::uint64_t raw_result) noexcept
{
    state.fpscr = (state.fpscr & ~fpscr::fprf_mask) |
        (static_cast<std::uint32_t>(binary64_result_fprf(raw_result)) << 12U);
}

[[nodiscard]] bool binary64_normal_or_zero(std::uint64_t raw) noexcept
{
    const std::uint64_t exponent = raw & binary64_exponent_mask;
    const std::uint64_t fraction = raw & binary64_fraction_mask;
    return exponent != binary64_exponent_mask && (exponent != 0U || fraction == 0U);
}

[[nodiscard]] double binary64_subtraction_error(
    double a,
    double b,
    double rounded_result) noexcept
{
    // Error-free TwoDiff decomposition for finite normal/zero operands/results.
    const double virtual_b = a - rounded_result;
    const double virtual_a = rounded_result + virtual_b;
    const double roundoff_b = virtual_b - b;
    const double roundoff_a = a - virtual_a;
    return roundoff_a + roundoff_b;
}

void set_single_arithmetic_result_status(
    CpuState& state,
    std::uint32_t single_bits,
    bool inexact,
    bool rounded_up) noexcept
{
    state.fpscr &= ~(fpscr::fprf_mask | fpscr::fi_mask | fpscr::fr_mask);
    state.fpscr |= static_cast<std::uint32_t>(single_result_fprf(single_bits)) << 12U;
    if (inexact)
    {
        state.fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
        if (rounded_up)
        {
            state.fpscr |= fpscr::fr_mask;
        }
    }
    update_fpscr_summaries(state);
}

void set_floating_compare_result(
    CpuState& state,
    std::uint8_t field,
    std::uint8_t result) noexcept
{
    const unsigned shift = (7U - field) * 4U;
    const std::uint32_t mask = 0xFU << shift;
    state.cr = (state.cr & ~mask) | (static_cast<std::uint32_t>(result) << shift);
    state.fpscr = (state.fpscr & ~fpscr::fpcc_mask) |
        (static_cast<std::uint32_t>(result) << 12U);
}

[[nodiscard]] const char* floating_compare_result_name(std::uint8_t result) noexcept
{
    switch (result)
    {
    case cr_less_than: return "LT";
    case cr_greater_than: return "GT";
    case cr_equal: return "EQ";
    case fp_unordered: return "UN";
    default: return "?";
    }
}

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

[[nodiscard]] std::string describe_memory_fault(
    const GuestMemoryFault& fault,
    std::uint32_t cia,
    bool has_instruction_word,
    std::uint32_t instruction_word,
    const CpuState& state)
{
    std::ostringstream detail;
    detail << (fault.access() == GuestMemoryAccess::read ? "read" : "write") << ' '
           << std::dec << fault.width() << " byte(s) at guest address 0x"
           << std::hex << std::uppercase << std::setw(8) << std::setfill('0')
           << fault.address() << "; CIA 0x" << std::setw(8) << cia;
    if (has_instruction_word)
    {
        detail << "; instruction 0x" << std::setw(8) << instruction_word;
        const DecodedInstruction instruction = decode(instruction_word);
        const bool indexed = instruction.opcode == Opcode::load_word_indexed ||
            instruction.opcode == Opcode::store_word_indexed ||
            instruction.opcode == Opcode::store_byte_indexed;
        const bool memory_instruction = indexed ||
            instruction.opcode == Opcode::load_word_zero ||
            instruction.opcode == Opcode::load_word_update ||
            instruction.opcode == Opcode::store_word ||
            instruction.opcode == Opcode::store_word_update ||
            instruction.opcode == Opcode::load_byte_zero ||
            instruction.opcode == Opcode::load_byte_update ||
            instruction.opcode == Opcode::store_byte ||
            instruction.opcode == Opcode::store_byte_update ||
            instruction.opcode == Opcode::load_halfword_zero ||
            instruction.opcode == Opcode::load_halfword_algebraic ||
            instruction.opcode == Opcode::store_halfword ||
            instruction.opcode == Opcode::store_halfword_update ||
            instruction.opcode == Opcode::load_multiple_word ||
            instruction.opcode == Opcode::load_string_word_immediate ||
            instruction.opcode == Opcode::store_string_word_immediate ||
            instruction.opcode == Opcode::store_multiple_word ||
            instruction.opcode == Opcode::load_single ||
            instruction.opcode == Opcode::load_single_update ||
            instruction.opcode == Opcode::store_single ||
            instruction.opcode == Opcode::store_single_update ||
            instruction.opcode == Opcode::store_double ||
            instruction.opcode == Opcode::load_double;
        if (memory_instruction)
        {
            detail << "; rA=" << std::dec << static_cast<unsigned>(instruction.base)
                   << " (0x" << std::hex << std::setw(8)
                   << (instruction.base == 0 ? 0U : state.gpr[instruction.base]) << ')';
            if (indexed)
            {
                detail << ", rB=" << std::dec << static_cast<unsigned>(instruction.source)
                       << " (0x" << std::hex << std::setw(8)
                       << state.gpr[instruction.source] << ')';
            }
        }
    }
    return detail.str();
}

[[nodiscard]] const char* opcode_name(Opcode opcode) noexcept
{
    switch (opcode)
    {
    case Opcode::multiply_low_immediate:
        return "mulli";
    case Opcode::multiply_low_word: return "mullw";
    case Opcode::multiply_high_word_unsigned: return "mulhwu";
    case Opcode::divide_word_unsigned: return "divwu";
    case Opcode::unsupported: return "unsupported";
    case Opcode::addi: return "addi";
    case Opcode::addis: return "addis";
    case Opcode::add_immediate_carry: return "addic";
    case Opcode::subtract_from_immediate_carry: return "subfic";
    case Opcode::add: return "add";
    case Opcode::subtract_from: return "subf";
    case Opcode::subtract_from_carrying: return "subfc";
    case Opcode::negate: return "neg";
    case Opcode::extend_sign_byte: return "extsb";
    case Opcode::ori: return "ori";
    case Opcode::or_immediate_shifted: return "oris";
    case Opcode::xor_immediate: return "xori";
    case Opcode::xor_immediate_shifted: return "xoris";
    case Opcode::count_leading_zeros: return "cntlzw";
    case Opcode::bitwise_or: return "or";
    case Opcode::bitwise_and: return "and";
    case Opcode::bitwise_and_complement: return "andc";
    case Opcode::bitwise_xor: return "xor";
    case Opcode::bitwise_equivalence: return "eqv";
    case Opcode::and_immediate_record: return "andi.";
    case Opcode::rotate_left_word_and_mask: return "rlwinm";
    case Opcode::rotate_left_word_and_mask_insert: return "rlwimi";
    case Opcode::arithmetic_shift_right_immediate: return "srawi";
    case Opcode::shift_left_word: return "slw";
    case Opcode::shift_right_word: return "srw";
    case Opcode::branch: return "b";
    case Opcode::instruction_sync: return "isync";
    case Opcode::floating_compare_unordered: return "fcmpu";
    case Opcode::floating_divide_single: return "fdivs";
    case Opcode::floating_add_single: return "fadds";
    case Opcode::floating_multiply_single: return "fmuls";
    case Opcode::floating_multiply_add_single: return "fmadds";
    case Opcode::floating_move_register: return "fmr";
    case Opcode::floating_negate: return "fneg";
    case Opcode::floating_subtract_double: return "fsub";
    case Opcode::floating_round_to_single: return "frsp";
    case Opcode::compare_signed_immediate: return "cmpwi";
    case Opcode::compare_signed_register: return "cmpw";
    case Opcode::compare_unsigned_immediate: return "cmplwi";
    case Opcode::compare_unsigned_register: return "cmplw";
    case Opcode::conditional_branch: return "bc";
    case Opcode::load_word_zero: return "lwz";
    case Opcode::load_multiple_word: return "lmw";
    case Opcode::load_string_word_immediate: return "lswi";
    case Opcode::store_string_word_immediate: return "stswi";
    case Opcode::load_word_update: return "lwzu";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::store_word: return "stw";
    case Opcode::store_multiple_word: return "stmw";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_byte_update: return "lbzu";
    case Opcode::store_byte: return "stb";
    case Opcode::store_byte_update: return "stbu";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::store_halfword: return "sth";
    case Opcode::store_halfword_update: return "sthu";
    case Opcode::store_word_update: return "stwu";
    case Opcode::move_from_link_register: return "mflr";
    case Opcode::move_to_link_register: return "mtlr";
    case Opcode::conditional_branch_to_link_register: return "bclr";
    case Opcode::move_from_count_register: return "mfctr";
    case Opcode::move_to_count_register: return "mtctr";
    case Opcode::conditional_branch_to_count_register: return "bcctr";
    case Opcode::load_single: return "lfs";
    case Opcode::load_single_update: return "lfsu";
    case Opcode::load_double: return "lfd";
    case Opcode::store_single: return "stfs";
    case Opcode::store_single_update: return "stfsu";
    case Opcode::store_double: return "stfd";
    case Opcode::paired_single_merge10: return "ps_merge10";
    }
    return "unknown";
}

void add_history_source(
    InstructionHistoryEntry& entry,
    const CpuState& state,
    std::uint8_t reg)
{
    for (std::uint8_t i = 0; i < entry.source_count; ++i)
    {
        if (entry.source_registers[i] == reg)
        {
            return;
        }
    }
    if (entry.source_count < entry.source_registers.size())
    {
        const std::size_t index = entry.source_count++;
        entry.source_registers[index] = reg;
        entry.source_values[index] = state.gpr[reg];
    }
}

[[nodiscard]] InstructionHistoryEntry make_history_entry(
    const CpuState& state,
    std::uint32_t cia,
    std::uint32_t word,
    const DecodedInstruction& instruction)
{
    InstructionHistoryEntry entry;
    entry.cia = cia;
    entry.instruction_word = word;
    entry.has_instruction_word = true;
    entry.opcode_name = opcode_name(instruction.opcode);
    if ((instruction.opcode == Opcode::multiply_low_word ||
         instruction.opcode == Opcode::multiply_high_word_unsigned ||
         instruction.opcode == Opcode::divide_word_unsigned ||
         instruction.opcode == Opcode::count_leading_zeros ||
         instruction.opcode == Opcode::negate ||
         instruction.opcode == Opcode::extend_sign_byte ||
         instruction.opcode == Opcode::rotate_left_word_and_mask_insert ||
         instruction.opcode == Opcode::rotate_left_word_and_mask ||
         instruction.opcode == Opcode::shift_right_word) && instruction.record)
    {
        entry.opcode_name += '.';
    }
    if (instruction.opcode == Opcode::multiply_low_immediate)
    {
        entry.has_immediate = true;
        entry.immediate = instruction.immediate;
    }
    else if (instruction.opcode == Opcode::or_immediate_shifted ||
             instruction.opcode == Opcode::xor_immediate ||
             instruction.opcode == Opcode::xor_immediate_shifted)
    {
        entry.has_immediate = true;
        entry.immediate_hex = true;
        entry.immediate = instruction.immediate;
    }
    if (instruction.opcode == Opcode::rotate_left_word_and_mask_insert)
    {
        entry.has_old_destination = true;
        entry.old_destination_register = instruction.destination;
        entry.old_destination_value = state.gpr[instruction.destination];
        entry.has_rotate_fields = true;
        entry.rotate_shift = instruction.shift;
        entry.rotate_mask_begin = instruction.mask_begin;
        entry.rotate_mask_end = instruction.mask_end;
    }

    const Opcode opcode = instruction.opcode;
    if (opcode == Opcode::paired_single_merge10)
    {
        entry.has_paired_fp_state = true;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = instruction.fp_source_a;
        entry.paired_fp_source_b_register = instruction.fp_source_b;
        entry.paired_fp_source_a_ps0 = state.fpr[instruction.fp_source_a];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        entry.paired_fp_source_b_ps0 = state.fpr[instruction.fp_source_b];
        entry.paired_fp_source_b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        return entry;
    }
    if (opcode == Opcode::floating_move_register || opcode == Opcode::floating_negate)
    {
        entry.has_fp_move = true;
        entry.fp_move_destination = instruction.fp_register;
        entry.fp_move_source = instruction.fp_source_b;
        entry.fp_move_source_value = state.fpr[instruction.fp_source_b];
        entry.fp_move_destination_ps1 = state.fpr_ps1[instruction.fp_register];
        return entry;
    }
    if (opcode == Opcode::floating_compare_unordered)
    {
        entry.has_fp_compare = true;
        entry.fp_compare_cr_field = instruction.cr_field;
        entry.fp_compare_a_register = instruction.fp_compare_a;
        entry.fp_compare_b_register = instruction.fp_compare_b;
        entry.fp_compare_a_raw = state.fpr[instruction.fp_compare_a];
        entry.fp_compare_b_raw = state.fpr[instruction.fp_compare_b];
        return entry;
    }
    if (opcode == Opcode::floating_divide_single || opcode == Opcode::floating_add_single ||
        opcode == Opcode::floating_multiply_single ||
        opcode == Opcode::floating_multiply_add_single ||
        opcode == Opcode::floating_subtract_double ||
        opcode == Opcode::floating_round_to_single)
    {
        entry.has_fp_arithmetic = true;
        entry.fp_arithmetic_destination = instruction.fp_register;
        entry.fp_arithmetic_source_a = instruction.fp_source_a;
        entry.fp_arithmetic_source_b = instruction.fp_source_b;
        entry.fp_arithmetic_a_raw = state.fpr[instruction.fp_source_a];
        entry.fp_arithmetic_b_raw = state.fpr[instruction.fp_source_b];
        entry.has_fp_arithmetic_single_bits = opcode != Opcode::floating_subtract_double;
        entry.fp_arithmetic_source_count = opcode == Opcode::floating_round_to_single
            ? 1U
            : opcode == Opcode::floating_multiply_add_single ? 3U : 2U;
        if (opcode == Opcode::floating_multiply_single ||
            opcode == Opcode::floating_multiply_add_single)
        {
            entry.has_fp_arithmetic_source_c = true;
            entry.fp_arithmetic_source_c = instruction.fp_source_c;
            entry.fp_arithmetic_c_raw = state.fpr[instruction.fp_source_c];
        }
        return entry;
    }
    if (opcode == Opcode::load_multiple_word || opcode == Opcode::store_multiple_word)
    {
        entry.has_register_range = true;
        entry.range_first_register = opcode == Opcode::load_multiple_word
            ? instruction.destination
            : instruction.source;
        entry.range_last_register = 31U;
        entry.memory_base_register = instruction.base;
        entry.memory_displacement = instruction.immediate;
        entry.has_effective_address = true;
        entry.effective_address = effective_address(state, instruction.base, instruction.immediate);
        return entry;
    }
    if (opcode == Opcode::load_string_word_immediate ||
        opcode == Opcode::store_string_word_immediate)
    {
        const bool store = opcode == Opcode::store_string_word_immediate;
        entry.has_string_transfer = true;
        entry.string_transfer_store = store;
        entry.string_transfer_first_register = store ? instruction.source : instruction.destination;
        entry.string_transfer_base_register = instruction.base;
        entry.string_transfer_base_value = instruction.base == 0 ? 0U : state.gpr[instruction.base];
        entry.string_transfer_byte_count = instruction.string_byte_count == 0
            ? 32U : instruction.string_byte_count;
        entry.string_transfer_register_count = static_cast<std::uint8_t>(
            (entry.string_transfer_byte_count + 3U) / 4U);
        entry.has_effective_address = true;
        entry.effective_address = entry.string_transfer_base_value;
        for (std::uint8_t i = 0; i < entry.string_transfer_register_count; ++i)
        {
            const std::uint8_t reg = static_cast<std::uint8_t>(
                (entry.string_transfer_first_register + i) & 31U);
            entry.string_transfer_registers[i] = reg;
            if (store)
            {
                entry.string_transfer_register_values[i] = state.gpr[reg];
            }
        }
        return entry;
    }
    const bool store = opcode == Opcode::store_word || opcode == Opcode::store_word_update ||
        opcode == Opcode::store_word_indexed || opcode == Opcode::store_byte ||
        opcode == Opcode::store_byte_update || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_halfword || opcode == Opcode::store_halfword_update;

    if (opcode == Opcode::load_single || opcode == Opcode::load_single_update ||
        opcode == Opcode::load_double ||
        opcode == Opcode::store_single ||
        opcode == Opcode::store_single_update || opcode == Opcode::store_double ||
        opcode == Opcode::load_halfword_algebraic || opcode == Opcode::store_halfword_update)
    {
        if (instruction.base != 0)
        {
            add_history_source(entry, state, instruction.base);
        }
        entry.has_effective_address = true;
        entry.effective_address = effective_address(state, instruction.base, instruction.immediate);
    }
    if (opcode == Opcode::load_single || opcode == Opcode::load_single_update ||
        opcode == Opcode::load_double)
    {
        entry.has_fp_destination = true;
        entry.fp_destination_register = instruction.fp_register;
    }
    else if (opcode == Opcode::store_single || opcode == Opcode::store_single_update ||
             opcode == Opcode::store_double)
    {
        entry.has_fp_source = true;
        entry.fp_source_register = instruction.fp_register;
        entry.fp_source_value = state.fpr[instruction.fp_register];
    }

    switch (opcode)
    {
    case Opcode::multiply_low_immediate:
        // Unlike D-form address calculations, mulli reads GPR[RA] even for RA=0.
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::addi:
    case Opcode::addis:
    case Opcode::load_word_zero:
    case Opcode::load_word_update:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::store_single_update:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::store_halfword:
    case Opcode::store_halfword_update:
        if (instruction.base != 0)
        {
            add_history_source(entry, state, instruction.base);
        }
        break;
    case Opcode::add_immediate_carry:
    case Opcode::subtract_from_immediate_carry:
    case Opcode::compare_signed_immediate:
    case Opcode::compare_unsigned_immediate:
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::ori:
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate:
    case Opcode::xor_immediate_shifted:
    case Opcode::count_leading_zeros:
    case Opcode::and_immediate_record:
    case Opcode::rotate_left_word_and_mask:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::arithmetic_shift_right_immediate:
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::add:
    case Opcode::subtract_from:
    case Opcode::subtract_from_carrying:
    case Opcode::multiply_low_word:
    case Opcode::multiply_high_word_unsigned:
    case Opcode::divide_word_unsigned:
    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_and_complement:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_equivalence:
    case Opcode::compare_signed_register:
    case Opcode::compare_unsigned_register:
        add_history_source(entry, state, instruction.base);
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::negate:
    case Opcode::extend_sign_byte:
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
        add_history_source(entry, state, instruction.source);
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::load_word_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
        if (instruction.base != 0)
        {
            add_history_source(entry, state, instruction.base);
        }
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::move_to_link_register:
    case Opcode::move_to_count_register:
        add_history_source(entry, state, instruction.destination);
        break;
    default:
        break;
    }
    if (store)
    {
        add_history_source(entry, state, instruction.destination);
    }

    switch (opcode)
    {
    case Opcode::multiply_low_immediate:
    case Opcode::addi:
    case Opcode::addis:
    case Opcode::add_immediate_carry:
    case Opcode::subtract_from_immediate_carry:
    case Opcode::add:
    case Opcode::subtract_from:
    case Opcode::subtract_from_carrying:
    case Opcode::multiply_low_word:
    case Opcode::multiply_high_word_unsigned:
    case Opcode::divide_word_unsigned:
    case Opcode::negate:
    case Opcode::extend_sign_byte:
    case Opcode::ori:
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate:
    case Opcode::xor_immediate_shifted:
    case Opcode::count_leading_zeros:
    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_and_complement:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_equivalence:
    case Opcode::and_immediate_record:
    case Opcode::rotate_left_word_and_mask:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::arithmetic_shift_right_immediate:
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::load_word_zero:
    case Opcode::load_word_update:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::move_from_link_register:
    case Opcode::move_from_count_register:
        entry.has_destination = true;
        entry.destination_register = instruction.destination;
        break;
    case Opcode::load_single:
        break;
    case Opcode::load_single_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::load_double:
        break;
    case Opcode::store_single:
        break;
    case Opcode::store_single_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::store_word_update:
    case Opcode::store_byte_update:
    case Opcode::store_halfword_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    default:
        break;
    }
    return entry;
}

}

void EspressoCore::append_instruction_history(InstructionHistoryEntry entry) noexcept
{
    instruction_history_[instruction_history_next_] = std::move(entry);
    instruction_history_next_ = (instruction_history_next_ + 1U) % instruction_history_.size();
    instruction_history_count_ = std::min(instruction_history_count_ + 1U,
                                          instruction_history_.size());
}

std::vector<InstructionHistoryEntry> EspressoCore::instruction_history_snapshot() const
{
    std::vector<InstructionHistoryEntry> result;
    result.reserve(instruction_history_count_);
    const std::size_t first =
        (instruction_history_next_ + instruction_history_.size() - instruction_history_count_) %
        instruction_history_.size();
    for (std::size_t i = 0; i < instruction_history_count_; ++i)
    {
        result.push_back(instruction_history_[(first + i) % instruction_history_.size()]);
    }
    return result;
}

std::string format_instruction_history(const RunResult& result)
{
    if (result.instruction_history.empty())
    {
        return {};
    }
    std::ostringstream text;
    text << "Recent instructions (oldest first):";
    for (const InstructionHistoryEntry& entry : result.instruction_history)
    {
        text << "\n  0x" << std::hex << std::uppercase << std::setw(8)
             << std::setfill('0') << entry.cia << ": ";
        if (entry.has_instruction_word)
        {
            text << "0x" << std::setw(8) << entry.instruction_word << ' ';
        }
        text << entry.opcode_name;
        if (entry.has_register_range)
        {
            text << " r" << std::dec << static_cast<unsigned>(entry.range_first_register)
                 << "-r" << static_cast<unsigned>(entry.range_last_register) << ", "
                 << entry.memory_displacement << "(r"
                 << static_cast<unsigned>(entry.memory_base_register) << ')';
        }
        else if (entry.has_string_transfer)
        {
            text << " r" << std::dec
                 << static_cast<unsigned>(entry.string_transfer_first_register)
                 << ",r" << static_cast<unsigned>(entry.string_transfer_base_register) << ','
                 << static_cast<unsigned>(entry.string_transfer_byte_count);
            if (entry.string_transfer_store)
            {
                for (std::uint8_t i = 0; i < entry.string_transfer_register_count; ++i)
                {
                    text << " r" << std::dec
                         << static_cast<unsigned>(entry.string_transfer_registers[i])
                         << "=0x" << std::hex << std::setw(8)
                         << entry.string_transfer_register_values[i];
                }
            }
            text << " r" << std::dec
                 << static_cast<unsigned>(entry.string_transfer_base_register)
                 << "=0x" << std::hex << std::setw(8) << entry.string_transfer_base_value;
        }
        else
        {
            for (std::uint8_t i = 0; i < entry.source_count; ++i)
            {
                text << " r" << std::dec << static_cast<unsigned>(entry.source_registers[i])
                     << "=0x" << std::hex << std::setw(8) << entry.source_values[i];
            }
        }
        if (entry.has_fp_compare)
        {
            const auto print_operand = [&](std::uint8_t reg, std::uint64_t raw) {
                text << " f" << std::dec << static_cast<unsigned>(reg) << '=';
                if (is_binary64_nan(raw))
                {
                    text << "NaN";
                }
                else
                {
                    text << std::setprecision(17) << std::bit_cast<double>(raw);
                }
                text << " [0x" << std::hex << std::setw(16) << raw << ']';
            };
            text << " cr" << std::dec
                 << static_cast<unsigned>(entry.fp_compare_cr_field);
            print_operand(entry.fp_compare_a_register, entry.fp_compare_a_raw);
            print_operand(entry.fp_compare_b_register, entry.fp_compare_b_raw);
            if (entry.completed)
            {
                text << " -> " << floating_compare_result_name(entry.fp_compare_result)
                     << " (cr" << std::dec
                     << static_cast<unsigned>(entry.fp_compare_cr_field) << "=0x"
                     << std::hex << static_cast<unsigned>(entry.fp_compare_result)
                     << ", fpcc=0x" << static_cast<unsigned>(entry.fp_compare_result) << ')';
            }
        }
        if (entry.has_fp_arithmetic)
        {
            const auto print_operand = [&](std::uint8_t reg, std::uint64_t raw) {
                text << " f" << std::dec << static_cast<unsigned>(reg) << '=';
                if (is_binary64_nan(raw))
                {
                    text << "NaN";
                }
                else
                {
                    text << std::setprecision(17) << std::bit_cast<double>(raw);
                }
                text << " [0x" << std::hex << std::setw(16) << raw << ']';
            };
            print_operand(entry.fp_arithmetic_source_a, entry.fp_arithmetic_a_raw);
            if (entry.has_fp_arithmetic_source_c)
            {
                // Multiply/FMA assembly uses frA and frC; FMA then adds frB.
                print_operand(entry.fp_arithmetic_source_c, entry.fp_arithmetic_c_raw);
                if (entry.fp_arithmetic_source_count == 3U)
                {
                    print_operand(entry.fp_arithmetic_source_b, entry.fp_arithmetic_b_raw);
                }
            }
            else
            {
                if (entry.fp_arithmetic_source_count > 1U)
                {
                    print_operand(entry.fp_arithmetic_source_b, entry.fp_arithmetic_b_raw);
                }
            }
            if (entry.has_fp_arithmetic_result)
            {
                const double value = std::bit_cast<double>(entry.fp_arithmetic_result_raw);
                text << " -> f" << std::dec
                     << static_cast<unsigned>(entry.fp_arithmetic_destination) << '='
                     << std::setprecision(17) << value << " [0x" << std::hex
                     << std::setw(16) << entry.fp_arithmetic_result_raw << ']';
                if (entry.has_fp_arithmetic_single_bits)
                {
                    text << " single=0x" << std::setw(8)
                         << entry.fp_arithmetic_single_bits;
                }
            }
            else if (entry.completed)
            {
                text << " -> f" << std::dec
                     << static_cast<unsigned>(entry.fp_arithmetic_destination)
                     << " (write suppressed)";
            }
        }
        if (entry.has_fp_move)
        {
            text << " f" << std::dec << static_cast<unsigned>(entry.fp_move_source)
                 << ".ps0=0x" << std::hex << std::setw(16)
                 << entry.fp_move_source_value;
            if (entry.completed)
            {
                text << " -> f" << std::dec
                     << static_cast<unsigned>(entry.fp_move_destination)
                     << ".ps0=0x" << std::hex << std::setw(16)
                     << entry.fp_move_destination_value
                     << " (ps1 unchanged=0x" << std::setw(16)
                     << entry.fp_move_destination_ps1 << ')';
            }
        }
        if (entry.has_immediate)
        {
            text << " imm=";
            if (entry.immediate_hex)
            {
                text << "0x" << std::hex << std::setw(4)
                     << static_cast<std::uint32_t>(entry.immediate);
            }
            else
            {
                text << std::dec << entry.immediate;
            }
        }
        if (entry.has_old_destination)
        {
            text << " old-r" << std::dec
                 << static_cast<unsigned>(entry.old_destination_register) << "=0x"
                 << std::hex << std::setw(8) << entry.old_destination_value;
        }
        if (entry.has_rotate_fields)
        {
            text << " sh=" << std::dec << static_cast<unsigned>(entry.rotate_shift)
                 << " mb=" << static_cast<unsigned>(entry.rotate_mask_begin)
                 << " me=" << static_cast<unsigned>(entry.rotate_mask_end);
        }
        if (entry.has_fp_source)
        {
            const double value = std::bit_cast<double>(entry.fp_source_value);
            text << " f" << std::dec << static_cast<unsigned>(entry.fp_source_register)
                 << '=' << std::setprecision(17) << value << " [0x" << std::hex
                 << std::setw(16) << entry.fp_source_value << ']';
        }
        if (entry.has_paired_fp_state)
        {
            const auto print_paired_source = [&](char operand, std::uint8_t reg,
                                                  std::uint64_t ps0, std::uint64_t ps1) {
                text << " f" << operand << "=f" << std::dec
                     << static_cast<unsigned>(reg) << "{ps0=0x" << std::hex
                     << std::setw(16) << ps0 << ",ps1=0x" << std::setw(16) << ps1 << '}';
            };
            print_paired_source('A', entry.paired_fp_source_a_register,
                                entry.paired_fp_source_a_ps0,
                                entry.paired_fp_source_a_ps1);
            print_paired_source('B', entry.paired_fp_source_b_register,
                                entry.paired_fp_source_b_ps0,
                                entry.paired_fp_source_b_ps1);
            text << " -> f" << std::dec
                 << static_cast<unsigned>(entry.paired_fp_destination_register);
            if (entry.completed)
            {
                text << "{ps0=0x" << std::hex << std::setw(16)
                     << entry.paired_fp_destination_ps0 << ",ps1=0x"
                     << std::setw(16) << entry.paired_fp_destination_ps1 << '}';
            }
            else
            {
                text << " (not written)";
            }
        }
        if (entry.has_effective_address)
        {
            text << " [0x" << std::hex << std::setw(8) << entry.effective_address << ']';
        }
        if (entry.has_string_transfer && entry.completed && !entry.string_transfer_store)
        {
            text << " ->";
            for (std::uint8_t i = 0; i < entry.string_transfer_register_count; ++i)
            {
                text << " r" << std::dec
                     << static_cast<unsigned>(entry.string_transfer_registers[i])
                     << "=0x" << std::hex << std::setw(8)
                     << entry.string_transfer_register_values[i];
            }
        }
        if (entry.has_string_transfer && entry.completed && entry.string_transfer_store)
        {
            text << " -> mem=";
            for (std::uint8_t i = 0; i < entry.string_transfer_byte_count; ++i)
            {
                if (i != 0)
                {
                    text << ' ';
                }
                text << std::hex << std::setw(2)
                     << static_cast<unsigned>(entry.string_transfer_bytes[i]);
            }
        }
        if (entry.has_destination)
        {
            text << " -> r" << std::dec << static_cast<unsigned>(entry.destination_register)
                 << (entry.completed
                         ? "=0x"
                         : " (not written)");
            if (entry.completed)
            {
                text << std::hex << std::setw(8) << entry.destination_value;
            }
        }
        if (entry.has_carry_result)
        {
            text << " CA=" << (entry.carry_result ? '1' : '0');
        }
        if (entry.has_fp_destination)
        {
            text << " -> f" << std::dec
                 << static_cast<unsigned>(entry.fp_destination_register);
            if (entry.completed)
            {
                const double value = std::bit_cast<double>(entry.fp_destination_value);
                text << '=' << std::setprecision(17) << value << " [0x" << std::hex
                     << std::setw(16) << entry.fp_destination_value << ']';
            }
            else
            {
                text << " (not written)";
            }
        }
        if (entry.has_stored_single_value && entry.completed)
        {
            text << " -> mem32=0x" << std::hex << std::setw(8)
                 << entry.stored_single_value;
        }
        if (entry.has_stored_double_value && entry.completed)
        {
            text << " -> mem64=0x" << std::hex << std::setw(16)
                 << entry.stored_double_value;
        }
        if (entry.has_stored_halfword_value && entry.completed)
        {
            text << " -> mem16=0x" << std::hex << std::setw(4)
                 << entry.stored_halfword_value;
        }
        if (!entry.completed)
        {
            text << " [did not complete]";
        }
    }
    return text.str();
}

StepResult EspressoCore::step()
{
    const std::uint32_t cia = state.cia;
    current_instruction_word_fetched_ = false;
    has_pending_history_entry_ = false;
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
    current_instruction_word_ = instruction_word;
    current_instruction_word_fetched_ = true;
    const DecodedInstruction instruction = decode(instruction_word);
    pending_history_entry_ = make_history_entry(state, cia, instruction_word, instruction);
    if (instruction.opcode == Opcode::subtract_from_carrying)
    {
        pending_history_entry_.has_carry_result = true;
    }
    has_pending_history_entry_ = true;

    if (instruction.opcode == Opcode::unsupported)
    {
        append_instruction_history(std::move(pending_history_entry_));
        has_pending_history_entry_ = false;
        return StepResult::unsupported_instruction;
    }

    const std::uint32_t fallthrough = cia + 4U;
    std::uint32_t next_cia = fallthrough;

    switch (instruction.opcode)
    {
    case Opcode::multiply_low_immediate:
    {
        const std::int64_t lhs = std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int64_t rhs = instruction.immediate;
        const std::uint32_t result = static_cast<std::uint32_t>(lhs * rhs);
        state.gpr[instruction.destination] = result;
        break;
    }

    case Opcode::multiply_low_word:
    {
        const std::int64_t lhs = std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int64_t rhs = std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        const std::int64_t product = lhs * rhs;
        const std::uint32_t result = static_cast<std::uint32_t>(product);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::multiply_high_word_unsigned:
    {
        const std::uint64_t lhs = state.gpr[instruction.base];
        const std::uint64_t rhs = state.gpr[instruction.source];
        const std::uint64_t product = lhs * rhs;
        const std::uint32_t result = static_cast<std::uint32_t>(product >> 32U);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::divide_word_unsigned:
    {
        const std::uint32_t dividend = state.gpr[instruction.base];
        const std::uint32_t divisor = state.gpr[instruction.source];
        const std::uint32_t result = divisor == 0U ? 0U : dividend / divisor;
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::negate:
    {
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t result = 0U - source;
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::extend_sign_byte:
    {
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t low_byte = source & 0xFFU;
        const std::uint32_t result = low_byte |
            ((low_byte & 0x80U) != 0U ? 0xFFFFFF00U : 0U);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

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
        // Unlike addi/addis, addic reads GPR[RA] even when RA is zero.
        const std::uint32_t lhs = state.gpr[instruction.base];
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

    case Opcode::subtract_from_carrying:
    {
        const std::uint32_t minuend = state.gpr[instruction.source];
        const std::uint32_t subtrahend = state.gpr[instruction.base];
        const std::uint32_t result = minuend - subtrahend;
        state.gpr[instruction.destination] = result;
        if (minuend >= subtrahend)
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

    case Opcode::ori:
        state.gpr[instruction.destination] =
            state.gpr[instruction.source] |
            static_cast<std::uint32_t>(instruction.immediate);
        break;

    case Opcode::or_immediate_shifted:
    {
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t operand =
            static_cast<std::uint32_t>(instruction.immediate) << 16U;
        state.gpr[instruction.destination] = source | operand;
        break;
    }

    case Opcode::count_leading_zeros:
    {
        const std::uint32_t result = static_cast<std::uint32_t>(
            std::countl_zero(state.gpr[instruction.source]));
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::xor_immediate:
    case Opcode::xor_immediate_shifted:
    {
        const std::uint32_t immediate = static_cast<std::uint32_t>(instruction.immediate);
        const std::uint32_t operand = instruction.opcode == Opcode::xor_immediate_shifted
            ? immediate << 16U
            : immediate;
        state.gpr[instruction.destination] = state.gpr[instruction.source] ^ operand;
        break;
    }

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

    case Opcode::rotate_left_word_and_mask_insert:
    {
        const std::uint32_t old_destination = state.gpr[instruction.destination];
        const std::uint32_t rotated = std::rotl(
            state.gpr[instruction.source], static_cast<int>(instruction.shift));
        const std::uint32_t mask = rotate_mask(instruction.mask_begin, instruction.mask_end);
        const std::uint32_t result = (rotated & mask) | (old_destination & ~mask);
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

    case Opcode::shift_right_word:
    {
        const std::uint32_t shift = state.gpr[instruction.base] & 0x3FU;
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t result = shift >= 32U ? 0U : source >> shift;
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

    case Opcode::floating_compare_unordered:
    {
        const std::uint64_t a_raw = state.fpr[instruction.fp_compare_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_compare_b];
        const bool a_nan = is_binary64_nan(a_raw);
        const bool b_nan = is_binary64_nan(b_raw);
        std::uint8_t result = fp_unordered;
        if (!a_nan && !b_nan)
        {
            const double a = std::bit_cast<double>(a_raw);
            const double b = std::bit_cast<double>(b_raw);
            result = a < b ? cr_less_than
                : a > b ? cr_greater_than
                        : cr_equal;
        }

        set_floating_compare_result(state, instruction.cr_field, result);
        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw))
        {
            raise_fpscr_exception(state, fpscr::vxsnan_mask);
        }
        pending_history_entry_.fp_compare_result = result;
        break;
    }

    case Opcode::floating_divide_single:
    {
        // Snapshot PS0 sources before any destination write; frD may alias either.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const auto commit_result = [&](std::uint32_t single_bits,
                                       bool inexact,
                                       bool rounded_up) {
            const float single_result = std::bit_cast<float>(single_bits);
            const double extended_result = static_cast<double>(single_result);
            const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(extended_result);
            state.fpr[instruction.fp_register] = raw_result;
            state.fpr_ps1[instruction.fp_register] = raw_result;
            set_single_arithmetic_result_status(state, single_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            pending_history_entry_.fp_arithmetic_single_bits = single_bits;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_single_bits(nan_source), false, false);
        };
        const auto signal_invalid = [&](std::uint32_t subexception,
                                        std::uint64_t nan_source) {
            raise_fpscr_exception(state, subexception);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_quiet_nan(nan_source);
            }
        };

        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw))
        {
            const std::uint64_t signaling_source = is_binary64_signaling_nan(a_raw)
                ? a_raw
                : b_raw;
            signal_invalid(fpscr::vxsnan_mask, signaling_source);
            break;
        }
        if (is_binary64_nan(a_raw) || is_binary64_nan(b_raw))
        {
            commit_quiet_nan(is_binary64_nan(a_raw) ? a_raw : b_raw);
            break;
        }

        const bool a_zero = is_binary64_zero(a_raw);
        const bool b_zero = is_binary64_zero(b_raw);
        const bool a_infinity = is_binary64_infinity(a_raw);
        const bool b_infinity = is_binary64_infinity(b_raw);
        if ((a_zero && b_zero) || (a_infinity && b_infinity))
        {
            const std::uint32_t invalid_subexception = a_zero
                ? fpscr::vxzdz_mask
                : fpscr::vxidi_mask;
            signal_invalid(invalid_subexception, 0U);
            break;
        }
        if (b_zero && !a_infinity)
        {
            raise_fpscr_exception(state, fpscr::zx_mask);
            if ((state.fpscr & fpscr::ze_mask) == 0U)
            {
                const std::uint64_t sign = (a_raw ^ b_raw) & binary64_sign_mask;
                const std::uint32_t single_bits =
                    (sign != 0U ? 0x80000000U : 0U) | 0x7F800000U;
                commit_result(single_bits, false, false);
            }
            break;
        }

        if (b_zero) // infinity / zero is signed infinity without ZX.
        {
            const std::uint64_t sign = (a_raw ^ b_raw) & binary64_sign_mask;
            const std::uint32_t single_bits =
                (sign != 0U ? 0x80000000U : 0U) | 0x7F800000U;
            commit_result(single_bits, false, false);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double exact_result = a / b;
        const float single_result = static_cast<float>(exact_result);
        const double extended_result = static_cast<double>(single_result);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
        const bool inexact = extended_result != exact_result;
        const bool rounded_up = inexact &&
            std::fabs(extended_result) > std::fabs(exact_result);

        // TODO: add exact FPSCR OX/UX behavior when broader FP exceptions are modeled.
        commit_result(single_bits, inexact, rounded_up);
        break;
    }

    case Opcode::floating_add_single:
    {
        // Capture PS0 before writing because frD may alias either source.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const auto commit_result = [&](std::uint32_t single_bits,
                                       bool inexact,
                                       bool rounded_up) {
            const float single_result = std::bit_cast<float>(single_bits);
            const double extended_result = static_cast<double>(single_result);
            const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(extended_result);
            state.fpr[instruction.fp_register] = raw_result;
            state.fpr_ps1[instruction.fp_register] = raw_result;
            set_single_arithmetic_result_status(state, single_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            pending_history_entry_.fp_arithmetic_single_bits = single_bits;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_single_bits(nan_source), false, false);
        };
        const auto signal_invalid = [&](std::uint32_t subexception,
                                        std::uint64_t nan_source) {
            raise_fpscr_exception(state, subexception);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_quiet_nan(nan_source);
            }
        };

        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw))
        {
            const std::uint64_t signaling_source = is_binary64_signaling_nan(a_raw)
                ? a_raw
                : b_raw;
            signal_invalid(fpscr::vxsnan_mask, signaling_source);
            break;
        }
        if (is_binary64_nan(a_raw) || is_binary64_nan(b_raw))
        {
            commit_quiet_nan(is_binary64_nan(a_raw) ? a_raw : b_raw);
            break;
        }

        if (is_binary64_infinity(a_raw) && is_binary64_infinity(b_raw) &&
            ((a_raw ^ b_raw) & binary64_sign_mask) != 0U)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double exact_result = a + b;
        const float single_result = static_cast<float>(exact_result);
        const double extended_result = static_cast<double>(single_result);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
        const bool inexact = extended_result != exact_result;
        const bool rounded_up = inexact &&
            std::fabs(extended_result) > std::fabs(exact_result);

        // TODO: add exact FPSCR OX/UX behavior when broader FP exceptions are modeled.
        commit_result(single_bits, inexact, rounded_up);
        break;
    }

    case Opcode::floating_multiply_single:
    {
        // Both architectural sources are PS0; snapshot them before a possible aliasing write.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint32_t single_bits,
                                       bool inexact,
                                       bool rounded_up) {
            const float single_result = std::bit_cast<float>(single_bits);
            const double extended_result = static_cast<double>(single_result);
            const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(extended_result);
            state.fpr[instruction.fp_register] = raw_result;
            state.fpr_ps1[instruction.fp_register] = raw_result;
            set_single_arithmetic_result_status(state, single_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            pending_history_entry_.fp_arithmetic_single_bits = single_bits;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_single_bits(nan_source), false, false);
        };
        const auto signal_invalid = [&](std::uint32_t subexception,
                                        std::uint64_t nan_source) {
            raise_fpscr_exception(state, subexception);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_quiet_nan(nan_source);
            }
        };

        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(c_raw))
        {
            signal_invalid(fpscr::vxsnan_mask,
                is_binary64_signaling_nan(a_raw) ? a_raw : c_raw);
            break;
        }
        if (is_binary64_nan(a_raw) || is_binary64_nan(c_raw))
        {
            commit_quiet_nan(is_binary64_nan(a_raw) ? a_raw : c_raw);
            break;
        }

        const bool a_zero = is_binary64_zero(a_raw);
        const bool c_zero = is_binary64_zero(c_raw);
        const bool a_infinity = is_binary64_infinity(a_raw);
        const bool c_infinity = is_binary64_infinity(c_raw);
        if ((a_infinity && c_zero) || (a_zero && c_infinity))
        {
            signal_invalid(fpscr::vximz_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double rounded_c = std::bit_cast<double>(round_multiply_operand_25bit(c_raw));
        const double exact_result = a * rounded_c;
        const float single_result = static_cast<float>(exact_result);
        const double extended_result = static_cast<double>(single_result);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
        const bool inexact = extended_result != exact_result;
        const bool rounded_up = inexact &&
            std::fabs(extended_result) > std::fabs(exact_result);

        // TODO: add exact FPSCR OX/UX behavior when broader FP exceptions are modeled.
        commit_result(single_bits, inexact, rounded_up);
        break;
    }

    case Opcode::floating_multiply_add_single:
    {
        // Snapshot all PS0 inputs before writing; frD may alias any source.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint32_t single_bits,
                                       bool inexact,
                                       bool rounded_up) {
            const float single_result = std::bit_cast<float>(single_bits);
            const double extended_result = static_cast<double>(single_result);
            const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(extended_result);
            state.fpr[instruction.fp_register] = raw_result;
            state.fpr_ps1[instruction.fp_register] = raw_result;
            set_single_arithmetic_result_status(state, single_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            pending_history_entry_.fp_arithmetic_single_bits = single_bits;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_single_bits(nan_source), false, false);
        };
        const auto signal_invalid = [&](std::uint32_t subexception,
                                        std::uint64_t nan_source) {
            raise_fpscr_exception(state, subexception);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_quiet_nan(nan_source);
            }
        };

        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw) ||
            is_binary64_signaling_nan(c_raw))
        {
            const std::uint64_t signaling_source = is_binary64_signaling_nan(a_raw)
                ? a_raw
                : is_binary64_signaling_nan(c_raw) ? c_raw : b_raw;
            signal_invalid(fpscr::vxsnan_mask, signaling_source);
            break;
        }
        if (is_binary64_nan(a_raw) || is_binary64_nan(c_raw) || is_binary64_nan(b_raw))
        {
            const std::uint64_t nan_source = is_binary64_nan(a_raw) ? a_raw
                : is_binary64_nan(c_raw) ? c_raw : b_raw;
            commit_quiet_nan(nan_source);
            break;
        }

        const bool a_zero = is_binary64_zero(a_raw);
        const bool c_zero = is_binary64_zero(c_raw);
        const bool a_infinity = is_binary64_infinity(a_raw);
        const bool c_infinity = is_binary64_infinity(c_raw);
        if ((a_infinity && c_zero) || (a_zero && c_infinity))
        {
            signal_invalid(fpscr::vximz_mask, 0U);
            break;
        }

        const bool product_is_infinite = a_infinity || c_infinity;
        const bool b_infinity = is_binary64_infinity(b_raw);
        const bool product_is_negative =
            ((a_raw ^ c_raw) & binary64_sign_mask) != 0U;
        const bool b_is_negative = (b_raw & binary64_sign_mask) != 0U;
        if (product_is_infinite && b_infinity && product_is_negative != b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double c = std::bit_cast<double>(round_multiply_operand_25bit(c_raw));
        const double b = std::bit_cast<double>(b_raw);
        const double exact_result = std::fma(a, c, b);
        const float single_result = static_cast<float>(exact_result);
        const double extended_result = static_cast<double>(single_result);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
        const bool inexact = extended_result != exact_result;
        const bool rounded_up = inexact &&
            std::fabs(extended_result) > std::fabs(exact_result);

        // TODO: add exact FPSCR OX/UX behavior when broader FP exceptions are modeled.
        commit_result(single_bits, inexact, rounded_up);
        break;
    }

    case Opcode::floating_move_register:
    {
        const std::uint64_t source = state.fpr[instruction.fp_source_b];
        state.fpr[instruction.fp_register] = source;
        pending_history_entry_.has_fp_move = true;
        pending_history_entry_.fp_move_destination = instruction.fp_register;
        pending_history_entry_.fp_move_source = instruction.fp_source_b;
        pending_history_entry_.fp_move_source_value = source;
        pending_history_entry_.fp_move_destination_value = source;
        pending_history_entry_.fp_move_destination_ps1 =
            state.fpr_ps1[instruction.fp_register];
        break;
    }

    case Opcode::floating_negate:
    {
        const std::uint64_t source = state.fpr[instruction.fp_source_b];
        const std::uint64_t result = source ^ binary64_sign_mask;
        state.fpr[instruction.fp_register] = result;
        pending_history_entry_.has_fp_move = true;
        pending_history_entry_.fp_move_destination = instruction.fp_register;
        pending_history_entry_.fp_move_source = instruction.fp_source_b;
        pending_history_entry_.fp_move_source_value = source;
        pending_history_entry_.fp_move_destination_value = result;
        pending_history_entry_.fp_move_destination_ps1 =
            state.fpr_ps1[instruction.fp_register];
        break;
    }

    case Opcode::floating_subtract_double:
    {
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const auto commit_result = [&](std::uint64_t raw_result,
                                       bool inexact,
                                       bool rounded_up) {
            state.fpr[instruction.fp_register] = raw_result;
            set_double_arithmetic_result_status(state, raw_result, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_binary64_bits(nan_source), false, false);
        };
        const auto signal_invalid = [&](std::uint32_t subexception,
                                        std::uint64_t nan_source) {
            raise_fpscr_exception(state, subexception);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_quiet_nan(nan_source);
            }
        };

        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw))
        {
            const std::uint64_t signaling_source = is_binary64_signaling_nan(a_raw)
                ? a_raw
                : b_raw;
            signal_invalid(fpscr::vxsnan_mask, signaling_source);
            break;
        }
        if (is_binary64_nan(a_raw) || is_binary64_nan(b_raw))
        {
            commit_quiet_nan(is_binary64_nan(a_raw) ? a_raw : b_raw);
            break;
        }

        const bool a_infinity = is_binary64_infinity(a_raw);
        const bool b_infinity = is_binary64_infinity(b_raw);
        if (a_infinity && b_infinity &&
            ((a_raw ^ b_raw) & binary64_sign_mask) == 0U)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double result = a - b;
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        if (a_infinity || b_infinity)
        {
            commit_result(raw_result, false, false);
            break;
        }

        if (binary64_normal_or_zero(a_raw) && binary64_normal_or_zero(b_raw) &&
            binary64_normal_or_zero(raw_result))
        {
            const double error = binary64_subtraction_error(a, b, result);
            const bool inexact = error != 0.0;
            const bool rounded_up = inexact && result != 0.0 &&
                ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
            commit_result(raw_result, inexact, rounded_up);
        }
        else
        {
            // TODO: classify double underflow/overflow and their FI/FR/XX details.
            // FPRF is still exact from the raw binary64 result; do not guess rounding flags.
            state.fpr[instruction.fp_register] = raw_result;
            set_double_arithmetic_fprf(state, raw_result);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        }
        break;
    }

    case Opcode::floating_round_to_single:
    {
        const std::uint64_t source_raw = state.fpr[instruction.fp_source_a];
        const auto commit_result = [&](std::uint32_t single_bits,
                                       bool inexact,
                                       bool rounded_up) {
            const float single_result = std::bit_cast<float>(single_bits);
            const double extended_result = static_cast<double>(single_result);
            const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(extended_result);
            state.fpr[instruction.fp_register] = raw_result;
            state.fpr_ps1[instruction.fp_register] = raw_result;
            set_single_arithmetic_result_status(state, single_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            pending_history_entry_.fp_arithmetic_single_bits = single_bits;
        };

        if (is_binary64_signaling_nan(source_raw))
        {
            raise_fpscr_exception(state, fpscr::vxsnan_mask);
            if ((state.fpscr & fpscr::ve_mask) == 0U)
            {
                commit_result(quiet_nan_single_bits(source_raw), false, false);
            }
            break;
        }
        if (is_binary64_nan(source_raw))
        {
            commit_result(quiet_nan_single_bits(source_raw), false, false);
            break;
        }

        const double source = std::bit_cast<double>(source_raw);
        // TODO: honor FPSCR.RN and model exact single overflow/underflow flags.
        const float single_result = static_cast<float>(source);
        const double extended_result = static_cast<double>(single_result);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
        const bool inexact = extended_result != source;
        const bool rounded_up = inexact &&
            std::fabs(extended_result) > std::fabs(source);
        commit_result(single_bits, inexact, rounded_up);
        break;
    }

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

    case Opcode::load_multiple_word:
    {
        std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        for (std::uint32_t reg = instruction.destination; reg < state.gpr.size(); ++reg)
        {
            state.gpr[reg] = memory.read32_be(address);
            address += sizeof(std::uint32_t);
        }
        break;
    }

    case Opcode::load_string_word_immediate:
    {
        // Snapshot the base and validate/read the complete string before any
        // destination GPR is changed. lswi may overwrite rA or wrap through r0.
        const std::uint32_t address = instruction.base == 0
            ? 0U : state.gpr[instruction.base];
        const std::size_t byte_count = instruction.string_byte_count == 0
            ? 32U : instruction.string_byte_count;
        std::array<std::uint8_t, 32> bytes{};
        memory.read_bytes(address, std::span<std::uint8_t>(bytes).first(byte_count));

        const std::size_t register_count = (byte_count + 3U) / 4U;
        std::array<std::uint32_t, 8> values{};
        for (std::size_t i = 0; i < byte_count; ++i)
        {
            const unsigned shift = static_cast<unsigned>(24U - (i % 4U) * 8U);
            values[i / 4U] |= static_cast<std::uint32_t>(bytes[i]) << shift;
        }
        for (std::size_t i = 0; i < register_count; ++i)
        {
            state.gpr[(instruction.destination + i) & 31U] = values[i];
        }
        break;
    }

    case Opcode::store_string_word_immediate:
    {
        const std::uint32_t address = instruction.base == 0
            ? 0U : state.gpr[instruction.base];
        const std::size_t byte_count = instruction.string_byte_count == 0
            ? 32U : instruction.string_byte_count;
        std::array<std::uint8_t, 32> bytes{};
        for (std::size_t i = 0; i < byte_count; ++i)
        {
            const std::uint32_t reg =
                (instruction.source + static_cast<std::uint32_t>(i / 4U)) & 31U;
            const unsigned shift = static_cast<unsigned>(24U - (i % 4U) * 8U);
            bytes[i] = static_cast<std::uint8_t>(state.gpr[reg] >> shift);
        }

        // Validate and commit the entire output span as one guest write.
        memory.write_bytes(address, std::span<const std::uint8_t>(bytes).first(byte_count));
        std::copy_n(bytes.begin(), byte_count,
                    pending_history_entry_.string_transfer_bytes.begin());
        break;
    }

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

    case Opcode::store_multiple_word:
    {
        std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        for (std::uint32_t reg = instruction.source; reg < state.gpr.size(); ++reg)
        {
            memory.write32_be(address, state.gpr[reg]);
            address += sizeof(std::uint32_t);
        }
        break;
    }

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

    case Opcode::load_halfword_algebraic:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::int16_t value = std::bit_cast<std::int16_t>(memory.read16_be(address));
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(value));
        break;
    }

    case Opcode::load_single:
    case Opcode::load_single_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint32_t single_bits = memory.read32_be(address);
        const float single_value = std::bit_cast<float>(single_bits);
        const double double_value = static_cast<double>(single_value);
        const std::uint64_t double_bits = std::bit_cast<std::uint64_t>(double_value);
        state.fpr[instruction.fp_register] = double_bits;
        state.fpr_ps1[instruction.fp_register] = double_bits;
        if (instruction.opcode == Opcode::load_single_update)
        {
            state.gpr[instruction.base] = address;
        }
        break;
    }

    case Opcode::load_double:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        state.fpr[instruction.fp_register] = memory.read64_be(address);
        break;
    }

    case Opcode::store_single:
    case Opcode::store_single_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const double double_value = std::bit_cast<double>(state.fpr[instruction.fp_register]);
        const float single_value = static_cast<float>(double_value);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_value);
        memory.write32_be(address, single_bits);
        pending_history_entry_.has_stored_single_value = true;
        pending_history_entry_.stored_single_value = single_bits;
        if (instruction.opcode == Opcode::store_single_update)
        {
            state.gpr[instruction.base] = address;
        }
        break;
    }

    case Opcode::store_double:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint64_t value = state.fpr[instruction.fp_register];
        memory.write64_be(address, value);
        pending_history_entry_.has_stored_double_value = true;
        pending_history_entry_.stored_double_value = value;
        break;
    }

    case Opcode::paired_single_merge10:
    {
        // Snapshot both source lanes first: any of the three FPR operands may
        // alias, including the all-equal form used by Wind Waker.
        struct PairedLanes
        {
            std::uint64_t ps0;
            std::uint64_t ps1;
        };
        const PairedLanes source_a{
            state.fpr[instruction.fp_source_a],
            state.fpr_ps1[instruction.fp_source_a]};
        const PairedLanes source_b{
            state.fpr[instruction.fp_source_b],
            state.fpr_ps1[instruction.fp_source_b]};
        state.fpr[instruction.fp_register] = source_a.ps1;
        state.fpr_ps1[instruction.fp_register] = source_b.ps0;
        break;
    }

    case Opcode::instruction_sync:
        // Instructions are fetched directly from guest memory. There is no
        // pipeline, I-cache, MMU synchronization state, or JIT cache to sync yet.
        break;

    case Opcode::store_halfword:
        memory.write16_be(
            effective_address(state, instruction.base, instruction.immediate),
            static_cast<std::uint16_t>(state.gpr[instruction.destination]));
        break;

    case Opcode::store_halfword_update:
    {
        const std::uint32_t old_base = state.gpr[instruction.base];
        const std::uint32_t source = state.gpr[instruction.destination];
        const std::uint32_t address =
            old_base + static_cast<std::uint32_t>(instruction.immediate);
        const std::uint16_t value = static_cast<std::uint16_t>(source);
        memory.write16_be(address, value);
        state.gpr[instruction.base] = address;
        pending_history_entry_.has_stored_halfword_value = true;
        pending_history_entry_.stored_halfword_value = value;
        break;
    }

    case Opcode::unsupported:
        return StepResult::unsupported_instruction;
    }

    pending_history_entry_.completed = true;
    if (pending_history_entry_.has_string_transfer &&
        !pending_history_entry_.string_transfer_store)
    {
        for (std::uint8_t i = 0; i < pending_history_entry_.string_transfer_register_count; ++i)
        {
            pending_history_entry_.string_transfer_register_values[i] =
                state.gpr[pending_history_entry_.string_transfer_registers[i]];
        }
    }
    if (pending_history_entry_.has_destination)
    {
        pending_history_entry_.destination_value =
            state.gpr[pending_history_entry_.destination_register];
    }
    if (pending_history_entry_.has_carry_result)
    {
        pending_history_entry_.carry_result = (state.xer & xer_carry_mask) != 0;
    }
    if (pending_history_entry_.has_fp_destination)
    {
        pending_history_entry_.fp_destination_value =
            state.fpr[pending_history_entry_.fp_destination_register];
    }
    if (pending_history_entry_.has_paired_fp_state)
    {
        pending_history_entry_.paired_fp_destination_ps0 =
            state.fpr[pending_history_entry_.paired_fp_destination_register];
        pending_history_entry_.paired_fp_destination_ps1 =
            state.fpr_ps1[pending_history_entry_.paired_fp_destination_register];
    }
    append_instruction_history(std::move(pending_history_entry_));
    has_pending_history_entry_ = false;
    state.cia = next_cia;
    return StepResult::executed;
}

RunResult EspressoCore::run(std::size_t max_steps)
{
    RunResult result{};
    result.cia = state.cia;
    const auto include_history = [&]() {
        result.instruction_history = instruction_history_snapshot();
    };

    while (result.steps < max_steps)
    {
        StepResult step_result{};
        try
        {
            step_result = step();
        }
        catch (const GuestMemoryFault& fault)
        {
            if (has_pending_history_entry_)
            {
                append_instruction_history(std::move(pending_history_entry_));
                has_pending_history_entry_ = false;
            }
            result.reason = StopReason::memory_fault;
            result.cia = state.cia;
            result.has_instruction_word = current_instruction_word_fetched_;
            if (result.has_instruction_word)
            {
                result.instruction_word = current_instruction_word_;
            }
            result.detail = describe_memory_fault(
                fault, result.cia, result.has_instruction_word,
                result.instruction_word, state);
            include_history();
            return result;
        }
        catch (const HleExecutionError& error)
        {
            if (has_pending_history_entry_)
            {
                append_instruction_history(std::move(pending_history_entry_));
                has_pending_history_entry_ = false;
            }
            result.reason = StopReason::hle_error;
            result.cia = state.cia;
            result.detail = error.what();
            include_history();
            return result;
        }
        catch (const std::exception& error)
        {
            if (has_pending_history_entry_)
            {
                append_instruction_history(std::move(pending_history_entry_));
                has_pending_history_entry_ = false;
            }
            result.reason = StopReason::memory_fault;
            result.cia = state.cia;
            result.detail = error.what();
            include_history();
            return result;
        }
        if (step_result == StepResult::unsupported_instruction)
        {
            result.reason = StopReason::unsupported_instruction;
            result.cia = state.cia;
            result.instruction_word = memory.read32_be(state.cia);
            result.has_instruction_word = true;
            include_history();
            return result;
        }
        if (step_result == StepResult::unimplemented_hle_call)
        {
            result.reason = StopReason::unimplemented_hle_call;
            result.cia = state.cia;
            result.hle_call = hle.last_unimplemented_call();
            include_history();
            return result;
        }

        ++result.steps;
    }

    result.reason = StopReason::instruction_limit;
    result.cia = state.cia;
    include_history();
    return result;
}

}
