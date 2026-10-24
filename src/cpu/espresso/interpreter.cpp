#include "cpu/espresso/interpreter.hpp"

#include "cpu/espresso/decoder.hpp"

#include <bit>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t xer_summary_overflow_mask = 0x80000000U;
constexpr std::uint32_t xer_overflow_mask = 0x40000000U;
constexpr std::uint32_t xer_carry_mask = 0x20000000U;
constexpr std::uint32_t espresso_data_cache_block_size = 32U;
constexpr std::uint64_t espresso_time_base_frequency = 62156250ULL;
constexpr std::uint8_t cr_less_than = 0x8U;
constexpr std::uint8_t cr_greater_than = 0x4U;
constexpr std::uint8_t cr_equal = 0x2U;
constexpr std::uint8_t fp_unordered = 0x1U;
constexpr std::uint64_t binary64_exponent_mask = 0x7FF0000000000000ULL;
constexpr std::uint64_t binary64_fraction_mask = 0x000FFFFFFFFFFFFFULL;
constexpr std::uint64_t binary64_quiet_nan_bit = 0x0008000000000000ULL;

[[nodiscard]] std::uint64_t elapsed_time_base_ticks(
    std::chrono::steady_clock::time_point origin) noexcept
{
    const auto elapsed = std::chrono::steady_clock::now() - origin;
    if (elapsed <= std::chrono::steady_clock::duration::zero())
    {
        return 0U;
    }

    const auto whole_seconds = std::chrono::duration_cast<std::chrono::seconds>(elapsed);
    const auto fractional_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(
        elapsed - whole_seconds).count();
    const auto whole_second_ticks =
        static_cast<std::uint64_t>(whole_seconds.count()) * espresso_time_base_frequency;
    const auto fractional_ticks =
        static_cast<std::uint64_t>(fractional_nanoseconds) *
        espresso_time_base_frequency / 1000000000ULL;
    // Unsigned addition naturally models the architectural 64-bit TB wrap.
    return whole_second_ticks + fractional_ticks;
}

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

[[nodiscard]] std::uint64_t encode_fctiw_result(
    std::uint32_t integer_bits,
    std::uint64_t source_raw) noexcept
{
    std::uint64_t encoded = 0xFFF8000000000000ULL | integer_bits;
    if (integer_bits == 0U && (source_raw & binary64_sign_mask) != 0U)
    {
        encoded |= 0x0000000100000000ULL;
    }
    return encoded;
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
    fpscr::vxidi_mask | fpscr::vxisi_mask | fpscr::vxcvi_mask |
    fpscr::vxvc_mask | fpscr::vxsqrt_mask | fpscr::vxsoft_mask;

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
        ((state.fpscr & fpscr::ox_mask) != 0U &&
         (state.fpscr & fpscr::oe_mask) != 0U) ||
        ((state.fpscr & fpscr::ux_mask) != 0U &&
         (state.fpscr & fpscr::ue_mask) != 0U) ||
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

[[nodiscard]] double binary64_addition_error(
    double a,
    double b,
    double rounded_result) noexcept
{
    // Error-free TwoSum decomposition for finite normal/zero operands/results.
    const double virtual_b = rounded_result - a;
    const double virtual_a = rounded_result - virtual_b;
    const double roundoff_b = b - virtual_b;
    const double roundoff_a = a - virtual_a;
    return roundoff_a + roundoff_b;
}

[[nodiscard]] double binary64_multiplication_error(
    double a,
    double c,
    double rounded_result) noexcept
{
    // One correctly rounded fused operation exposes the product's rounding residual.
    return std::fma(a, c, -rounded_result);
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

void set_xer_carry(CpuState& state, bool carry) noexcept
{
    if (carry)
    {
        state.xer |= xer_carry_mask;
    }
    else
    {
        state.xer &= ~xer_carry_mask;
    }
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
            instruction.opcode == Opcode::load_word_reserve_indexed ||
            instruction.opcode == Opcode::load_word_indexed_update ||
            instruction.opcode == Opcode::load_word_byte_reverse_indexed ||
            instruction.opcode == Opcode::load_single_indexed ||
            instruction.opcode == Opcode::load_single_indexed_update ||
            instruction.opcode == Opcode::load_double_indexed ||
            instruction.opcode == Opcode::load_double_indexed_update ||
            instruction.opcode == Opcode::load_byte_zero_indexed ||
            instruction.opcode == Opcode::load_byte_zero_indexed_update ||
            instruction.opcode == Opcode::load_halfword_zero_indexed ||
            instruction.opcode == Opcode::load_halfword_zero_indexed_update ||
            instruction.opcode == Opcode::load_halfword_byte_reverse_indexed ||
            instruction.opcode == Opcode::load_halfword_algebraic_indexed ||
            instruction.opcode == Opcode::load_halfword_algebraic_indexed_update ||
            instruction.opcode == Opcode::store_word_indexed ||
            instruction.opcode == Opcode::store_word_conditional_indexed ||
            instruction.opcode == Opcode::store_word_indexed_update ||
            instruction.opcode == Opcode::store_word_byte_reverse_indexed ||
            instruction.opcode == Opcode::store_byte_indexed ||
            instruction.opcode == Opcode::store_byte_indexed_update ||
            instruction.opcode == Opcode::store_halfword_indexed ||
            instruction.opcode == Opcode::store_halfword_indexed_update ||
            instruction.opcode == Opcode::store_halfword_byte_reverse_indexed ||
            instruction.opcode == Opcode::store_single_indexed ||
            instruction.opcode == Opcode::store_single_indexed_update ||
            instruction.opcode == Opcode::store_double_indexed ||
            instruction.opcode == Opcode::store_double_indexed_update ||
            instruction.opcode == Opcode::store_floating_as_integer_word_indexed ||
            instruction.opcode == Opcode::load_string_word_indexed ||
            instruction.opcode == Opcode::store_string_word_indexed ||
            instruction.opcode == Opcode::data_cache_block_zero ||
            instruction.opcode == Opcode::data_cache_block_store ||
            instruction.opcode == Opcode::data_cache_block_flush;
        const bool memory_instruction = indexed ||
            instruction.opcode == Opcode::load_word_zero ||
            instruction.opcode == Opcode::load_word_update ||
            instruction.opcode == Opcode::load_word_indexed_update ||
            instruction.opcode == Opcode::store_word ||
            instruction.opcode == Opcode::store_word_update ||
            instruction.opcode == Opcode::load_byte_zero ||
            instruction.opcode == Opcode::load_byte_update ||
            instruction.opcode == Opcode::store_byte ||
            instruction.opcode == Opcode::store_byte_update ||
            instruction.opcode == Opcode::load_halfword_zero ||
            instruction.opcode == Opcode::load_halfword_zero_update ||
            instruction.opcode == Opcode::paired_single_quantized_load ||
            instruction.opcode == Opcode::paired_single_quantized_store ||
            instruction.opcode == Opcode::load_halfword_algebraic ||
            instruction.opcode == Opcode::store_halfword ||
            instruction.opcode == Opcode::store_halfword_update ||
            instruction.opcode == Opcode::store_halfword_indexed ||
            instruction.opcode == Opcode::store_floating_as_integer_word_indexed ||
            instruction.opcode == Opcode::load_multiple_word ||
            instruction.opcode == Opcode::load_string_word_immediate ||
            instruction.opcode == Opcode::load_string_word_indexed ||
            instruction.opcode == Opcode::store_string_word_immediate ||
            instruction.opcode == Opcode::store_string_word_indexed ||
            instruction.opcode == Opcode::store_multiple_word ||
            instruction.opcode == Opcode::load_single ||
            instruction.opcode == Opcode::load_single_update ||
            instruction.opcode == Opcode::load_single_indexed ||
            instruction.opcode == Opcode::load_single_indexed_update ||
            instruction.opcode == Opcode::load_double_indexed ||
            instruction.opcode == Opcode::load_double_indexed_update ||
            instruction.opcode == Opcode::store_single ||
            instruction.opcode == Opcode::store_single_update ||
            instruction.opcode == Opcode::store_single_indexed ||
            instruction.opcode == Opcode::store_single_indexed_update ||
            instruction.opcode == Opcode::store_double_indexed ||
            instruction.opcode == Opcode::store_double_indexed_update ||
            instruction.opcode == Opcode::store_double ||
            instruction.opcode == Opcode::store_double_update ||
            instruction.opcode == Opcode::load_double ||
            instruction.opcode == Opcode::load_double_update ||
            instruction.opcode == Opcode::data_cache_block_zero ||
            instruction.opcode == Opcode::data_cache_block_store ||
            instruction.opcode == Opcode::data_cache_block_flush;
        if (memory_instruction)
        {
            detail << "; rA=" << std::dec << static_cast<unsigned>(instruction.base)
                   << " (0x" << std::hex << std::setw(8)
                   << (instruction.base == 0 ? 0U : state.gpr[instruction.base]) << ')';
            if (indexed)
            {
                const std::uint8_t index_register =
                    instruction.opcode == Opcode::store_string_word_indexed
                        ? instruction.destination
                        : instruction.source;
                detail << ", rB=" << std::dec << static_cast<unsigned>(index_register)
                       << " (0x" << std::hex << std::setw(8)
                       << state.gpr[index_register] << ')';
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
    case Opcode::multiply_low_word_overflow: return "mullwo";
    case Opcode::multiply_high_word_unsigned: return "mulhwu";
    case Opcode::multiply_high_word_signed: return "mulhw";
    case Opcode::divide_word_unsigned: return "divwu";
    case Opcode::divide_word_unsigned_overflow: return "divwuo";
    case Opcode::divide_word_signed: return "divw";
    case Opcode::divide_word_signed_overflow: return "divwo";
    case Opcode::unsupported: return "unsupported";
    case Opcode::addi: return "addi";
    case Opcode::addis: return "addis";
    case Opcode::add_immediate_carry: return "addic";
    case Opcode::add_carrying: return "addc";
    case Opcode::add_carrying_overflow: return "addco";
    case Opcode::add_to_zero_extended: return "addze";
    case Opcode::add_to_zero_extended_overflow: return "addzeo";
    case Opcode::add_extended: return "adde";
    case Opcode::add_extended_overflow: return "addeo";
    case Opcode::add_to_minus_one_extended: return "addme";
    case Opcode::add_to_minus_one_extended_overflow: return "addmeo";
    case Opcode::subtract_from_immediate_carry: return "subfic";
    case Opcode::add: return "add";
    case Opcode::add_overflow: return "addo";
    case Opcode::subtract_from: return "subf";
    case Opcode::subtract_from_overflow: return "subfo";
    case Opcode::subtract_from_carrying: return "subfc";
    case Opcode::subtract_from_carrying_overflow: return "subfco";
    case Opcode::subtract_from_extended: return "subfe";
    case Opcode::subtract_from_extended_overflow: return "subfeo";
    case Opcode::subtract_from_zero_extended: return "subfze";
    case Opcode::subtract_from_zero_extended_overflow: return "subfzeo";
    case Opcode::subtract_from_minus_one_extended: return "subfme";
    case Opcode::subtract_from_minus_one_extended_overflow: return "subfmeo";
    case Opcode::negate: return "neg";
    case Opcode::negate_overflow: return "nego";
    case Opcode::extend_sign_byte: return "extsb";
    case Opcode::extend_sign_halfword: return "extsh";
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
    case Opcode::bitwise_nand: return "nand";
    case Opcode::bitwise_nor: return "nor";
    case Opcode::bitwise_or_complement: return "orc";
    case Opcode::and_immediate_record: return "andi.";
    case Opcode::and_immediate_shifted_record: return "andis.";
    case Opcode::rotate_left_word_and_mask: return "rlwinm";
    case Opcode::rotate_left_word_and_mask_register: return "rlwnm";
    case Opcode::rotate_left_word_and_mask_insert: return "rlwimi";
    case Opcode::arithmetic_shift_right_immediate: return "srawi";
    case Opcode::arithmetic_shift_right_word: return "sraw";
    case Opcode::shift_left_word: return "slw";
    case Opcode::shift_right_word: return "srw";
    case Opcode::branch: return "b";
    case Opcode::instruction_sync: return "isync";
    case Opcode::enforce_in_order_io: return "eieio";
    case Opcode::instruction_cache_block_invalidate: return "icbi";
    case Opcode::data_cache_block_touch: return "dcbt";
    case Opcode::data_cache_block_touch_for_store: return "dcbtst";
    case Opcode::data_cache_block_store: return "dcbst";
    case Opcode::data_cache_block_flush: return "dcbf";
    case Opcode::data_cache_block_zero: return "dcbz";
    case Opcode::floating_compare_unordered: return "fcmpu";
    case Opcode::floating_compare_ordered: return "fcmpo";
    case Opcode::floating_divide_single: return "fdivs";
    case Opcode::floating_divide_double: return "fdiv";
    case Opcode::floating_add_single: return "fadds";
    case Opcode::floating_subtract_single: return "fsubs";
    case Opcode::floating_add_double: return "fadd";
    case Opcode::floating_multiply_single: return "fmuls";
    case Opcode::floating_multiply_double: return "fmul";
    case Opcode::floating_select: return "fsel";
    case Opcode::floating_multiply_add_single: return "fmadds";
    case Opcode::floating_multiply_subtract_single: return "fmsubs";
    case Opcode::floating_negative_multiply_subtract_single: return "fnmsubs";
    case Opcode::floating_negative_multiply_add_single: return "fnmadds";
    case Opcode::floating_multiply_add_double: return "fmadd";
    case Opcode::floating_multiply_subtract_double: return "fmsub";
    case Opcode::floating_negative_multiply_add_double: return "fnmadd";
    case Opcode::floating_negative_multiply_subtract_double: return "fnmsub";
    case Opcode::floating_reciprocal_estimate_single: return "fres";
    case Opcode::floating_reciprocal_square_root_estimate: return "frsqrte";
    case Opcode::floating_move_register: return "fmr";
    case Opcode::floating_negate: return "fneg";
    case Opcode::floating_absolute_value: return "fabs";
    case Opcode::floating_negative_absolute_value: return "fnabs";
    case Opcode::floating_move_from_fpscr: return "mffs";
    case Opcode::floating_move_to_fpscr_bit_zero: return "mtfsb0";
    case Opcode::floating_move_to_fpscr_bit_one: return "mtfsb1";
    case Opcode::floating_move_to_fpscr_field_immediate: return "mtfsfi";
    case Opcode::floating_move_to_fpscr_fields: return "mtfsf";
    case Opcode::floating_convert_to_integer_word: return "fctiw";
    case Opcode::floating_convert_to_integer_word_zero: return "fctiwz";
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
    case Opcode::load_string_word_indexed: return "lswx";
    case Opcode::store_string_word_immediate: return "stswi";
    case Opcode::store_string_word_indexed: return "stswx";
    case Opcode::load_word_update: return "lwzu";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::load_word_reserve_indexed: return "lwarx";
    case Opcode::load_word_indexed_update: return "lwzux";
    case Opcode::load_word_byte_reverse_indexed: return "lwbrx";
    case Opcode::load_byte_zero_indexed: return "lbzx";
    case Opcode::load_byte_zero_indexed_update: return "lbzux";
    case Opcode::load_halfword_zero_indexed: return "lhzx";
    case Opcode::load_halfword_zero_indexed_update: return "lhzux";
    case Opcode::load_halfword_byte_reverse_indexed: return "lhbrx";
    case Opcode::load_halfword_algebraic_indexed: return "lhax";
    case Opcode::load_halfword_algebraic_indexed_update: return "lhaux";
    case Opcode::store_word: return "stw";
    case Opcode::store_multiple_word: return "stmw";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_word_conditional_indexed: return "stwcx.";
    case Opcode::store_word_indexed_update: return "stwux";
    case Opcode::store_word_byte_reverse_indexed: return "stwbrx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::store_byte_indexed_update: return "stbux";
    case Opcode::store_halfword_indexed: return "sthx";
    case Opcode::store_halfword_indexed_update: return "sthux";
    case Opcode::store_halfword_byte_reverse_indexed: return "sthbrx";
    case Opcode::store_floating_as_integer_word_indexed: return "stfiwx";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_byte_update: return "lbzu";
    case Opcode::store_byte: return "stb";
    case Opcode::store_byte_update: return "stbu";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_zero_update: return "lhzu";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::store_halfword: return "sth";
    case Opcode::store_halfword_update: return "sthu";
    case Opcode::store_word_update: return "stwu";
    case Opcode::move_from_link_register: return "mflr";
    case Opcode::move_to_link_register: return "mtlr";
    case Opcode::conditional_branch_to_link_register: return "bclr";
    case Opcode::move_from_count_register: return "mfctr";
    case Opcode::move_to_count_register: return "mtctr";
    case Opcode::move_from_xer: return "mfxer";
    case Opcode::move_to_xer: return "mtxer";
    case Opcode::move_from_gqr: return "mfgqr";
    case Opcode::move_to_gqr: return "mtgqr";
    case Opcode::move_from_time_base_lower: return "mftb";
    case Opcode::move_from_time_base_upper: return "mftbu";
    case Opcode::conditional_branch_to_count_register: return "bcctr";
    case Opcode::load_single: return "lfs";
    case Opcode::load_single_update: return "lfsu";
    case Opcode::load_single_indexed: return "lfsx";
    case Opcode::load_single_indexed_update: return "lfsux";
    case Opcode::load_double: return "lfd";
    case Opcode::load_double_update: return "lfdu";
    case Opcode::load_double_indexed: return "lfdx";
    case Opcode::load_double_indexed_update: return "lfdux";
    case Opcode::store_single: return "stfs";
    case Opcode::store_single_update: return "stfsu";
    case Opcode::store_single_indexed: return "stfsx";
    case Opcode::store_single_indexed_update: return "stfsux";
    case Opcode::store_double: return "stfd";
    case Opcode::store_double_indexed: return "stfdx";
    case Opcode::store_double_indexed_update: return "stfdux";
    case Opcode::store_double_update: return "stfdu";
    case Opcode::paired_single_merge00: return "ps_merge00";
    case Opcode::paired_single_merge01: return "ps_merge01";
    case Opcode::paired_single_merge10: return "ps_merge10";
    case Opcode::paired_single_merge11: return "ps_merge11";
    case Opcode::paired_single_add: return "ps_add";
    case Opcode::paired_single_subtract: return "ps_sub";
    case Opcode::paired_single_multiply: return "ps_mul";
    case Opcode::paired_single_multiply_add: return "ps_madd";
    case Opcode::paired_single_multiply_subtract: return "ps_msub";
    case Opcode::paired_single_multiply_add_scalar0: return "ps_madds0";
    case Opcode::paired_single_multiply_add_scalar1: return "ps_madds1";
    case Opcode::paired_single_negative_multiply_add: return "ps_nmadd";
    case Opcode::paired_single_negative_multiply_subtract: return "ps_nmsub";
    case Opcode::paired_single_multiply_scalar0: return "ps_muls0";
    case Opcode::paired_single_multiply_scalar1: return "ps_muls1";
    case Opcode::paired_single_sum0: return "ps_sum0";
    case Opcode::paired_single_sum1: return "ps_sum1";
    case Opcode::paired_single_divide: return "ps_div";
    case Opcode::paired_single_compare_unordered0: return "ps_cmpu0";
    case Opcode::paired_single_compare_unordered1: return "ps_cmpu1";
    case Opcode::paired_single_compare_ordered0: return "ps_cmpo0";
    case Opcode::paired_single_compare_ordered1: return "ps_cmpo1";
    case Opcode::paired_single_reciprocal_estimate: return "ps_res";
    case Opcode::paired_single_reciprocal_square_root_estimate: return "ps_rsqrte";
    case Opcode::paired_single_move_register: return "ps_mr";
    case Opcode::paired_single_negate: return "ps_neg";
    case Opcode::paired_single_absolute_value: return "ps_abs";
    case Opcode::paired_single_negative_absolute_value: return "ps_nabs";
    case Opcode::paired_single_select: return "ps_sel";
    case Opcode::paired_single_quantized_load: return "psq_l";
    case Opcode::paired_single_quantized_store: return "psq_st";
    case Opcode::condition_bit_and: return "crand";
    case Opcode::condition_bit_and_complement: return "crandc";
    case Opcode::condition_bit_or: return "cror";
    case Opcode::condition_bit_or_complement: return "crorc";
    case Opcode::condition_bit_nor: return "crnor";
    case Opcode::condition_bit_xor: return "crxor";
    case Opcode::condition_bit_nand: return "crnand";
    case Opcode::condition_bit_equivalence: return "creqv";
    case Opcode::move_condition_register_field: return "mcrf";
    case Opcode::move_condition_register_from_fpscr: return "mcrfs";
    case Opcode::move_from_condition_register: return "mfcr";
    case Opcode::move_to_condition_register_fields: return "mtcrf";
    case Opcode::move_condition_register_from_xer: return "mcrxr";
    case Opcode::memory_synchronize: return "sync";
    case Opcode::memory_synchronize_lightweight: return "lwsync";
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
         instruction.opcode == Opcode::multiply_low_word_overflow ||
         instruction.opcode == Opcode::multiply_high_word_unsigned ||
         instruction.opcode == Opcode::multiply_high_word_signed ||
         instruction.opcode == Opcode::divide_word_unsigned ||
         instruction.opcode == Opcode::divide_word_signed ||
         instruction.opcode == Opcode::add_carrying ||
         instruction.opcode == Opcode::subtract_from ||
         instruction.opcode == Opcode::add_extended ||
         instruction.opcode == Opcode::add_to_minus_one_extended ||
         instruction.opcode == Opcode::add_to_zero_extended ||
         instruction.opcode == Opcode::subtract_from_extended ||
         instruction.opcode == Opcode::subtract_from_minus_one_extended ||
         instruction.opcode == Opcode::subtract_from_zero_extended ||
         instruction.opcode == Opcode::count_leading_zeros ||
         instruction.opcode == Opcode::add_immediate_carry ||
         instruction.opcode == Opcode::negate ||
         instruction.opcode == Opcode::extend_sign_byte ||
         instruction.opcode == Opcode::extend_sign_halfword ||
         instruction.opcode == Opcode::rotate_left_word_and_mask_insert ||
         instruction.opcode == Opcode::rotate_left_word_and_mask ||
         instruction.opcode == Opcode::rotate_left_word_and_mask_register ||
         instruction.opcode == Opcode::shift_right_word ||
         instruction.opcode == Opcode::arithmetic_shift_right_word ||
         instruction.opcode == Opcode::bitwise_nand ||
         instruction.opcode == Opcode::bitwise_nor ||
         instruction.opcode == Opcode::bitwise_or_complement ||
         instruction.opcode == Opcode::bitwise_or ||
         instruction.opcode == Opcode::bitwise_and ||
         instruction.opcode == Opcode::bitwise_xor) && instruction.record)
    {
        entry.opcode_name += '.';
    }
    if ((instruction.opcode == Opcode::add_overflow ||
         instruction.opcode == Opcode::subtract_from_overflow ||
         instruction.opcode == Opcode::subtract_from_carrying_overflow ||
         instruction.opcode == Opcode::subtract_from_extended_overflow ||
         instruction.opcode == Opcode::add_carrying_overflow ||
         instruction.opcode == Opcode::add_extended_overflow ||
         instruction.opcode == Opcode::add_to_zero_extended_overflow ||
         instruction.opcode == Opcode::add_to_minus_one_extended_overflow ||
         instruction.opcode == Opcode::subtract_from_zero_extended_overflow ||
         instruction.opcode == Opcode::subtract_from_minus_one_extended_overflow ||
         instruction.opcode == Opcode::negate_overflow ||
         instruction.opcode == Opcode::divide_word_signed_overflow ||
         instruction.opcode == Opcode::divide_word_unsigned_overflow) && instruction.record)
    {
        entry.opcode_name += '.';
    }
    if (instruction.opcode == Opcode::multiply_low_immediate ||
        instruction.opcode == Opcode::add_immediate_carry)
    {
        entry.has_immediate = true;
        entry.immediate = instruction.immediate;
    }
    else if (instruction.opcode == Opcode::or_immediate_shifted ||
             instruction.opcode == Opcode::xor_immediate ||
             instruction.opcode == Opcode::xor_immediate_shifted ||
             instruction.opcode == Opcode::and_immediate_shifted_record)
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
    else if (instruction.opcode == Opcode::rotate_left_word_and_mask_register)
    {
        entry.has_rotate_fields = true;
        entry.rotate_shift = static_cast<std::uint8_t>(state.gpr[instruction.base] & 0x1FU);
        entry.rotate_mask_begin = instruction.mask_begin;
        entry.rotate_mask_end = instruction.mask_end;
    }

    const Opcode opcode = instruction.opcode;
    if (opcode == Opcode::paired_single_merge00 ||
        opcode == Opcode::paired_single_merge01 ||
        opcode == Opcode::paired_single_merge10 ||
        opcode == Opcode::paired_single_merge11 ||
        opcode == Opcode::paired_single_reciprocal_estimate ||
        opcode == Opcode::paired_single_reciprocal_square_root_estimate ||
        opcode == Opcode::paired_single_add ||
        opcode == Opcode::paired_single_subtract ||
        opcode == Opcode::paired_single_divide ||
        opcode == Opcode::paired_single_move_register ||
        opcode == Opcode::paired_single_negate ||
        opcode == Opcode::paired_single_absolute_value ||
        opcode == Opcode::paired_single_negative_absolute_value)
    {
        entry.has_paired_fp_state = true;
        entry.paired_fp_unary = opcode == Opcode::paired_single_move_register ||
            opcode == Opcode::paired_single_reciprocal_estimate ||
            opcode == Opcode::paired_single_reciprocal_square_root_estimate ||
            opcode == Opcode::paired_single_negate ||
            opcode == Opcode::paired_single_absolute_value ||
            opcode == Opcode::paired_single_negative_absolute_value;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = entry.paired_fp_unary
            ? instruction.fp_source_b
            : instruction.fp_source_a;
        entry.paired_fp_source_b_register = instruction.fp_source_b;
        entry.paired_fp_source_a_ps0 = state.fpr[entry.paired_fp_source_a_register];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[entry.paired_fp_source_a_register];
        entry.paired_fp_source_b_ps0 = state.fpr[instruction.fp_source_b];
        entry.paired_fp_source_b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        return entry;
    }
    if (opcode == Opcode::paired_single_multiply ||
        opcode == Opcode::paired_single_multiply_scalar0 ||
        opcode == Opcode::paired_single_multiply_scalar1)
    {
        entry.has_paired_fp_state = true;
        entry.paired_fp_scalar_multiply =
            opcode == Opcode::paired_single_multiply_scalar0 ||
            opcode == Opcode::paired_single_multiply_scalar1;
        entry.paired_fp_scalar_source_ps1 =
            opcode == Opcode::paired_single_multiply_scalar1;
        entry.paired_fp_vector_multiply = opcode == Opcode::paired_single_multiply;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = instruction.fp_source_a;
        entry.paired_fp_source_b_register = instruction.fp_source_c;
        entry.paired_fp_source_a_ps0 = state.fpr[instruction.fp_source_a];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        entry.paired_fp_source_b_ps0 = state.fpr[instruction.fp_source_c];
        entry.paired_fp_source_b_ps1 = state.fpr_ps1[instruction.fp_source_c];
        return entry;
    }
    if (opcode == Opcode::paired_single_multiply_add ||
        opcode == Opcode::paired_single_multiply_subtract ||
        opcode == Opcode::paired_single_multiply_add_scalar0 ||
        opcode == Opcode::paired_single_multiply_add_scalar1 ||
        opcode == Opcode::paired_single_negative_multiply_add ||
        opcode == Opcode::paired_single_negative_multiply_subtract)
    {
        entry.has_paired_fp_state = true;
        entry.paired_fp_multiply_add = opcode == Opcode::paired_single_multiply_add;
        entry.paired_fp_multiply_subtract =
            opcode == Opcode::paired_single_multiply_subtract;
        entry.paired_fp_multiply_add_scalar0 =
            opcode == Opcode::paired_single_multiply_add_scalar0;
        entry.paired_fp_multiply_add_scalar1 =
            opcode == Opcode::paired_single_multiply_add_scalar1;
        entry.paired_fp_negative_multiply_add =
            opcode == Opcode::paired_single_negative_multiply_add;
        entry.paired_fp_negative_multiply_subtract =
            opcode == Opcode::paired_single_negative_multiply_subtract;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = instruction.fp_source_a;
        entry.paired_fp_source_b_register = instruction.fp_source_b;
        entry.paired_fp_source_c_register = instruction.fp_source_c;
        entry.paired_fp_source_a_ps0 = state.fpr[instruction.fp_source_a];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        entry.paired_fp_source_b_ps0 = state.fpr[instruction.fp_source_b];
        entry.paired_fp_source_b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        entry.paired_fp_source_c_ps0 = state.fpr[instruction.fp_source_c];
        entry.paired_fp_source_c_ps1 = state.fpr_ps1[instruction.fp_source_c];
        return entry;
    }
    if (opcode == Opcode::paired_single_sum0 || opcode == Opcode::paired_single_sum1)
    {
        entry.has_paired_fp_state = true;
        entry.paired_fp_sum0 = opcode == Opcode::paired_single_sum0;
        entry.paired_fp_sum1 = opcode == Opcode::paired_single_sum1;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = instruction.fp_source_a;
        entry.paired_fp_source_b_register = instruction.fp_source_b;
        entry.paired_fp_source_c_register = instruction.fp_source_c;
        entry.paired_fp_source_a_ps0 = state.fpr[instruction.fp_source_a];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        entry.paired_fp_source_b_ps0 = state.fpr[instruction.fp_source_b];
        entry.paired_fp_source_b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        entry.paired_fp_source_c_ps0 = state.fpr[instruction.fp_source_c];
        entry.paired_fp_source_c_ps1 = state.fpr_ps1[instruction.fp_source_c];
        return entry;
    }
    if (opcode == Opcode::paired_single_select)
    {
        entry.has_paired_fp_select = true;
        entry.paired_fp_select_destination = instruction.fp_register;
        entry.paired_fp_select_a = instruction.fp_source_a;
        entry.paired_fp_select_b = instruction.fp_source_b;
        entry.paired_fp_select_c = instruction.fp_source_c;
        entry.paired_fp_select_a_ps0 = state.fpr[instruction.fp_source_a];
        entry.paired_fp_select_a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        entry.paired_fp_select_b_ps0 = state.fpr[instruction.fp_source_b];
        entry.paired_fp_select_b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        entry.paired_fp_select_c_ps0 = state.fpr[instruction.fp_source_c];
        entry.paired_fp_select_c_ps1 = state.fpr_ps1[instruction.fp_source_c];
        return entry;
    }
    if (opcode == Opcode::condition_bit_and ||
        opcode == Opcode::condition_bit_and_complement ||
        opcode == Opcode::condition_bit_or ||
        opcode == Opcode::condition_bit_or_complement ||
        opcode == Opcode::condition_bit_nor ||
        opcode == Opcode::condition_bit_xor ||
        opcode == Opcode::condition_bit_nand ||
        opcode == Opcode::condition_bit_equivalence)
    {
        entry.has_cr_bit_operation = true;
        entry.cr_bit_destination = instruction.cr_destination_bit;
        entry.cr_bit_source_a = instruction.cr_source_a_bit;
        entry.cr_bit_source_b = instruction.cr_source_b_bit;
        entry.cr_bit_source_a_value = read_cr_bit(state, instruction.cr_source_a_bit);
        entry.cr_bit_source_b_value = read_cr_bit(state, instruction.cr_source_b_bit);
        return entry;
    }
    if (opcode == Opcode::move_condition_register_field)
    {
        entry.has_cr_field_move = true;
        entry.cr_field_destination = instruction.cr_field;
        entry.cr_field_source = instruction.cr_field_source;
        const std::uint32_t shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.cr_field_source);
        entry.cr_field_value = static_cast<std::uint8_t>((state.cr >> shift) & 0xFU);
        return entry;
    }
    if (opcode == Opcode::move_condition_register_from_fpscr)
    {
        entry.has_fpscr_cr_move = true;
        entry.fpscr_cr_destination = instruction.cr_field;
        entry.fpscr_cr_source = instruction.fpscr_field;
        const std::uint32_t shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.fpscr_field);
        const std::uint32_t valid_field_mask =
            instruction.fpscr_field == 5U ? 0x7U : 0xFU;
        entry.fpscr_cr_value = static_cast<std::uint8_t>(
            (state.fpscr >> shift) & valid_field_mask);
        entry.fpscr_before_cr_move = state.fpscr;
        return entry;
    }
    if (opcode == Opcode::move_to_condition_register_fields)
    {
        add_history_source(entry, state, instruction.destination);
        return entry;
    }
    if (opcode == Opcode::paired_single_quantized_load)
    {
        entry.has_effective_address = true;
        entry.effective_address = effective_address(
            state, instruction.base, instruction.immediate);
        if (instruction.base != 0U)
        {
            add_history_source(entry, state, instruction.base);
        }
        entry.has_paired_fp_state = true;
        entry.paired_fp_destination_register = instruction.fp_register;
        return entry;
    }
    if (opcode == Opcode::paired_single_quantized_store)
    {
        entry.has_effective_address = true;
        entry.effective_address = effective_address(
            state, instruction.base, instruction.immediate);
        if (instruction.base != 0U)
        {
            add_history_source(entry, state, instruction.base);
        }
        entry.has_paired_fp_state = true;
        entry.paired_fp_quantized_store = true;
        entry.paired_fp_destination_register = instruction.fp_register;
        entry.paired_fp_source_a_register = instruction.fp_register;
        entry.paired_fp_source_a_ps0 = state.fpr[instruction.fp_register];
        entry.paired_fp_source_a_ps1 = state.fpr_ps1[instruction.fp_register];
        entry.paired_fp_store_gqr = instruction.quantized_register;
        entry.paired_fp_store_type = static_cast<std::uint8_t>(
            state.gqr[instruction.quantized_register] & 0x7U);
        entry.paired_fp_store_w = instruction.paired_single_w;
        return entry;
    }
    if (opcode == Opcode::move_condition_register_from_xer)
    {
        entry.has_xer_cr_move = true;
        entry.xer_cr_field = instruction.cr_field;
        entry.xer_cr_value = static_cast<std::uint8_t>(state.xer >> 28U);
        return entry;
    }
    if (opcode == Opcode::floating_move_register || opcode == Opcode::floating_negate ||
        opcode == Opcode::floating_absolute_value ||
        opcode == Opcode::floating_negative_absolute_value)
    {
        entry.has_fp_move = true;
        entry.fp_move_destination = instruction.fp_register;
        entry.fp_move_source = instruction.fp_source_b;
        entry.fp_move_source_value = state.fpr[instruction.fp_source_b];
        entry.fp_move_destination_ps1 = state.fpr_ps1[instruction.fp_register];
        return entry;
    }
    if (opcode == Opcode::floating_move_from_fpscr)
    {
        entry.has_fp_status_move = true;
        entry.fp_status_move_destination = instruction.fp_register;
        entry.fp_status_move_source = state.fpscr;
        entry.fp_status_move_destination_ps1 = state.fpr_ps1[instruction.fp_register];
        return entry;
    }
    if (opcode == Opcode::floating_move_to_fpscr_bit_zero ||
        opcode == Opcode::floating_move_to_fpscr_bit_one)
    {
        entry.has_fpscr_bit_operation = true;
        entry.fpscr_bit_set = opcode == Opcode::floating_move_to_fpscr_bit_one;
        entry.fpscr_bit_number = instruction.fpscr_bit;
        entry.fpscr_before_bit_operation = state.fpscr;
        return entry;
    }
    if (opcode == Opcode::floating_move_to_fpscr_field_immediate)
    {
        entry.has_fpscr_field_write = true;
        entry.fpscr_field_number = instruction.fpscr_field;
        entry.fpscr_field_immediate = instruction.fpscr_immediate;
        entry.fpscr_before_field_write = state.fpscr;
        return entry;
    }
    if (opcode == Opcode::floating_move_to_fpscr_fields)
    {
        entry.has_fpscr_masked_write = true;
        entry.fpscr_masked_write_source = instruction.fp_source_b;
        entry.fpscr_masked_write_mask = instruction.fpscr_field_mask;
        entry.fpscr_masked_write_source_raw = state.fpr[instruction.fp_source_b];
        entry.fpscr_before_masked_write = state.fpscr;
        return entry;
    }
    if (opcode == Opcode::floating_select)
    {
        entry.has_fp_select = true;
        entry.fp_select_destination = instruction.fp_register;
        entry.fp_select_a = instruction.fp_source_a;
        entry.fp_select_b = instruction.fp_source_b;
        entry.fp_select_c = instruction.fp_source_c;
        entry.fp_select_a_raw = state.fpr[instruction.fp_source_a];
        entry.fp_select_b_raw = state.fpr[instruction.fp_source_b];
        entry.fp_select_c_raw = state.fpr[instruction.fp_source_c];
        return entry;
    }
    if (opcode == Opcode::floating_convert_to_integer_word ||
        opcode == Opcode::floating_convert_to_integer_word_zero)
    {
        entry.has_fp_integer_conversion = true;
        entry.fp_integer_conversion_source = instruction.fp_source_b;
        entry.fp_integer_conversion_source_raw = state.fpr[instruction.fp_source_b];
        entry.fp_integer_conversion_destination = instruction.fp_register;
        return entry;
    }
    if (opcode == Opcode::floating_reciprocal_estimate_single ||
        opcode == Opcode::floating_reciprocal_square_root_estimate)
    {
        entry.has_fp_source = true;
        entry.fp_source_register = instruction.fp_source_b;
        entry.fp_source_value = state.fpr[instruction.fp_source_b];
        entry.has_fp_destination = true;
        entry.fp_destination_register = instruction.fp_register;
        return entry;
    }
    if (opcode == Opcode::floating_compare_unordered ||
        opcode == Opcode::floating_compare_ordered ||
        opcode == Opcode::paired_single_compare_unordered0 ||
        opcode == Opcode::paired_single_compare_unordered1 ||
        opcode == Opcode::paired_single_compare_ordered0 ||
        opcode == Opcode::paired_single_compare_ordered1)
    {
        entry.has_fp_compare = true;
        entry.fp_compare_cr_field = instruction.cr_field;
        entry.fp_compare_a_register = instruction.fp_compare_a;
        entry.fp_compare_b_register = instruction.fp_compare_b;
        entry.fp_compare_a_raw = instruction.fp_compare_uses_ps1
            ? state.fpr_ps1[instruction.fp_compare_a]
            : state.fpr[instruction.fp_compare_a];
        entry.fp_compare_b_raw = instruction.fp_compare_uses_ps1
            ? state.fpr_ps1[instruction.fp_compare_b]
            : state.fpr[instruction.fp_compare_b];
        entry.fp_compare_uses_ps1 = instruction.fp_compare_uses_ps1;
        return entry;
    }
    if (opcode == Opcode::floating_divide_single || opcode == Opcode::floating_divide_double ||
        opcode == Opcode::floating_add_single ||
        opcode == Opcode::floating_subtract_single ||
        opcode == Opcode::floating_add_double ||
        opcode == Opcode::floating_multiply_single ||
        opcode == Opcode::floating_multiply_double ||
        opcode == Opcode::floating_multiply_add_single ||
        opcode == Opcode::floating_multiply_subtract_single ||
        opcode == Opcode::floating_negative_multiply_subtract_single ||
        opcode == Opcode::floating_negative_multiply_add_single ||
        opcode == Opcode::floating_multiply_add_double ||
        opcode == Opcode::floating_multiply_subtract_double ||
        opcode == Opcode::floating_negative_multiply_add_double ||
        opcode == Opcode::floating_negative_multiply_subtract_double ||
        opcode == Opcode::floating_subtract_double ||
        opcode == Opcode::floating_round_to_single)
    {
        entry.has_fp_arithmetic = true;
        entry.fp_arithmetic_destination = instruction.fp_register;
        entry.fp_arithmetic_source_a = instruction.fp_source_a;
        entry.fp_arithmetic_source_b = instruction.fp_source_b;
        entry.fp_arithmetic_a_raw = state.fpr[instruction.fp_source_a];
        entry.fp_arithmetic_b_raw = state.fpr[instruction.fp_source_b];
        entry.has_fp_arithmetic_single_bits =
            opcode != Opcode::floating_subtract_double &&
            opcode != Opcode::floating_divide_double &&
            opcode != Opcode::floating_add_double &&
            opcode != Opcode::floating_multiply_double &&
            opcode != Opcode::floating_multiply_add_double &&
            opcode != Opcode::floating_multiply_subtract_double &&
            opcode != Opcode::floating_negative_multiply_add_double &&
            opcode != Opcode::floating_negative_multiply_subtract_double;
        entry.fp_arithmetic_source_count = opcode == Opcode::floating_round_to_single
            ? 1U
            : (opcode == Opcode::floating_multiply_add_single ||
               opcode == Opcode::floating_multiply_subtract_single ||
               opcode == Opcode::floating_negative_multiply_subtract_single ||
               opcode == Opcode::floating_negative_multiply_add_single ||
               opcode == Opcode::floating_multiply_add_double ||
               opcode == Opcode::floating_multiply_subtract_double ||
               opcode == Opcode::floating_negative_multiply_add_double ||
               opcode == Opcode::floating_negative_multiply_subtract_double) ? 3U : 2U;
        if (opcode == Opcode::floating_multiply_single ||
            opcode == Opcode::floating_multiply_double ||
            opcode == Opcode::floating_multiply_add_single ||
            opcode == Opcode::floating_multiply_subtract_single ||
            opcode == Opcode::floating_negative_multiply_subtract_single ||
            opcode == Opcode::floating_negative_multiply_add_single ||
            opcode == Opcode::floating_multiply_add_double ||
            opcode == Opcode::floating_multiply_subtract_double ||
            opcode == Opcode::floating_negative_multiply_add_double ||
            opcode == Opcode::floating_negative_multiply_subtract_double)
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
        opcode == Opcode::load_string_word_indexed ||
        opcode == Opcode::store_string_word_immediate ||
        opcode == Opcode::store_string_word_indexed)
    {
        const bool indexed = opcode == Opcode::load_string_word_indexed ||
            opcode == Opcode::store_string_word_indexed;
        const bool store = opcode == Opcode::store_string_word_immediate ||
            opcode == Opcode::store_string_word_indexed;
        entry.has_string_transfer = true;
        entry.string_transfer_store = store;
        entry.string_transfer_indexed = indexed;
        entry.string_transfer_first_register = store ? instruction.source : instruction.destination;
        entry.string_transfer_base_register = instruction.base;
        entry.string_transfer_base_value = instruction.base == 0 ? 0U : state.gpr[instruction.base];
        if (indexed)
        {
            entry.string_transfer_index_register = store
                ? instruction.destination
                : instruction.source;
            entry.string_transfer_index_value =
                state.gpr[entry.string_transfer_index_register];
            entry.string_transfer_byte_count =
                static_cast<std::uint8_t>(state.xer & 0x7FU);
            if (instruction.base != 0U)
            {
                add_history_source(entry, state, instruction.base);
            }
            add_history_source(entry, state, entry.string_transfer_index_register);
            entry.has_effective_address = true;
            entry.effective_address = entry.string_transfer_base_value +
                entry.string_transfer_index_value;
        }
        else
        {
            entry.string_transfer_byte_count = instruction.string_byte_count == 0
                ? 32U : instruction.string_byte_count;
            entry.has_effective_address = true;
            entry.effective_address = entry.string_transfer_base_value;
        }
        entry.string_transfer_register_count = static_cast<std::uint8_t>(
            (entry.string_transfer_byte_count + 3U) / 4U);
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
        opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_word_conditional_indexed ||
        opcode == Opcode::store_word_indexed_update ||
        opcode == Opcode::store_word_byte_reverse_indexed || opcode == Opcode::store_byte ||
        opcode == Opcode::store_byte_update || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_byte_indexed_update ||
        opcode == Opcode::store_halfword || opcode == Opcode::store_halfword_update ||
        opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::store_halfword_indexed_update ||
        opcode == Opcode::store_halfword_byte_reverse_indexed;

    if (opcode == Opcode::load_word_indexed ||
        opcode == Opcode::load_word_reserve_indexed ||
        opcode == Opcode::load_word_indexed_update ||
        opcode == Opcode::load_word_byte_reverse_indexed ||
        opcode == Opcode::load_single_indexed ||
        opcode == Opcode::load_single_indexed_update ||
        opcode == Opcode::load_double_indexed ||
        opcode == Opcode::load_double_indexed_update ||
        opcode == Opcode::load_byte_zero_indexed ||
        opcode == Opcode::load_byte_zero_indexed_update ||
        opcode == Opcode::load_halfword_zero_indexed ||
        opcode == Opcode::load_halfword_zero_indexed_update ||
        opcode == Opcode::load_halfword_byte_reverse_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed_update ||
        opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_word_conditional_indexed ||
        opcode == Opcode::store_word_indexed_update ||
        opcode == Opcode::store_word_byte_reverse_indexed ||
        opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_byte_indexed_update ||
        opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::store_halfword_indexed_update ||
        opcode == Opcode::store_halfword_byte_reverse_indexed ||
        opcode == Opcode::store_single_indexed ||
        opcode == Opcode::store_single_indexed_update ||
        opcode == Opcode::store_double_indexed ||
        opcode == Opcode::store_double_indexed_update ||
        opcode == Opcode::store_floating_as_integer_word_indexed ||
        opcode == Opcode::instruction_cache_block_invalidate ||
        opcode == Opcode::data_cache_block_touch ||
        opcode == Opcode::data_cache_block_touch_for_store ||
        opcode == Opcode::data_cache_block_store ||
        opcode == Opcode::data_cache_block_flush ||
        opcode == Opcode::data_cache_block_zero)
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        entry.has_effective_address = true;
        entry.effective_address = base + state.gpr[instruction.source];
        if (opcode == Opcode::data_cache_block_zero)
        {
            entry.effective_address &= ~(espresso_data_cache_block_size - 1U);
        }
    }

    if (opcode == Opcode::load_single || opcode == Opcode::load_single_update ||
        opcode == Opcode::load_double || opcode == Opcode::load_double_update ||
        opcode == Opcode::store_single ||
        opcode == Opcode::store_single_update || opcode == Opcode::store_double ||
        opcode == Opcode::store_double_update ||
        opcode == Opcode::load_halfword_zero_update ||
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
        opcode == Opcode::load_single_indexed || opcode == Opcode::load_single_indexed_update ||
        opcode == Opcode::load_double || opcode == Opcode::load_double_update ||
        opcode == Opcode::load_double_indexed ||
        opcode == Opcode::load_double_indexed_update)
    {
        entry.has_fp_destination = true;
        entry.fp_destination_register = instruction.fp_register;
    }
    else if (opcode == Opcode::store_single || opcode == Opcode::store_single_update ||
             opcode == Opcode::store_single_indexed || opcode == Opcode::store_double ||
             opcode == Opcode::store_single_indexed_update ||
             opcode == Opcode::store_double_indexed ||
             opcode == Opcode::store_double_indexed_update ||
             opcode == Opcode::store_double_update ||
             opcode == Opcode::store_floating_as_integer_word_indexed)
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
    case Opcode::load_halfword_zero_update:
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
    case Opcode::and_immediate_shifted_record:
    case Opcode::rotate_left_word_and_mask:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::arithmetic_shift_right_immediate:
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::add:
    case Opcode::add_overflow:
    case Opcode::add_carrying:
    case Opcode::add_carrying_overflow:
    case Opcode::add_extended:
    case Opcode::add_extended_overflow:
    case Opcode::subtract_from:
    case Opcode::subtract_from_overflow:
    case Opcode::subtract_from_carrying_overflow:
    case Opcode::subtract_from_carrying:
    case Opcode::subtract_from_extended:
    case Opcode::subtract_from_extended_overflow:
    case Opcode::multiply_low_word:
    case Opcode::multiply_low_word_overflow:
    case Opcode::multiply_high_word_unsigned:
    case Opcode::multiply_high_word_signed:
    case Opcode::divide_word_unsigned:
    case Opcode::divide_word_unsigned_overflow:
    case Opcode::divide_word_signed:
    case Opcode::divide_word_signed_overflow:
    case Opcode::compare_signed_register:
    case Opcode::compare_unsigned_register:
        add_history_source(entry, state, instruction.base);
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_and_complement:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_equivalence:
    case Opcode::bitwise_nand:
    case Opcode::bitwise_nor:
    case Opcode::bitwise_or_complement:
        add_history_source(entry, state, instruction.source);
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::add_to_zero_extended:
    case Opcode::add_to_zero_extended_overflow:
    case Opcode::add_to_minus_one_extended:
    case Opcode::add_to_minus_one_extended_overflow:
    case Opcode::subtract_from_zero_extended:
    case Opcode::subtract_from_zero_extended_overflow:
    case Opcode::subtract_from_minus_one_extended:
    case Opcode::subtract_from_minus_one_extended_overflow:
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::negate:
    case Opcode::negate_overflow:
    case Opcode::extend_sign_byte:
    case Opcode::extend_sign_halfword:
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::arithmetic_shift_right_word:
    case Opcode::rotate_left_word_and_mask_register:
        add_history_source(entry, state, instruction.source);
        add_history_source(entry, state, instruction.base);
        break;
    case Opcode::load_word_indexed:
    case Opcode::load_word_reserve_indexed:
    case Opcode::load_word_indexed_update:
    case Opcode::load_word_byte_reverse_indexed:
    case Opcode::load_single_indexed:
    case Opcode::load_single_indexed_update:
    case Opcode::load_double_indexed:
    case Opcode::load_double_indexed_update:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_byte_zero_indexed_update:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_zero_indexed_update:
    case Opcode::load_halfword_byte_reverse_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::load_halfword_algebraic_indexed_update:
    case Opcode::store_word_indexed:
    case Opcode::store_word_conditional_indexed:
    case Opcode::store_word_indexed_update:
    case Opcode::store_word_byte_reverse_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_byte_indexed_update:
    case Opcode::store_halfword_indexed:
    case Opcode::store_halfword_indexed_update:
    case Opcode::store_halfword_byte_reverse_indexed:
    case Opcode::store_single_indexed:
    case Opcode::store_single_indexed_update:
    case Opcode::store_double_indexed:
    case Opcode::store_double_indexed_update:
    case Opcode::store_floating_as_integer_word_indexed:
        if (instruction.base != 0)
        {
            add_history_source(entry, state, instruction.base);
        }
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::data_cache_block_zero:
    case Opcode::data_cache_block_touch:
    case Opcode::data_cache_block_touch_for_store:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
    case Opcode::instruction_cache_block_invalidate:
        if (instruction.base != 0U)
        {
            add_history_source(entry, state, instruction.base);
        }
        add_history_source(entry, state, instruction.source);
        break;
    case Opcode::move_to_link_register:
    case Opcode::move_to_count_register:
    case Opcode::move_to_xer:
    case Opcode::move_to_gqr:
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
    case Opcode::add_overflow:
    case Opcode::add_carrying_overflow:
    case Opcode::add_extended_overflow:
    case Opcode::subtract_from_overflow:
    case Opcode::subtract_from_carrying_overflow:
    case Opcode::subtract_from_extended_overflow:
    case Opcode::add_carrying:
    case Opcode::add_extended:
    case Opcode::add_to_zero_extended:
    case Opcode::add_to_zero_extended_overflow:
    case Opcode::add_to_minus_one_extended:
    case Opcode::add_to_minus_one_extended_overflow:
    case Opcode::subtract_from:
    case Opcode::subtract_from_carrying:
    case Opcode::subtract_from_extended:
    case Opcode::subtract_from_minus_one_extended:
    case Opcode::subtract_from_minus_one_extended_overflow:
    case Opcode::subtract_from_zero_extended:
    case Opcode::subtract_from_zero_extended_overflow:
    case Opcode::multiply_low_word:
    case Opcode::multiply_low_word_overflow:
    case Opcode::multiply_high_word_unsigned:
    case Opcode::multiply_high_word_signed:
    case Opcode::divide_word_unsigned:
    case Opcode::divide_word_unsigned_overflow:
    case Opcode::divide_word_signed:
    case Opcode::divide_word_signed_overflow:
    case Opcode::negate:
    case Opcode::negate_overflow:
    case Opcode::extend_sign_byte:
    case Opcode::extend_sign_halfword:
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
    case Opcode::bitwise_nand:
    case Opcode::bitwise_nor:
    case Opcode::bitwise_or_complement:
    case Opcode::and_immediate_record:
    case Opcode::and_immediate_shifted_record:
    case Opcode::rotate_left_word_and_mask:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::rotate_left_word_and_mask_register:
    case Opcode::arithmetic_shift_right_immediate:
    case Opcode::arithmetic_shift_right_word:
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::load_word_zero:
    case Opcode::load_word_update:
    case Opcode::load_word_indexed:
    case Opcode::load_word_reserve_indexed:
    case Opcode::load_word_indexed_update:
    case Opcode::load_word_byte_reverse_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_byte_zero_indexed_update:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_zero_indexed_update:
    case Opcode::load_halfword_byte_reverse_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::load_halfword_algebraic_indexed_update:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_zero_update:
    case Opcode::load_halfword_algebraic:
    case Opcode::move_from_link_register:
    case Opcode::move_from_count_register:
    case Opcode::move_from_xer:
    case Opcode::move_from_gqr:
    case Opcode::move_from_time_base_lower:
    case Opcode::move_from_time_base_upper:
        entry.has_destination = true;
        entry.destination_register = instruction.destination;
        break;
    case Opcode::load_single:
        break;
    case Opcode::load_single_indexed:
        break;
    case Opcode::load_single_update:
    case Opcode::load_single_indexed_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::load_double_indexed:
        break;
    case Opcode::load_double_indexed_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::load_double_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::load_double:
        break;
    case Opcode::store_single:
        break;
    case Opcode::store_single_update:
    case Opcode::store_single_indexed_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    case Opcode::store_single_indexed:
        break;
    case Opcode::store_word_update:
    case Opcode::store_word_indexed_update:
    case Opcode::store_byte_update:
    case Opcode::store_byte_indexed_update:
    case Opcode::store_halfword_update:
    case Opcode::store_halfword_indexed_update:
    case Opcode::store_double_update:
    case Opcode::store_double_indexed_update:
        entry.has_destination = true;
        entry.destination_register = instruction.base;
        break;
    default:
        break;
    }

    return entry;
}

}

const char* opcode_mnemonic(Opcode opcode) noexcept
{
    return opcode_name(opcode);
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
        if (entry.has_cr_bit_operation)
        {
            text << " cr" << std::dec
                 << static_cast<unsigned>(entry.cr_bit_source_a) << '='
                 << (entry.cr_bit_source_a_value ? '1' : '0')
                 << " cr" << static_cast<unsigned>(entry.cr_bit_source_b) << '='
                 << (entry.cr_bit_source_b_value ? '1' : '0');
            if (entry.completed)
            {
                text << " -> cr" << static_cast<unsigned>(entry.cr_bit_destination)
                     << '=' << (entry.cr_bit_result ? '1' : '0');
            }
        }
        if (entry.has_cr_field_move)
        {
            text << " cr" << std::dec
                 << static_cast<unsigned>(entry.cr_field_source) << "=0x"
                 << std::hex << static_cast<unsigned>(entry.cr_field_value);
            if (entry.completed)
            {
                text << " -> cr" << std::dec
                     << static_cast<unsigned>(entry.cr_field_destination)
                     << "=0x" << std::hex
                     << static_cast<unsigned>(entry.cr_field_value);
            }
        }
        if (entry.has_fpscr_cr_move)
        {
            text << " FPSCR" << std::dec
                 << static_cast<unsigned>(entry.fpscr_cr_source) << "=0x"
                 << std::hex << static_cast<unsigned>(entry.fpscr_cr_value)
                 << " -> cr" << std::dec
                 << static_cast<unsigned>(entry.fpscr_cr_destination) << "=0x"
                 << std::hex << static_cast<unsigned>(entry.fpscr_cr_value)
                 << " FPSCR=0x" << std::setw(8)
                 << entry.fpscr_before_cr_move;
            if (entry.completed)
            {
                text << " -> 0x" << std::hex << std::setw(8)
                     << entry.fpscr_after_cr_move;
            }
        }
        if (entry.has_xer_cr_move)
        {
            text << " XER[0:3]=0x" << std::hex
                 << static_cast<unsigned>(entry.xer_cr_value);
            if (entry.completed)
            {
                text << " -> cr" << std::dec
                     << static_cast<unsigned>(entry.xer_cr_field)
                     << "=0x" << std::hex
                     << static_cast<unsigned>(entry.xer_cr_value);
            }
        }
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
                 << ",r" << static_cast<unsigned>(entry.string_transfer_base_register);
            if (entry.string_transfer_indexed)
            {
                text << "=0x" << std::hex << std::setw(8)
                     << entry.string_transfer_base_value
                     << " r" << std::dec
                     << static_cast<unsigned>(entry.string_transfer_index_register)
                     << "=0x" << std::hex << std::setw(8)
                     << entry.string_transfer_index_value
                     << " bytes=" << std::dec
                     << static_cast<unsigned>(entry.string_transfer_byte_count);
            }
            else
            {
                text << ','
                     << static_cast<unsigned>(entry.string_transfer_byte_count);
            }
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
        if (entry.has_fp_select)
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
            print_operand(entry.fp_select_a, entry.fp_select_a_raw);
            print_operand(entry.fp_select_c, entry.fp_select_c_raw);
            print_operand(entry.fp_select_b, entry.fp_select_b_raw);
            if (entry.completed)
            {
                text << " -> f" << std::dec
                     << static_cast<unsigned>(entry.fp_select_destination) << "=0x"
                     << std::hex << std::setw(16) << entry.fp_select_result_raw;
            }
        }
        if (entry.has_fp_integer_conversion)
        {
            text << " f" << std::dec
                 << static_cast<unsigned>(entry.fp_integer_conversion_source) << '=';
            if (is_binary64_nan(entry.fp_integer_conversion_source_raw))
            {
                text << "NaN";
            }
            else
            {
                text << std::setprecision(17)
                     << std::bit_cast<double>(entry.fp_integer_conversion_source_raw);
            }
            text << " [0x" << std::hex << std::setw(16)
                 << entry.fp_integer_conversion_source_raw << ']';
            if (entry.completed)
            {
                const std::int32_t integer = std::bit_cast<std::int32_t>(
                    entry.fp_integer_conversion_result);
                if (entry.fp_integer_conversion_write_suppressed)
                {
                    text << " -> f" << std::dec
                         << static_cast<unsigned>(entry.fp_integer_conversion_destination)
                         << " (write suppressed; int32(" << integer << ") raw=0x"
                         << std::hex << std::setw(16)
                         << entry.fp_integer_conversion_encoded_raw;
                }
                else
                {
                    text << " -> f" << std::dec
                         << static_cast<unsigned>(entry.fp_integer_conversion_destination)
                         << "=int32(" << integer << ") raw=0x" << std::hex
                         << std::setw(16) << entry.fp_integer_conversion_encoded_raw;
                }
                if (entry.fp_integer_conversion_invalid)
                {
                    text << " invalid";
                }
                if (entry.fp_integer_conversion_write_suppressed)
                {
                    text << ')';
                }
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
        if (entry.has_fp_status_move)
        {
            text << " FPSCR=0x" << std::hex << std::setw(8)
                 << entry.fp_status_move_source;
            if (entry.completed)
            {
                text << " -> f" << std::dec
                     << static_cast<unsigned>(entry.fp_status_move_destination)
                     << ".ps0=0x" << std::hex << std::setw(16)
                     << entry.fp_status_move_result
                     << " (ps1 unchanged=0x" << std::setw(16)
                     << entry.fp_status_move_destination_ps1 << ')';
            }
        }
        if (entry.has_fpscr_bit_operation)
        {
            text << " BT=" << std::dec
                 << static_cast<unsigned>(entry.fpscr_bit_number)
                 << " FPSCR=0x" << std::hex << std::setw(8)
                 << entry.fpscr_before_bit_operation;
            if (entry.completed)
            {
                text << " -> 0x" << std::hex << std::setw(8)
                     << entry.fpscr_after_bit_operation;
            }
        }
        if (entry.has_fpscr_field_write)
        {
            text << " BF=" << std::dec
                 << static_cast<unsigned>(entry.fpscr_field_number)
                 << " U=" << static_cast<unsigned>(entry.fpscr_field_immediate)
                 << " FPSCR=0x" << std::hex << std::setw(8)
                 << entry.fpscr_before_field_write;
            if (entry.completed)
            {
                text << " -> 0x" << std::hex << std::setw(8)
                     << entry.fpscr_after_field_write;
            }
        }
        if (entry.has_fpscr_masked_write)
        {
            text << " f" << std::dec
                 << static_cast<unsigned>(entry.fpscr_masked_write_source)
                 << ".ps0=0x" << std::hex << std::setw(16)
                 << entry.fpscr_masked_write_source_raw
                 << " FM=0x" << std::setw(2)
                 << static_cast<unsigned>(entry.fpscr_masked_write_mask)
                 << " FPSCR=0x" << std::setw(8)
                 << entry.fpscr_before_masked_write;
            if (entry.completed)
            {
                text << " -> 0x" << std::hex << std::setw(8)
                     << entry.fpscr_after_masked_write;
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
        if (entry.has_carry_input)
        {
            text << " CAin=" << (entry.carry_input ? '1' : '0');
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
            if (entry.paired_fp_quantized_store)
            {
                print_paired_source('S', entry.paired_fp_source_a_register,
                                    entry.paired_fp_source_a_ps0,
                                    entry.paired_fp_source_a_ps1);
                text << " GQR" << std::dec
                     << static_cast<unsigned>(entry.paired_fp_store_gqr)
                     << " type=" << static_cast<unsigned>(entry.paired_fp_store_type)
                     << " W=" << (entry.paired_fp_store_w ? '1' : '0')
                     << " [0x" << std::hex << std::setw(8) << entry.effective_address << ']';
                if (entry.completed)
                {
                    text << " -> mem=0x" << std::hex << std::setw(8)
                         << entry.paired_fp_store_value0;
                    if (!entry.paired_fp_store_w)
                    {
                        text << ",0x" << std::setw(8)
                             << entry.paired_fp_store_value1;
                    }
                }
                else
                {
                    text << " (not written)";
                }
            }
            else if (entry.paired_fp_multiply_add ||
                     entry.paired_fp_multiply_subtract ||
                     entry.paired_fp_multiply_add_scalar0 ||
                     entry.paired_fp_multiply_add_scalar1 ||
                     entry.paired_fp_negative_multiply_add ||
                     entry.paired_fp_negative_multiply_subtract ||
                     entry.paired_fp_sum0 || entry.paired_fp_sum1)
            {
                print_paired_source('A', entry.paired_fp_source_a_register,
                                    entry.paired_fp_source_a_ps0,
                                    entry.paired_fp_source_a_ps1);
                print_paired_source('C', entry.paired_fp_source_c_register,
                                    entry.paired_fp_source_c_ps0,
                                    entry.paired_fp_source_c_ps1);
                print_paired_source('B', entry.paired_fp_source_b_register,
                                    entry.paired_fp_source_b_ps0,
                                    entry.paired_fp_source_b_ps1);
            }
            else if (entry.paired_fp_vector_multiply)
            {
                print_paired_source('A', entry.paired_fp_source_a_register,
                                    entry.paired_fp_source_a_ps0,
                                    entry.paired_fp_source_a_ps1);
                print_paired_source('C', entry.paired_fp_source_b_register,
                                    entry.paired_fp_source_b_ps0,
                                    entry.paired_fp_source_b_ps1);
            }
            else if (entry.paired_fp_scalar_multiply)
            {
                print_paired_source('A', entry.paired_fp_source_a_register,
                                    entry.paired_fp_source_a_ps0,
                                    entry.paired_fp_source_a_ps1);
                text << " fC=f" << std::dec
                     << static_cast<unsigned>(entry.paired_fp_source_b_register)
                     << ".ps" << (entry.paired_fp_scalar_source_ps1 ? '1' : '0')
                     << "=0x" << std::hex << std::setw(16)
                     << (entry.paired_fp_scalar_source_ps1
                             ? entry.paired_fp_source_b_ps1
                             : entry.paired_fp_source_b_ps0);
            }
            else if (entry.paired_fp_unary)
            {
                print_paired_source('B', entry.paired_fp_source_b_register,
                                    entry.paired_fp_source_b_ps0,
                                    entry.paired_fp_source_b_ps1);
            }
            else
            {
                print_paired_source('A', entry.paired_fp_source_a_register,
                                    entry.paired_fp_source_a_ps0,
                                    entry.paired_fp_source_a_ps1);
                print_paired_source('B', entry.paired_fp_source_b_register,
                                    entry.paired_fp_source_b_ps0,
                                    entry.paired_fp_source_b_ps1);
            }
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
        if (entry.has_paired_fp_select)
        {
            const auto print_select_source = [&](char operand, std::uint8_t reg,
                                                  std::uint64_t ps0, std::uint64_t ps1) {
                text << " f" << operand << "=f" << std::dec
                     << static_cast<unsigned>(reg) << "{ps0=0x" << std::hex
                     << std::setw(16) << ps0 << ",ps1=0x" << std::setw(16) << ps1 << '}';
            };
            print_select_source('A', entry.paired_fp_select_a,
                                entry.paired_fp_select_a_ps0,
                                entry.paired_fp_select_a_ps1);
            print_select_source('B', entry.paired_fp_select_b,
                                entry.paired_fp_select_b_ps0,
                                entry.paired_fp_select_b_ps1);
            print_select_source('C', entry.paired_fp_select_c,
                                entry.paired_fp_select_c_ps0,
                                entry.paired_fp_select_c_ps1);
            text << " -> f" << std::dec
                 << static_cast<unsigned>(entry.paired_fp_select_destination);
            if (entry.completed)
            {
                text << "{ps0=0x" << std::hex << std::setw(16)
                     << entry.paired_fp_select_result_ps0 << ",ps1=0x"
                     << std::setw(16) << entry.paired_fp_select_result_ps1 << '}';
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
        if (entry.has_overflow_result)
        {
            text << " OV=" << (entry.overflow_result ? '1' : '0')
                 << " SO=" << (entry.summary_overflow_result ? '1' : '0');
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
        if (entry.has_stored_integer_word_value && entry.completed)
        {
            text << " -> mem32=0x" << std::hex << std::setw(8)
                 << entry.stored_integer_word_value;
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
    if (memory.has_write_watchpoint())
    {
        memory.set_current_guest_cia(cia);
    }
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
    if (instruction.opcode == Opcode::add_overflow ||
        instruction.opcode == Opcode::subtract_from_overflow ||
        instruction.opcode == Opcode::add_carrying_overflow ||
        instruction.opcode == Opcode::add_extended_overflow ||
        instruction.opcode == Opcode::add_to_zero_extended_overflow ||
        instruction.opcode == Opcode::add_to_minus_one_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_zero_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_minus_one_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_carrying_overflow ||
        instruction.opcode == Opcode::subtract_from_extended_overflow ||
        instruction.opcode == Opcode::negate_overflow ||
        instruction.opcode == Opcode::multiply_low_word_overflow ||
        instruction.opcode == Opcode::divide_word_signed_overflow ||
        instruction.opcode == Opcode::divide_word_unsigned_overflow)
    {
        pending_history_entry_.has_overflow_result = true;
    }
    if (instruction.opcode == Opcode::subtract_from_carrying ||
        instruction.opcode == Opcode::subtract_from_carrying_overflow ||
        instruction.opcode == Opcode::subtract_from_extended ||
        instruction.opcode == Opcode::subtract_from_extended_overflow ||
        instruction.opcode == Opcode::add_carrying ||
        instruction.opcode == Opcode::add_carrying_overflow ||
        instruction.opcode == Opcode::add_extended_overflow ||
        instruction.opcode == Opcode::add_to_zero_extended ||
        instruction.opcode == Opcode::add_to_zero_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_zero_extended ||
        instruction.opcode == Opcode::subtract_from_zero_extended_overflow ||
        instruction.opcode == Opcode::add_extended ||
        instruction.opcode == Opcode::add_to_minus_one_extended ||
        instruction.opcode == Opcode::add_to_minus_one_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_minus_one_extended ||
        instruction.opcode == Opcode::subtract_from_minus_one_extended_overflow ||
        instruction.opcode == Opcode::arithmetic_shift_right_word)
    {
        pending_history_entry_.has_carry_result = true;
    }
    if (instruction.opcode == Opcode::subtract_from_extended ||
        instruction.opcode == Opcode::subtract_from_extended_overflow ||
        instruction.opcode == Opcode::add_to_zero_extended ||
        instruction.opcode == Opcode::add_to_zero_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_zero_extended ||
        instruction.opcode == Opcode::subtract_from_zero_extended_overflow ||
        instruction.opcode == Opcode::add_extended ||
        instruction.opcode == Opcode::add_extended_overflow ||
        instruction.opcode == Opcode::add_to_minus_one_extended ||
        instruction.opcode == Opcode::add_to_minus_one_extended_overflow ||
        instruction.opcode == Opcode::subtract_from_minus_one_extended ||
        instruction.opcode == Opcode::subtract_from_minus_one_extended_overflow)
    {
        pending_history_entry_.has_carry_input = true;
        pending_history_entry_.carry_input = (state.xer & xer_carry_mask) != 0U;
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

    case Opcode::multiply_low_word_overflow:
    {
        const std::int64_t a = std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int64_t b = std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        const std::int64_t product = a * b;
        const std::uint32_t result = static_cast<std::uint32_t>(product);
        const bool overflow = product < std::numeric_limits<std::int32_t>::min() ||
                              product > std::numeric_limits<std::int32_t>::max();
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
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

    case Opcode::divide_word_unsigned_overflow:
    {
        const std::uint32_t dividend = state.gpr[instruction.base];
        const std::uint32_t divisor = state.gpr[instruction.source];
        // PowerPC leaves the quotient undefined for division by zero. Match
        // Cemu's stable result while exposing the architecturally defined OV.
        const bool overflow = divisor == 0U;
        const std::uint32_t result = overflow ? 0U : dividend / divisor;
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::multiply_high_word_signed:
    {
        const std::int64_t lhs = std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int64_t rhs = std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        const std::int64_t product = lhs * rhs;
        const std::uint32_t result = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(product) >> 32U);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::divide_word_signed:
    {
        const std::int32_t dividend = std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int32_t divisor = std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        std::uint32_t result = 0U;
        if (divisor == 0)
        {
            // Deterministic Cemu-compatible non-OE behavior for architecturally
            // exceptional division; this is not asserted to be ISA-mandated.
            result = dividend < 0 ? 0xFFFFFFFFU : 0U;
        }
        else if (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1)
        {
            result = 0xFFFFFFFFU;
        }
        else
        {
            result = static_cast<std::uint32_t>(dividend / divisor);
        }
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::divide_word_signed_overflow:
    {
        const std::int32_t dividend =
            std::bit_cast<std::int32_t>(state.gpr[instruction.base]);
        const std::int32_t divisor =
            std::bit_cast<std::int32_t>(state.gpr[instruction.source]);
        const bool overflow = divisor == 0 ||
            (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1);
        std::uint32_t result = 0U;
        if (divisor == 0)
        {
            // The quotient is architecturally undefined for divide-by-zero;
            // retain Cemu's deterministic result convention.
            result = dividend < 0 ? 0xFFFFFFFFU : 0U;
        }
        else if (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1)
        {
            result = 0xFFFFFFFFU;
        }
        else
        {
            result = static_cast<std::uint32_t>(dividend / divisor);
        }
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
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

    case Opcode::negate_overflow:
    {
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t result = 0U - source;
        const bool overflow = source == 0x80000000U;
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
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

    case Opcode::extend_sign_halfword:
    {
        const std::uint32_t source = state.gpr[instruction.source];
        const std::uint32_t low_halfword = source & 0xFFFFU;
        const std::uint32_t result = low_halfword |
            ((low_halfword & 0x8000U) != 0U ? 0xFFFF0000U : 0U);
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

    case Opcode::add_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const std::uint32_t result = a + b;
        const bool overflow = ((~(a ^ b) & (a ^ result)) & 0x80000000U) != 0U;
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const std::uint32_t result = b - a;
        const bool overflow = (((b ^ a) & (b ^ result)) & 0x80000000U) != 0U;
        state.gpr[instruction.destination] = result;
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_carrying:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + b;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = result;
        if ((sum >> 32U) != 0U)
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

    case Opcode::add_carrying_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + b;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const bool carry_out = (sum >> 32U) != 0U;
        const bool overflow = ((~(a ^ b) & (a ^ result)) & 0x80000000U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_to_zero_extended:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t carry_in =
            (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = result;
        if ((sum >> 32U) != 0U)
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

    case Opcode::add_to_zero_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) +
                                  static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t exact_signed =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) +
            static_cast<std::int64_t>(carry_in);
        const bool overflow = exact_signed > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_extended:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const std::uint32_t carry_in =
            (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + b + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, (sum >> 32U) != 0U);
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + b +
                                  static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t exact_signed =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) +
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(b)) +
            static_cast<std::int64_t>(carry_in);
        const bool overflow = exact_signed < std::numeric_limits<std::int32_t>::min() ||
                              exact_signed > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_to_minus_one_extended:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t carry_in =
            (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) + 0xFFFFFFFFULL + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, (sum >> 32U) != 0U);
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::add_to_minus_one_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(a) +
                                  0xFFFFFFFFULL +
                                  static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t exact_signed =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) - 1 +
            static_cast<std::int64_t>(carry_in);
        const bool overflow = exact_signed < std::numeric_limits<std::int32_t>::min() ||
                              exact_signed > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
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

    case Opcode::subtract_from_carrying_overflow:
    {
        const std::uint32_t minuend = state.gpr[instruction.source];
        const std::uint32_t subtrahend = state.gpr[instruction.base];
        const std::uint32_t result = minuend - subtrahend;
        const bool carry_out = minuend >= subtrahend;
        const bool overflow = (((minuend ^ subtrahend) &
                                (minuend ^ result)) & 0x80000000U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from_extended:
    {
        const std::uint32_t ra = state.gpr[instruction.base];
        const std::uint32_t rb = state.gpr[instruction.source];
        const std::uint32_t carry_in = (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(static_cast<std::uint32_t>(~ra)) +
            static_cast<std::uint64_t>(rb) + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        if (carry_out)
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

    case Opcode::subtract_from_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t b = state.gpr[instruction.source];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint64_t extended_sum =
            static_cast<std::uint64_t>(~a) + b + static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(extended_sum);
        const std::int64_t signed_result =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(b)) -
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) -
            static_cast<std::int64_t>(!carry_in);
        const bool overflow = signed_result < std::numeric_limits<std::int32_t>::min() ||
                              signed_result > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (extended_sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from_zero_extended:
    {
        const std::uint32_t ra = state.gpr[instruction.base];
        const std::uint32_t carry_in =
            (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint32_t complement = ~ra;
        const std::uint64_t sum =
            static_cast<std::uint64_t>(complement) + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        if (carry_out)
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

    case Opcode::subtract_from_zero_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint32_t complement = ~a;
        const std::uint64_t sum = static_cast<std::uint64_t>(complement) +
                                  static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t exact_signed =
            -static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) - 1 +
            static_cast<std::int64_t>(carry_in);
        const bool overflow = exact_signed < std::numeric_limits<std::int32_t>::min() ||
                              exact_signed > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
        }
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from_minus_one_extended:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const std::uint32_t carry_in =
            (state.xer & xer_carry_mask) != 0U ? 1U : 0U;
        const std::uint32_t complement = ~a;
        const std::uint64_t sum =
            static_cast<std::uint64_t>(complement) + 0xFFFFFFFFULL + carry_in;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, (sum >> 32U) != 0U);
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::subtract_from_minus_one_extended_overflow:
    {
        const std::uint32_t a = state.gpr[instruction.base];
        const bool carry_in = (state.xer & xer_carry_mask) != 0U;
        const std::uint32_t complement = ~a;
        const std::uint64_t sum = static_cast<std::uint64_t>(complement) +
                                  0xFFFFFFFFULL +
                                  static_cast<std::uint32_t>(carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t exact_signed =
            -static_cast<std::int64_t>(std::bit_cast<std::int32_t>(a)) - 2 +
            static_cast<std::int64_t>(carry_in);
        const bool overflow = exact_signed < std::numeric_limits<std::int32_t>::min() ||
                              exact_signed > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (sum >> 32U) != 0U;
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry_out);
        if (overflow)
        {
            state.xer |= xer_overflow_mask | xer_summary_overflow_mask;
        }
        else
        {
            state.xer &= ~xer_overflow_mask;
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
    case Opcode::bitwise_nand:
    case Opcode::bitwise_nor:
    case Opcode::bitwise_or_complement:
    {
        const std::uint32_t lhs = state.gpr[instruction.source];
        const std::uint32_t rhs = state.gpr[instruction.base];
        const std::uint32_t result = instruction.opcode == Opcode::bitwise_or
            ? lhs | rhs
            : instruction.opcode == Opcode::bitwise_and ? lhs & rhs
            : instruction.opcode == Opcode::bitwise_and_complement ? lhs & ~rhs
            : instruction.opcode == Opcode::bitwise_equivalence ? ~(lhs ^ rhs)
            : instruction.opcode == Opcode::bitwise_nand ? ~(lhs & rhs)
            : instruction.opcode == Opcode::bitwise_nor ? ~(lhs | rhs)
            : instruction.opcode == Opcode::bitwise_or_complement ? lhs | ~rhs
                                                                  : lhs ^ rhs;
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::and_immediate_record:
    case Opcode::and_immediate_shifted_record:
    {
        const std::uint32_t immediate = static_cast<std::uint32_t>(instruction.immediate);
        const std::uint32_t operand = instruction.opcode == Opcode::and_immediate_shifted_record
            ? immediate << 16U
            : immediate;
        const std::uint32_t result = state.gpr[instruction.source] & operand;
        state.gpr[instruction.destination] = result;
        set_record_result(state, result);
        break;
    }

    case Opcode::rotate_left_word_and_mask:
    case Opcode::rotate_left_word_and_mask_register:
    {
        const std::uint32_t shift = instruction.opcode == Opcode::rotate_left_word_and_mask_register
            ? state.gpr[instruction.base] & 0x1FU
            : instruction.shift;
        const std::uint32_t rotated = std::rotl(
            state.gpr[instruction.source], static_cast<int>(shift));
        const std::uint32_t result =
            rotated & rotate_mask(instruction.mask_begin, instruction.mask_end);
        state.gpr[instruction.destination] = result;
        if (instruction.record)
        {
            set_record_result(state, result);
        }
        break;
    }

    case Opcode::arithmetic_shift_right_word:
    {
        const std::uint32_t source_bits = state.gpr[instruction.source];
        const std::int32_t source = std::bit_cast<std::int32_t>(source_bits);
        const std::uint32_t shift = state.gpr[instruction.base] & 0x3FU;
        const bool negative = source < 0;
        std::uint32_t result = source_bits;
        bool carry = false;
        if (shift == 0U)
        {
            carry = false;
        }
        else if (shift < 32U)
        {
            result = static_cast<std::uint32_t>(source >> shift);
            const std::uint32_t dropped_mask = (std::uint32_t{1} << shift) - 1U;
            carry = negative && (source_bits & dropped_mask) != 0U;
        }
        else
        {
            result = negative ? 0xFFFFFFFFU : 0U;
            carry = negative;
        }
        state.gpr[instruction.destination] = result;
        set_xer_carry(state, carry);
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

    case Opcode::paired_single_compare_unordered0:
    case Opcode::paired_single_compare_unordered1:
    case Opcode::paired_single_compare_ordered0:
    case Opcode::paired_single_compare_ordered1:
    case Opcode::floating_compare_unordered:
    case Opcode::floating_compare_ordered:
    {
        const std::uint64_t a_raw = instruction.fp_compare_uses_ps1
            ? state.fpr_ps1[instruction.fp_compare_a]
            : state.fpr[instruction.fp_compare_a];
        const std::uint64_t b_raw = instruction.fp_compare_uses_ps1
            ? state.fpr_ps1[instruction.fp_compare_b]
            : state.fpr[instruction.fp_compare_b];
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
        const bool ordered = instruction.opcode == Opcode::floating_compare_ordered ||
            instruction.opcode == Opcode::paired_single_compare_ordered0 ||
            instruction.opcode == Opcode::paired_single_compare_ordered1;
        if (is_binary64_signaling_nan(a_raw) || is_binary64_signaling_nan(b_raw))
        {
            raise_fpscr_exception(state, fpscr::vxsnan_mask);
            if (ordered && (state.fpscr & fpscr::ve_mask) == 0U)
            {
                raise_fpscr_exception(state, fpscr::vxvc_mask);
            }
        }
        else if (ordered && (a_nan || b_nan))
        {
            raise_fpscr_exception(state, fpscr::vxvc_mask);
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

    case Opcode::floating_divide_double:
    {
        // frD can alias either input; preserve the pre-instruction PS0 values.
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

        const bool a_zero = is_binary64_zero(a_raw);
        const bool b_zero = is_binary64_zero(b_raw);
        const bool a_infinity = is_binary64_infinity(a_raw);
        const bool b_infinity = is_binary64_infinity(b_raw);
        if (a_zero && b_zero)
        {
            signal_invalid(fpscr::vxzdz_mask, 0U);
            break;
        }
        if (a_infinity && b_infinity)
        {
            signal_invalid(fpscr::vxidi_mask, 0U);
            break;
        }
        if (b_zero)
        {
            raise_fpscr_exception(state, fpscr::zx_mask);
            if ((state.fpscr & fpscr::ze_mask) == 0U)
            {
                const std::uint64_t sign = (a_raw ^ b_raw) & binary64_sign_mask;
                commit_result(sign | binary64_infinity_bits, false, false);
            }
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double result = a / b;
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        if (a_infinity || b_infinity)
        {
            commit_result(raw_result, false, false);
            break;
        }

        const bool subnormal_or_overflow_case =
            !binary64_normal_or_zero(a_raw) || !binary64_normal_or_zero(b_raw) ||
            !binary64_normal_or_zero(raw_result) ||
            (is_binary64_zero(raw_result) && !a_zero);
        if (subnormal_or_overflow_case)
        {
            // TODO: classify exact binary64 division FI/FR and OX/UX in extreme cases.
            state.fpr[instruction.fp_register] = raw_result;
            set_double_arithmetic_fprf(state, raw_result);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            break;
        }

        const double remainder = std::fma(-result, b, a);
        const bool inexact = remainder != 0.0;
        const bool error_negative = std::signbit(remainder) != std::signbit(b);
        const bool rounded_up = inexact && result != 0.0 &&
            ((result > 0.0 && error_negative) ||
             (result < 0.0 && !error_negative));
        commit_result(raw_result, inexact, rounded_up);
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

    case Opcode::floating_subtract_single:
    {
        // fsubs computes from the full PS0 operands, rounds once to binary32,
        // and writes the extended result to both paired lanes.
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
            ((a_raw ^ b_raw) & binary64_sign_mask) == 0U)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double exact_result = a - b;
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

    case Opcode::floating_negative_multiply_add_single:
    {
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
        const double temporary = std::fma(a, c, b);
        std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
        if (!is_binary64_nan(temporary_raw))
        {
            temporary_raw ^= binary64_sign_mask;
        }
        const double exact_result = std::bit_cast<double>(temporary_raw);
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

    case Opcode::floating_negative_multiply_subtract_single:
    {
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
        if (product_is_infinite && b_infinity && product_is_negative == b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double c = std::bit_cast<double>(round_multiply_operand_25bit(c_raw));
        const double b = std::bit_cast<double>(b_raw);
        const double temporary = std::fma(a, c, -b);
        std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
        if (!is_binary64_nan(temporary_raw))
        {
            temporary_raw ^= binary64_sign_mask;
        }
        const double exact_result = std::bit_cast<double>(temporary_raw);
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

    case Opcode::floating_multiply_subtract_single:
    {
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
        if (product_is_infinite && b_infinity && product_is_negative == b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double c = std::bit_cast<double>(round_multiply_operand_25bit(c_raw));
        const double b = std::bit_cast<double>(b_raw);
        const double exact_result = std::fma(a, c, -b);
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

    case Opcode::floating_absolute_value:
    {
        const std::uint64_t source = state.fpr[instruction.fp_source_b];
        const std::uint64_t result = source & ~binary64_sign_mask;
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

    case Opcode::floating_negative_absolute_value:
    {
        const std::uint64_t source = state.fpr[instruction.fp_source_b];
        const std::uint64_t result = source | binary64_sign_mask;
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

    case Opcode::floating_move_from_fpscr:
    {
        const std::uint64_t result = 0xFFFFFFFF00000000ULL | state.fpscr;
        state.fpr[instruction.fp_register] = result;
        pending_history_entry_.fp_status_move_result = result;
        break;
    }

    case Opcode::floating_move_to_fpscr_bit_zero:
    case Opcode::floating_move_to_fpscr_bit_one:
    {
        // FPSCR bits 1 (FEX) and 2 (VX) are summary bits and cannot be written
        // directly; update_fpscr_summaries derives them from exception state.
        if (instruction.fpscr_bit != 1U && instruction.fpscr_bit != 2U)
        {
            const std::uint32_t bit_mask =
                1U << (31U - instruction.fpscr_bit);
            if (instruction.opcode == Opcode::floating_move_to_fpscr_bit_zero)
            {
                state.fpscr &= ~bit_mask;
            }
            else
            {
                state.fpscr |= bit_mask;
            }
            update_fpscr_summaries(state);
        }
        pending_history_entry_.fpscr_after_bit_operation = state.fpscr;
        break;
    }

    case Opcode::floating_move_to_fpscr_field_immediate:
    {
        const std::uint32_t shift = 28U -
            (static_cast<std::uint32_t>(instruction.fpscr_field) * 4U);
        const std::uint32_t field_mask = 0xFU << shift;
        std::uint32_t writable_mask = field_mask;
        if (instruction.fpscr_field == 0U)
        {
            // FEX and VX are summary bits; writes to those positions are ignored.
            writable_mask &= ~(fpscr::fex_mask | fpscr::vx_mask);
        }
        const std::uint32_t value =
            static_cast<std::uint32_t>(instruction.fpscr_immediate) << shift;
        state.fpscr = (state.fpscr & ~writable_mask) | (value & writable_mask);
        update_fpscr_summaries(state);
        pending_history_entry_.fpscr_after_field_write = state.fpscr;
        break;
    }

    case Opcode::floating_move_to_fpscr_fields:
    {
        const std::uint32_t source =
            static_cast<std::uint32_t>(state.fpr[instruction.fp_source_b]);
        std::uint32_t write_mask = 0U;
        for (std::uint32_t field_index = 0U; field_index < 8U; ++field_index)
        {
            const std::uint32_t mask_bit = 1U << (7U - field_index);
            if ((instruction.fpscr_field_mask & mask_bit) == 0U)
            {
                continue;
            }

            const std::uint32_t shift = 28U - field_index * 4U;
            write_mask |= 0xFU << shift;
        }
        // FEX and VX are derived summary bits and cannot be written directly.
        write_mask &= ~(fpscr::fex_mask | fpscr::vx_mask);
        state.fpscr = (state.fpscr & ~write_mask) | (source & write_mask);
        update_fpscr_summaries(state);
        pending_history_entry_.fpscr_after_masked_write = state.fpscr;
        break;
    }

    case Opcode::floating_select:
    {
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const double a = std::bit_cast<double>(a_raw);
        const std::uint64_t result_raw =
            !is_binary64_nan(a_raw) && a >= 0.0 ? c_raw : b_raw;

        state.fpr[instruction.fp_register] = result_raw;
        pending_history_entry_.fp_select_result_raw = result_raw;
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

    case Opcode::floating_add_double:
    {
        // frD can alias either source, so capture both PS0 values up front.
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
            ((a_raw ^ b_raw) & binary64_sign_mask) != 0U)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double result = a + b;
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        if (a_infinity || b_infinity)
        {
            commit_result(raw_result, false, false);
            break;
        }

        if (binary64_normal_or_zero(a_raw) && binary64_normal_or_zero(b_raw) &&
            binary64_normal_or_zero(raw_result))
        {
            const double error = binary64_addition_error(a, b, result);
            const bool inexact = error != 0.0;
            const bool rounded_up = inexact && result != 0.0 &&
                ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
            commit_result(raw_result, inexact, rounded_up);
        }
        else
        {
            // TODO: classify double underflow/overflow and their FI/FR/XX details.
            state.fpr[instruction.fp_register] = raw_result;
            set_double_arithmetic_fprf(state, raw_result);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        }
        break;
    }

    case Opcode::floating_multiply_double:
    {
        // fmul consumes frA.PS0 and frC.PS0 and writes only destination PS0.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
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
        const double c = std::bit_cast<double>(c_raw);
        const double result = a * c;
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        if (a_infinity || c_infinity)
        {
            commit_result(raw_result, false, false);
            break;
        }

        const bool subnormal_or_overflow_case =
            !binary64_normal_or_zero(a_raw) || !binary64_normal_or_zero(c_raw) ||
            !binary64_normal_or_zero(raw_result) ||
            (is_binary64_zero(raw_result) && !a_zero && !c_zero);
        if (subnormal_or_overflow_case)
        {
            // TODO: model exact binary64 underflow/overflow exception and rounding flags.
            state.fpr[instruction.fp_register] = raw_result;
            set_double_arithmetic_fprf(state, raw_result);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
            break;
        }

        const double error = binary64_multiplication_error(a, c, result);
        const bool inexact = error != 0.0;
        const bool rounded_up = inexact && result != 0.0 &&
            ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
        commit_result(raw_result, inexact, rounded_up);
        break;
    }

    case Opcode::floating_multiply_add_double:
    {
        // frD may alias any operand, so snapshot the complete A, B, C tuple.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint64_t raw_result,
                                       bool known_exact) {
            state.fpr[instruction.fp_register] = raw_result;
            if (known_exact)
            {
                set_double_arithmetic_result_status(state, raw_result, false, false);
            }
            else
            {
                // TODO: classify exact binary64 FMA FI/FR/XX and OX/UX behavior.
                set_double_arithmetic_fprf(state, raw_result);
            }
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_binary64_bits(nan_source), true);
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

        const bool b_infinity = is_binary64_infinity(b_raw);
        const bool product_is_infinite = a_infinity || c_infinity;
        const bool product_is_negative =
            ((a_raw ^ c_raw) & binary64_sign_mask) != 0U;
        const bool b_is_negative = (b_raw & binary64_sign_mask) != 0U;
        if (product_is_infinite && b_infinity && product_is_negative != b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(c_raw);
        const double result = std::fma(a, c, b);
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
        commit_result(raw_result, known_exact);
        break;
    }

    case Opcode::floating_multiply_subtract_double:
    {
        // Snapshot the full source tuple; frD is allowed to alias any source.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint64_t raw_result,
                                       bool known_exact) {
            state.fpr[instruction.fp_register] = raw_result;
            if (known_exact)
            {
                set_double_arithmetic_result_status(state, raw_result, false, false);
            }
            else
            {
                // TODO: classify exact binary64 FMA FI/FR/XX and OX/UX behavior.
                set_double_arithmetic_fprf(state, raw_result);
            }
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_binary64_bits(nan_source), true);
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

        const bool b_infinity = is_binary64_infinity(b_raw);
        const bool product_is_infinite = a_infinity || c_infinity;
        const bool product_is_negative =
            ((a_raw ^ c_raw) & binary64_sign_mask) != 0U;
        const bool b_is_negative = (b_raw & binary64_sign_mask) != 0U;
        if (product_is_infinite && b_infinity && product_is_negative == b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(c_raw);
        const double result = std::fma(a, c, -b);
        const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
        const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
        commit_result(raw_result, known_exact);
        break;
    }

    case Opcode::floating_negative_multiply_add_double:
    {
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint64_t raw_result,
                                       bool known_exact) {
            state.fpr[instruction.fp_register] = raw_result;
            if (known_exact)
            {
                set_double_arithmetic_result_status(state, raw_result, false, false);
            }
            else
            {
                // TODO: classify exact binary64 FMA FI/FR/XX and OX/UX behavior.
                set_double_arithmetic_fprf(state, raw_result);
            }
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_binary64_bits(nan_source), true);
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

        const bool b_infinity = is_binary64_infinity(b_raw);
        const bool product_is_infinite = a_infinity || c_infinity;
        const bool product_is_negative =
            ((a_raw ^ c_raw) & binary64_sign_mask) != 0U;
        const bool b_is_negative = (b_raw & binary64_sign_mask) != 0U;
        if (product_is_infinite && b_infinity && product_is_negative != b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(c_raw);
        const double temporary = std::fma(a, c, b);
        const std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
        const std::uint64_t raw_result = is_binary64_nan(temporary_raw)
            ? temporary_raw
            : temporary_raw ^ binary64_sign_mask;
        const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
        commit_result(raw_result, known_exact);
        break;
    }

    case Opcode::floating_negative_multiply_subtract_double:
    {
        // Snapshot all three PS0 inputs before writing; frD may alias any operand.
        const std::uint64_t a_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t b_raw = state.fpr[instruction.fp_source_b];
        const std::uint64_t c_raw = state.fpr[instruction.fp_source_c];
        const auto commit_result = [&](std::uint64_t raw_result,
                                       bool known_exact) {
            state.fpr[instruction.fp_register] = raw_result;
            if (known_exact)
            {
                set_double_arithmetic_result_status(state, raw_result, false, false);
            }
            else
            {
                // TODO: classify exact binary64 FMA FI/FR/XX and OX/UX behavior.
                set_double_arithmetic_fprf(state, raw_result);
            }
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = raw_result;
        };
        const auto commit_quiet_nan = [&](std::uint64_t nan_source) {
            commit_result(quiet_nan_binary64_bits(nan_source), true);
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

        const bool b_infinity = is_binary64_infinity(b_raw);
        const bool product_is_infinite = a_infinity || c_infinity;
        const bool product_is_negative =
            ((a_raw ^ c_raw) & binary64_sign_mask) != 0U;
        const bool b_is_negative = (b_raw & binary64_sign_mask) != 0U;
        if (product_is_infinite && b_infinity && product_is_negative == b_is_negative)
        {
            signal_invalid(fpscr::vxisi_mask, 0U);
            break;
        }

        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(c_raw);
        const double temporary = std::fma(a, c, -b);
        const std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
        const std::uint64_t raw_result = is_binary64_nan(temporary_raw)
            ? temporary_raw
            : temporary_raw ^ binary64_sign_mask;

        const bool known_exact = c_zero || a_infinity || c_infinity || b_infinity;
        commit_result(raw_result, known_exact);
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

    case Opcode::floating_reciprocal_square_root_estimate:
    {
        const std::uint64_t source_raw = state.fpr[instruction.fp_source_b];
        const double source = std::bit_cast<double>(source_raw);
        std::uint64_t result_raw{};
        bool suppress_write = false;

        if (is_binary64_signaling_nan(source_raw))
        {
            raise_fpscr_exception(state, fpscr::vxsnan_mask);
            suppress_write = (state.fpscr & fpscr::ve_mask) != 0U;
            result_raw = quiet_nan_binary64_bits(source_raw);
        }
        else if (is_binary64_nan(source_raw))
        {
            result_raw = quiet_nan_binary64_bits(source_raw);
        }
        else if (source < 0.0)
        {
            raise_fpscr_exception(state, fpscr::vxsqrt_mask);
            suppress_write = (state.fpscr & fpscr::ve_mask) != 0U;
            result_raw = quiet_nan_binary64_bits(0U);
        }
        else if (source == 0.0)
        {
            raise_fpscr_exception(state, fpscr::zx_mask);
            suppress_write = (state.fpscr & fpscr::ze_mask) != 0U;
            result_raw = ((source_raw & binary64_sign_mask) != 0U
                    ? binary64_sign_mask : 0U) |
                binary64_exponent_mask;
        }
        else if (std::isinf(source))
        {
            result_raw = 0U;
        }
        else
        {
            // PowerPC specifies an estimate (relative error <= 1/32), not a
            // uniquely rounded value. A host double estimate is within that bound.
            result_raw = std::bit_cast<std::uint64_t>(1.0 / std::sqrt(source));
        }

        if (!suppress_write)
        {
            state.fpr[instruction.fp_register] = result_raw;
            set_double_arithmetic_fprf(state, result_raw);
        }
        break;
    }

    case Opcode::floating_reciprocal_estimate_single:
    {
        const std::uint64_t source_raw = state.fpr[instruction.fp_source_b];
        const double source = std::bit_cast<double>(source_raw);
        bool suppress_write = false;
        std::uint32_t result_bits{};
        bool inexact = false;
        bool rounded_up = false;

        if (is_binary64_signaling_nan(source_raw))
        {
            raise_fpscr_exception(state, fpscr::vxsnan_mask);
            suppress_write = (state.fpscr & fpscr::ve_mask) != 0U;
            result_bits = quiet_nan_single_bits(source_raw);
        }
        else if (is_binary64_nan(source_raw))
        {
            result_bits = quiet_nan_single_bits(source_raw);
        }
        else if (source == 0.0)
        {
            raise_fpscr_exception(state, fpscr::zx_mask);
            suppress_write = (state.fpscr & fpscr::ze_mask) != 0U;
            const std::uint32_t sign =
                (source_raw & binary64_sign_mask) != 0U ? 0x80000000U : 0U;
            result_bits = sign | 0x7F800000U;
        }
        else if (std::isinf(source))
        {
            result_bits = (source_raw & binary64_sign_mask) != 0U
                ? 0x80000000U
                : 0U;
        }
        else
        {
            // fres permits implementation-dependent estimates. Correctly round
            // the host binary64 reciprocal to single precision; this is within
            // the architectural 1/256 relative-error bound for normal results.
            const double reciprocal = 1.0 / source;
            const float estimate = static_cast<float>(reciprocal);
            const double extended_estimate = static_cast<double>(estimate);
            result_bits = std::bit_cast<std::uint32_t>(estimate);
            inexact = extended_estimate != reciprocal;
            rounded_up = inexact &&
                std::fabs(extended_estimate) > std::fabs(reciprocal);
            // TODO: model fres OX/UX behavior for extreme reciprocal overflow
            // and underflow, including OE/UE destination suppression.
        }

        if (!suppress_write)
        {
            const float result = std::bit_cast<float>(result_bits);
            const std::uint64_t extended_result =
                std::bit_cast<std::uint64_t>(static_cast<double>(result));
            state.fpr[instruction.fp_register] = extended_result;
            state.fpr_ps1[instruction.fp_register] = extended_result;
            set_single_arithmetic_result_status(state, result_bits, inexact, rounded_up);
            pending_history_entry_.has_fp_arithmetic_result = true;
            pending_history_entry_.fp_arithmetic_result_raw = extended_result;
            pending_history_entry_.fp_arithmetic_single_bits = result_bits;
        }
        break;
    }

    case Opcode::floating_convert_to_integer_word:
    case Opcode::floating_convert_to_integer_word_zero:
    {
        const std::uint64_t source_raw = state.fpr[instruction.fp_source_b];
        const double source = std::bit_cast<double>(source_raw);
        constexpr double minimum_signed_word = -2147483648.0;
        constexpr double one_past_maximum_signed_word = 2147483648.0;
        std::uint32_t integer_bits{};
        bool invalid = false;
        bool inexact = false;
        bool rounded_up = false;

        if (is_binary64_nan(source_raw))
        {
            invalid = true;
            integer_bits = 0x80000000U;
            raise_fpscr_exception(state, fpscr::vxcvi_mask);
            if (is_binary64_signaling_nan(source_raw))
            {
                raise_fpscr_exception(state, fpscr::vxsnan_mask);
            }
        }
        else
        {
            const std::uint32_t rounding_mode =
                instruction.opcode == Opcode::floating_convert_to_integer_word_zero
                    ? 1U
                    : state.fpscr & fpscr::rn_mask;
            double rounded{};
            switch (rounding_mode)
            {
            case 0U:
            {
                const double lower = std::floor(source);
                const double fraction = source - lower;
                if (fraction < 0.5)
                {
                    rounded = lower;
                }
                else if (fraction > 0.5)
                {
                    rounded = lower + 1.0;
                }
                else
                {
                    // Nearest with ties to even; beyond the exact-integer
                    // range, representable doubles are already even integers.
                    rounded = std::fmod(std::fabs(lower), 2.0) == 0.0
                        ? lower
                        : lower + 1.0;
                }
                break;
            }
            case 1U: rounded = std::trunc(source); break;
            case 2U: rounded = std::ceil(source); break;
            case 3U: rounded = std::floor(source); break;
            default: break;
            }

            if (rounded >= one_past_maximum_signed_word)
            {
                invalid = true;
                integer_bits = 0x7FFFFFFFU;
            }
            else if (rounded < minimum_signed_word)
            {
                invalid = true;
                integer_bits = 0x80000000U;
            }
            else
            {
                const std::int32_t integer = static_cast<std::int32_t>(rounded);
                integer_bits = std::bit_cast<std::uint32_t>(integer);
                inexact = rounded != source;
                rounded_up = inexact &&
                    std::fabs(rounded) > std::fabs(source);
            }

            if (invalid)
            {
                raise_fpscr_exception(state, fpscr::vxcvi_mask);
            }
        }

        state.fpscr &= ~(fpscr::fi_mask | fpscr::fr_mask);
        if (!invalid && inexact)
        {
            state.fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
            if (rounded_up)
            {
                state.fpscr |= fpscr::fr_mask;
            }
        }
        update_fpscr_summaries(state);

        const std::uint64_t encoded = encode_fctiw_result(integer_bits, source_raw);
        const bool write_suppressed = invalid && (state.fpscr & fpscr::ve_mask) != 0U;
        if (!write_suppressed)
        {
            state.fpr[instruction.fp_register] = encoded;
        }

        pending_history_entry_.has_fp_integer_conversion = true;
        pending_history_entry_.fp_integer_conversion_source = instruction.fp_source_b;
        pending_history_entry_.fp_integer_conversion_source_raw = source_raw;
        pending_history_entry_.fp_integer_conversion_destination = instruction.fp_register;
        pending_history_entry_.fp_integer_conversion_result = integer_bits;
        pending_history_entry_.fp_integer_conversion_encoded_raw = encoded;
        pending_history_entry_.fp_integer_conversion_invalid = invalid;
        pending_history_entry_.fp_integer_conversion_write_suppressed = write_suppressed;
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

    case Opcode::move_from_xer:
        state.gpr[instruction.destination] = state.xer;
        break;

    case Opcode::move_from_gqr:
        state.gpr[instruction.destination] =
            state.gqr[instruction.quantized_register];
        break;

    case Opcode::move_from_time_base_lower:
    case Opcode::move_from_time_base_upper:
    {
        const std::uint64_t time_base = elapsed_time_base_ticks(timebase_origin_);
        state.gpr[instruction.destination] =
            instruction.opcode == Opcode::move_from_time_base_lower
                ? static_cast<std::uint32_t>(time_base)
                : static_cast<std::uint32_t>(time_base >> 32U);
        break;
    }

    case Opcode::move_to_count_register:
        state.ctr = state.gpr[instruction.destination];
        break;

    case Opcode::move_to_xer:
        state.xer = state.gpr[instruction.destination];
        break;

    case Opcode::move_to_gqr:
        state.gqr[instruction.quantized_register] =
            state.gpr[instruction.destination];
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
    case Opcode::load_string_word_indexed:
    {
        const bool indexed = instruction.opcode == Opcode::load_string_word_indexed;
        // Snapshot indexed EA and count before any destination GPR is changed.
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = indexed
            ? base + state.gpr[instruction.source]
            : base;
        const std::size_t byte_count = indexed
            ? static_cast<std::size_t>(state.xer & 0x7FU)
            : instruction.string_byte_count == 0U
                ? 32U : instruction.string_byte_count;
        std::array<std::uint8_t, 127> bytes{};
        if (byte_count != 0U)
        {
            memory.read_bytes(address, std::span<std::uint8_t>(bytes).first(byte_count));
        }

        const std::size_t register_count = (byte_count + 3U) / 4U;
        std::array<std::uint32_t, 32> values{};
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
    case Opcode::store_string_word_indexed:
    {
        const bool indexed = instruction.opcode == Opcode::store_string_word_indexed;
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = indexed
            ? base + state.gpr[instruction.destination]
            : base;
        const std::size_t byte_count = indexed
            ? static_cast<std::size_t>(state.xer & 0x7FU)
            : instruction.string_byte_count == 0U
                ? 32U : instruction.string_byte_count;
        std::array<std::uint8_t, 127> bytes{};
        for (std::size_t i = 0; i < byte_count; ++i)
        {
            const std::uint32_t reg =
                (instruction.source + static_cast<std::uint32_t>(i / 4U)) & 31U;
            const unsigned shift = static_cast<unsigned>(24U - (i % 4U) * 8U);
            bytes[i] = static_cast<std::uint8_t>(state.gpr[reg] >> shift);
        }

        // Validate and commit the entire output span as one guest write.
        if (byte_count != 0U)
        {
            memory.write_bytes(address, std::span<const std::uint8_t>(bytes).first(byte_count));
            std::copy_n(bytes.begin(), byte_count,
                        pending_history_entry_.string_transfer_bytes.begin());
        }
        break;
    }

    case Opcode::load_word_indexed:
    {
        const std::uint32_t base = instruction.base == 0 ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        state.gpr[instruction.destination] = memory.read32_be(address);
        break;
    }

    case Opcode::load_word_reserve_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        if ((address & 3U) != 0U)
        {
            throw GuestMemoryFault(address, sizeof(std::uint32_t), GuestMemoryAccess::read);
        }
        const std::uint32_t value = memory.read32_be(address);
        state.gpr[instruction.destination] = value;
        state.reservation_address = memory.reservation_address(address);
        state.reservation_valid = true;
        break;
    }

    case Opcode::load_word_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint32_t value = memory.read32_be(address);
        state.gpr[instruction.destination] = value;
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::load_word_byte_reverse_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint32_t value = memory.read32_be(address);
        state.gpr[instruction.destination] = std::byteswap(value);
        break;
    }

    case Opcode::load_halfword_byte_reverse_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint16_t value = memory.read16_be(address);
        state.gpr[instruction.destination] = std::byteswap(value);
        break;
    }

    case Opcode::load_byte_zero_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint8_t value = memory.read8(address);
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(value);
        break;
    }

    case Opcode::load_byte_zero_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint8_t value = memory.read8(address);
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(value);
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::load_halfword_zero_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint16_t value = memory.read16_be(address);
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(value);
        break;
    }

    case Opcode::load_halfword_zero_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint16_t value = memory.read16_be(address);
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(value);
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::load_halfword_algebraic_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint32_t halfword = memory.read16_be(address);
        const std::uint32_t result = halfword |
            ((halfword & 0x8000U) != 0U ? 0xFFFF0000U : 0U);
        state.gpr[instruction.destination] = result;
        break;
    }

    case Opcode::load_halfword_algebraic_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint16_t halfword = memory.read16_be(address);
        const std::uint32_t value = static_cast<std::uint32_t>(
            static_cast<std::int32_t>(std::bit_cast<std::int16_t>(halfword)));
        state.gpr[instruction.destination] = value;
        state.gpr[instruction.base] = address;
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

    case Opcode::store_word_conditional_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint32_t value = state.gpr[instruction.destination];
        if ((address & 3U) != 0U)
        {
            throw GuestMemoryFault(address, sizeof(std::uint32_t), GuestMemoryAccess::write);
        }
        const std::uint32_t canonical_address = memory.reservation_address(address);
        const bool success = state.reservation_valid &&
            state.reservation_address == canonical_address;
        if (success)
        {
            memory.write32_be(address, value);
        }
        // Each conditional-store attempt consumes the single per-core
        // reservation, whether the address matched or not.
        state.reservation_valid = false;
        const std::uint8_t cr0 = static_cast<std::uint8_t>(
            (success ? cr_equal : 0U) |
            ((state.xer & xer_summary_overflow_mask) != 0U ? 1U : 0U));
        state.cr = (state.cr & 0x0FFFFFFFU) |
            (static_cast<std::uint32_t>(cr0) << 28U);
        break;
    }

    case Opcode::store_word_indexed_update:
    {
        const std::uint32_t source = state.gpr[instruction.destination];
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        memory.write32_be(address, source);
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::store_word_byte_reverse_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        memory.write32_be(address, std::byteswap(state.gpr[instruction.destination]));
        break;
    }

    case Opcode::store_byte_indexed:
    {
        const std::uint32_t base = instruction.base == 0 ? 0 : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        memory.write8(address, static_cast<std::uint8_t>(state.gpr[instruction.destination]));
        break;
    }

    case Opcode::store_byte_indexed_update:
    {
        const std::uint32_t source = state.gpr[instruction.destination];
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t index = state.gpr[instruction.source];
        const std::uint32_t address = base + index;
        memory.write8(address, static_cast<std::uint8_t>(source));
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::store_halfword_indexed:
    {
        const std::uint32_t source = state.gpr[instruction.destination];
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t index = state.gpr[instruction.source];
        const std::uint32_t address = base + index;
        const std::uint16_t value = static_cast<std::uint16_t>(source);
        memory.write16_be(address, value);
        pending_history_entry_.has_stored_halfword_value = true;
        pending_history_entry_.stored_halfword_value = value;
        break;
    }

    case Opcode::store_halfword_indexed_update:
    {
        const std::uint32_t source = state.gpr[instruction.destination];
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t index = state.gpr[instruction.source];
        const std::uint32_t address = base + index;
        const std::uint16_t value = static_cast<std::uint16_t>(source);
        memory.write16_be(address, value);
        state.gpr[instruction.base] = address;
        pending_history_entry_.has_stored_halfword_value = true;
        pending_history_entry_.stored_halfword_value = value;
        break;
    }

    case Opcode::store_halfword_byte_reverse_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint16_t value = std::byteswap(
            static_cast<std::uint16_t>(state.gpr[instruction.destination]));
        memory.write16_be(address, value);
        pending_history_entry_.has_stored_halfword_value = true;
        pending_history_entry_.stored_halfword_value = value;
        break;
    }

    case Opcode::data_cache_block_zero:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t effective = base + state.gpr[instruction.source];
        const std::uint32_t block_address =
            effective & ~(espresso_data_cache_block_size - 1U);
        const std::array<std::uint8_t, espresso_data_cache_block_size> zero_block{};

        // With no separately modeled Espresso D-cache, make the coherent guest
        // memory effect visible directly. write_bytes validates all 32 bytes
        // before changing any backing region.
        memory.write_bytes(block_address, zero_block);
        break;
    }

    case Opcode::store_floating_as_integer_word_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t index = state.gpr[instruction.source];
        const std::uint32_t address = base + index;
        const std::uint64_t raw_source = state.fpr[instruction.fp_register];
        const std::uint32_t value = static_cast<std::uint32_t>(raw_source);
        memory.write32_be(address, value);
        pending_history_entry_.has_stored_integer_word_value = true;
        pending_history_entry_.stored_integer_word_value = value;
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

    case Opcode::load_halfword_zero_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint16_t value = memory.read16_be(address);
        state.gpr[instruction.destination] = static_cast<std::uint32_t>(value);
        state.gpr[instruction.base] = address;
        break;
    }

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
    case Opcode::load_single_indexed:
    case Opcode::load_single_indexed_update:
    {
        const bool indexed = instruction.opcode == Opcode::load_single_indexed ||
            instruction.opcode == Opcode::load_single_indexed_update;
        const std::uint32_t address = indexed
            ? (instruction.base == 0U ? 0U : state.gpr[instruction.base]) +
                state.gpr[instruction.source]
            : effective_address(state, instruction.base, instruction.immediate);
        const std::uint32_t single_bits = memory.read32_be(address);
        const float single_value = std::bit_cast<float>(single_bits);
        const double double_value = static_cast<double>(single_value);
        const std::uint64_t double_bits = std::bit_cast<std::uint64_t>(double_value);
        state.fpr[instruction.fp_register] = double_bits;
        state.fpr_ps1[instruction.fp_register] = double_bits;
        if (instruction.opcode == Opcode::load_single_update ||
            instruction.opcode == Opcode::load_single_indexed_update)
        {
            state.gpr[instruction.base] = address;
        }
        break;
    }

    case Opcode::load_double_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        state.fpr[instruction.fp_register] = memory.read64_be(address);
        break;
    }

    case Opcode::load_double_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint64_t value = memory.read64_be(address);
        state.fpr[instruction.fp_register] = value;
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::load_double:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        state.fpr[instruction.fp_register] = memory.read64_be(address);
        break;
    }

    case Opcode::load_double_update:
    {
        const std::uint32_t old_base = state.gpr[instruction.base];
        const std::uint32_t address =
            old_base + static_cast<std::uint32_t>(instruction.immediate);
        const std::uint64_t value = memory.read64_be(address);
        state.fpr[instruction.fp_register] = value;
        state.gpr[instruction.base] = address;
        break;
    }

    case Opcode::store_single:
    case Opcode::store_single_update:
    case Opcode::store_single_indexed:
    case Opcode::store_single_indexed_update:
    {
        const bool indexed = instruction.opcode == Opcode::store_single_indexed ||
            instruction.opcode == Opcode::store_single_indexed_update;
        const std::uint32_t address = indexed
            ? (instruction.base == 0U ? 0U : state.gpr[instruction.base]) +
                state.gpr[instruction.source]
            : effective_address(state, instruction.base, instruction.immediate);
        const double double_value = std::bit_cast<double>(state.fpr[instruction.fp_register]);
        const float single_value = static_cast<float>(double_value);
        const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_value);
        memory.write32_be(address, single_bits);
        pending_history_entry_.has_stored_single_value = true;
        pending_history_entry_.stored_single_value = single_bits;
        if (instruction.opcode == Opcode::store_single_update ||
            instruction.opcode == Opcode::store_single_indexed_update)
        {
            state.gpr[instruction.base] = address;
        }
        break;
    }

    case Opcode::store_double_indexed:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint64_t value = state.fpr[instruction.fp_register];
        memory.write64_be(address, value);
        pending_history_entry_.has_stored_double_value = true;
        pending_history_entry_.stored_double_value = value;
        break;
    }

    case Opcode::store_double_indexed_update:
    {
        const std::uint32_t base = state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        const std::uint64_t value = state.fpr[instruction.fp_register];
        memory.write64_be(address, value);
        state.gpr[instruction.base] = address;
        pending_history_entry_.has_stored_double_value = true;
        pending_history_entry_.stored_double_value = value;
        break;
    }

    case Opcode::store_double:
    case Opcode::store_double_update:
    {
        const std::uint32_t address =
            effective_address(state, instruction.base, instruction.immediate);
        const std::uint64_t value = state.fpr[instruction.fp_register];
        memory.write64_be(address, value);
        pending_history_entry_.has_stored_double_value = true;
        pending_history_entry_.stored_double_value = value;
        if (instruction.opcode == Opcode::store_double_update)
        {
            state.gpr[instruction.base] = address;
        }
        break;
    }

    case Opcode::paired_single_quantized_load:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address =
            base + static_cast<std::uint32_t>(instruction.immediate);
        const std::uint32_t gqr =
            state.gqr[instruction.quantized_register];
        const std::uint32_t load_type = (gqr >> 16U) & 0x7U;
        std::int32_t scale = static_cast<std::int32_t>((gqr >> 24U) & 0x3FU);
        if ((scale & 0x20) != 0)
        {
            scale -= 0x40;
        }

        const auto load_lane = [&](std::uint32_t lane_address) -> std::uint64_t {
            float value = 0.0F;
            switch (load_type)
            {
            case 0U:
                value = std::bit_cast<float>(memory.read32_be(lane_address));
                break;
            case 4U:
                value = static_cast<float>(memory.read8(lane_address));
                value = std::ldexp(value, -scale);
                break;
            case 5U:
                value = static_cast<float>(memory.read16_be(lane_address));
                value = std::ldexp(value, -scale);
                break;
            case 6U:
                value = static_cast<float>(std::bit_cast<std::int8_t>(
                    memory.read8(lane_address)));
                value = std::ldexp(value, -scale);
                break;
            case 7U:
                value = static_cast<float>(std::bit_cast<std::int16_t>(
                    memory.read16_be(lane_address)));
                value = std::ldexp(value, -scale);
                break;
            default:
                break;
            }
            return std::bit_cast<std::uint64_t>(static_cast<double>(value));
        };

        if (load_type < 4U)
        {
            // GQR load types 1-3 are reserved and have no defined result.
            if (load_type != 0U)
            {
                pending_history_entry_.completed = false;
                append_instruction_history(std::move(pending_history_entry_));
                has_pending_history_entry_ = false;
                return StepResult::unsupported_instruction;
            }
        }
        const std::uint64_t ps0 = load_lane(address);
        const std::uint64_t ps1 = instruction.paired_single_w
            ? std::bit_cast<std::uint64_t>(1.0)
            : load_lane(address + (load_type == 0U ? 4U :
                (load_type == 4U || load_type == 6U ? 1U : 2U)));
        state.fpr[instruction.fp_register] = ps0;
        state.fpr_ps1[instruction.fp_register] = ps1;
        break;
    }

    case Opcode::paired_single_quantized_store:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address =
            base + static_cast<std::uint32_t>(instruction.immediate);
        const std::uint32_t gqr = state.gqr[instruction.quantized_register];
        const std::uint32_t store_type = gqr & 0x7U;
        std::int32_t scale = static_cast<std::int32_t>((gqr >> 8U) & 0x3FU);
        if ((scale & 0x20) != 0)
        {
            scale -= 0x40;
        }

        if (store_type >= 1U && store_type <= 3U)
        {
            pending_history_entry_.completed = false;
            append_instruction_history(std::move(pending_history_entry_));
            has_pending_history_entry_ = false;
            return StepResult::unsupported_instruction;
        }

        const std::uint64_t ps0_raw = state.fpr[instruction.fp_register];
        const std::uint64_t ps1_raw = state.fpr_ps1[instruction.fp_register];
        const float ps0 = static_cast<float>(std::bit_cast<double>(ps0_raw));
        const float ps1 = static_cast<float>(std::bit_cast<double>(ps1_raw));
        const auto quantize_lane = [&](float lane) -> std::optional<std::uint32_t> {
            if (store_type == 0U)
            {
                return std::bit_cast<std::uint32_t>(lane);
            }
            if (std::isnan(lane))
            {
                return std::nullopt;
            }

            const float scaled = lane * std::ldexp(1.0F, scale);
            switch (store_type)
            {
            case 4U:
                return static_cast<std::uint32_t>(static_cast<std::uint8_t>(
                    static_cast<std::uint32_t>(std::clamp(scaled, 0.0F, 255.0F))));
            case 5U:
                return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
                    static_cast<std::uint32_t>(std::clamp(scaled, 0.0F, 65535.0F))));
            case 6U:
                return static_cast<std::uint32_t>(static_cast<std::uint8_t>(
                    static_cast<std::int32_t>(std::clamp(scaled, -128.0F, 127.0F))));
            case 7U:
                return static_cast<std::uint32_t>(static_cast<std::uint16_t>(
                    static_cast<std::int32_t>(std::clamp(scaled, -32768.0F, 32767.0F))));
            default:
                return std::nullopt;
            }
        };

        const auto value0 = quantize_lane(ps0);
        const auto value1 = instruction.paired_single_w
            ? std::optional<std::uint32_t>{0U}
            : quantize_lane(ps1);
        if (!value0 || !value1)
        {
            // Integer quantization of NaN is implementation-specific here; avoid
            // invoking undefined host conversions or fabricating guest bytes.
            pending_history_entry_.completed = false;
            append_instruction_history(std::move(pending_history_entry_));
            has_pending_history_entry_ = false;
            return StepResult::unsupported_instruction;
        }

        const std::size_t lane_width = store_type == 0U ? 4U
            : (store_type == 4U || store_type == 6U) ? 1U : 2U;
        const std::size_t access_width = instruction.paired_single_w
            ? lane_width : lane_width * 2U;
        memory.validate_write_range(address, access_width);
        const auto store_lane = [&](std::uint32_t lane_address, std::uint32_t value) {
            switch (store_type)
            {
            case 0U: memory.write32_be(lane_address, value); break;
            case 4U:
            case 6U: memory.write8(lane_address, static_cast<std::uint8_t>(value)); break;
            case 5U:
            case 7U: memory.write16_be(lane_address, static_cast<std::uint16_t>(value)); break;
            default: break;
            }
        };
        store_lane(address, *value0);
        if (!instruction.paired_single_w)
        {
            store_lane(address + static_cast<std::uint32_t>(lane_width), *value1);
        }
        pending_history_entry_.paired_fp_store_value0 = *value0;
        pending_history_entry_.paired_fp_store_value1 = *value1;
        break;
    }

    case Opcode::paired_single_merge00:
    case Opcode::paired_single_merge01:
    case Opcode::paired_single_merge10:
    case Opcode::paired_single_merge11:
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
        state.fpr[instruction.fp_register] =
            (instruction.opcode == Opcode::paired_single_merge10 ||
             instruction.opcode == Opcode::paired_single_merge11)
                ? source_a.ps1
                : source_a.ps0;
        state.fpr_ps1[instruction.fp_register] =
            (instruction.opcode == Opcode::paired_single_merge01 ||
             instruction.opcode == Opcode::paired_single_merge11)
                ? source_b.ps1
                : source_b.ps0;
        break;
    }

    case Opcode::paired_single_move_register:
    {
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        state.fpr[instruction.fp_register] = source_ps0;
        state.fpr_ps1[instruction.fp_register] = source_ps1;
        break;
    }

    case Opcode::paired_single_reciprocal_square_root_estimate:
    {
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        const auto estimate = [](std::uint64_t raw) {
            const double source = std::bit_cast<double>(raw);
            const float result = static_cast<float>(1.0 / std::sqrt(source));
            return std::bit_cast<std::uint64_t>(static_cast<double>(result));
        };
        state.fpr[instruction.fp_register] = estimate(source_ps0);
        state.fpr_ps1[instruction.fp_register] = estimate(source_ps1);
        break;
    }

    case Opcode::paired_single_reciprocal_estimate:
    {
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        const auto estimate = [](std::uint64_t raw) {
            const double source = std::bit_cast<double>(raw);
            const float result = static_cast<float>(1.0 / source);
            return std::bit_cast<std::uint64_t>(static_cast<double>(result));
        };
        state.fpr[instruction.fp_register] = estimate(source_ps0);
        state.fpr_ps1[instruction.fp_register] = estimate(source_ps1);
        break;
    }

    case Opcode::paired_single_select:
    {
        const std::uint64_t a_ps0 = state.fpr[instruction.fp_source_a];
        const std::uint64_t a_ps1 = state.fpr_ps1[instruction.fp_source_a];
        const std::uint64_t b_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t b_ps1 = state.fpr_ps1[instruction.fp_source_b];
        const std::uint64_t c_ps0 = state.fpr[instruction.fp_source_c];
        const std::uint64_t c_ps1 = state.fpr_ps1[instruction.fp_source_c];
        const double a0 = std::bit_cast<double>(a_ps0);
        const double a1 = std::bit_cast<double>(a_ps1);
        const std::uint64_t result_ps0 =
            !is_binary64_nan(a_ps0) && a0 >= 0.0 ? c_ps0 : b_ps0;
        const std::uint64_t result_ps1 =
            !is_binary64_nan(a_ps1) && a1 >= 0.0 ? c_ps1 : b_ps1;
        state.fpr[instruction.fp_register] = result_ps0;
        state.fpr_ps1[instruction.fp_register] = result_ps1;
        pending_history_entry_.paired_fp_select_result_ps0 = result_ps0;
        pending_history_entry_.paired_fp_select_result_ps1 = result_ps1;
        break;
    }

    case Opcode::paired_single_add:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const float result_ps0 = static_cast<float>(a_ps0 + b_ps0);
        const float result_ps1 = static_cast<float>(a_ps1 + b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_subtract:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const float result_ps0 = static_cast<float>(a_ps0 - b_ps0);
        const float result_ps1 = static_cast<float>(a_ps1 - b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_divide:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const float result_ps0 = static_cast<float>(a_ps0 / b_ps0);
        const float result_ps1 = static_cast<float>(a_ps1 / b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_multiply:
    case Opcode::paired_single_multiply_scalar0:
    case Opcode::paired_single_multiply_scalar1:
    {
        const std::uint64_t a_ps0_raw = state.fpr[instruction.fp_source_a];
        const std::uint64_t a_ps1_raw = state.fpr_ps1[instruction.fp_source_a];
        const std::uint64_t c_ps0_raw = state.fpr[instruction.fp_source_c];
        const std::uint64_t c_ps1_raw = state.fpr_ps1[instruction.fp_source_c];
        const double a_ps0 = std::bit_cast<double>(a_ps0_raw);
        const double a_ps1 = std::bit_cast<double>(a_ps1_raw);
        const bool scalar0 = instruction.opcode == Opcode::paired_single_multiply_scalar0;
        const bool scalar1 = instruction.opcode == Opcode::paired_single_multiply_scalar1;
        const bool scalar = scalar0 || scalar1;
        const std::uint64_t scalar_c_raw = scalar1 ? c_ps1_raw : c_ps0_raw;
        const double c_ps0 = std::bit_cast<double>(round_multiply_operand_25bit(
            scalar ? scalar_c_raw : c_ps0_raw));
        const double c_ps1 = scalar
            ? c_ps0
            : std::bit_cast<double>(round_multiply_operand_25bit(c_ps1_raw));
        const auto multiply_lane = [](double a, double c, bool flush_denormals) {
            float result = static_cast<float>(a * c);
            std::uint32_t bits = std::bit_cast<std::uint32_t>(result);
            if (flush_denormals && (bits & 0x7F800000U) == 0U &&
                (bits & 0x007FFFFFU) != 0U)
            {
                bits &= 0x80000000U;
                result = std::bit_cast<float>(bits);
            }
            return result;
        };
        const bool flush_denormals = !scalar;
        const float result_ps0 = multiply_lane(a_ps0, c_ps0, flush_denormals);
        const float result_ps1 = multiply_lane(a_ps1, c_ps1, flush_denormals);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_multiply_add:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const double c_ps0 = std::bit_cast<double>(round_multiply_operand_25bit(
            state.fpr[instruction.fp_source_c]));
        const double c_ps1 = std::bit_cast<double>(round_multiply_operand_25bit(
            state.fpr_ps1[instruction.fp_source_c]));
        const auto multiply_add_lane = [](double a, double c, double b) {
            const float rounded_product = static_cast<float>(a * c);
            float result = static_cast<float>(static_cast<double>(rounded_product) + b);
            std::uint32_t bits = std::bit_cast<std::uint32_t>(result);
            if ((bits & 0x7F800000U) == 0U && (bits & 0x007FFFFFU) != 0U)
            {
                bits &= 0x80000000U;
                result = std::bit_cast<float>(bits);
            }
            return result;
        };
        const float result_ps0 = multiply_add_lane(a_ps0, c_ps0, b_ps0);
        const float result_ps1 = multiply_add_lane(a_ps1, c_ps1, b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_multiply_add_scalar0:
    case Opcode::paired_single_multiply_add_scalar1:
    case Opcode::paired_single_negative_multiply_add:
    case Opcode::paired_single_negative_multiply_subtract:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        if (instruction.opcode == Opcode::paired_single_negative_multiply_add ||
            instruction.opcode == Opcode::paired_single_negative_multiply_subtract)
        {
            const double c_ps0 = std::bit_cast<double>(round_multiply_operand_25bit(
                state.fpr[instruction.fp_source_c]));
            const double c_ps1 = std::bit_cast<double>(round_multiply_operand_25bit(
                state.fpr_ps1[instruction.fp_source_c]));
            const bool subtract = instruction.opcode ==
                Opcode::paired_single_negative_multiply_subtract;
            const float result_ps0 = static_cast<float>(subtract
                ? -(a_ps0 * c_ps0 - b_ps0)
                : -(a_ps0 * c_ps0 + b_ps0));
            const float result_ps1 = static_cast<float>(subtract
                ? -(a_ps1 * c_ps1 - b_ps1)
                : -(a_ps1 * c_ps1 + b_ps1));
            state.fpr[instruction.fp_register] =
                std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
            state.fpr_ps1[instruction.fp_register] =
                std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
            break;
        }
        const std::uint64_t scalar_raw = instruction.opcode ==
                Opcode::paired_single_multiply_add_scalar0
            ? state.fpr[instruction.fp_source_c]
            : state.fpr_ps1[instruction.fp_source_c];
        const double c = std::bit_cast<double>(round_multiply_operand_25bit(scalar_raw));
        const float result_ps0 = static_cast<float>(a_ps0 * c + b_ps0);
        const float result_ps1 = static_cast<float>(a_ps1 * c + b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_multiply_subtract:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double a_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_a]);
        const double b_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_b]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const double c_ps0 = std::bit_cast<double>(round_multiply_operand_25bit(
            state.fpr[instruction.fp_source_c]));
        const double c_ps1 = std::bit_cast<double>(round_multiply_operand_25bit(
            state.fpr_ps1[instruction.fp_source_c]));
        const auto multiply_subtract_lane = [](double a, double c, double b) {
            return static_cast<float>(a * c - b);
        };
        const float result_ps0 = multiply_subtract_lane(a_ps0, c_ps0, b_ps0);
        const float result_ps1 = multiply_subtract_lane(a_ps1, c_ps1, b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_sum0:
    case Opcode::paired_single_sum1:
    {
        const double a_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_a]);
        const double b_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_b]);
        const double c_ps0 = std::bit_cast<double>(state.fpr[instruction.fp_source_c]);
        const double c_ps1 = std::bit_cast<double>(state.fpr_ps1[instruction.fp_source_c]);
        const bool sum0 = instruction.opcode == Opcode::paired_single_sum0;
        const float result_ps0 = static_cast<float>(sum0 ? a_ps0 + b_ps1 : c_ps0);
        const float result_ps1 = static_cast<float>(sum0 ? c_ps1 : a_ps0 + b_ps1);
        state.fpr[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps0));
        state.fpr_ps1[instruction.fp_register] =
            std::bit_cast<std::uint64_t>(static_cast<double>(result_ps1));
        break;
    }

    case Opcode::paired_single_negate:
    {
        constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        state.fpr[instruction.fp_register] = source_ps0 ^ sign_mask;
        state.fpr_ps1[instruction.fp_register] = source_ps1 ^ sign_mask;
        break;
    }

    case Opcode::paired_single_absolute_value:
    {
        constexpr std::uint64_t magnitude_mask = 0x7FFFFFFFFFFFFFFFULL;
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        state.fpr[instruction.fp_register] = source_ps0 & magnitude_mask;
        state.fpr_ps1[instruction.fp_register] = source_ps1 & magnitude_mask;
        break;
    }

    case Opcode::paired_single_negative_absolute_value:
    {
        constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
        const std::uint64_t source_ps0 = state.fpr[instruction.fp_source_b];
        const std::uint64_t source_ps1 = state.fpr_ps1[instruction.fp_source_b];
        state.fpr[instruction.fp_register] = source_ps0 | sign_mask;
        state.fpr_ps1[instruction.fp_register] = source_ps1 | sign_mask;
        break;
    }

    case Opcode::instruction_sync:
        // Instructions are fetched directly from guest memory. There is no
        // pipeline, I-cache, MMU synchronization state, or JIT cache to sync yet.
        break;

    case Opcode::memory_synchronize:
    case Opcode::memory_synchronize_lightweight:
        // Guest memory operations complete synchronously on the interpreter
        // lane. Keep the architectural ordering point visible to the host
        // memory model as well. A sequentially-consistent fence is stronger
        // than the lightweight lwsync ordering, but valid for both forms.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        break;

    case Opcode::enforce_in_order_io:
        // Guest memory accesses execute synchronously and in program order on
        // Affogato's single interpreter lane; there is no cache-inhibited I/O
        // mapping or asynchronous memory queue to order separately.
        break;

    case Opcode::data_cache_block_touch:
        // dcbt is a non-faulting prefetch hint. Affogato has no modeled D-cache
        // or cacheability attributes, so the hint has no observable effect.
        break;

    case Opcode::data_cache_block_touch_for_store:
        // dcbtst is likewise a non-faulting prefetch hint (for a future store).
        // With no modeled D-cache or cacheability attributes, it has no effect.
        break;

    case Opcode::data_cache_block_store:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        // dcbst is treated as a load for address translation/protection. Reads
        // are already immediately coherent in GuestMemory, so there is no
        // cache state to write back; validate the addressed byte nonetheless.
        static_cast<void>(memory.read8(address));
        break;
    }

    case Opcode::data_cache_block_flush:
    {
        const std::uint32_t base = instruction.base == 0U
            ? 0U : state.gpr[instruction.base];
        const std::uint32_t address = base + state.gpr[instruction.source];
        // dcbf writes back and invalidates a D-cache block. GuestMemory is
        // directly coherent and has no modeled D-cache, but the instruction
        // still performs a load-like address translation/protection check.
        static_cast<void>(memory.read8(address));
        break;
    }

    case Opcode::instruction_cache_block_invalidate:
        // Instructions are fetched from GuestMemory on every step; there is no
        // separate I-cache or translated-code cache whose block can be stale.
        break;

    case Opcode::condition_bit_and:
    case Opcode::condition_bit_and_complement:
    case Opcode::condition_bit_or:
    case Opcode::condition_bit_or_complement:
    case Opcode::condition_bit_nor:
    case Opcode::condition_bit_xor:
    case Opcode::condition_bit_nand:
    case Opcode::condition_bit_equivalence:
    {
        const bool source_a = read_cr_bit(state, instruction.cr_source_a_bit);
        const bool source_b = read_cr_bit(state, instruction.cr_source_b_bit);
        const bool result = instruction.opcode == Opcode::condition_bit_and
            ? source_a && source_b
            : instruction.opcode == Opcode::condition_bit_and_complement
                ? source_a && !source_b
                : instruction.opcode == Opcode::condition_bit_or
                    ? source_a || source_b
                    : instruction.opcode == Opcode::condition_bit_or_complement
                        ? source_a || !source_b
                        : instruction.opcode == Opcode::condition_bit_nor
                            ? !(source_a || source_b)
                            : instruction.opcode == Opcode::condition_bit_xor
                                ? source_a != source_b
                                : instruction.opcode == Opcode::condition_bit_nand
                                    ? !(source_a && source_b)
                                    : source_a == source_b;
        const std::uint32_t destination_mask =
            1U << (31U - instruction.cr_destination_bit);
        if (result)
        {
            state.cr |= destination_mask;
        }
        else
        {
            state.cr &= ~destination_mask;
        }
        pending_history_entry_.cr_bit_result = result;
        break;
    }

    case Opcode::move_condition_register_field:
    {
        const std::uint32_t source_shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.cr_field_source);
        const std::uint32_t destination_shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.cr_field);
        const std::uint32_t value = (state.cr >> source_shift) & 0xFU;
        const std::uint32_t destination_mask = 0xFU << destination_shift;
        state.cr = (state.cr & ~destination_mask) | (value << destination_shift);
        pending_history_entry_.cr_field_value = static_cast<std::uint8_t>(value);
        break;
    }

    case Opcode::move_condition_register_from_fpscr:
    {
        const std::uint32_t shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.fpscr_field);
        const std::uint32_t valid_field_mask =
            instruction.fpscr_field == 5U ? 0x7U : 0xFU;
        const std::uint32_t source_value =
            (state.fpscr >> shift) & valid_field_mask;
        const std::uint32_t cr_shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.cr_field);
        const std::uint32_t cr_mask = 0xFU << cr_shift;
        state.cr = (state.cr & ~cr_mask) | (source_value << cr_shift);

        std::uint32_t clear_mask = 0U;
        switch (instruction.fpscr_field)
        {
        case 0U:
            clear_mask = fpscr::fx_mask | fpscr::ox_mask;
            break;
        case 1U:
            clear_mask = fpscr::ux_mask | fpscr::zx_mask | fpscr::xx_mask |
                fpscr::vxsnan_mask;
            break;
        case 2U:
            clear_mask = fpscr::vxisi_mask | fpscr::vxidi_mask |
                fpscr::vxzdz_mask | fpscr::vximz_mask;
            break;
        case 3U:
            clear_mask = fpscr::vxvc_mask;
            break;
        case 5U:
            clear_mask = fpscr::vxsoft_mask | fpscr::vxsqrt_mask |
                fpscr::vxcvi_mask;
            break;
        default:
            break;
        }
        state.fpscr &= ~clear_mask;
        update_fpscr_summaries(state);
        pending_history_entry_.fpscr_after_cr_move = state.fpscr;
        pending_history_entry_.cr_field_value = static_cast<std::uint8_t>(source_value);
        break;
    }

    case Opcode::move_from_condition_register:
        state.gpr[instruction.destination] = state.cr;
        break;

    case Opcode::move_to_condition_register_fields:
    {
        const std::uint32_t source = state.gpr[instruction.destination];
        for (std::uint32_t field_index = 0U; field_index < 8U; ++field_index)
        {
            const std::uint32_t mask_bit = 1U << (7U - field_index);
            if ((instruction.cr_field_mask & mask_bit) == 0U)
            {
                continue;
            }

            const std::uint32_t shift = 28U - 4U * field_index;
            const std::uint32_t field_mask = 0xFU << shift;
            state.cr = (state.cr & ~field_mask) | (source & field_mask);
        }
        break;
    }

    case Opcode::move_condition_register_from_xer:
    {
        const std::uint32_t value = state.xer >> 28U;
        const std::uint32_t shift =
            28U - 4U * static_cast<std::uint32_t>(instruction.cr_field);
        const std::uint32_t cr_mask = 0xFU << shift;
        state.cr = (state.cr & ~cr_mask) | ((value & 0xFU) << shift);
        state.xer &= 0x0FFFFFFFU;
        pending_history_entry_.xer_cr_value = static_cast<std::uint8_t>(value);
        break;
    }

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
    if (pending_history_entry_.has_overflow_result)
    {
        pending_history_entry_.overflow_result = (state.xer & xer_overflow_mask) != 0U;
        pending_history_entry_.summary_overflow_result =
            (state.xer & xer_summary_overflow_mask) != 0U;
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
    if (pending_history_entry_.has_paired_fp_select)
    {
        pending_history_entry_.paired_fp_select_result_ps0 =
            state.fpr[pending_history_entry_.paired_fp_select_destination];
        pending_history_entry_.paired_fp_select_result_ps1 =
            state.fpr_ps1[pending_history_entry_.paired_fp_select_destination];
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
