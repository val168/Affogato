#include "cpu/espresso/recompiler.hpp"

#include <asmjit/core.h>
#include <asmjit/x86.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
#include <optional>
#include <type_traits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t xer_carry_mask = 0x20000000U;

struct NativeMemoryRecord
{
    std::uint32_t address{};
    std::uint32_t value{};
    std::uint64_t fp_value{};
    std::uint32_t base_value{};
    std::uint32_t index_value{};
    bool indexed{};
};

struct NativeFpCompareRecord
{
    std::uint64_t a_raw{};
    std::uint64_t b_raw{};
    std::uint8_t result{};
};

struct NativeFpIntegerConversionRecord
{
    std::uint64_t source_raw{};
    std::uint32_t integer_bits{};
    std::uint64_t encoded_raw{};
    bool invalid{};
    bool write_suppressed{};
};

struct NativeFrspRecord
{
    std::uint64_t source_raw{};
    std::uint64_t result_raw{};
    std::uint32_t single_bits{};
    bool wrote_result{};
};

struct NativeFmulRecord
{
    std::uint64_t a_raw{};
    std::uint64_t c_raw{};
    std::uint64_t result_raw{};
    bool wrote_result{};
};

struct NativeFmulsRecord
{
    std::uint64_t a_raw{};
    std::uint64_t c_raw{};
    std::uint64_t result_raw{};
    std::uint32_t single_bits{};
    bool wrote_result{};
};

struct NativeFmaddRecord
{
    std::uint64_t a_raw{};
    std::uint64_t b_raw{};
    std::uint64_t c_raw{};
    std::uint64_t result_raw{};
    bool wrote_result{};
};

struct NativeFsubRecord
{
    std::uint64_t a_raw{};
    std::uint64_t b_raw{};
    std::uint64_t result_raw{};
    bool wrote_result{};
};

struct NativePsqStoreRecord
{
    std::uint64_t source_ps0{};
    std::uint64_t source_ps1{};
    std::uint32_t value0{};
    std::uint32_t value1{};
    std::uint32_t gqr{};
};

struct NativePsqLoadRecord
{
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
    std::uint32_t gqr{};
};

struct NativeMulliRecord
{
    std::uint32_t source_value{};
    std::uint32_t result{};
};

struct NativeMullwRecord
{
    std::uint32_t source_a{};
    std::uint32_t source_b{};
    std::uint32_t result{};
};

struct NativeExecutionContext
{
    GuestMemory* memory{};
    std::uint32_t block_cia{};
    std::array<NativeMemoryRecord, 64> memory_records{};
    std::array<NativeFpCompareRecord, 64> fp_compare_records{};
    std::array<NativeFpIntegerConversionRecord, 64> fp_integer_conversion_records{};
    std::array<NativeFrspRecord, 64> frsp_records{};
    std::array<NativeFmulRecord, 64> fmul_records{};
    std::array<NativeFmulsRecord, 64> fmuls_records{};
    std::array<NativeFmaddRecord, 64> fmadd_records{};
    std::array<NativeFsubRecord, 64> fsub_records{};
    std::array<NativePsqStoreRecord, 64> psq_store_records{};
    std::array<NativePsqLoadRecord, 64> psq_load_records{};
    std::array<NativeMulliRecord, 64> mulli_records{};
    std::array<NativeMullwRecord, 64> mullw_records{};
    std::uint64_t memory_access_calls{};
    bool failed{};
    bool guest_memory_fault{};
    bool unsupported_instruction{};
    std::exception_ptr exception{};
    std::uint32_t fault_cia{};
    std::uint32_t failed_address{};
    std::uint32_t failed_width{};
    std::uint32_t failed_flags{};
    std::uint32_t failed_instruction_index{};
};

extern "C" std::uint32_t native_psq_load(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t address,
    std::uint32_t fp_register,
    std::uint32_t quantized_register,
    std::uint32_t single_element,
    std::uint32_t instruction_index) noexcept
{
    if (instruction_index >= context->psq_load_records.size() ||
        quantized_register >= state->gqr.size() || fp_register >= state->fpr.size())
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 0U;
        context->failed_flags = 0U;
        context->failed_instruction_index = instruction_index;
        return 2U;
    }

    const std::uint32_t gqr = state->gqr[quantized_register];
    const std::uint32_t load_type = (gqr >> 16U) & 0x7U;
    std::int32_t scale = static_cast<std::int32_t>((gqr >> 24U) & 0x3FU);
    if ((scale & 0x20) != 0)
    {
        scale -= 0x40;
    }
    if (load_type >= 1U && load_type <= 3U)
    {
        context->unsupported_instruction = true;
        return 1U;
    }

    const std::uint32_t lane_width = load_type == 0U ? 4U
        : (load_type == 4U || load_type == 6U) ? 1U : 2U;
    const auto load_lane = [&](std::uint32_t lane_address) -> std::uint64_t {
        ++context->memory_access_calls;
        float value = 0.0F;
        switch (load_type)
        {
        case 0U:
            value = std::bit_cast<float>(context->memory->read32_be(lane_address));
            break;
        case 4U:
            value = static_cast<float>(context->memory->read8(lane_address));
            value = std::ldexp(value, -scale);
            break;
        case 5U:
            value = static_cast<float>(context->memory->read16_be(lane_address));
            value = std::ldexp(value, -scale);
            break;
        case 6U:
            value = static_cast<float>(std::bit_cast<std::int8_t>(
                context->memory->read8(lane_address)));
            value = std::ldexp(value, -scale);
            break;
        case 7U:
            value = static_cast<float>(std::bit_cast<std::int16_t>(
                context->memory->read16_be(lane_address)));
            value = std::ldexp(value, -scale);
            break;
        default:
            break;
        }
        return std::bit_cast<std::uint64_t>(static_cast<double>(value));
    };

    NativePsqLoadRecord& record = context->psq_load_records[instruction_index];
    try
    {
        record.destination_ps0 = load_lane(address);
        record.destination_ps1 = single_element != 0U
            ? std::bit_cast<std::uint64_t>(1.0)
            : load_lane(address + lane_width);
        record.gqr = gqr;
        // Commit both lanes only after every guest read has succeeded.
        state->fpr[fp_register] = record.destination_ps0;
        state->fpr_ps1[fp_register] = record.destination_ps1;
        NativeMemoryRecord& memory_record = context->memory_records[instruction_index];
        memory_record.address = address;
        memory_record.indexed = false;
        return 0U;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = lane_width;
        context->failed_flags = 0U;
        context->failed_instruction_index = instruction_index;
        context->fault_cia = context->block_cia + instruction_index * 4U;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault& fault)
        {
            context->guest_memory_fault = true;
            context->failed_address = fault.address();
        }
        catch (...)
        {
        }
        return 2U;
    }
}

extern "C" std::uint32_t native_psq_store(
    NativeExecutionContext* context,
    const CpuState* state,
    std::uint32_t address,
    std::uint32_t fp_register,
    std::uint32_t quantized_register,
    std::uint32_t single_element,
    std::uint32_t instruction_index) noexcept
{
    if (instruction_index >= context->psq_store_records.size() ||
        quantized_register >= state->gqr.size() || fp_register >= state->fpr.size())
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 0U;
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        return 2U;
    }

    const std::uint32_t gqr = state->gqr[quantized_register];
    const std::uint32_t store_type = gqr & 0x7U;
    std::int32_t scale = static_cast<std::int32_t>((gqr >> 8U) & 0x3FU);
    if ((scale & 0x20) != 0)
    {
        scale -= 0x40;
    }
    if (store_type >= 1U && store_type <= 3U)
    {
        context->unsupported_instruction = true;
        return 1U;
    }

    NativePsqStoreRecord& record = context->psq_store_records[instruction_index];
    record.source_ps0 = state->fpr[fp_register];
    record.source_ps1 = state->fpr_ps1[fp_register];
    record.gqr = gqr;
    const float ps0 = static_cast<float>(std::bit_cast<double>(record.source_ps0));
    const float ps1 = static_cast<float>(std::bit_cast<double>(record.source_ps1));
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
    const auto value1 = single_element != 0U
        ? std::optional<std::uint32_t>{0U}
        : quantize_lane(ps1);
    if (!value0 || !value1)
    {
        // Match the interpreter: unsupported quantization does not touch memory.
        context->unsupported_instruction = true;
        return 1U;
    }

    const std::uint32_t lane_width = store_type == 0U ? 4U
        : (store_type == 4U || store_type == 6U) ? 1U : 2U;
    const std::uint32_t access_width = single_element != 0U
        ? lane_width : lane_width * 2U;
    try
    {
        // Validate the complete architectural transfer before either lane is
        // written, matching the interpreter's all-or-nothing fault behavior.
        context->memory->validate_write_range(address, access_width);
        record.value0 = *value0;
        record.value1 = *value1;
        for (std::uint32_t lane = 0U; lane < (single_element != 0U ? 1U : 2U); ++lane)
        {
            const std::uint32_t lane_address = address + lane * lane_width;
            const std::uint32_t value = lane == 0U ? *value0 : *value1;
            if (context->memory->has_write_watchpoint())
            {
                context->memory->set_current_guest_cia(
                    context->block_cia + instruction_index * 4U);
            }
            ++context->memory_access_calls;
            switch (store_type)
            {
            case 0U: context->memory->write32_be(lane_address, value); break;
            case 4U:
            case 6U: context->memory->write8(lane_address, static_cast<std::uint8_t>(value)); break;
            case 5U:
            case 7U: context->memory->write16_be(lane_address, static_cast<std::uint16_t>(value)); break;
            default: break;
            }
        }
        NativeMemoryRecord& memory_record = context->memory_records[instruction_index];
        memory_record.address = address;
        memory_record.fp_value = record.source_ps0;
        memory_record.value = *value0;
        memory_record.indexed = false;
        return 0U;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = access_width;
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        context->fault_cia = context->block_cia + instruction_index * 4U;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 2U;
    }
}

static_assert(std::is_standard_layout_v<NativeExecutionContext>);

using NativeBlock = std::uint32_t (*)(CpuState*, NativeExecutionContext*);

void native_update_fpscr_summaries(CpuState* state) noexcept
{
    constexpr std::uint32_t implemented_invalid_masks =
        fpscr::vxsnan_mask | fpscr::vxzdz_mask | fpscr::vximz_mask |
        fpscr::vxidi_mask | fpscr::vxisi_mask | fpscr::vxcvi_mask |
        fpscr::vxvc_mask | fpscr::vxsqrt_mask | fpscr::vxsoft_mask;
    state->fpscr &= ~(fpscr::vx_mask | fpscr::fex_mask);
    if ((state->fpscr & implemented_invalid_masks) != 0U)
    {
        state->fpscr |= fpscr::vx_mask;
    }
    const bool enabled_exception =
        ((state->fpscr & fpscr::vx_mask) != 0U &&
         (state->fpscr & fpscr::ve_mask) != 0U) ||
        ((state->fpscr & fpscr::zx_mask) != 0U &&
         (state->fpscr & fpscr::ze_mask) != 0U) ||
        ((state->fpscr & fpscr::ox_mask) != 0U &&
         (state->fpscr & fpscr::oe_mask) != 0U) ||
        ((state->fpscr & fpscr::ux_mask) != 0U &&
         (state->fpscr & fpscr::ue_mask) != 0U) ||
        ((state->fpscr & fpscr::xx_mask) != 0U &&
         (state->fpscr & fpscr::xe_mask) != 0U);
    if (enabled_exception)
    {
        state->fpscr |= fpscr::fex_mask;
    }
}

[[nodiscard]] std::uint8_t native_single_result_fprf(std::uint32_t bits) noexcept
{
    const bool negative = (bits & 0x80000000U) != 0U;
    const std::uint32_t exponent = bits & 0x7F800000U;
    const std::uint32_t fraction = bits & 0x007FFFFFU;
    if (exponent == 0x7F800000U)
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

[[nodiscard]] std::uint8_t native_double_result_fprf(std::uint64_t bits) noexcept
{
    const bool negative = (bits & 0x8000000000000000ULL) != 0U;
    const std::uint64_t exponent = bits & 0x7FF0000000000000ULL;
    const std::uint64_t fraction = bits & 0x000FFFFFFFFFFFFFULL;
    if (exponent == 0x7FF0000000000000ULL)
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

void native_set_double_result_status(
    CpuState* state, std::uint64_t raw_result, bool inexact, bool rounded_up) noexcept
{
    state->fpscr &= ~(fpscr::fprf_mask | fpscr::fi_mask | fpscr::fr_mask);
    state->fpscr |= static_cast<std::uint32_t>(
        native_double_result_fprf(raw_result)) << 12U;
    if (inexact)
    {
        state->fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
        if (rounded_up)
        {
            state->fpscr |= fpscr::fr_mask;
        }
    }
    native_update_fpscr_summaries(state);
}

void native_set_double_fprf(CpuState* state, std::uint64_t raw_result) noexcept
{
    state->fpscr = (state->fpscr & ~fpscr::fprf_mask) |
        (static_cast<std::uint32_t>(native_double_result_fprf(raw_result)) << 12U);
}

[[nodiscard]] bool native_binary64_normal_or_zero(std::uint64_t raw) noexcept
{
    const std::uint64_t exponent = raw & 0x7FF0000000000000ULL;
    const std::uint64_t fraction = raw & 0x000FFFFFFFFFFFFFULL;
    return exponent != 0x7FF0000000000000ULL &&
        (exponent != 0U || fraction == 0U);
}

extern "C" void native_raise_fpscr_exception(
    CpuState* state, std::uint32_t subexception) noexcept
{
    state->fpscr |= subexception | fpscr::fx_mask;
    native_update_fpscr_summaries(state);
}

extern "C" void native_raise_fpscr_snan(CpuState* state) noexcept
{
    native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
}

extern "C" void native_raise_fpscr_snan_if_present(
    CpuState* state, std::uint64_t a_raw, std::uint64_t b_raw) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    const auto is_signaling_nan = [&](std::uint64_t value) {
        return (value & exponent_mask) == exponent_mask &&
            (value & fraction_mask) != 0U && (value & quiet_bit) == 0U;
    };
    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw))
    {
        native_raise_fpscr_snan(state);
    }
}

extern "C" void native_fctiwz(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_register,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t quiet_nan_mask = 0x0008000000000000ULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t integer_encoding_prefix = 0xFFF8000000000000ULL;
    constexpr double minimum_signed_word = -2147483648.0;
    constexpr double one_past_maximum_signed_word = 2147483648.0;

    const std::uint64_t source_raw = state->fpr[source_register];
    const double source = std::bit_cast<double>(source_raw);
    const bool is_nan = (source_raw & exponent_mask) == exponent_mask &&
        (source_raw & fraction_mask) != 0U;
    const bool is_signaling_nan = is_nan && (source_raw & quiet_nan_mask) == 0U;
    std::uint32_t integer_bits = 0x80000000U;
    bool invalid = false;
    bool inexact = false;

    if (is_nan)
    {
        invalid = true;
        native_raise_fpscr_exception(state, fpscr::vxcvi_mask);
        if (is_signaling_nan)
        {
            native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
        }
    }
    else
    {
        const double rounded = std::trunc(source);
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
            integer_bits = std::bit_cast<std::uint32_t>(
                static_cast<std::int32_t>(rounded));
            inexact = rounded != source;
        }

        if (invalid)
        {
            native_raise_fpscr_exception(state, fpscr::vxcvi_mask);
        }
    }

    state->fpscr &= ~(fpscr::fi_mask | fpscr::fr_mask);
    if (!invalid && inexact)
    {
        state->fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
    }
    native_update_fpscr_summaries(state);

    const std::uint64_t encoded = integer_encoding_prefix | integer_bits |
        (integer_bits == 0U && (source_raw & sign_mask) != 0U
            ? 0x0000000100000000ULL
            : 0U);
    const bool write_suppressed = invalid &&
        (state->fpscr & fpscr::ve_mask) != 0U;
    if (!write_suppressed)
    {
        state->fpr[destination] = encoded;
    }

    if (instruction_index < context->fp_integer_conversion_records.size())
    {
        NativeFpIntegerConversionRecord& record =
            context->fp_integer_conversion_records[instruction_index];
        record.source_raw = source_raw;
        record.integer_bits = integer_bits;
        record.encoded_raw = encoded;
        record.invalid = invalid;
        record.write_suppressed = write_suppressed;
    }
}

extern "C" void native_frsp(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_register,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    if (destination >= state->fpr.size() || source_register >= state->fpr.size() ||
        instruction_index >= context->frsp_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFrspRecord& record = context->frsp_records[instruction_index];
    record.source_raw = state->fpr[source_register];
    const std::uint64_t source_raw = record.source_raw;
    const bool is_nan = (source_raw & exponent_mask) == exponent_mask &&
        (source_raw & fraction_mask) != 0U;
    const bool is_signaling_nan = is_nan && (source_raw & quiet_bit) == 0U;
    bool inexact = false;
    bool rounded_up = false;
    if (is_signaling_nan)
    {
        native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
        if ((state->fpscr & fpscr::ve_mask) != 0U)
        {
            return;
        }
    }

    float rounded{};
    if (is_nan)
    {
        const std::uint32_t sign = (source_raw & sign_mask) != 0U ? 0x80000000U : 0U;
        const std::uint32_t payload = static_cast<std::uint32_t>(
            (source_raw & fraction_mask) >> 29U);
        record.single_bits = sign | 0x7F800000U | payload | 0x00400000U;
        rounded = std::bit_cast<float>(record.single_bits);
    }
    else
    {
        // Match the interpreter's currently modeled host conversion policy;
        // FPSCR.RN and exact overflow/underflow exception modeling remain TODOs there.
        rounded = static_cast<float>(std::bit_cast<double>(source_raw));
        record.single_bits = std::bit_cast<std::uint32_t>(rounded);
        const double extended = static_cast<double>(rounded);
        inexact = extended != std::bit_cast<double>(source_raw);
        rounded_up = inexact &&
            std::fabs(extended) > std::fabs(std::bit_cast<double>(source_raw));
    }

    state->fpscr &= ~(fpscr::fprf_mask | fpscr::fi_mask | fpscr::fr_mask);
    state->fpscr |= static_cast<std::uint32_t>(
        native_single_result_fprf(record.single_bits)) << 12U;
    if (inexact)
    {
        state->fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
        if (rounded_up)
        {
            state->fpscr |= fpscr::fr_mask;
        }
    }
    native_update_fpscr_summaries(state);

    record.result_raw = std::bit_cast<std::uint64_t>(static_cast<double>(rounded));
    state->fpr[destination] = record.result_raw;
    state->fpr_ps1[destination] = record.result_raw;
    record.wrote_result = true;
}

extern "C" void native_fmul(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_c >= state->fpr.size() || instruction_index >= context->fmul_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFmulRecord& record = context->fmul_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.c_raw = state->fpr[source_c];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t c_raw = record.c_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask &&
            (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_zero = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0U;
    };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0x7FF0000000000000ULL;
    };
    const auto commit = [&](std::uint64_t raw_result, bool inexact, bool rounded_up) {
        state->fpr[destination] = raw_result;
        native_set_double_result_status(state, raw_result, inexact, rounded_up);
        record.result_raw = raw_result;
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        const std::uint64_t quieted = is_nan(source_raw)
            ? source_raw | quiet_bit : 0x7FF8000000000000ULL;
        commit(quieted, false, false);
    };
    const auto signal_invalid = [&](std::uint32_t exception, std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(c_raw))
    {
        signal_invalid(fpscr::vxsnan_mask,
            is_signaling_nan(a_raw) ? a_raw : c_raw);
        return;
    }
    if (is_nan(a_raw) || is_nan(c_raw))
    {
        commit_nan(is_nan(a_raw) ? a_raw : c_raw);
        return;
    }

    const bool a_zero = is_zero(a_raw);
    const bool c_zero = is_zero(c_raw);
    const bool a_infinity = is_infinity(a_raw);
    const bool c_infinity = is_infinity(c_raw);
    if ((a_infinity && c_zero) || (a_zero && c_infinity))
    {
        signal_invalid(fpscr::vximz_mask, 0U);
        return;
    }

    const double a = std::bit_cast<double>(a_raw);
    const double c = std::bit_cast<double>(c_raw);
    const double result = a * c;
    const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
    if (a_infinity || c_infinity)
    {
        commit(raw_result, false, false);
        return;
    }

    const bool subnormal_or_overflow_case =
        !native_binary64_normal_or_zero(a_raw) ||
        !native_binary64_normal_or_zero(c_raw) ||
        !native_binary64_normal_or_zero(raw_result) ||
        (is_zero(raw_result) && !a_zero && !c_zero);
    if (subnormal_or_overflow_case)
    {
        // Match the interpreter's scoped policy: exact raw result and FPRF,
        // without guessing FI/FR/XX/OX/UX for subnormal/overflow cases.
        state->fpr[destination] = raw_result;
        native_set_double_fprf(state, raw_result);
        record.result_raw = raw_result;
        record.wrote_result = true;
        return;
    }

    const double error = std::fma(a, c, -result);
    const bool inexact = error != 0.0;
    const bool rounded_up = inexact && result != 0.0 &&
        ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
    commit(raw_result, inexact, rounded_up);
}

extern "C" void native_fmuls(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_c >= state->fpr.size() || instruction_index >= context->fmuls_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFmulsRecord& record = context->fmuls_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.c_raw = state->fpr[source_c];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t c_raw = record.c_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_zero = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0U;
    };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0x7FF0000000000000ULL;
    };
    const auto quiet_single = [&](std::uint64_t raw) {
        if (!is_nan(raw))
        {
            raw = 0x7FF8000000000000ULL;
        }
        const std::uint32_t sign = (raw & sign_mask) != 0U ? 0x80000000U : 0U;
        const std::uint32_t payload = static_cast<std::uint32_t>(
            (raw & fraction_mask) >> 29U);
        return sign | 0x7F800000U | payload | 0x00400000U;
    };
    const auto commit = [&](std::uint32_t single_bits, bool inexact, bool rounded_up) {
        record.single_bits = single_bits;
        const double extended = static_cast<double>(std::bit_cast<float>(single_bits));
        record.result_raw = std::bit_cast<std::uint64_t>(extended);
        state->fpr[destination] = record.result_raw;
        state->fpr_ps1[destination] = record.result_raw;
        state->fpscr &= ~(fpscr::fprf_mask | fpscr::fi_mask | fpscr::fr_mask);
        state->fpscr |= static_cast<std::uint32_t>(
            native_single_result_fprf(single_bits)) << 12U;
        if (inexact)
        {
            state->fpscr |= fpscr::fi_mask | fpscr::xx_mask | fpscr::fx_mask;
            if (rounded_up)
            {
                state->fpscr |= fpscr::fr_mask;
            }
        }
        native_update_fpscr_summaries(state);
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        commit(quiet_single(source_raw), false, false);
    };
    const auto signal_invalid = [&](std::uint32_t exception, std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(c_raw))
    {
        signal_invalid(fpscr::vxsnan_mask,
            is_signaling_nan(a_raw) ? a_raw : c_raw);
        return;
    }
    if (is_nan(a_raw) || is_nan(c_raw))
    {
        commit_nan(is_nan(a_raw) ? a_raw : c_raw);
        return;
    }

    const bool a_zero = is_zero(a_raw);
    const bool c_zero = is_zero(c_raw);
    const bool a_infinity = is_infinity(a_raw);
    const bool c_infinity = is_infinity(c_raw);
    if ((a_infinity && c_zero) || (a_zero && c_infinity))
    {
        signal_invalid(fpscr::vximz_mask, 0U);
        return;
    }

    std::uint64_t rounded_c_raw = c_raw;
    const std::uint64_t c_exponent = c_raw & exponent_mask;
    const std::uint64_t c_fraction = c_raw & fraction_mask;
    if (c_exponent == 0U && c_fraction != 0U)
    {
        const unsigned shift = static_cast<unsigned>(std::countl_zero(c_fraction) - 11);
        const std::int64_t signed_mask =
            std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
        const std::uint64_t keep_mask = static_cast<std::uint64_t>(signed_mask >> shift);
        rounded_c_raw = (c_raw & keep_mask) + (c_raw & (0x0000000008000000ULL >> shift));
    }
    else
    {
        rounded_c_raw = (c_raw & 0xFFFFFFFFF8000000ULL) +
            (c_raw & 0x0000000008000000ULL);
    }

    const double a = std::bit_cast<double>(a_raw);
    const double rounded_c = std::bit_cast<double>(rounded_c_raw);
    const double exact_result = a * rounded_c;
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fmadd(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->fmadd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFmaddRecord& record = context->fmadd_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    record.c_raw = state->fpr[source_c];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const std::uint64_t c_raw = record.c_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask &&
            (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_zero = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0U;
    };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0x7FF0000000000000ULL;
    };
    const auto commit = [&](std::uint64_t raw_result, bool known_exact) {
        state->fpr[destination] = raw_result;
        if (known_exact)
        {
            native_set_double_result_status(state, raw_result, false, false);
        }
        else
        {
            // Mirror the interpreter's deliberately scoped FMA status policy.
            native_set_double_fprf(state, raw_result);
        }
        record.result_raw = raw_result;
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        const std::uint64_t quieted = is_nan(source_raw)
            ? source_raw | quiet_bit : 0x7FF8000000000000ULL;
        commit(quieted, true);
    };
    const auto signal_invalid = [&](std::uint32_t exception, std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw) || is_signaling_nan(c_raw))
    {
        const std::uint64_t source = is_signaling_nan(a_raw) ? a_raw
            : is_signaling_nan(c_raw) ? c_raw : b_raw;
        signal_invalid(fpscr::vxsnan_mask, source);
        return;
    }
    if (is_nan(a_raw) || is_nan(c_raw) || is_nan(b_raw))
    {
        const std::uint64_t source = is_nan(a_raw) ? a_raw
            : is_nan(c_raw) ? c_raw : b_raw;
        commit_nan(source);
        return;
    }

    const bool a_zero = is_zero(a_raw);
    const bool c_zero = is_zero(c_raw);
    const bool a_infinity = is_infinity(a_raw);
    const bool c_infinity = is_infinity(c_raw);
    if ((a_infinity && c_zero) || (a_zero && c_infinity))
    {
        signal_invalid(fpscr::vximz_mask, 0U);
        return;
    }

    const bool b_infinity = is_infinity(b_raw);
    const bool product_is_infinite = a_infinity || c_infinity;
    const bool product_is_negative = ((a_raw ^ c_raw) & sign_mask) != 0U;
    const bool b_is_negative = (b_raw & sign_mask) != 0U;
    if (product_is_infinite && b_infinity && product_is_negative != b_is_negative)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double a = std::bit_cast<double>(a_raw);
    const double b = std::bit_cast<double>(b_raw);
    const double c = std::bit_cast<double>(c_raw);
    const double result = std::fma(a, c, b);
    const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
    const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
    commit(raw_result, known_exact);
}

extern "C" void native_fsub(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->fsub_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFsubRecord& record = context->fsub_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask &&
            (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == 0x7FF0000000000000ULL;
    };
    const auto commit = [&](std::uint64_t raw_result, bool inexact, bool rounded_up) {
        state->fpr[destination] = raw_result;
        native_set_double_result_status(state, raw_result, inexact, rounded_up);
        record.result_raw = raw_result;
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        const std::uint64_t quieted = is_nan(source_raw)
            ? source_raw | quiet_bit : 0x7FF8000000000000ULL;
        commit(quieted, false, false);
    };
    const auto signal_invalid = [&](std::uint32_t exception, std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw))
    {
        signal_invalid(fpscr::vxsnan_mask,
            is_signaling_nan(a_raw) ? a_raw : b_raw);
        return;
    }
    if (is_nan(a_raw) || is_nan(b_raw))
    {
        commit_nan(is_nan(a_raw) ? a_raw : b_raw);
        return;
    }

    const bool a_infinity = is_infinity(a_raw);
    const bool b_infinity = is_infinity(b_raw);
    if (a_infinity && b_infinity && ((a_raw ^ b_raw) & sign_mask) == 0U)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double a = std::bit_cast<double>(a_raw);
    const double b = std::bit_cast<double>(b_raw);
    const double result = a - b;
    const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
    if (a_infinity || b_infinity)
    {
        commit(raw_result, false, false);
        return;
    }

    if (native_binary64_normal_or_zero(a_raw) &&
        native_binary64_normal_or_zero(b_raw) &&
        native_binary64_normal_or_zero(raw_result))
    {
        const double virtual_b = a - result;
        const double virtual_a = result + virtual_b;
        const double roundoff_b = virtual_b - b;
        const double roundoff_a = a - virtual_a;
        const double error = roundoff_a + roundoff_b;
        const bool inexact = error != 0.0;
        const bool rounded_up = inexact && result != 0.0 &&
            ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
        commit(raw_result, inexact, rounded_up);
    }
    else
    {
        // Match the interpreter's current subnormal/overflow status scope.
        state->fpr[destination] = raw_result;
        native_set_double_fprf(state, raw_result);
        record.result_raw = raw_result;
        record.wrote_result = true;
    }
}

extern "C" std::uint32_t native_memory_access(
    NativeExecutionContext* context,
    std::uint32_t address,
    std::uint32_t value,
    std::uint32_t width,
    std::uint32_t flags,
    std::uint32_t instruction_index) noexcept
{
    try
    {
        ++context->memory_access_calls;
        if (instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = width;
            context->failed_flags = flags;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        if (context->memory->has_write_watchpoint())
        {
            context->memory->set_current_guest_cia(
                context->block_cia + instruction_index * 4U);
        }
        record.address = address;
        record.indexed = (flags & 4U) != 0U;
        record.value = value;
        if ((flags & 1U) != 0U)
        {
            switch (width)
            {
            case 1: context->memory->write8(address, static_cast<std::uint8_t>(value)); break;
            case 2: context->memory->write16_be(address, static_cast<std::uint16_t>(value)); break;
            case 4: context->memory->write32_be(address, value); break;
            default:
                context->failed = true;
                context->failed_address = address;
                context->failed_width = width;
                context->failed_flags = flags;
                context->failed_instruction_index = instruction_index;
                return 0U;
            }
            return 0;
        }

        std::uint32_t loaded{};
        switch (width)
        {
        case 1: loaded = context->memory->read8(address); break;
        case 2: loaded = context->memory->read16_be(address); break;
        case 4: loaded = context->memory->read32_be(address); break;
        default:
            context->failed = true;
            context->failed_address = address;
            context->failed_width = width;
            context->failed_flags = flags;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        if ((flags & 2U) != 0U)
        {
            if (width == 1U)
                loaded = static_cast<std::uint32_t>(static_cast<std::int32_t>(
                    std::bit_cast<std::int8_t>(static_cast<std::uint8_t>(loaded))));
            else if (width == 2U)
                loaded = static_cast<std::uint32_t>(static_cast<std::int32_t>(
                    std::bit_cast<std::int16_t>(static_cast<std::uint16_t>(loaded))));
        }
        record.value = loaded;
        return loaded;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = width;
        context->failed_flags = flags;
        context->failed_instruction_index = instruction_index;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 0;
    }
}

extern "C" void native_load_multiple_words(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t address,
    std::uint32_t first_register,
    std::uint32_t instruction_index) noexcept
{
    if (instruction_index >= context->memory_records.size())
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = sizeof(std::uint32_t);
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeMemoryRecord& record = context->memory_records[instruction_index];
    record.address = address;
    record.value = 0U;
    record.indexed = false;
    for (std::uint32_t reg = first_register; reg < state->gpr.size(); ++reg)
    {
        ++context->memory_access_calls;
        try
        {
            state->gpr[reg] = context->memory->read32_be(address);
        }
        catch (...)
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = sizeof(std::uint32_t);
            context->failed_flags = 0U;
            context->failed_instruction_index = instruction_index;
            context->exception = std::current_exception();
            try
            {
                std::rethrow_exception(context->exception);
            }
            catch (const GuestMemoryFault&)
            {
                context->guest_memory_fault = true;
            }
            catch (...)
            {
            }
            return;
        }
        address += sizeof(std::uint32_t);
    }
}

extern "C" void native_store_multiple_words(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t address,
    std::uint32_t first_register,
    std::uint32_t instruction_index) noexcept
{
    if (instruction_index >= context->memory_records.size())
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = sizeof(std::uint32_t);
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeMemoryRecord& record = context->memory_records[instruction_index];
    record.address = address;
    record.value = 0U;
    record.indexed = false;
    for (std::uint32_t reg = first_register; reg < state->gpr.size(); ++reg)
    {
        ++context->memory_access_calls;
        if (context->memory->has_write_watchpoint())
        {
            context->memory->set_current_guest_cia(
                context->block_cia + instruction_index * 4U);
        }
        try
        {
            context->memory->write32_be(address, state->gpr[reg]);
        }
        catch (...)
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = sizeof(std::uint32_t);
            context->failed_flags = 1U;
            context->failed_instruction_index = instruction_index;
            context->exception = std::current_exception();
            try
            {
                std::rethrow_exception(context->exception);
            }
            catch (const GuestMemoryFault&)
            {
                context->guest_memory_fault = true;
            }
            catch (...)
            {
            }
            return;
        }
        address += sizeof(std::uint32_t);
    }
}

extern "C" std::uint64_t native_load_double(
    NativeExecutionContext* context,
    std::uint32_t address,
    std::uint32_t instruction_index) noexcept
{
    try
    {
        ++context->memory_access_calls;
        if (instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = 8U;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.indexed = false;
        record.fp_value = context->memory->read64_be(address);
        return record.fp_value;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 8U;
        context->failed_flags = 0U;
        context->failed_instruction_index = instruction_index;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 0U;
    }
}

extern "C" std::uint32_t native_store_single(
    NativeExecutionContext* context,
    const CpuState* state,
    std::uint32_t address,
    std::uint32_t fp_register,
    std::uint32_t instruction_index) noexcept
{
    try
    {
        ++context->memory_access_calls;
        if (instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = 4U;
            context->failed_flags = 1U;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        if (context->memory->has_write_watchpoint())
        {
            context->memory->set_current_guest_cia(
                context->block_cia + instruction_index * 4U);
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.indexed = false;
        record.fp_value = state->fpr[fp_register];
        const float single_value = static_cast<float>(
            std::bit_cast<double>(record.fp_value));
        record.value = std::bit_cast<std::uint32_t>(single_value);
        context->memory->write32_be(address, record.value);
        return record.value;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 4U;
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 0U;
    }
}

extern "C" std::uint32_t native_store_double(
    NativeExecutionContext* context,
    const CpuState* state,
    std::uint32_t address,
    std::uint32_t fp_register,
    std::uint32_t instruction_index) noexcept
{
    try
    {
        ++context->memory_access_calls;
        if (instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = 8U;
            context->failed_flags = 1U;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        if (context->memory->has_write_watchpoint())
        {
            context->memory->set_current_guest_cia(
                context->block_cia + instruction_index * 4U);
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.indexed = false;
        record.fp_value = state->fpr[fp_register];
        context->memory->write64_be(address, record.fp_value);
        return 0U;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 8U;
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 0U;
    }
}

extern "C" std::uint64_t native_load_single(
    NativeExecutionContext* context,
    std::uint32_t address,
    std::uint32_t instruction_index) noexcept
{
    try
    {
        ++context->memory_access_calls;
        if (instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = 4U;
            context->failed_instruction_index = instruction_index;
            return 0U;
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.indexed = false;
        const float single_value = std::bit_cast<float>(context->memory->read32_be(address));
        record.fp_value = std::bit_cast<std::uint64_t>(
            static_cast<double>(single_value));
        return record.fp_value;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = 4U;
        context->failed_flags = 0U;
        context->failed_instruction_index = instruction_index;
        context->exception = std::current_exception();
        try
        {
            std::rethrow_exception(context->exception);
        }
        catch (const GuestMemoryFault&)
        {
            context->guest_memory_fault = true;
        }
        catch (...)
        {
        }
        return 0U;
    }
}

struct BlockCode
{
    std::uint32_t address{};
    std::vector<std::uint32_t> words;
    std::vector<DecodedInstruction> instructions;
    NativeBlock function{};
};

[[nodiscard]] constexpr std::size_t gpr_offset(std::uint8_t reg) noexcept
{
    return offsetof(CpuState, gpr) + sizeof(std::uint32_t) * reg;
}

[[nodiscard]] constexpr std::size_t fpr_offset(std::uint8_t reg) noexcept
{
    return offsetof(CpuState, fpr) + sizeof(std::uint64_t) * reg;
}

[[nodiscard]] bool is_native_instruction(const DecodedInstruction& insn) noexcept
{
    if (insn.record && (insn.opcode == Opcode::bitwise_or ||
        insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor ||
        insn.opcode == Opcode::bitwise_nor))
    {
        return true;
    }
    if (insn.opcode == Opcode::add_immediate_carry && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::negate && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::multiply_low_word && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_carrying)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_extended)
    {
        return true;
    }
    if ((insn.opcode == Opcode::shift_left_word ||
         insn.opcode == Opcode::shift_right_word) && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::arithmetic_shift_right_immediate && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::count_leading_zeros && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::rotate_left_word_and_mask_insert && insn.record)
    {
        return true;
    }
    switch (insn.opcode)
    {
    case Opcode::addi:
    case Opcode::addis:
    case Opcode::multiply_low_immediate:
    case Opcode::multiply_low_word:
    case Opcode::negate:
    case Opcode::add_immediate_carry:
    case Opcode::subtract_from_immediate_carry:
    case Opcode::ori:
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate:
    case Opcode::xor_immediate_shifted:
    case Opcode::add:
    case Opcode::subtract_from:
    case Opcode::subtract_from_carrying:
    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_nor:
    case Opcode::move_condition_register_field:
    case Opcode::move_from_condition_register:
    case Opcode::move_to_condition_register_fields:
    case Opcode::move_condition_register_from_xer:
    case Opcode::compare_signed_immediate:
    case Opcode::compare_unsigned_immediate:
    case Opcode::compare_signed_register:
    case Opcode::compare_unsigned_register:
    case Opcode::floating_compare_unordered:
    case Opcode::floating_convert_to_integer_word_zero:
    case Opcode::floating_round_to_single:
    case Opcode::floating_multiply_single:
    case Opcode::floating_multiply_double:
    case Opcode::floating_multiply_add_double:
    case Opcode::floating_subtract_double:
    case Opcode::branch:
    case Opcode::conditional_branch:
    case Opcode::conditional_branch_to_link_register:
    case Opcode::conditional_branch_to_count_register:
    case Opcode::move_from_link_register:
    case Opcode::move_to_link_register:
    case Opcode::move_from_count_register:
    case Opcode::move_to_count_register:
    case Opcode::move_from_xer:
    case Opcode::move_to_xer:
    case Opcode::move_from_gqr:
    case Opcode::move_to_gqr:
    case Opcode::floating_move_register:
    case Opcode::floating_negate:
    case Opcode::floating_absolute_value:
    case Opcode::paired_single_merge00:
    case Opcode::paired_single_merge10:
    case Opcode::paired_single_quantized_load:
    case Opcode::paired_single_quantized_store:
    case Opcode::instruction_sync:
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::arithmetic_shift_right_immediate:
    case Opcode::count_leading_zeros:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::load_word_zero:
    case Opcode::load_multiple_word:
    case Opcode::store_multiple_word:
    case Opcode::load_word_update:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_double:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::store_halfword:
    case Opcode::store_single:
    case Opcode::store_single_update:
    case Opcode::store_double:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_halfword_indexed:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
    case Opcode::condition_bit_and:
    case Opcode::condition_bit_and_complement:
    case Opcode::condition_bit_or:
    case Opcode::condition_bit_or_complement:
    case Opcode::condition_bit_nor:
    case Opcode::condition_bit_xor:
    case Opcode::condition_bit_nand:
    case Opcode::condition_bit_equivalence:
        return !insn.record;
    case Opcode::rotate_left_word_and_mask:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] const char* name(Opcode opcode) noexcept
{
    switch (opcode)
    {
    case Opcode::addi: return "addi";
    case Opcode::addis: return "addis";
    case Opcode::multiply_low_immediate: return "mulli";
    case Opcode::multiply_low_word: return "mullw";
    case Opcode::negate: return "neg";
    case Opcode::add_immediate_carry: return "addic";
    case Opcode::subtract_from_immediate_carry: return "subfic";
    case Opcode::ori: return "ori";
    case Opcode::or_immediate_shifted: return "oris";
    case Opcode::xor_immediate: return "xori";
    case Opcode::xor_immediate_shifted: return "xoris";
    case Opcode::add: return "add";
    case Opcode::subtract_from: return "subf";
    case Opcode::subtract_from_carrying: return "subfc";
    case Opcode::subtract_from_extended: return "subfe";
    case Opcode::rotate_left_word_and_mask_insert: return "rlwimi";
    case Opcode::paired_single_merge00: return "ps_merge00";
    case Opcode::paired_single_merge10: return "ps_merge10";
    case Opcode::paired_single_quantized_store: return "psq_st";
    case Opcode::paired_single_quantized_load: return "psq_l";
    case Opcode::instruction_sync: return "isync";
    case Opcode::bitwise_or: return "or";
    case Opcode::bitwise_and: return "and";
    case Opcode::bitwise_xor: return "xor";
    case Opcode::bitwise_nor: return "nor";
    case Opcode::compare_signed_immediate: return "cmpwi";
    case Opcode::compare_unsigned_immediate: return "cmplwi";
    case Opcode::compare_signed_register: return "cmpw";
    case Opcode::compare_unsigned_register: return "cmplw";
    case Opcode::floating_compare_unordered: return "fcmpu";
    case Opcode::move_condition_register_field: return "mcrf";
    case Opcode::move_from_condition_register: return "mfcr";
    case Opcode::move_to_condition_register_fields: return "mtcrf";
    case Opcode::move_condition_register_from_xer: return "mcrxr";
    case Opcode::branch: return "b";
    case Opcode::conditional_branch: return "bc";
    case Opcode::conditional_branch_to_link_register: return "bclr";
    case Opcode::conditional_branch_to_count_register: return "bcctr";
    case Opcode::move_from_link_register: return "mflr";
    case Opcode::move_to_link_register: return "mtlr";
    case Opcode::move_from_count_register: return "mfctr";
    case Opcode::move_to_count_register: return "mtctr";
    case Opcode::move_from_xer: return "mfxer";
    case Opcode::move_to_xer: return "mtxer";
    case Opcode::move_from_gqr: return "mfgqr";
    case Opcode::move_to_gqr: return "mtgqr";
    case Opcode::floating_move_register: return "fmr";
    case Opcode::floating_negate: return "fneg";
    case Opcode::floating_absolute_value: return "fabs";
    case Opcode::floating_convert_to_integer_word_zero: return "fctiwz";
    case Opcode::floating_round_to_single: return "frsp";
    case Opcode::floating_multiply_single: return "fmuls";
    case Opcode::floating_multiply_double: return "fmul";
    case Opcode::floating_multiply_add_double: return "fmadd";
    case Opcode::floating_subtract_double: return "fsub";
    case Opcode::shift_left_word: return "slw";
    case Opcode::shift_right_word: return "srw";
    case Opcode::arithmetic_shift_right_immediate: return "srawi";
    case Opcode::count_leading_zeros: return "cntlzw";
    case Opcode::load_word_zero: return "lwz";
    case Opcode::load_multiple_word: return "lmw";
    case Opcode::store_multiple_word: return "stmw";
    case Opcode::load_word_update: return "lwzu";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_byte_update: return "lbzu";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::load_single: return "lfs";
    case Opcode::load_double: return "lfd";
    case Opcode::store_word: return "stw";
    case Opcode::store_word_update: return "stwu";
    case Opcode::store_byte: return "stb";
    case Opcode::store_byte_update: return "stbu";
    case Opcode::store_halfword: return "sth";
    case Opcode::store_single: return "stfs";
    case Opcode::store_single_update: return "stfsu";
    case Opcode::store_double: return "stfd";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::load_byte_zero_indexed: return "lbzx";
    case Opcode::load_halfword_zero_indexed: return "lhzx";
    case Opcode::load_halfword_algebraic_indexed: return "lhax";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::store_halfword_indexed: return "sthx";
    case Opcode::data_cache_block_store: return "dcbst";
    case Opcode::data_cache_block_flush: return "dcbf";
    case Opcode::condition_bit_and: return "crand";
    case Opcode::condition_bit_and_complement: return "crandc";
    case Opcode::condition_bit_or: return "cror";
    case Opcode::condition_bit_or_complement: return "crorc";
    case Opcode::condition_bit_nor: return "crnor";
    case Opcode::condition_bit_xor: return "crxor";
    case Opcode::condition_bit_nand: return "crnand";
    case Opcode::condition_bit_equivalence: return "creqv";
    case Opcode::rotate_left_word_and_mask: return "rlwinm";
    default: return "<native>";
    }
}

[[nodiscard]] bool is_memory_instruction(Opcode opcode) noexcept
{
    switch (opcode)
    {
    case Opcode::load_word_zero:
    case Opcode::load_multiple_word:
    case Opcode::store_multiple_word:
    case Opcode::load_word_update:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_double:
    case Opcode::store_single:
    case Opcode::store_single_update:
    case Opcode::store_double:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::store_halfword:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_halfword_indexed:
    case Opcode::paired_single_quantized_store:
    case Opcode::paired_single_quantized_load:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
        return true;
    default:
        return false;
    }
}

[[nodiscard]] std::uint32_t memory_width(Opcode opcode) noexcept
{
    switch (opcode)
    {
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::load_byte_zero_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
        return 1U;
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::store_halfword:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_halfword_indexed:
        return 2U;
    case Opcode::store_double:
        return 8U;
    default:
        return 4U;
    }
}

[[nodiscard]] bool is_memory_store(Opcode opcode) noexcept
{
    return opcode == Opcode::store_word || opcode == Opcode::store_byte ||
        opcode == Opcode::store_halfword || opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_byte_update || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::store_single || opcode == Opcode::store_double ||
        opcode == Opcode::store_word_update || opcode == Opcode::store_single_update;
}

[[nodiscard]] bool is_indexed_memory(Opcode opcode) noexcept
{
    return opcode == Opcode::load_word_indexed || opcode == Opcode::load_byte_zero_indexed ||
        opcode == Opcode::load_halfword_zero_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed ||
        opcode == Opcode::store_word_indexed || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::data_cache_block_store ||
        opcode == Opcode::data_cache_block_flush;
}

[[nodiscard]] std::uint32_t immediate_value(const DecodedInstruction& insn) noexcept
{
    switch (insn.opcode)
    {
    case Opcode::addis:
        return static_cast<std::uint32_t>(insn.immediate) << 16U;
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate_shifted:
        return static_cast<std::uint32_t>(insn.immediate) << 16U;
    default:
        return static_cast<std::uint32_t>(insn.immediate);
    }
}

[[nodiscard]] std::string memory_fault_text(
    const GuestMemoryFault& fault, std::uint32_t cia)
{
    std::ostringstream text;
    text << (fault.access() == GuestMemoryAccess::read ? "read " : "write ")
         << fault.width() << " byte(s) at guest address 0x" << std::hex
         << std::uppercase << std::setw(8) << std::setfill('0') << fault.address()
         << "; CIA 0x" << std::setw(8) << cia;
    return text.str();
}

[[nodiscard]] InstructionHistoryEntry native_history(
    const DecodedInstruction& insn,
    std::uint32_t cia)
{
    InstructionHistoryEntry entry{};
    entry.cia = cia;
    entry.instruction_word = insn.raw;
    entry.has_instruction_word = true;
    entry.opcode_name = name(insn.opcode);
    if (insn.record)
    {
        entry.opcode_name += '.';
    }
    entry.completed = true;

    // Native blocks may contain dependencies between guest instructions. Until
    // per-instruction snapshots are emitted into the JIT context, retain the
    // exact native trace identity without reporting potentially stale values.
    if (insn.opcode == Opcode::addi || insn.opcode == Opcode::addis ||
        insn.opcode == Opcode::multiply_low_immediate ||
        insn.opcode == Opcode::add_immediate_carry ||
        insn.opcode == Opcode::subtract_from_immediate_carry ||
        insn.opcode == Opcode::ori || insn.opcode == Opcode::or_immediate_shifted ||
        insn.opcode == Opcode::xor_immediate ||
        insn.opcode == Opcode::xor_immediate_shifted)
    {
        entry.has_immediate = true;
        entry.immediate = insn.immediate;
        entry.immediate_hex = insn.opcode == Opcode::ori ||
            insn.opcode == Opcode::or_immediate_shifted ||
            insn.opcode == Opcode::xor_immediate ||
            insn.opcode == Opcode::xor_immediate_shifted;
    }
    return entry;
}

}

struct Recompiler::Impl
{
    static constexpr std::size_t fast_lookup_cache_size = 16'384U;

    struct CacheEntry
    {
        BlockCode block;
        const GuestMemory* memory_owner{};
        std::uint32_t first_code_page{};
        std::uint32_t second_code_page{};
        std::uint64_t first_page_write_generation{};
        std::uint64_t second_page_write_generation{};
        bool spans_two_pages{};
    };

    struct FastLookupEntry
    {
        std::uint32_t address{};
        CacheEntry* entry{};
    };

    asmjit::JitRuntime runtime;
    std::unordered_map<std::uint32_t, CacheEntry> cache;
    std::array<FastLookupEntry, fast_lookup_cache_size> fast_lookup_cache{};
    CacheEntry* last_cache_entry{};
    std::uint32_t last_cache_address{};
    RecompilerStatistics stats{};

    static void capture_code_page_generations(
        CacheEntry& entry, const GuestMemory& memory) noexcept
    {
        entry.first_code_page = entry.block.address >> 12U;
        const std::uint32_t last_byte = entry.block.address +
            static_cast<std::uint32_t>(entry.block.words.size() * sizeof(std::uint32_t) - 1U);
        entry.second_code_page = last_byte >> 12U;
        entry.spans_two_pages = entry.second_code_page != entry.first_code_page;
        entry.first_page_write_generation = memory.page_write_generation(
            entry.first_code_page << 12U);
        entry.second_page_write_generation = entry.spans_two_pages
            ? memory.page_write_generation(entry.second_code_page << 12U)
            : 0U;
        entry.memory_owner = &memory;
    }

    [[nodiscard]] static bool code_page_generations_match(
        const CacheEntry& entry, const GuestMemory& memory) noexcept
    {
        return entry.first_page_write_generation == memory.page_write_generation(
                   entry.first_code_page << 12U) &&
            (!entry.spans_two_pages ||
                entry.second_page_write_generation == memory.page_write_generation(
                    entry.second_code_page << 12U));
    }

    ~Impl()
    {
        clear();
    }

    void clear() noexcept
    {
        last_cache_entry = nullptr;
        last_cache_address = 0U;
        fast_lookup_cache = {};
        for (auto& [address, entry] : cache)
        {
            static_cast<void>(address);
            if (entry.block.function != nullptr)
            {
                static_cast<void>(runtime.release(entry.block.function));
            }
        }
        cache.clear();
        stats = {};
    }

    [[nodiscard]] bool compile(BlockCode& block)
    {
        if (block.instructions.empty())
        {
            return false;
        }

        asmjit::CodeHolder code;
        if (const asmjit::Error error = code.init(runtime.environment()); error != asmjit::kErrorOk)
        {
            return false;
        }
        asmjit::x86::Compiler compiler(&code);
        auto* function = compiler.add_func(
            asmjit::FuncSignature::build<std::uint32_t, CpuState*, NativeExecutionContext*>(
                asmjit::CallConvId::kCDecl));
        asmjit::x86::Gp state = compiler.new_gp_ptr("state");
        asmjit::x86::Gp context = compiler.new_gp_ptr("context");
        function->set_arg(0, state);
        function->set_arg(1, context);

        using namespace asmjit;
        std::array<x86::Gp, 32> guest_gprs{};
        std::array<bool, 32> guest_gpr_initialized{};
        const auto read_guest_gpr = [&](std::uint8_t reg) -> x86::Gp {
            if (!guest_gpr_initialized[reg])
            {
                guest_gprs[reg] = compiler.new_gp32("guest_gpr");
                compiler.mov(guest_gprs[reg], x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(reg))));
                guest_gpr_initialized[reg] = true;
            }
            return guest_gprs[reg];
        };
        const auto write_guest_gpr = [&](std::uint8_t reg, x86::Gp value) {
            if (!guest_gpr_initialized[reg])
            {
                guest_gprs[reg] = compiler.new_gp32("guest_gpr");
                guest_gpr_initialized[reg] = true;
            }
            compiler.mov(guest_gprs[reg], value);
        };
        const auto flush_guest_gprs = [&]() {
            for (std::uint8_t reg = 0; reg < guest_gprs.size(); ++reg)
            {
                if (guest_gpr_initialized[reg])
                {
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(reg))), guest_gprs[reg]);
                }
            }
        };

        for (std::size_t index = 0; index < block.instructions.size(); ++index)
        {
            const DecodedInstruction& insn = block.instructions[index];
            using namespace asmjit;
            switch (insn.opcode)
            {
            case Opcode::addi:
            case Opcode::addis:
            {
                if (insn.base != 0U && insn.destination == insn.base)
                {
                    compiler.add(read_guest_gpr(insn.destination),
                                 imm(immediate_value(insn)));
                    break;
                }
                x86::Gp value = compiler.new_gp32("value");
                if (insn.base == 0U)
                {
                    compiler.mov(value, imm(0));
                }
                else
                {
                    compiler.mov(value, read_guest_gpr(insn.base));
                }
                compiler.add(value, imm(immediate_value(insn)));
                write_guest_gpr(insn.destination, value);
                break;
            }
            case Opcode::subtract_from_immediate_carry:
            {
                const x86::Gp source = read_guest_gpr(insn.base);
                x86::Gp result = compiler.new_gp32("subfic_result");
                compiler.mov(result, imm(immediate_value(insn)));
                compiler.sub(result, source);

                x86::Gp carry = compiler.new_gp32("subfic_carry");
                compiler.mov(carry, imm(0U));
                compiler.setae(carry.r8());
                compiler.shl(carry, imm(29U));
                x86::Gp xer = compiler.new_gp32("subfic_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                compiler.and_(xer, imm(~0x20000000U));
                compiler.or_(xer, carry);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))), xer);
                write_guest_gpr(insn.destination, result);
                break;
            }
            case Opcode::shift_left_word:
            case Opcode::shift_right_word:
            {
                x86::Gp shift = compiler.new_gp32("slw_shift");
                compiler.mov(shift, read_guest_gpr(insn.base));
                compiler.and_(shift, imm(0x3FU));

                x86::Gp result = compiler.new_gp32("slw_result");
                compiler.mov(result, read_guest_gpr(insn.source));
                Label zero_result = compiler.new_label();
                Label shift_done = compiler.new_label();
                compiler.test(shift, imm(0x20U));
                compiler.jnz(zero_result);
                if (insn.opcode == Opcode::shift_left_word)
                {
                    compiler.shl(result, shift);
                }
                else
                {
                    compiler.shr(result, shift);
                }
                compiler.jmp(shift_done);
                compiler.bind(zero_result);
                compiler.mov(result, imm(0U));
                compiler.bind(shift_done);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("slw_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("slw_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("slw_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);

                    x86::Gp xer_so = compiler.new_gp32("slw_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("slw_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::arithmetic_shift_right_immediate:
            {
                x86::Gp source = compiler.new_gp32("srawi_source");
                compiler.mov(source, read_guest_gpr(insn.source));
                x86::Gp result = compiler.new_gp32("srawi_result");
                compiler.mov(result, source);
                compiler.sar(result, imm(insn.shift));
                write_guest_gpr(insn.destination, result);

                x86::Gp carry = compiler.new_gp32("srawi_carry");
                compiler.mov(carry, imm(0U));
                if (insn.shift != 0U)
                {
                    const std::uint32_t discarded_mask =
                        (std::uint32_t{1U} << insn.shift) - 1U;
                    Label no_carry = compiler.new_label();
                    Label carry_done = compiler.new_label();
                    compiler.test(source, imm(0x80000000U));
                    compiler.jz(no_carry);
                    compiler.test(source, imm(discarded_mask));
                    compiler.setne(carry.r8());
                    compiler.jmp(carry_done);
                    compiler.bind(no_carry);
                    compiler.bind(carry_done);
                }
                compiler.shl(carry, imm(29U));
                x86::Gp xer = compiler.new_gp32("srawi_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                compiler.and_(xer, imm(~0x20000000U));
                compiler.or_(xer, carry);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))), xer);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("srawi_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("srawi_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("srawi_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);
                    x86::Gp xer_so = compiler.new_gp32("srawi_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("srawi_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::count_leading_zeros:
            {
                x86::Gp source = read_guest_gpr(insn.source);
                x86::Gp result = compiler.new_gp32("cntlzw_result");
                Label zero_source = compiler.new_label();
                Label count_done = compiler.new_label();
                compiler.test(source, source);
                compiler.jz(zero_source);
                compiler.bsr(result, source);
                compiler.xor_(result, imm(31U));
                compiler.jmp(count_done);
                compiler.bind(zero_source);
                compiler.mov(result, imm(32U));
                compiler.bind(count_done);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("cntlzw_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("cntlzw_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("cntlzw_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);
                    x86::Gp xer_so = compiler.new_gp32("cntlzw_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("cntlzw_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::multiply_low_immediate:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext, mulli_records) +
                    index * sizeof(NativeMulliRecord);
                x86::Gp source = read_guest_gpr(insn.base);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMulliRecord, source_value))), source);
                x86::Gp result = compiler.new_gp32("mulli_result");
                compiler.imul(result, source, imm(insn.immediate));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMulliRecord, result))), result);
                write_guest_gpr(insn.destination, result);
                break;
            }
            case Opcode::multiply_low_word:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext, mullw_records) +
                    index * sizeof(NativeMullwRecord);
                const x86::Gp source_a = read_guest_gpr(insn.base);
                const x86::Gp source_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMullwRecord, source_a))), source_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMullwRecord, source_b))), source_b);
                x86::Gp result = compiler.new_gp32("mullw_result");
                compiler.mov(result, source_a);
                compiler.imul(result, source_b);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMullwRecord, result))), result);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("mullw_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("mullw_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("mullw_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);

                    x86::Gp xer_so = compiler.new_gp32("mullw_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("mullw_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::negate:
            {
                x86::Gp result = compiler.new_gp32("neg_result");
                compiler.mov(result, read_guest_gpr(insn.source));
                compiler.neg(result);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("neg_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("neg_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("neg_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);

                    x86::Gp xer_so = compiler.new_gp32("neg_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("neg_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_immediate_carry:
            {
                const x86::Gp source = read_guest_gpr(insn.base);
                x86::Gp result = source;
                if (insn.destination != insn.base)
                {
                    result = compiler.new_gp32("addic_result");
                    compiler.mov(result, source);
                }
                compiler.add(result, imm(immediate_value(insn)));

                x86::Gp carry = compiler.new_gp32("addic_carry");
                compiler.mov(carry, imm(0U));
                compiler.setb(carry.r8());
                compiler.shl(carry, imm(29U));
                x86::Gp xer = compiler.new_gp32("addic_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                compiler.and_(xer, imm(~0x20000000U));
                compiler.or_(xer, carry);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))), xer);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addic_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addic_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addic_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);

                    x86::Gp xer_so = compiler.new_gp32("addic_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addic_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::ori:
            case Opcode::or_immediate_shifted:
            case Opcode::xor_immediate:
            case Opcode::xor_immediate_shifted:
            {
                x86::Gp value = read_guest_gpr(insn.source);
                if (insn.destination != insn.source)
                {
                    value = compiler.new_gp32("value");
                    compiler.mov(value, read_guest_gpr(insn.source));
                }
                if (insn.opcode == Opcode::ori || insn.opcode == Opcode::or_immediate_shifted)
                    compiler.or_(value, imm(immediate_value(insn)));
                else
                    compiler.xor_(value, imm(immediate_value(insn)));
                write_guest_gpr(insn.destination, value);
                break;
            }
            case Opcode::add:
            case Opcode::subtract_from:
            case Opcode::bitwise_or:
            case Opcode::bitwise_and:
            case Opcode::bitwise_xor:
            case Opcode::bitwise_nor:
            {
                x86::Gp lhs = compiler.new_gp32("lhs");
                x86::Gp source_a = read_guest_gpr(insn.base);
                x86::Gp source_b = read_guest_gpr(insn.source);
                if (insn.opcode == Opcode::add || insn.opcode == Opcode::bitwise_or ||
                    insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor ||
                    insn.opcode == Opcode::bitwise_nor)
                {
                    x86::Gp rhs = source_b;
                    if (insn.destination == insn.base)
                    {
                        lhs = source_a;
                    }
                    else if (insn.destination == insn.source)
                    {
                        lhs = source_b;
                        rhs = source_a;
                    }
                    else
                    {
                        compiler.mov(lhs, source_a);
                    }
                    if (insn.opcode == Opcode::add) compiler.add(lhs, rhs);
                    else if (insn.opcode == Opcode::bitwise_or) compiler.or_(lhs, rhs);
                    else if (insn.opcode == Opcode::bitwise_and) compiler.and_(lhs, rhs);
                    else if (insn.opcode == Opcode::bitwise_xor) compiler.xor_(lhs, rhs);
                    else
                    {
                        compiler.or_(lhs, rhs);
                        compiler.not_(lhs);
                    }
                    write_guest_gpr(insn.destination, lhs);
                    if (insn.record)
                    {
                        compiler.test(lhs, lhs);
                        x86::Gp less = compiler.new_gp32("logical_cr0_lt");
                        x86::Gp greater = compiler.new_gp32("logical_cr0_gt");
                        x86::Gp equal = compiler.new_gp32("logical_cr0_eq");
                        compiler.mov(less, imm(0U));
                        compiler.setl(less.r8());
                        compiler.mov(greater, imm(0U));
                        compiler.setg(greater.r8());
                        compiler.mov(equal, imm(0U));
                        compiler.sete(equal.r8());
                        compiler.shl(less, imm(3U));
                        compiler.shl(greater, imm(2U));
                        compiler.shl(equal, imm(1U));
                        compiler.or_(less, greater);
                        compiler.or_(less, equal);

                        x86::Gp xer_so = compiler.new_gp32("logical_cr0_so");
                        compiler.mov(xer_so, x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                        compiler.shr(xer_so, imm(31U));
                        compiler.or_(less, xer_so);
                        compiler.shl(less, imm(28U));
                        x86::Gp cr = compiler.new_gp32("logical_cr0");
                        compiler.mov(cr, x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                        compiler.and_(cr, imm(0x0FFFFFFFU));
                        compiler.or_(cr, less);
                        compiler.mov(x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                    }
                }
                else if (insn.opcode == Opcode::subtract_from)
                {
                    if (insn.destination == insn.source)
                    {
                        compiler.sub(source_b, source_a);
                        write_guest_gpr(insn.destination, source_b);
                    }
                    else
                    {
                        compiler.mov(lhs, source_b);
                        compiler.sub(lhs, source_a);
                        write_guest_gpr(insn.destination, lhs);
                    }
                }
                break;
            }
            case Opcode::subtract_from_carrying:
            {
                // PPC subfc computes rB-rA and sets CA when that unsigned
                // subtraction does not borrow. Capture both operands before
                // writing rD so all destination aliases remain well-defined.
                x86::Gp minuend = read_guest_gpr(insn.source);
                x86::Gp subtrahend = read_guest_gpr(insn.base);
                x86::Gp result = compiler.new_gp32("subfc_result");
                compiler.mov(result, minuend);
                compiler.sub(result, subtrahend);

                x86::Gp carry = compiler.new_gp32("subfc_carry");
                compiler.mov(carry, imm(0U));
                compiler.setae(carry.r8());
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("subfc_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                compiler.shl(carry, imm(29U));
                compiler.or_(xer, carry);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp field = compiler.new_gp32("subfc_cr0_field");
                    x86::Gp less = compiler.new_gp32("subfc_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("subfc_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("subfc_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);

                    x86::Gp xer_so = compiler.new_gp32("subfc_xer_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.mov(field, less);
                    compiler.shl(field, imm(28U));

                    x86::Gp cr = compiler.new_gp32("subfc_cr");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, field);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::subtract_from_extended:
            {
                // subfe computes rB-rA-(1-CAin). Preserve the first borrow
                // before subtracting the borrow-in, then combine both borrow
                // bits to produce the architectural no-borrow CA result.
                x86::Gp minuend = read_guest_gpr(insn.source);
                x86::Gp subtrahend = read_guest_gpr(insn.base);
                x86::Gp old_carry = compiler.new_gp32("subfe_old_ca");
                compiler.mov(old_carry, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                compiler.shr(old_carry, imm(29U));
                compiler.and_(old_carry, imm(1U));
                compiler.xor_(old_carry, imm(1U));

                x86::Gp result = compiler.new_gp32("subfe_result");
                compiler.mov(result, minuend);
                compiler.sub(result, subtrahend);
                x86::Gp first_borrow = compiler.new_gp32("subfe_first_borrow");
                compiler.mov(first_borrow, imm(0U));
                compiler.setb(first_borrow.r8());
                compiler.sub(result, old_carry);
                x86::Gp second_borrow = compiler.new_gp32("subfe_second_borrow");
                compiler.mov(second_borrow, imm(0U));
                compiler.setb(second_borrow.r8());
                compiler.or_(first_borrow, second_borrow);
                compiler.xor_(first_borrow, imm(1U));

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("subfe_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                compiler.shl(first_borrow, imm(29U));
                compiler.or_(xer, first_borrow);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("subfe_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("subfe_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("subfe_cr0_eq");
                    compiler.mov(less, imm(0U));
                    compiler.setl(less.r8());
                    compiler.mov(greater, imm(0U));
                    compiler.setg(greater.r8());
                    compiler.mov(equal, imm(0U));
                    compiler.sete(equal.r8());
                    compiler.shl(less, imm(3U));
                    compiler.shl(greater, imm(2U));
                    compiler.shl(equal, imm(1U));
                    compiler.or_(less, greater);
                    compiler.or_(less, equal);
                    x86::Gp xer_so = compiler.new_gp32("subfe_xer_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("subfe_cr");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::compare_signed_immediate:
            case Opcode::compare_unsigned_immediate:
            case Opcode::compare_signed_register:
            case Opcode::compare_unsigned_register:
            {
                x86::Gp lhs = read_guest_gpr(insn.base);
                if (insn.opcode == Opcode::compare_signed_immediate ||
                    insn.opcode == Opcode::compare_unsigned_immediate)
                {
                    compiler.cmp(lhs, imm(static_cast<std::uint32_t>(insn.immediate)));
                }
                else
                {
                    compiler.cmp(lhs, read_guest_gpr(insn.source));
                }
                const bool signed_compare =
                    insn.opcode == Opcode::compare_signed_immediate ||
                    insn.opcode == Opcode::compare_signed_register;

                // SETcc writes only the low byte, so zero-extend each result
                // before combining the PPC LT/GT/EQ bits.
                x86::Gp less = compiler.new_gp32("cmp_less");
                x86::Gp greater = compiler.new_gp32("cmp_greater");
                x86::Gp equal = compiler.new_gp32("cmp_equal");
                compiler.mov(less, imm(0));
                if (signed_compare) compiler.setl(less.r8());
                else compiler.setb(less.r8());
                compiler.mov(greater, imm(0));
                if (signed_compare) compiler.setg(greater.r8());
                else compiler.seta(greater.r8());
                compiler.mov(equal, imm(0));
                compiler.sete(equal.r8());

                compiler.shl(less, imm(3U));
                compiler.shl(greater, imm(2U));
                compiler.shl(equal, imm(1U));
                compiler.or_(less, greater);
                compiler.or_(less, equal);

                const std::uint32_t field_shift =
                    (7U - insn.cr_field) * 4U;
                x86::Gp xer = compiler.new_gp32("cmp_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                compiler.shr(xer, imm(31U));
                compiler.or_(less, xer);
                compiler.shl(less, imm(field_shift));

                constexpr std::uint32_t cr_mask = 0xFU;
                const std::uint32_t shifted_mask = cr_mask << field_shift;
                x86::Gp cr = compiler.new_gp32("cmp_cr");
                compiler.mov(cr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                compiler.and_(cr, imm(~shifted_mask));
                compiler.or_(cr, less);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                break;
            }
            case Opcode::floating_compare_unordered:
            {
                const std::uint32_t compare_record_offset = static_cast<std::uint32_t>(
                    offsetof(NativeExecutionContext, fp_compare_records) +
                    index * sizeof(NativeFpCompareRecord));
                x86::Gp a_raw = compiler.new_gp64("fcmpu_a_raw");
                x86::Gp b_raw = compiler.new_gp64("fcmpu_b_raw");
                compiler.mov(a_raw, x86::qword_ptr(
                    state, static_cast<std::int32_t>(fpr_offset(insn.fp_compare_a))));
                compiler.mov(b_raw, x86::qword_ptr(
                    state, static_cast<std::int32_t>(fpr_offset(insn.fp_compare_b))));

                x86::Vec a = compiler.new_xmm("fcmpu_a");
                x86::Vec b = compiler.new_xmm("fcmpu_b");
                compiler.movq(a, a_raw);
                compiler.movq(b, b_raw);
                compiler.ucomisd(a, b);

                x86::Gp less = compiler.new_gp32("fcmpu_less");
                x86::Gp greater = compiler.new_gp32("fcmpu_greater");
                x86::Gp equal = compiler.new_gp32("fcmpu_equal");
                x86::Gp unordered = compiler.new_gp32("fcmpu_unordered");
                x86::Gp comparison = compiler.new_gp32("fcmpu_result");
                compiler.mov(less, imm(0));
                compiler.setb(less.r8());
                compiler.mov(greater, imm(0));
                compiler.seta(greater.r8());
                compiler.mov(equal, imm(0));
                compiler.sete(equal.r8());
                compiler.mov(unordered, imm(0));
                compiler.setp(unordered.r8());

                compiler.mov(comparison, imm(0));
                compiler.shl(less, imm(3U));
                compiler.shl(greater, imm(2U));
                compiler.shl(equal, imm(1U));
                compiler.or_(comparison, less);
                compiler.or_(comparison, greater);
                compiler.or_(comparison, equal);

                Label ordered = compiler.new_label();
                compiler.test(unordered, unordered);
                compiler.jz(ordered);
                compiler.mov(comparison, imm(1U));
                auto* invoke_snan = static_cast<InvokeNode*>(nullptr);
                const auto snan_helper = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_raise_fpscr_snan_if_present));
                compiler.invoke(Out(invoke_snan), snan_helper,
                    FuncSignature::build<void, CpuState*, std::uint64_t, std::uint64_t>(
                        CallConvId::kCDecl));
                invoke_snan->set_arg(0, state);
                invoke_snan->set_arg(1, a_raw);
                invoke_snan->set_arg(2, b_raw);
                compiler.bind(ordered);

                const std::uint32_t cr_shift =
                    (7U - insn.cr_field) * 4U;
                x86::Gp cr = compiler.new_gp32("fcmpu_cr");
                compiler.mov(cr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                compiler.and_(cr, imm(~(0xFU << cr_shift)));
                x86::Gp cr_bits = compiler.new_gp32("fcmpu_cr_bits");
                compiler.mov(cr_bits, comparison);
                compiler.shl(cr_bits, imm(cr_shift));
                compiler.or_(cr, cr_bits);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);

                x86::Gp fpscr = compiler.new_gp32("fcmpu_fpscr");
                compiler.mov(fpscr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.and_(fpscr, imm(~fpscr::fpcc_mask));
                x86::Gp fpcc = compiler.new_gp32("fcmpu_fpcc");
                compiler.mov(fpcc, comparison);
                compiler.shl(fpcc, imm(12U));
                compiler.or_(fpscr, fpcc);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, fpscr))), fpscr);
                compiler.mov(x86::qword_ptr(context, static_cast<std::int32_t>(
                    compare_record_offset + offsetof(NativeFpCompareRecord, a_raw))), a_raw);
                compiler.mov(x86::qword_ptr(context, static_cast<std::int32_t>(
                    compare_record_offset + offsetof(NativeFpCompareRecord, b_raw))), b_raw);
                compiler.mov(x86::byte_ptr(context, static_cast<std::int32_t>(
                    compare_record_offset + offsetof(NativeFpCompareRecord, result))),
                    comparison.r8());
                break;
            }
            case Opcode::move_condition_register_field:
            {
                constexpr std::uint32_t cr_offset = offsetof(CpuState, cr);
                const std::uint32_t source_shift =
                    (7U - insn.cr_field_source) * 4U;
                const std::uint32_t destination_shift =
                    (7U - insn.cr_field) * 4U;
                x86::Gp field_value = compiler.new_gp32("mcrf_value");
                compiler.mov(field_value, x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)));
                compiler.shr(field_value, imm(source_shift));
                compiler.and_(field_value, imm(0xFU));
                compiler.shl(field_value, imm(destination_shift));
                x86::Gp cr = compiler.new_gp32("mcrf_cr");
                compiler.mov(cr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)));
                compiler.and_(cr, imm(~(0xFU << destination_shift)));
                compiler.or_(cr, field_value);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)), cr);
                break;
            }
            case Opcode::move_from_condition_register:
            {
                x86::Gp value = compiler.new_gp32("mfcr_value");
                compiler.mov(value, x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                write_guest_gpr(insn.destination, value);
                break;
            }
            case Opcode::move_to_condition_register_fields:
            {
                std::uint32_t cr_mask = 0U;
                for (std::uint32_t field_index = 0U; field_index < 8U; ++field_index)
                {
                    if ((insn.cr_field_mask & (1U << (7U - field_index))) != 0U)
                    {
                        cr_mask |= 0xFU << (28U - 4U * field_index);
                    }
                }

                if (cr_mask != 0U)
                {
                    x86::Gp value = compiler.new_gp32("mtcrf_value");
                    compiler.mov(value, read_guest_gpr(insn.destination));
                    compiler.and_(value, imm(cr_mask));
                    x86::Gp cr = compiler.new_gp32("mtcrf_cr");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(~cr_mask));
                    compiler.or_(cr, value);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::move_condition_register_from_xer:
            {
                constexpr std::uint32_t xer_offset = offsetof(CpuState, xer);
                constexpr std::uint32_t cr_offset = offsetof(CpuState, cr);
                const std::uint32_t destination_shift =
                    (7U - insn.cr_field) * 4U;
                const std::uint32_t field_mask = 0xFU << destination_shift;

                x86::Gp value = compiler.new_gp32("mcrxr_value");
                compiler.mov(value, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(value, imm(28U));
                compiler.shl(value, imm(destination_shift));

                x86::Gp cr = compiler.new_gp32("mcrxr_cr");
                compiler.mov(cr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)));
                compiler.and_(cr, imm(~field_mask));
                compiler.or_(cr, value);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)), cr);

                x86::Gp xer = compiler.new_gp32("mcrxr_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(0x0FFFFFFFU));
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                break;
            }
            case Opcode::condition_bit_and:
            case Opcode::condition_bit_and_complement:
            case Opcode::condition_bit_or:
            case Opcode::condition_bit_or_complement:
            case Opcode::condition_bit_nor:
            case Opcode::condition_bit_xor:
            case Opcode::condition_bit_nand:
            case Opcode::condition_bit_equivalence:
            {
                constexpr std::size_t cr_offset = offsetof(CpuState, cr);
                x86::Gp cr = compiler.new_gp32("cr");
                compiler.mov(cr, x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)));
                x86::Gp lhs = compiler.new_gp32("cr_lhs");
                compiler.mov(lhs, cr);
                compiler.shr(lhs, imm(31U - insn.cr_source_a_bit));
                compiler.and_(lhs, imm(1U));
                x86::Gp rhs = compiler.new_gp32("cr_rhs");
                compiler.mov(rhs, cr);
                compiler.shr(rhs, imm(31U - insn.cr_source_b_bit));
                compiler.and_(rhs, imm(1U));

                if (insn.opcode == Opcode::condition_bit_and ||
                    insn.opcode == Opcode::condition_bit_and_complement ||
                    insn.opcode == Opcode::condition_bit_nand)
                {
                    if (insn.opcode == Opcode::condition_bit_and_complement)
                    {
                        compiler.xor_(rhs, imm(1U));
                    }
                    compiler.and_(lhs, rhs);
                    if (insn.opcode == Opcode::condition_bit_nand)
                    {
                        compiler.xor_(lhs, imm(1U));
                    }
                }
                else if (insn.opcode == Opcode::condition_bit_or ||
                         insn.opcode == Opcode::condition_bit_or_complement ||
                         insn.opcode == Opcode::condition_bit_nor)
                {
                    if (insn.opcode == Opcode::condition_bit_or_complement)
                    {
                        compiler.xor_(rhs, imm(1U));
                    }
                    compiler.or_(lhs, rhs);
                    if (insn.opcode == Opcode::condition_bit_nor)
                    {
                        compiler.xor_(lhs, imm(1U));
                    }
                }
                else
                {
                    compiler.xor_(lhs, rhs);
                    if (insn.opcode == Opcode::condition_bit_equivalence)
                    {
                        compiler.xor_(lhs, imm(1U));
                    }
                }

                constexpr std::size_t cr_bit_count = 32U;
                const std::uint32_t destination_shift =
                    static_cast<std::uint32_t>(cr_bit_count - 1U - insn.cr_destination_bit);
                const std::uint32_t destination_mask = 1U << destination_shift;
                compiler.and_(cr, imm(~destination_mask));
                compiler.shl(lhs, imm(destination_shift));
                compiler.or_(cr, lhs);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(cr_offset)), cr);
                break;
            }
            case Opcode::rotate_left_word_and_mask:
            case Opcode::rotate_left_word_and_mask_insert:
            {
                const bool insert =
                    insn.opcode == Opcode::rotate_left_word_and_mask_insert;
                x86::Gp old_destination{};
                x86::Gp value = compiler.new_gp32("rotate_mask_value");
                if (insert)
                {
                    old_destination = read_guest_gpr(insn.destination);
                }
                compiler.mov(value, read_guest_gpr(insn.source));
                if (insn.shift != 0U)
                {
                    compiler.rol(value, imm(insn.shift));
                }

                std::uint32_t mask = 0U;
                for (std::uint32_t bit = 0U; bit < 32U; ++bit)
                {
                    const bool selected = insn.mask_begin <= insn.mask_end
                        ? bit >= insn.mask_begin && bit <= insn.mask_end
                        : bit >= insn.mask_begin || bit <= insn.mask_end;
                    if (selected)
                    {
                        mask |= 1U << (31U - bit);
                    }
                }
                compiler.and_(value, imm(mask));
                if (insert)
                {
                    compiler.and_(old_destination, imm(~mask));
                    compiler.or_(value, old_destination);
                }
                write_guest_gpr(insn.destination, value);

                if (insn.record)
                {
                    constexpr std::size_t cr_offset = offsetof(CpuState, cr);
                    constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                    x86::Gp field = compiler.new_gp32("record_cr0");
                    compiler.mov(field, value);
                    compiler.shr(field, imm(28U));
                    compiler.and_(field, imm(8U));

                    x86::Gp equal = compiler.new_gp32("record_equal");
                    x86::Gp greater = compiler.new_gp32("record_greater");
                    compiler.test(value, value);
                    compiler.setz(equal.r8());
                    compiler.setg(greater.r8());
                    compiler.shl(equal, imm(1U));
                    compiler.or_(field, equal);
                    compiler.shl(greater, imm(2U));
                    compiler.or_(field, greater);

                    x86::Gp summary_overflow = compiler.new_gp32("record_so");
                    compiler.mov(summary_overflow, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(summary_overflow, imm(31U));
                    compiler.or_(field, summary_overflow);
                    compiler.shl(field, imm(28U));

                    x86::Gp cr = compiler.new_gp32("record_cr");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(cr_offset)));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, field);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(cr_offset)), cr);
                }
                break;
            }
            case Opcode::instruction_sync:
                // Guest memory writes and instruction fetch are synchronous
                // on this core. Block construction terminates at isync so
                // subsequent instructions are resolved from current memory.
                break;
            case Opcode::branch:
            {
                if (insn.link)
                {
                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(offsetof(CpuState, lr))),
                                 imm(block.address + static_cast<std::uint32_t>((index + 1U) * 4U)));
                }
                const std::uint32_t cia = block.address + static_cast<std::uint32_t>(index * 4U);
                const std::uint32_t target = insn.absolute
                    ? static_cast<std::uint32_t>(insn.immediate)
                    : cia + static_cast<std::uint32_t>(insn.immediate);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(offsetof(CpuState, cia))),
                             imm(target));
                break;
            }
            case Opcode::conditional_branch:
            case Opcode::conditional_branch_to_link_register:
            case Opcode::conditional_branch_to_count_register:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::uint32_t fallthrough = cia + 4U;
                const bool ignore_condition = (insn.branch_options & 0x10U) != 0U;
                const bool condition_sense = (insn.branch_options & 0x08U) != 0U;
                const bool ignore_count = (insn.branch_options & 0x04U) != 0U;
                const bool count_sense = (insn.branch_options & 0x02U) != 0U;
                Label not_taken = compiler.new_label();
                Label done = compiler.new_label();

                x86::Gp target = compiler.new_gp32("target");
                if (insn.opcode == Opcode::conditional_branch_to_link_register)
                {
                    compiler.mov(target, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, lr))));
                    compiler.and_(target, imm(0xFFFFFFFCU));
                }
                else if (insn.opcode == Opcode::conditional_branch_to_count_register)
                {
                    compiler.mov(target, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, ctr))));
                    compiler.and_(target, imm(0xFFFFFFFCU));
                }
                if (insn.link)
                {
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, lr))),
                        imm(fallthrough));
                }

                if (!ignore_count)
                {
                    x86::Gp ctr = compiler.new_gp32("ctr");
                    compiler.mov(ctr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, ctr))));
                    compiler.sub(ctr, imm(1));
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, ctr))), ctr);
                    compiler.test(ctr, ctr);
                    if (count_sense) compiler.jnz(not_taken);
                    else compiler.jz(not_taken);
                }
                if (!ignore_condition)
                {
                    x86::Gp cr = compiler.new_gp32("cr");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.shr(cr, imm(31U - insn.condition_bit));
                    compiler.and_(cr, imm(1));
                    compiler.cmp(cr, imm(condition_sense ? 1 : 0));
                    compiler.jne(not_taken);
                }

                if (insn.opcode == Opcode::conditional_branch)
                {
                    const std::uint32_t branch_target = insn.absolute
                        ? static_cast<std::uint32_t>(insn.immediate)
                        : cia + static_cast<std::uint32_t>(insn.immediate);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cia))),
                        imm(branch_target));
                }
                else
                {
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cia))), target);
                }
                compiler.jmp(done);
                compiler.bind(not_taken);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(offsetof(CpuState, cia))),
                    imm(fallthrough));
                compiler.bind(done);
                break;
            }
            case Opcode::move_from_link_register:
            case Opcode::move_from_count_register:
            case Opcode::move_from_xer:
            case Opcode::move_to_link_register:
            case Opcode::move_to_count_register:
            case Opcode::move_to_xer:
            {
                const std::size_t special_offset =
                    insn.opcode == Opcode::move_from_link_register ||
                    insn.opcode == Opcode::move_to_link_register
                    ? offsetof(CpuState, lr)
                    : insn.opcode == Opcode::move_from_xer ||
                      insn.opcode == Opcode::move_to_xer
                        ? offsetof(CpuState, xer)
                        : offsetof(CpuState, ctr);
                if (insn.opcode == Opcode::move_from_link_register ||
                    insn.opcode == Opcode::move_from_count_register ||
                    insn.opcode == Opcode::move_from_xer)
                {
                    x86::Gp value = compiler.new_gp32("special");
                    compiler.mov(value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(special_offset)));
                    write_guest_gpr(insn.destination, value);
                }
                else
                {
                    x86::Gp value = compiler.new_gp32("gpr");
                    compiler.mov(value, read_guest_gpr(insn.destination));
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(special_offset)), value);
                }
                break;
            }
            case Opcode::move_from_gqr:
            {
                const std::size_t gqr_offset = offsetof(CpuState, gqr) +
                    sizeof(std::uint32_t) * insn.quantized_register;
                x86::Gp value = compiler.new_gp32("gqr");
                compiler.mov(value, x86::dword_ptr(
                    state, static_cast<std::int32_t>(gqr_offset)));
                write_guest_gpr(insn.destination, value);
                break;
            }
            case Opcode::move_to_gqr:
            {
                const std::size_t gqr_offset = offsetof(CpuState, gqr) +
                    sizeof(std::uint32_t) * insn.quantized_register;
                x86::Gp value = compiler.new_gp32("gpr");
                compiler.mov(value, read_guest_gpr(insn.destination));
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(gqr_offset)), value);
                break;
            }
            case Opcode::floating_move_register:
            case Opcode::floating_negate:
            case Opcode::floating_absolute_value:
            {
                x86::Gp value = compiler.new_gp64("fpr");
                compiler.mov(value, x86::qword_ptr(
                    state, static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                if (insn.opcode == Opcode::floating_negate)
                {
                    compiler.btc(value, imm(63U));
                }
                else if (insn.opcode == Opcode::floating_absolute_value)
                {
                    compiler.btr(value, imm(63U));
                }
                compiler.mov(x86::qword_ptr(
                    state, static_cast<std::int32_t>(fpr_offset(insn.fp_register))), value);
                break;
            }
            case Opcode::floating_convert_to_integer_word_zero:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fctiwz));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_b));
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::floating_round_to_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_frsp));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::floating_multiply_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fmuls));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_c));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::floating_multiply_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fmul));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_c));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::floating_multiply_add_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fmadd));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(insn.fp_source_c));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::floating_subtract_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fsub));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::paired_single_merge00:
            case Opcode::paired_single_merge10:
            {
                // Snapshot both lanes before either destination lane is
                // written; frD may alias either or both sources.
                x86::Gp source_a_lane = compiler.new_gp64("ps_merge_a_lane");
                x86::Gp source_b_ps0 = compiler.new_gp64("ps_merge_b_ps0");
                const std::size_t source_a_offset =
                    insn.opcode == Opcode::paired_single_merge10
                        ? offsetof(CpuState, fpr_ps1) : offsetof(CpuState, fpr);
                compiler.mov(source_a_lane, x86::qword_ptr(state,
                    static_cast<std::int32_t>(source_a_offset +
                        sizeof(std::uint64_t) * insn.fp_source_a)));
                compiler.mov(source_b_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))),
                    source_a_lane);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)),
                    source_b_ps0);
                break;
            }
            case Opcode::load_multiple_word:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext,
                    memory_records) + index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                x86::Gp address = compiler.new_gp32("lmw_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, base_value))), address);
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // The helper writes the loaded register range directly to
                // CpuState, so flush prior cached mappings before calling it.
                flush_guest_gprs();
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_load_multiple_words));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, address);
                invoke->set_arg(3, imm(insn.destination));
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));

                guest_gpr_initialized.fill(false);
                x86::Gp failed = compiler.new_gp32("lmw_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                compiler.jnz(fault);
                Label completed = compiler.new_label();
                compiler.jmp(completed);
                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                x86::Gp executed = compiler.new_gp32("lmw_completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
                break;
            }
            case Opcode::store_multiple_word:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext,
                    memory_records) + index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::uint32_t byte_count =
                    static_cast<std::uint32_t>(32U - insn.source) * 4U;
                x86::Gp address = compiler.new_gp32("stmw_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                }
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, address))), address);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, base_value))), address);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // The helper reads the source range from CpuState and writes
                // stores sequentially, preserving partial stores on a fault.
                flush_guest_gprs();
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_store_multiple_words));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, address);
                invoke->set_arg(3, imm(insn.source));
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));

                guest_gpr_initialized.fill(false);
                x86::Gp failed = compiler.new_gp32("stmw_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                compiler.jnz(fault);

                // A store into already translated code requires leaving this
                // block so its next dispatch validates the changed page.
                Label store_is_safe = compiler.new_label();
                Label self_modifying = compiler.new_label();
                compiler.cmp(address, imm(0xFFFFFFFFU - byte_count + 1U));
                compiler.ja(self_modifying);
                const std::uint32_t block_end = block.address +
                    static_cast<std::uint32_t>(block.words.size() * sizeof(std::uint32_t));
                compiler.cmp(address, imm(block_end));
                compiler.jae(store_is_safe);
                x86::Gp store_end = compiler.new_gp32("stmw_end");
                compiler.mov(store_end, address);
                compiler.add(store_end, imm(byte_count));
                compiler.cmp(store_end, imm(block.address));
                compiler.jbe(store_is_safe);
                compiler.jmp(self_modifying);

                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                x86::Gp fault_executed = compiler.new_gp32("stmw_fault_count");
                compiler.mov(fault_executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(fault_executed);

                compiler.bind(self_modifying);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia + 4U));
                x86::Gp completed_executed = compiler.new_gp32("stmw_completed_count");
                compiler.mov(completed_executed,
                    imm(static_cast<std::uint32_t>(index + 1U)));
                compiler.ret(completed_executed);
                compiler.bind(store_is_safe);
                break;
            }
            case Opcode::paired_single_quantized_store:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                x86::Gp address = compiler.new_gp32("psq_st_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_psq_store));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                        const CpuState*, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, address);
                invoke->set_arg(3, imm(insn.fp_register));
                invoke->set_arg(4, imm(insn.quantized_register));
                invoke->set_arg(5, imm(insn.paired_single_w ? 1U : 0U));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                x86::Gp status = compiler.new_gp32("psq_st_status");
                invoke->set_ret(0, status);

                Label store_completed = compiler.new_label();
                compiler.test(status, status);
                compiler.jz(store_completed);
                // The helper has either rejected a runtime GQR/NaN case before
                // touching memory or captured a fault. Resume at this guest
                // instruction after synchronizing the block's cached GPRs.
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                flush_guest_gprs();
                x86::Gp stopped_at_store = compiler.new_gp32("psq_st_stop_count");
                compiler.mov(stopped_at_store, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(stopped_at_store);
                compiler.bind(store_completed);

                // A store overlapping this translated block must leave now so
                // the next lookup can validate the code-page generation.
                Label store_is_safe = compiler.new_label();
                Label self_modifying = compiler.new_label();
                compiler.cmp(address, imm(0xFFFFFFF8U));
                compiler.ja(self_modifying);
                const std::uint32_t block_end = block.address +
                    static_cast<std::uint32_t>(block.words.size() * sizeof(std::uint32_t));
                compiler.cmp(address, imm(block_end));
                compiler.jae(store_is_safe);
                x86::Gp store_end = compiler.new_gp32("psq_st_end");
                compiler.mov(store_end, address);
                compiler.add(store_end, imm(8U));
                compiler.cmp(store_end, imm(block.address));
                compiler.jbe(store_is_safe);
                compiler.jmp(self_modifying);
                compiler.bind(self_modifying);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia + 4U));
                x86::Gp completed_store = compiler.new_gp32("psq_st_completed_count");
                compiler.mov(completed_store, imm(static_cast<std::uint32_t>(index + 1U)));
                compiler.ret(completed_store);
                compiler.bind(store_is_safe);
                break;
            }
            case Opcode::paired_single_quantized_load:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                x86::Gp address = compiler.new_gp32("psq_l_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_psq_load));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                        CpuState*, std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, address);
                invoke->set_arg(3, imm(insn.fp_register));
                invoke->set_arg(4, imm(insn.quantized_register));
                invoke->set_arg(5, imm(insn.paired_single_w ? 1U : 0U));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                x86::Gp status = compiler.new_gp32("psq_l_status");
                invoke->set_ret(0, status);

                Label load_completed = compiler.new_label();
                compiler.test(status, status);
                compiler.jz(load_completed);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                flush_guest_gprs();
                x86::Gp stopped_at_load = compiler.new_gp32("psq_l_stop_count");
                compiler.mov(stopped_at_load, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(stopped_at_load);
                compiler.bind(load_completed);
                break;
            }
            case Opcode::load_single:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                x86::Gp address = compiler.new_gp32("lfs_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_load_single));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint64_t, NativeExecutionContext*,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, address);
                invoke->set_arg(2, imm(static_cast<std::uint32_t>(index)));
                x86::Gp value = compiler.new_gp64("lfs_value");
                invoke->set_ret(0, value);

                x86::Gp failed = compiler.new_gp32("lfs_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                Label completed = compiler.new_label();
                compiler.jnz(fault);
                compiler.mov(x86::qword_ptr(state, static_cast<std::int32_t>(
                    fpr_offset(insn.fp_register))), value);
                compiler.mov(x86::qword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, fpr_ps1) + sizeof(std::uint64_t) * insn.fp_register)),
                    value);
                compiler.jmp(completed);
                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                flush_guest_gprs();
                x86::Gp executed = compiler.new_gp32("lfs_completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
                break;
            }
            case Opcode::load_double:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                x86::Gp address = compiler.new_gp32("lfd_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_load_double));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint64_t, NativeExecutionContext*,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, address);
                invoke->set_arg(2, imm(static_cast<std::uint32_t>(index)));
                x86::Gp value = compiler.new_gp64("lfd_value");
                invoke->set_ret(0, value);

                x86::Gp failed = compiler.new_gp32("lfd_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                Label completed = compiler.new_label();
                compiler.jnz(fault);
                compiler.mov(x86::qword_ptr(state, static_cast<std::int32_t>(
                    fpr_offset(insn.fp_register))), value);
                compiler.jmp(completed);
                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                flush_guest_gprs();
                x86::Gp executed = compiler.new_gp32("lfd_completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
                break;
            }
            case Opcode::load_word_zero:
            case Opcode::load_word_update:
            case Opcode::load_byte_zero:
            case Opcode::load_byte_update:
            case Opcode::load_halfword_zero:
            case Opcode::load_halfword_algebraic:
            case Opcode::store_word:
            case Opcode::store_word_update:
            case Opcode::store_byte:
            case Opcode::store_byte_update:
            case Opcode::store_halfword:
            case Opcode::store_single:
            case Opcode::store_single_update:
            case Opcode::store_double:
            case Opcode::load_word_indexed:
            case Opcode::load_byte_zero_indexed:
            case Opcode::load_halfword_zero_indexed:
            case Opcode::load_halfword_algebraic_indexed:
            case Opcode::store_word_indexed:
            case Opcode::store_byte_indexed:
            case Opcode::store_halfword_indexed:
            case Opcode::data_cache_block_store:
            case Opcode::data_cache_block_flush:
            {
                const bool indexed = is_indexed_memory(insn.opcode);
                const bool store = is_memory_store(insn.opcode);
                const bool validation_only =
                    insn.opcode == Opcode::data_cache_block_store ||
                    insn.opcode == Opcode::data_cache_block_flush;
                const bool destination_was_cached =
                    !store && !validation_only &&
                    guest_gpr_initialized[insn.destination];
                const bool single_store = insn.opcode == Opcode::store_single ||
                    insn.opcode == Opcode::store_single_update;
                const bool double_store = insn.opcode == Opcode::store_double;
                const bool update_base = insn.opcode == Opcode::load_byte_update ||
                    insn.opcode == Opcode::load_word_update ||
                    insn.opcode == Opcode::store_byte_update ||
                    insn.opcode == Opcode::store_word_update ||
                    insn.opcode == Opcode::store_single_update;
                const bool sign_extend =
                    insn.opcode == Opcode::load_halfword_algebraic ||
                    insn.opcode == Opcode::load_halfword_algebraic_indexed;
                const std::uint32_t width = memory_width(insn.opcode);
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);

                x86::Gp address = compiler.new_gp32("address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                if (indexed)
                {
                    x86::Gp index_value = compiler.new_gp32("index");
                    compiler.mov(index_value, read_guest_gpr(insn.source));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, index_value))), index_value);
                    compiler.add(address, index_value);
                }
                else
                {
                    compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                }

                x86::Gp stored_value = compiler.new_gp32("stored");
                if (store)
                {
                    if (single_store || double_store)
                    {
                        compiler.mov(stored_value, imm(0U));
                    }
                    else
                    {
                        compiler.mov(stored_value, read_guest_gpr(insn.destination));
                    }
                }
                else
                {
                    compiler.mov(stored_value, imm(0));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const std::uint32_t flags = (store ? 1U : 0U) |
                    (sign_extend ? 2U : 0U) | (indexed ? 4U : 0U);
                if (single_store)
                {
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_store_single));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                            CpuState*, std::uint32_t, std::uint32_t,
                            std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, imm(insn.fp_register));
                    invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));
                }
                else if (double_store)
                {
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_store_double));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                            CpuState*, std::uint32_t, std::uint32_t,
                            std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, imm(insn.fp_register));
                    invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));
                }
                else
                {
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_memory_access));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                            std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, address);
                    invoke->set_arg(2, stored_value);
                    invoke->set_arg(3, imm(width));
                    invoke->set_arg(4, imm(flags));
                    invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
                }
                x86::Gp value = compiler.new_gp32("loaded");
                invoke->set_ret(0, value);

                x86::Gp failed = compiler.new_gp32("failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                Label completed = compiler.new_label();
                compiler.jnz(fault);
                if (!store && !validation_only)
                {
                    write_guest_gpr(insn.destination, value);
                    if (update_base)
                    {
                        write_guest_gpr(insn.base, address);
                    }
                }
                else
                {
                    if (update_base)
                    {
                        write_guest_gpr(insn.base, address);
                    }
                    // A store can only make the instructions already decoded
                    // into this native block stale if its byte range overlaps
                    // this block. Other cached blocks validate guest page
                    // generations when next looked up, so keep running here.
                    Label store_is_safe = compiler.new_label();
                    const std::uint32_t block_end = block.address +
                        static_cast<std::uint32_t>(block.words.size() * sizeof(std::uint32_t));
                    compiler.cmp(address, imm(block_end));
                    compiler.jae(store_is_safe);
                    x86::Gp store_end = compiler.new_gp32("store_end");
                    compiler.mov(store_end, address);
                    compiler.add(store_end, imm(width));
                    compiler.cmp(store_end, imm(block.address));
                    compiler.jbe(store_is_safe);

                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                        offsetof(CpuState, cia))), imm(cia + 4U));
                    flush_guest_gprs();
                    x86::Gp executed = compiler.new_gp32("self_modifying_store");
                    compiler.mov(executed, imm(static_cast<std::uint32_t>(index + 1U)));
                    compiler.ret(executed);
                    compiler.bind(store_is_safe);
                }
                compiler.jmp(completed);
                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                if (!store && !validation_only && !destination_was_cached)
                {
                    // The load destination is only assigned on the success
                    // edge. If the helper faults, initialize its cached host
                    // register from the architectural pre-instruction value
                    // before flushing the block's dirty GPRs.
                    compiler.mov(guest_gprs[insn.destination], x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.destination))));
                }
                flush_guest_gprs();
                x86::Gp executed = compiler.new_gp32("completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
                break;
            }
            default:
                return false;
            }
        }

        const DecodedInstruction& last = block.instructions.back();
        if (last.opcode != Opcode::branch &&
            last.opcode != Opcode::conditional_branch &&
            last.opcode != Opcode::conditional_branch_to_link_register &&
            last.opcode != Opcode::conditional_branch_to_count_register)
        {
            const std::uint32_t next = block.address +
                static_cast<std::uint32_t>(block.instructions.size() * 4U);
            compiler.mov(asmjit::x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, cia))),
                         asmjit::imm(next));
        }
        flush_guest_gprs();
        asmjit::x86::Gp return_value = compiler.new_gp32("step_count");
        compiler.mov(return_value, asmjit::imm(static_cast<std::uint32_t>(block.instructions.size())));
        compiler.ret(return_value);
        compiler.end_func();
        if (compiler.finalize() != asmjit::kErrorOk)
        {
            return false;
        }
        if (runtime.add(&block.function, &code) != asmjit::kErrorOk)
        {
            block.function = nullptr;
            return false;
        }
        return true;
    }
};

Recompiler::Recompiler() : impl_(std::make_unique<Impl>()) {}
Recompiler::~Recompiler() = default;
Recompiler::Recompiler(Recompiler&&) noexcept = default;
Recompiler& Recompiler::operator=(Recompiler&&) noexcept = default;

void Recompiler::clear_cache() noexcept
{
    if (impl_) impl_->clear();
}

const RecompilerStatistics& Recompiler::statistics() const noexcept
{
    return impl_->stats;
}

RunResult Recompiler::run(EspressoCore& core, std::size_t max_steps)
{
    RunResult result{};
    result.cia = core.state.cia;
    const auto include_history = [&]() {
        result.instruction_history = core.instruction_history_snapshot();
    };
    const auto count_fallback = [&](bool is_hle) {
        ++impl_->stats.interpreter_fallbacks;
        if (!is_hle)
        {
            try
            {
                const DecodedInstruction decoded = decode(core.memory.read32_be(core.state.cia));
                ++impl_->stats.fallbacks_by_opcode[
                    static_cast<std::uint16_t>(decoded.opcode)];
            }
            catch (const GuestMemoryFault&)
            {
                // The interpreter immediately reports this fetch fault.
            }
        }
    };

    while (result.steps < max_steps)
    {
        ++impl_->stats.dispatcher_entries;
        const std::uint32_t start = core.state.cia;
        if (core.hle.is_import_address(start))
        {
            count_fallback(true);
            StepResult step_result{};
            try
            {
                step_result = core.step();
            }
            catch (const GuestMemoryFault& fault)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.detail = memory_fault_text(fault, result.cia);
                include_history();
                return result;
            }
            catch (const HleExecutionError& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::hle_error;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
            catch (const std::exception& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
            if (step_result == StepResult::unimplemented_hle_call)
            {
                result.reason = StopReason::unimplemented_hle_call;
                result.cia = core.state.cia;
                result.hle_call = core.hle.last_unimplemented_call();
                include_history();
                return result;
            }
            if (step_result == StepResult::unsupported_instruction)
            {
                result.reason = StopReason::unsupported_instruction;
                result.cia = core.state.cia;
                result.instruction_word = core.memory.read32_be(core.state.cia);
                result.has_instruction_word = true;
                include_history();
                return result;
            }
            ++result.steps;
            continue;
        }

        BlockCode* block = nullptr;
        try
        {
            Impl::FastLookupEntry& fast_entry = impl_->fast_lookup_cache[
                (start >> 2U) & (Impl::fast_lookup_cache_size - 1U)];
            if (impl_->last_cache_entry != nullptr &&
                impl_->last_cache_address == start &&
                impl_->last_cache_entry->memory_owner == &core.memory &&
                Impl::code_page_generations_match(
                    *impl_->last_cache_entry, core.memory))
            {
                ++impl_->stats.cache_hits;
                ++impl_->stats.last_block_cache_hits;
                block = &impl_->last_cache_entry->block;
            }
            else if (fast_entry.address == start && fast_entry.entry != nullptr &&
                fast_entry.entry->memory_owner == &core.memory &&
                Impl::code_page_generations_match(*fast_entry.entry, core.memory))
            {
                ++impl_->stats.cache_hits;
                ++impl_->stats.fast_lookup_hits;
                block = &fast_entry.entry->block;
                impl_->last_cache_entry = fast_entry.entry;
                impl_->last_cache_address = start;
            }
            else
            {
                auto found = impl_->cache.find(start);
                bool valid = found != impl_->cache.end();
                if (valid && (found->second.memory_owner != &core.memory ||
                        !Impl::code_page_generations_match(found->second, core.memory)))
                {
                    const auto& words = found->second.block.words;
                    for (std::size_t i = 0; i < words.size(); ++i)
                    {
                        if (core.memory.read32_be(
                                start + static_cast<std::uint32_t>(i * 4U)) != words[i])
                        {
                            valid = false;
                            break;
                        }
                    }
                    if (!valid)
                    {
                        if (fast_entry.address == start &&
                            fast_entry.entry == &found->second)
                        {
                            fast_entry = {};
                        }
                        if (impl_->last_cache_entry == &found->second)
                        {
                            impl_->last_cache_entry = nullptr;
                            impl_->last_cache_address = 0U;
                        }
                        if (found->second.block.function != nullptr)
                            static_cast<void>(impl_->runtime.release(found->second.block.function));
                        impl_->cache.erase(found);
                        found = impl_->cache.end();
                    }
                    else
                    {
                        Impl::capture_code_page_generations(found->second, core.memory);
                    }
                }

                if (valid)
                {
                    ++impl_->stats.cache_hits;
                    block = &found->second.block;
                    impl_->last_cache_entry = &found->second;
                    impl_->last_cache_address = start;
                    fast_entry = {start, &found->second};
                }
                else
                {
                    ++impl_->stats.block_lookup_misses;
                    BlockCode candidate{};
                    candidate.address = start;
                    constexpr std::size_t max_block_instructions = 64U;
                    for (std::size_t i = 0; i < max_block_instructions; ++i)
                    {
                        const std::uint32_t cia = start + static_cast<std::uint32_t>(i * 4U);
                        if (core.hle.is_import_address(cia))
                        {
                            break;
                        }
                        std::uint32_t word{};
                        try
                        {
                            word = core.memory.read32_be(cia);
                        }
                        catch (const GuestMemoryFault&)
                        {
                            if (candidate.instructions.empty()) throw;
                            break;
                        }
                        const DecodedInstruction decoded = decode(word);
                        if (!is_native_instruction(decoded))
                        {
                            break;
                        }
                        candidate.words.push_back(word);
                        candidate.instructions.push_back(decoded);
                        if (decoded.opcode == Opcode::instruction_sync ||
                            decoded.opcode == Opcode::branch ||
                            decoded.opcode == Opcode::conditional_branch ||
                            decoded.opcode == Opcode::conditional_branch_to_link_register ||
                            decoded.opcode == Opcode::conditional_branch_to_count_register)
                        {
                            break;
                        }
                    }
                    bool compiled = false;
                    if (!candidate.instructions.empty())
                    {
#ifndef NDEBUG
                        const auto compile_start = std::chrono::steady_clock::now();
#endif
                        compiled = impl_->compile(candidate);
#ifndef NDEBUG
                        impl_->stats.compilation_time_ns += static_cast<std::uint64_t>(
                            std::chrono::duration_cast<std::chrono::nanoseconds>(
                                std::chrono::steady_clock::now() - compile_start).count());
#endif
                    }
                    if (compiled)
                    {
                        Impl::CacheEntry entry{};
                        entry.block = std::move(candidate);
                        Impl::capture_code_page_generations(entry, core.memory);
                        auto [inserted, did_insert] = impl_->cache.emplace(start,
                                                                          std::move(entry));
                        static_cast<void>(did_insert);
                        ++impl_->stats.compiled_blocks;
                        block = &inserted->second.block;
                        impl_->last_cache_entry = &inserted->second;
                        impl_->last_cache_address = start;
                        fast_entry = {start, &inserted->second};
                    }
                }
            }
        }
        catch (const GuestMemoryFault& fault)
        {
            result.reason = StopReason::memory_fault;
            result.cia = core.state.cia;
            result.detail = memory_fault_text(fault, core.state.cia);
            include_history();
            return result;
        }

        if (block == nullptr || block->function == nullptr ||
            block->instructions.size() > max_steps - result.steps)
        {
            count_fallback(false);
            StepResult step_result{};
            try
            {
                step_result = core.step();
            }
            catch (const GuestMemoryFault& fault)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.has_instruction_word = core.current_instruction_word_fetched_;
                if (result.has_instruction_word)
                    result.instruction_word = core.current_instruction_word_;
                result.detail = memory_fault_text(fault, result.cia);
                include_history();
                return result;
            }
            catch (const HleExecutionError& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::hle_error;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
            catch (const std::exception& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
            if (step_result == StepResult::unsupported_instruction)
            {
                result.reason = StopReason::unsupported_instruction;
                result.cia = core.state.cia;
                result.instruction_word = core.memory.read32_be(core.state.cia);
                result.has_instruction_word = true;
                include_history();
                return result;
            }
            if (step_result == StepResult::unimplemented_hle_call)
            {
                result.reason = StopReason::unimplemented_hle_call;
                result.cia = core.state.cia;
                result.hle_call = core.hle.last_unimplemented_call();
                include_history();
                return result;
            }
            ++result.steps;
            continue;
        }

        NativeExecutionContext context{};
        context.memory = &core.memory;
        context.block_cia = block->address;
#ifndef NDEBUG
        const auto native_start = std::chrono::steady_clock::now();
#endif
        const std::uint32_t executed = block->function(&core.state, &context);
#ifndef NDEBUG
        impl_->stats.native_execution_time_ns += static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - native_start).count());
#endif
        impl_->stats.memory_slow_path_calls += context.memory_access_calls;
        impl_->stats.native_instructions += executed;
        // RunResult exposes only the most recent 32 history records. A native
        // block can execute 64 instructions, so records older than the final
        // history window are guaranteed to be overwritten before observation.
        const std::size_t first_history_index = executed > instruction_history_capacity
            ? static_cast<std::size_t>(executed) - instruction_history_capacity
            : 0U;
        for (std::size_t i = first_history_index; i < executed; ++i)
        {
            const std::uint32_t cia = block->address + static_cast<std::uint32_t>(i * 4U);
            const DecodedInstruction& insn = block->instructions[i];
            if (insn.opcode == Opcode::floating_round_to_single)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFrspRecord& record = context.frsp_records[i];
                entry.has_fp_arithmetic = true;
                entry.fp_arithmetic_destination = insn.fp_register;
                entry.fp_arithmetic_source_a = insn.fp_source_a;
                entry.fp_arithmetic_a_raw = record.source_raw;
                entry.fp_arithmetic_source_count = 1U;
                if (record.wrote_result)
                {
                    entry.has_fp_arithmetic_result = true;
                    entry.fp_arithmetic_result_raw = record.result_raw;
                    entry.has_fp_arithmetic_single_bits = true;
                    entry.fp_arithmetic_single_bits = record.single_bits;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_multiply_double)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFmulRecord& record = context.fmul_records[i];
                entry.has_fp_arithmetic = true;
                entry.fp_arithmetic_destination = insn.fp_register;
                entry.fp_arithmetic_source_a = insn.fp_source_a;
                entry.fp_arithmetic_source_c = insn.fp_source_c;
                entry.fp_arithmetic_a_raw = record.a_raw;
                entry.fp_arithmetic_c_raw = record.c_raw;
                entry.fp_arithmetic_source_count = 2U;
                entry.has_fp_arithmetic_source_c = true;
                if (record.wrote_result)
                {
                    entry.has_fp_arithmetic_result = true;
                    entry.fp_arithmetic_result_raw = record.result_raw;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_multiply_single)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFmulsRecord& record = context.fmuls_records[i];
                entry.has_fp_arithmetic = true;
                entry.fp_arithmetic_destination = insn.fp_register;
                entry.fp_arithmetic_source_a = insn.fp_source_a;
                entry.fp_arithmetic_source_c = insn.fp_source_c;
                entry.fp_arithmetic_a_raw = record.a_raw;
                entry.fp_arithmetic_c_raw = record.c_raw;
                entry.fp_arithmetic_source_count = 2U;
                entry.has_fp_arithmetic_source_c = true;
                if (record.wrote_result)
                {
                    entry.has_fp_arithmetic_result = true;
                    entry.fp_arithmetic_result_raw = record.result_raw;
                    entry.has_fp_arithmetic_single_bits = true;
                    entry.fp_arithmetic_single_bits = record.single_bits;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_multiply_add_double)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFmaddRecord& record = context.fmadd_records[i];
                entry.has_fp_arithmetic = true;
                entry.fp_arithmetic_destination = insn.fp_register;
                entry.fp_arithmetic_source_a = insn.fp_source_a;
                entry.fp_arithmetic_source_b = insn.fp_source_b;
                entry.fp_arithmetic_source_c = insn.fp_source_c;
                entry.fp_arithmetic_a_raw = record.a_raw;
                entry.fp_arithmetic_b_raw = record.b_raw;
                entry.fp_arithmetic_c_raw = record.c_raw;
                entry.fp_arithmetic_source_count = 3U;
                entry.has_fp_arithmetic_source_c = true;
                if (record.wrote_result)
                {
                    entry.has_fp_arithmetic_result = true;
                    entry.fp_arithmetic_result_raw = record.result_raw;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_subtract_double)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFsubRecord& record = context.fsub_records[i];
                entry.has_fp_arithmetic = true;
                entry.fp_arithmetic_destination = insn.fp_register;
                entry.fp_arithmetic_source_a = insn.fp_source_a;
                entry.fp_arithmetic_source_b = insn.fp_source_b;
                entry.fp_arithmetic_a_raw = record.a_raw;
                entry.fp_arithmetic_b_raw = record.b_raw;
                entry.fp_arithmetic_source_count = 2U;
                if (record.wrote_result)
                {
                    entry.has_fp_arithmetic_result = true;
                    entry.fp_arithmetic_result_raw = record.result_raw;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_convert_to_integer_word_zero)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFpIntegerConversionRecord& record =
                    context.fp_integer_conversion_records[i];
                entry.has_fp_integer_conversion = true;
                entry.fp_integer_conversion_source = insn.fp_source_b;
                entry.fp_integer_conversion_source_raw = record.source_raw;
                entry.fp_integer_conversion_destination = insn.fp_register;
                entry.fp_integer_conversion_result = record.integer_bits;
                entry.fp_integer_conversion_encoded_raw = record.encoded_raw;
                entry.fp_integer_conversion_invalid = record.invalid;
                entry.fp_integer_conversion_write_suppressed = record.write_suppressed;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_compare_unordered)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFpCompareRecord& record = context.fp_compare_records[i];
                entry.has_fp_compare = true;
                entry.fp_compare_cr_field = insn.cr_field;
                entry.fp_compare_a_register = insn.fp_compare_a;
                entry.fp_compare_b_register = insn.fp_compare_b;
                entry.fp_compare_a_raw = record.a_raw;
                entry.fp_compare_b_raw = record.b_raw;
                entry.fp_compare_result = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::multiply_low_immediate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMulliRecord& record = context.mulli_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.source_value;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::multiply_low_word)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMullwRecord& record = context.mullw_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.source_a;
                entry.source_registers[1] = insn.source;
                entry.source_values[1] = record.source_b;
                entry.source_count = 2U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_quantized_load)
            {
                const NativeMemoryRecord& memory_record = context.memory_records[i];
                const NativePsqLoadRecord& record = context.psq_load_records[i];
                InstructionHistoryEntry entry = native_history(insn, cia);
                entry.completed = true;
                entry.has_effective_address = true;
                entry.effective_address = memory_record.address;
                if (insn.base != 0U)
                {
                    entry.source_registers[entry.source_count] = insn.base;
                    entry.source_values[entry.source_count++] = memory_record.base_value;
                }
                entry.has_paired_fp_state = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_quantized_store)
            {
                const NativeMemoryRecord& memory_record = context.memory_records[i];
                const NativePsqStoreRecord& record = context.psq_store_records[i];
                InstructionHistoryEntry entry = native_history(insn, cia);
                entry.completed = true;
                entry.has_effective_address = true;
                entry.effective_address = memory_record.address;
                if (insn.base != 0U)
                {
                    entry.source_registers[entry.source_count] = insn.base;
                    entry.source_values[entry.source_count++] = memory_record.base_value;
                }
                entry.has_paired_fp_state = true;
                entry.paired_fp_quantized_store = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_register;
                entry.paired_fp_source_a_ps0 = record.source_ps0;
                entry.paired_fp_source_a_ps1 = record.source_ps1;
                entry.paired_fp_store_gqr = insn.quantized_register;
                entry.paired_fp_store_type = static_cast<std::uint8_t>(record.gqr & 0x7U);
                entry.paired_fp_store_w = insn.paired_single_w;
                entry.paired_fp_store_value0 = record.value0;
                entry.paired_fp_store_value1 = record.value1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (!is_memory_instruction(insn.opcode))
            {
                core.append_instruction_history(native_history(insn, cia));
                continue;
            }

            const NativeMemoryRecord& record = context.memory_records[i];
            InstructionHistoryEntry entry{};
            entry.cia = cia;
            entry.instruction_word = insn.raw;
            entry.has_instruction_word = true;
            entry.opcode_name = name(insn.opcode);
            entry.completed = true;
            entry.has_effective_address = true;
            entry.effective_address = record.address;
            if (insn.opcode == Opcode::load_multiple_word ||
                insn.opcode == Opcode::store_multiple_word)
            {
                entry.has_register_range = true;
                entry.range_first_register = insn.opcode == Opcode::load_multiple_word
                    ? insn.destination : insn.source;
                entry.range_last_register = 31U;
                entry.memory_base_register = insn.base;
                entry.memory_displacement = insn.immediate;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (record.indexed)
            {
                if (insn.base != 0U)
                {
                    entry.source_registers[entry.source_count] = insn.base;
                    entry.source_values[entry.source_count++] = record.base_value;
                }
                entry.source_registers[entry.source_count] = insn.source;
                entry.source_values[entry.source_count++] = record.index_value;
            }
            else
            {
                if (insn.base != 0U)
                {
                    entry.source_registers[entry.source_count] = insn.base;
                    entry.source_values[entry.source_count++] = record.base_value;
                }
                entry.has_immediate = true;
                entry.immediate = insn.immediate;
            }
            if (insn.opcode == Opcode::data_cache_block_store ||
                insn.opcode == Opcode::data_cache_block_flush)
            {
                // Cache-maintenance instructions have no architectural
                // destination in Affogato's
                // coherent-memory model; the byte read above exists only to
                // retain its translation/protection fault behavior.
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::load_single || insn.opcode == Opcode::load_double)
            {
                entry.has_fp_destination = true;
                entry.fp_destination_register = insn.fp_register;
                entry.fp_destination_value = record.fp_value;
            }
            else if (insn.opcode == Opcode::store_single ||
                insn.opcode == Opcode::store_single_update)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = record.fp_value;
                entry.has_stored_single_value = true;
                entry.stored_single_value = record.value;
                if (insn.opcode == Opcode::store_single_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
            }
            else if (insn.opcode == Opcode::store_double)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = record.fp_value;
                entry.has_stored_double_value = true;
                entry.stored_double_value = record.fp_value;
            }
            else if (is_memory_store(insn.opcode))
            {
                entry.source_registers[entry.source_count] = insn.destination;
                entry.source_values[entry.source_count++] = record.value;
                if (insn.opcode == Opcode::store_halfword ||
                    insn.opcode == Opcode::store_halfword_indexed)
                {
                    entry.has_stored_halfword_value = true;
                    entry.stored_halfword_value = static_cast<std::uint16_t>(record.value);
                }
                if (insn.opcode == Opcode::store_word ||
                    insn.opcode == Opcode::store_word_indexed ||
                    insn.opcode == Opcode::store_word_update)
                {
                    entry.has_stored_integer_word_value = true;
                    entry.stored_integer_word_value = record.value;
                }
                if (insn.opcode == Opcode::store_word_update ||
                    insn.opcode == Opcode::store_byte_update ||
                    insn.opcode == Opcode::store_single_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
            }
            else
            {
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.value;
            }
            core.append_instruction_history(std::move(entry));
        }
        result.steps += executed;
        result.cia = core.state.cia;
        if (context.unsupported_instruction)
        {
            ++impl_->stats.interpreter_fallbacks;
            const StepResult fallback_result = core.step();
            if (fallback_result != StepResult::unsupported_instruction)
            {
                result.reason = StopReason::hle_error;
                result.detail = "native psq_st fallback did not reproduce its unsupported result";
                result.cia = core.state.cia;
                include_history();
                return result;
            }
            result.reason = StopReason::unsupported_instruction;
            result.cia = core.state.cia;
            result.instruction_word = core.memory.read32_be(core.state.cia);
            result.has_instruction_word = true;
            include_history();
            return result;
        }
        if (context.failed)
        {
            try
            {
                if (context.guest_memory_fault)
                {
                    const DecodedInstruction& faulting_instruction =
                        block->instructions[context.failed_instruction_index];
                    if (faulting_instruction.opcode == Opcode::store_multiple_word)
                    {
                        InstructionHistoryEntry entry{};
                        entry.cia = block->address + static_cast<std::uint32_t>(
                            context.failed_instruction_index * 4U);
                        entry.instruction_word = faulting_instruction.raw;
                        entry.has_instruction_word = true;
                        entry.opcode_name = "stmw";
                        entry.completed = false;
                        entry.has_effective_address = true;
                        entry.effective_address = context.memory_records[
                            context.failed_instruction_index].address;
                        entry.has_register_range = true;
                        entry.range_first_register = faulting_instruction.source;
                        entry.range_last_register = 31U;
                        entry.memory_base_register = faulting_instruction.base;
                        entry.memory_displacement = faulting_instruction.immediate;
                        core.append_instruction_history(std::move(entry));
                        std::rethrow_exception(context.exception);
                    }
                    static_cast<void>(core.step());
                }
                else if (context.exception)
                    std::rethrow_exception(context.exception);
                result.reason = StopReason::hle_error;
                {
                    std::ostringstream detail;
                    detail << "native memory helper failed without an exception: address=0x"
                           << std::hex << context.failed_address << " width=" << std::dec
                           << context.failed_width << " flags=0x" << std::hex
                           << context.failed_flags << " instruction_index=" << std::dec
                           << context.failed_instruction_index << " calls="
                           << context.memory_access_calls << " guest_memory_fault="
                           << context.guest_memory_fault << " exception="
                           << static_cast<bool>(context.exception);
                    result.detail = detail.str();
                }
                include_history();
                return result;
            }
            catch (const GuestMemoryFault& fault)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.detail = memory_fault_text(fault, context.fault_cia);
                include_history();
                return result;
            }
            catch (const HleExecutionError& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::hle_error;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
            catch (const std::exception& error)
            {
                if (core.has_pending_history_entry_)
                {
                    core.append_instruction_history(std::move(core.pending_history_entry_));
                    core.has_pending_history_entry_ = false;
                }
                result.reason = StopReason::memory_fault;
                result.cia = core.state.cia;
                result.detail = error.what();
                include_history();
                return result;
            }
        }
    }

    result.reason = StopReason::instruction_limit;
    result.cia = core.state.cia;
    include_history();
    return result;
}

}
