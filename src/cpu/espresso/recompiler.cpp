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
#include <span>
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
constexpr std::uint32_t xer_summary_overflow_mask = 0x80000000U;
constexpr std::uint32_t xer_overflow_mask = 0x40000000U;

struct NativeMemoryRecord
{
    std::uint32_t address{};
    std::uint32_t value{};
    std::uint32_t source_value{};
    std::uint64_t fp_value{};
    std::uint32_t base_value{};
    std::uint32_t index_value{};
    bool indexed{};
};

struct NativeStringLoadRecord
{
    std::uint32_t address{};
    std::uint32_t base_value{};
    std::uint32_t index_value{};
    std::uint8_t byte_count{};
    std::uint8_t register_count{};
    bool indexed{};
    std::array<std::uint8_t, 127> bytes{};
    std::array<std::uint8_t, 32> registers{};
    std::array<std::uint32_t, 32> values{};
};

struct NativeStringStoreRecord
{
    std::uint32_t address{};
    std::uint32_t base_value{};
    std::uint32_t index_value{};
    std::uint8_t byte_count{};
    std::uint8_t register_count{};
    bool indexed{};
    std::array<std::uint8_t, 127> bytes{};
    std::array<std::uint8_t, 32> registers{};
    std::array<std::uint32_t, 32> values{};
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

struct NativeMffsRecord
{
    std::uint32_t fpscr{};
    std::uint64_t result{};
};

struct NativeMtfsbRecord
{
    std::uint32_t before{};
    std::uint32_t after{};
};

using NativeMtfsfiRecord = NativeMtfsbRecord;

struct NativeMtfsfRecord
{
    std::uint64_t source_raw{};
    std::uint32_t before{};
    std::uint32_t after{};
};

struct NativeMcrfsRecord
{
    std::uint32_t before{};
    std::uint32_t after{};
    std::uint8_t value{};
};

struct NativeFrspRecord
{
    std::uint64_t source_raw{};
    std::uint64_t result_raw{};
    std::uint32_t single_bits{};
    bool wrote_result{};
};

struct NativeFrsqrteRecord
{
    std::uint64_t source_raw{};
    std::uint64_t result_raw{};
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

struct NativeFmaddsRecord
{
    std::uint64_t a_raw{};
    std::uint64_t b_raw{};
    std::uint64_t c_raw{};
    std::uint64_t result_raw{};
    std::uint32_t single_bits{};
    bool wrote_result{};
};

struct NativeFaddsRecord
{
    std::uint64_t a_raw{};
    std::uint64_t b_raw{};
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

struct NativePsSumRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_b_ps0{};
    std::uint64_t source_b_ps1{};
    std::uint64_t source_c_ps0{};
    std::uint64_t source_c_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
};

struct NativePsSelectRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_b_ps0{};
    std::uint64_t source_b_ps1{};
    std::uint64_t source_c_ps0{};
    std::uint64_t source_c_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
};

struct NativePsBinaryRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_b_ps0{};
    std::uint64_t source_b_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
};

struct NativePsMergeRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_b_ps0{};
    std::uint64_t source_b_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
};

struct NativePsMulRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_c_ps0{};
    std::uint64_t source_c_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
};

struct NativePsMaddRecord
{
    std::uint64_t source_a_ps0{};
    std::uint64_t source_a_ps1{};
    std::uint64_t source_b_ps0{};
    std::uint64_t source_b_ps1{};
    std::uint64_t source_c_ps0{};
    std::uint64_t source_c_ps1{};
    std::uint64_t destination_ps0{};
    std::uint64_t destination_ps1{};
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
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

struct NativeDivwuRecord
{
    std::uint32_t dividend{};
    std::uint32_t divisor{};
    std::uint32_t result{};
};

struct NativeDivwRecord
{
    std::uint32_t dividend{};
    std::uint32_t divisor{};
    std::uint32_t result{};
};

struct NativeSubfRecord
{
    std::uint32_t source_a{};
    std::uint32_t source_b{};
    std::uint32_t result{};
};

struct NativeAddOverflowRecord
{
    std::uint32_t operand_a{};
    std::uint32_t operand_b{};
    std::uint32_t result{};
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

using NativeSubfOverflowRecord = NativeAddOverflowRecord;

struct NativeNegateOverflowRecord
{
    std::uint32_t source{};
    std::uint32_t result{};
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

struct NativeFselRecord
{
    std::uint64_t source_a_raw{};
    std::uint64_t source_b_raw{};
    std::uint64_t source_c_raw{};
    std::uint64_t result_raw{};
};

struct NativeMulhwuRecord
{
    std::uint32_t source_a{};
    std::uint32_t source_b{};
    std::uint32_t result{};
};

struct NativeAddcRecord
{
    std::uint32_t operand_a{};
    std::uint32_t operand_b{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
};

struct NativeAddcoRecord
{
    std::uint32_t operand_a{};
    std::uint32_t operand_b{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

using NativeSubfcoRecord = NativeAddcoRecord;

struct NativeAddeRecord
{
    std::uint32_t operand_a{};
    std::uint32_t operand_b{};
    std::uint32_t carry_in{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
};

struct NativeAddeoRecord
{
    std::uint32_t operand_a{};
    std::uint32_t operand_b{};
    std::uint32_t carry_in{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

struct NativeAddzeRecord
{
    std::uint32_t operand{};
    std::uint32_t carry_in{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
};

struct NativeAddzeoRecord
{
    std::uint32_t operand{};
    std::uint32_t carry_in{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
    std::uint32_t overflow{};
    std::uint32_t summary_overflow{};
};

using NativeAddmeoRecord = NativeAddzeoRecord;

using NativeSubfzeRecord = NativeAddzeRecord;

struct NativeSrawRecord
{
    std::uint32_t source{};
    std::uint32_t shift{};
    std::uint32_t result{};
    std::uint32_t carry_out{};
};

struct NativeExtsbRecord
{
    std::uint32_t source{};
    std::uint32_t result{};
};

using NativeExtshRecord = NativeExtsbRecord;

struct NativeTimeBaseRecord
{
    std::uint32_t result{};
};

struct NativeExecutionContext
{
    EspressoCore* core{};
    GuestMemory* memory{};
    const std::uint8_t* flat_memory_data{};
    std::size_t flat_memory_size{};
    std::uint32_t block_cia{};
    std::array<NativeMemoryRecord, 64> memory_records{};
    std::array<NativeStringLoadRecord, 64> lswi_records{};
    std::array<NativeStringStoreRecord, 64> stswi_records{};
    std::array<NativeFpCompareRecord, 64> fp_compare_records{};
    std::array<NativeFpIntegerConversionRecord, 64> fp_integer_conversion_records{};
    std::array<NativeMffsRecord, 64> mffs_records{};
    std::array<NativeMtfsbRecord, 64> mtfsb_records{};
    std::array<NativeMtfsfiRecord, 64> mtfsfi_records{};
    std::array<NativeMtfsfRecord, 64> mtfsf_records{};
    std::array<NativeMcrfsRecord, 64> mcrfs_records{};
    std::array<NativeFrspRecord, 64> frsp_records{};
    std::array<NativeFrspRecord, 64> fres_records{};
    std::array<NativeFrsqrteRecord, 64> frsqrte_records{};
    std::array<NativeFmulRecord, 64> fmul_records{};
    std::array<NativeFsubRecord, 64> fdiv_records{};
    std::array<NativePsSumRecord, 64> ps_sum_records{};
    std::array<NativePsSelectRecord, 64> ps_select_records{};
    std::array<NativePsBinaryRecord, 64> ps_add_records{};
    std::array<NativePsBinaryRecord, 64> ps_sub_records{};
    std::array<NativePsBinaryRecord, 64> ps_div_records{};
    std::array<NativePsMergeRecord, 64> ps_merge_records{};
    std::array<NativePsMergeRecord, 64> ps_move_records{};
    std::array<NativePsMergeRecord, 64> ps_neg_records{};
    std::array<NativePsMergeRecord, 64> ps_abs_records{};
    std::array<NativePsMergeRecord, 64> ps_nabs_records{};
    std::array<NativePsMergeRecord, 64> ps_res_records{};
    std::array<NativePsMergeRecord, 64> ps_rsqrte_records{};
    std::array<NativePsMulRecord, 64> ps_mul_records{};
    std::array<NativePsMulRecord, 64> ps_muls_records{};
    std::array<NativePsMaddRecord, 64> ps_madd_records{};
    std::array<NativeFmulsRecord, 64> fmuls_records{};
    std::array<NativeFmaddsRecord, 64> fmadds_records{};
    std::array<NativeFmaddsRecord, 64> fnmsubs_records{};
    std::array<NativeFmaddsRecord, 64> fmsubs_records{};
    std::array<NativeFaddsRecord, 64> fadds_records{};
    std::array<NativeFaddsRecord, 64> fsubs_records{};
    std::array<NativeFaddsRecord, 64> fdivs_records{};
    std::array<NativeFmaddRecord, 64> fmadd_records{};
    std::array<NativeFsubRecord, 64> fsub_records{};
    std::array<NativeFsubRecord, 64> fadd_records{};
    std::array<NativePsqStoreRecord, 64> psq_store_records{};
    std::array<NativePsqLoadRecord, 64> psq_load_records{};
    std::array<NativeMulliRecord, 64> mulli_records{};
    std::array<NativeMullwRecord, 64> mullw_records{};
    std::array<NativeDivwuRecord, 64> divwu_records{};
    std::array<NativeDivwRecord, 64> divw_records{};
    std::array<NativeSubfRecord, 64> subf_records{};
    std::array<NativeAddOverflowRecord, 64> add_overflow_records{};
    std::array<NativeSubfOverflowRecord, 64> subf_overflow_records{};
    std::array<NativeNegateOverflowRecord, 64> negate_overflow_records{};
    std::array<NativeFselRecord, 64> fsel_records{};
    std::array<NativeMulhwuRecord, 64> mulhwu_records{};
    std::array<NativeAddcRecord, 64> addc_records{};
    std::array<NativeAddcoRecord, 64> addco_records{};
    std::array<NativeSubfcoRecord, 64> subfco_records{};
    std::array<NativeAddeRecord, 64> adde_records{};
    std::array<NativeAddeoRecord, 64> addeo_records{};
    std::array<NativeAddzeRecord, 64> addze_records{};
    std::array<NativeAddzeoRecord, 64> addzeo_records{};
    std::array<NativeAddzeRecord, 64> addme_records{};
    std::array<NativeAddmeoRecord, 64> addmeo_records{};
    std::array<NativeAddzeRecord, 64> subfme_records{};
    std::array<NativeSubfzeRecord, 64> subfze_records{};
    std::array<NativeSrawRecord, 64> sraw_records{};
    std::array<NativeExtsbRecord, 64> extsb_records{};
    std::array<NativeExtshRecord, 64> extsh_records{};
    std::array<NativeTimeBaseRecord, 64> time_base_records{};
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

    void begin_block(
        GuestMemory& guest_memory,
        std::uint32_t guest_cia,
        std::span<const DecodedInstruction> instructions) noexcept
    {
        memory = &guest_memory;
        flat_memory_data = guest_memory.flat_data();
        flat_memory_size = guest_memory.size();
        block_cia = guest_cia;
        memory_access_calls = 0U;
        failed = false;
        guest_memory_fault = false;
        unsupported_instruction = false;
        exception = nullptr;
        fault_cia = 0U;
        failed_address = 0U;
        failed_width = 0U;
        failed_flags = 0U;
        failed_instruction_index = 0U;

        for (std::size_t index = 0U; index < instructions.size(); ++index)
        {
            switch (instructions[index].opcode)
            {
            case Opcode::floating_round_to_single:
                frsp_records[index].wrote_result = false;
                break;
            case Opcode::floating_reciprocal_estimate_single:
                fres_records[index].wrote_result = false;
                break;
            case Opcode::floating_reciprocal_square_root_estimate:
                frsqrte_records[index].wrote_result = false;
                break;
            case Opcode::floating_multiply_double:
                fmul_records[index].wrote_result = false;
                break;
            case Opcode::floating_multiply_single:
                fmuls_records[index].wrote_result = false;
                break;
            case Opcode::floating_multiply_add_single:
                fmadds_records[index].wrote_result = false;
                break;
            case Opcode::floating_multiply_subtract_single:
                fmsubs_records[index].wrote_result = false;
                break;
            case Opcode::floating_negative_multiply_subtract_single:
            case Opcode::floating_negative_multiply_add_single:
                fnmsubs_records[index].wrote_result = false;
                break;
            case Opcode::floating_add_single:
                fadds_records[index].wrote_result = false;
                break;
            case Opcode::floating_subtract_single:
                fsubs_records[index].wrote_result = false;
                break;
            case Opcode::floating_divide_single:
                fdivs_records[index].wrote_result = false;
                break;
            case Opcode::floating_multiply_add_double:
            case Opcode::floating_multiply_subtract_double:
            case Opcode::floating_negative_multiply_add_double:
            case Opcode::floating_negative_multiply_subtract_double:
                fmadd_records[index].wrote_result = false;
                break;
            case Opcode::floating_subtract_double:
                fsub_records[index].wrote_result = false;
                break;
            case Opcode::floating_add_double:
                fadd_records[index].wrote_result = false;
                break;
            case Opcode::floating_divide_double:
                fdiv_records[index].wrote_result = false;
                break;
            default:
                break;
            }
        }
    }
};

extern "C" std::uint32_t native_read_time_base(
    NativeExecutionContext* context, std::uint32_t upper) noexcept
{
    if (context->core == nullptr)
    {
        return 0U;
    }
    const std::uint64_t ticks = context->core->time_base_ticks();
    return upper != 0U
        ? static_cast<std::uint32_t>(ticks >> 32U)
        : static_cast<std::uint32_t>(ticks);
}

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

extern "C" void native_raise_fpscr_compare_invalid_if_present(
    CpuState* state,
    std::uint64_t a_raw,
    std::uint64_t b_raw,
    std::uint32_t ordered) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    const auto is_nan = [&](std::uint64_t value) {
        return (value & exponent_mask) == exponent_mask &&
            (value & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t value) {
        return is_nan(value) && (value & quiet_bit) == 0U;
    };
    const bool signaling_nan =
        is_signaling_nan(a_raw) || is_signaling_nan(b_raw);
    const bool any_nan = is_nan(a_raw) || is_nan(b_raw);
    if (signaling_nan)
    {
        native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
    }
    if (ordered != 0U &&
        ((signaling_nan && (state->fpscr & fpscr::ve_mask) == 0U) ||
         (!signaling_nan && any_nan)))
    {
        native_raise_fpscr_exception(state, fpscr::vxvc_mask);
    }
}

extern "C" void native_fctiw(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_register,
    std::uint32_t instruction_index,
    std::uint32_t round_toward_zero) noexcept
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
    bool rounded_up = false;

    const auto round_source = [source](std::uint32_t rounding_mode) noexcept {
        switch (rounding_mode)
        {
        case 0U:
        {
            const double lower = std::floor(source);
            const double fraction = source - lower;
            if (fraction < 0.5)
            {
                return lower;
            }
            if (fraction > 0.5)
            {
                return lower + 1.0;
            }
            return std::fmod(std::fabs(lower), 2.0) == 0.0
                ? lower
                : lower + 1.0;
        }
        case 1U: return std::trunc(source);
        case 2U: return std::ceil(source);
        case 3U: return std::floor(source);
        default: return source;
        }
    };

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
        const std::uint32_t rounding_mode = round_toward_zero != 0U
            ? 1U
            : state->fpscr & fpscr::rn_mask;
        const double rounded = round_source(rounding_mode);
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
            rounded_up = inexact && std::fabs(rounded) > std::fabs(source);
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
        if (rounded_up)
        {
            state->fpscr |= fpscr::fr_mask;
        }
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

extern "C" void native_frsqrte(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_register,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    constexpr std::uint64_t infinity = 0x7FF0000000000000ULL;
    if (destination >= state->fpr.size() || source_register >= state->fpr.size() ||
        instruction_index >= context->frsqrte_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFrsqrteRecord& record = context->frsqrte_records[instruction_index];
    record.source_raw = state->fpr[source_register];
    record.wrote_result = false;
    const std::uint64_t source_raw = record.source_raw;
    const double source = std::bit_cast<double>(source_raw);
    const bool is_nan = (source_raw & exponent_mask) == exponent_mask &&
        (source_raw & fraction_mask) != 0U;
    const bool is_signaling_nan = is_nan && (source_raw & quiet_bit) == 0U;
    std::uint64_t result_raw{};
    bool suppress_write = false;

    if (is_signaling_nan)
    {
        native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
        suppress_write = (state->fpscr & fpscr::ve_mask) != 0U;
        result_raw = source_raw | quiet_bit;
    }
    else if (is_nan)
    {
        result_raw = source_raw | quiet_bit;
    }
    else if (source < 0.0)
    {
        native_raise_fpscr_exception(state, fpscr::vxsqrt_mask);
        suppress_write = (state->fpscr & fpscr::ve_mask) != 0U;
        result_raw = 0x7FF8000000000000ULL;
    }
    else if (source == 0.0)
    {
        native_raise_fpscr_exception(state, fpscr::zx_mask);
        suppress_write = (state->fpscr & fpscr::ze_mask) != 0U;
        result_raw = ((source_raw & sign_mask) != 0U ? sign_mask : 0U) | infinity;
    }
    else if (std::isinf(source))
    {
        result_raw = 0U;
    }
    else
    {
        // frsqrte is architecturally an estimate (relative error <= 1/32),
        // matching the interpreter's host estimate within that allowance.
        result_raw = std::bit_cast<std::uint64_t>(1.0 / std::sqrt(source));
    }

    if (!suppress_write)
    {
        record.result_raw = result_raw;
        state->fpr[destination] = result_raw;
        native_set_double_fprf(state, result_raw);
        record.wrote_result = true;
    }
}

extern "C" void native_fres(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_register,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_register >= state->fpr.size() ||
        instruction_index >= context->fres_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFrspRecord& record = context->fres_records[instruction_index];
    record.source_raw = state->fpr[source_register];
    record.wrote_result = false;
    const std::uint64_t source_raw = record.source_raw;
    const double source = std::bit_cast<double>(source_raw);
    const bool is_nan = (source_raw & exponent_mask) == exponent_mask &&
        (source_raw & fraction_mask) != 0U;
    const bool is_signaling_nan = is_nan && (source_raw & quiet_bit) == 0U;
    bool suppress_write = false;
    bool inexact = false;
    bool rounded_up = false;

    if (is_signaling_nan)
    {
        native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
        suppress_write = (state->fpscr & fpscr::ve_mask) != 0U;
    }

    if (!suppress_write)
    {
        if (is_nan)
        {
            const std::uint32_t sign = (source_raw & sign_mask) != 0U
                ? 0x80000000U : 0U;
            const std::uint32_t payload = static_cast<std::uint32_t>(
                (source_raw & fraction_mask) >> 29U);
            record.single_bits = sign | 0x7F800000U | payload | 0x00400000U;
        }
        else if (source == 0.0)
        {
            native_raise_fpscr_exception(state, fpscr::zx_mask);
            suppress_write = (state->fpscr & fpscr::ze_mask) != 0U;
            record.single_bits = ((source_raw & sign_mask) != 0U
                    ? 0x80000000U : 0U) | 0x7F800000U;
        }
        else if (std::isinf(source))
        {
            record.single_bits = (source_raw & sign_mask) != 0U
                ? 0x80000000U : 0U;
        }
        else
        {
            // fres permits implementation-dependent estimates; match the
            // interpreter's correctly-rounded single reciprocal policy.
            const double reciprocal = 1.0 / source;
            const float estimate = static_cast<float>(reciprocal);
            const double extended_estimate = static_cast<double>(estimate);
            record.single_bits = std::bit_cast<std::uint32_t>(estimate);
            inexact = extended_estimate != reciprocal;
            rounded_up = inexact &&
                std::fabs(extended_estimate) > std::fabs(reciprocal);
            // TODO: model fres OX/UX behavior for extreme reciprocal overflow
            // and underflow, matching the interpreter's current scope.
        }
    }

    if (suppress_write)
    {
        return;
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

    const float result = std::bit_cast<float>(record.single_bits);
    record.result_raw = std::bit_cast<std::uint64_t>(static_cast<double>(result));
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

extern "C" void native_fmadds(
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
        instruction_index >= context->fmadds_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFmaddsRecord& record = context->fmadds_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    record.c_raw = state->fpr[source_c];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
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

    if (is_signaling_nan(a_raw) || is_signaling_nan(c_raw) || is_signaling_nan(b_raw))
    {
        const std::uint64_t selected = is_signaling_nan(a_raw) ? a_raw
            : is_signaling_nan(c_raw) ? c_raw : b_raw;
        signal_invalid(fpscr::vxsnan_mask, selected);
        return;
    }
    if (is_nan(a_raw) || is_nan(c_raw) || is_nan(b_raw))
    {
        const std::uint64_t selected = is_nan(a_raw) ? a_raw
            : is_nan(c_raw) ? c_raw : b_raw;
        commit_nan(selected);
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
    const bool product_infinity = a_infinity || c_infinity;
    const bool b_infinity = is_infinity(b_raw);
    const bool product_negative = ((a_raw ^ c_raw) & sign_mask) != 0U;
    const bool b_negative = (b_raw & sign_mask) != 0U;
    if (product_infinity && b_infinity && product_negative != b_negative)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
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

    const double exact_result = std::fma(std::bit_cast<double>(a_raw),
        std::bit_cast<double>(rounded_c_raw), std::bit_cast<double>(b_raw));
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fnmsubs(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index,
    std::uint32_t subtract_addend,
    std::uint32_t negate_output) noexcept
{
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->fnmsubs_records.size() ||
        instruction_index >= context->fmsubs_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFmaddsRecord& record = negate_output != 0U
        ? context->fnmsubs_records[instruction_index]
        : context->fmsubs_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    record.c_raw = state->fpr[source_c];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const std::uint64_t c_raw = record.c_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_zero = [&](std::uint64_t raw) { return (raw & ~sign_mask) == 0U; };
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
        record.result_raw = std::bit_cast<std::uint64_t>(
            static_cast<double>(std::bit_cast<float>(single_bits)));
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

    if (is_signaling_nan(a_raw) || is_signaling_nan(c_raw) || is_signaling_nan(b_raw))
    {
        const std::uint64_t selected = is_signaling_nan(a_raw) ? a_raw
            : is_signaling_nan(c_raw) ? c_raw : b_raw;
        signal_invalid(fpscr::vxsnan_mask, selected);
        return;
    }
    if (is_nan(a_raw) || is_nan(c_raw) || is_nan(b_raw))
    {
        const std::uint64_t selected = is_nan(a_raw) ? a_raw
            : is_nan(c_raw) ? c_raw : b_raw;
        commit_nan(selected);
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
    const bool product_infinity = a_infinity || c_infinity;
    const bool b_infinity = is_infinity(b_raw);
    const bool product_negative = ((a_raw ^ c_raw) & sign_mask) != 0U;
    const bool b_negative = (b_raw & sign_mask) != 0U;
    if (product_infinity && b_infinity &&
        (subtract_addend != 0U
            ? product_negative == b_negative
            : product_negative != b_negative))
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
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
        rounded_c_raw = (c_raw & keep_mask) +
            (c_raw & (0x0000000008000000ULL >> shift));
    }
    else
    {
        rounded_c_raw = (c_raw & 0xFFFFFFFFF8000000ULL) +
            (c_raw & 0x0000000008000000ULL);
    }

    const double addend = std::bit_cast<double>(b_raw);
    const double temporary = std::fma(std::bit_cast<double>(a_raw),
        std::bit_cast<double>(rounded_c_raw),
        subtract_addend != 0U ? -addend : addend);
    std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
    if (negate_output != 0U && !is_nan(temporary_raw))
    {
        temporary_raw ^= sign_mask;
    }
    const double exact_result = std::bit_cast<double>(temporary_raw);
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fadds(
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
        source_b >= state->fpr.size() || instruction_index >= context->fadds_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFaddsRecord& record = context->fadds_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
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
        record.result_raw = std::bit_cast<std::uint64_t>(
            static_cast<double>(std::bit_cast<float>(single_bits)));
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
    if (is_infinity(a_raw) && is_infinity(b_raw) &&
        ((a_raw ^ b_raw) & sign_mask) != 0U)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double exact_result = std::bit_cast<double>(a_raw) +
        std::bit_cast<double>(b_raw);
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fsubs(
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
        source_b >= state->fpr.size() || instruction_index >= context->fsubs_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFaddsRecord& record = context->fsubs_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
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
        record.result_raw = std::bit_cast<std::uint64_t>(
            static_cast<double>(std::bit_cast<float>(single_bits)));
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
    if (is_infinity(a_raw) && is_infinity(b_raw) &&
        ((a_raw ^ b_raw) & sign_mask) == 0U)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double exact_result = std::bit_cast<double>(a_raw) -
        std::bit_cast<double>(b_raw);
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fdivs(
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
        source_b >= state->fpr.size() || instruction_index >= context->fdivs_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFaddsRecord& record = context->fdivs_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
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
        record.result_raw = std::bit_cast<std::uint64_t>(
            static_cast<double>(std::bit_cast<float>(single_bits)));
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

    const bool a_zero = is_zero(a_raw);
    const bool b_zero = is_zero(b_raw);
    const bool a_infinity = is_infinity(a_raw);
    const bool b_infinity = is_infinity(b_raw);
    if ((a_zero && b_zero) || (a_infinity && b_infinity))
    {
        signal_invalid(a_zero ? fpscr::vxzdz_mask : fpscr::vxidi_mask, 0U);
        return;
    }
    if (b_zero && !a_infinity)
    {
        native_raise_fpscr_exception(state, fpscr::zx_mask);
        if ((state->fpscr & fpscr::ze_mask) == 0U)
        {
            const std::uint32_t sign = ((a_raw ^ b_raw) & sign_mask) != 0U
                ? 0x80000000U : 0U;
            commit(sign | 0x7F800000U, false, false);
        }
        return;
    }
    if (b_zero) // Match the interpreter: infinity / zero does not raise ZX.
    {
        const std::uint32_t sign = ((a_raw ^ b_raw) & sign_mask) != 0U
            ? 0x80000000U : 0U;
        commit(sign | 0x7F800000U, false, false);
        return;
    }

    const double exact_result = std::bit_cast<double>(a_raw) /
        std::bit_cast<double>(b_raw);
    const float single_result = static_cast<float>(exact_result);
    const double extended_result = static_cast<double>(single_result);
    const std::uint32_t single_bits = std::bit_cast<std::uint32_t>(single_result);
    const bool inexact = extended_result != exact_result;
    const bool rounded_up = inexact &&
        std::fabs(extended_result) > std::fabs(exact_result);
    commit(single_bits, inexact, rounded_up);
}

extern "C" void native_fdiv(
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
    constexpr std::uint64_t infinity_bits = 0x7FF0000000000000ULL;
    constexpr std::uint64_t quiet_bit = 0x0008000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->fdiv_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFsubRecord& record = context->fdiv_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_zero = [&](std::uint64_t raw) { return (raw & ~sign_mask) == 0U; };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == infinity_bits;
    };
    const auto commit = [&](std::uint64_t raw_result, bool inexact, bool rounded_up) {
        record.result_raw = raw_result;
        state->fpr[destination] = raw_result;
        native_set_double_result_status(state, raw_result, inexact, rounded_up);
        record.wrote_result = true;
    };
    const auto quiet_nan = [&](std::uint64_t source_raw) {
        if (!is_nan(source_raw))
        {
            source_raw = 0x7FF8000000000000ULL;
        }
        return source_raw | quiet_bit;
    };
    const auto signal_invalid = [&](std::uint32_t exception, std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit(quiet_nan(source_raw), false, false);
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
        commit(quiet_nan(is_nan(a_raw) ? a_raw : b_raw), false, false);
        return;
    }

    const bool a_zero = is_zero(a_raw);
    const bool b_zero = is_zero(b_raw);
    const bool a_infinity = is_infinity(a_raw);
    const bool b_infinity = is_infinity(b_raw);
    if (a_zero && b_zero)
    {
        signal_invalid(fpscr::vxzdz_mask, 0U);
        return;
    }
    if (a_infinity && b_infinity)
    {
        signal_invalid(fpscr::vxidi_mask, 0U);
        return;
    }
    if (b_zero)
    {
        native_raise_fpscr_exception(state, fpscr::zx_mask);
        if ((state->fpscr & fpscr::ze_mask) == 0U)
        {
            commit(((a_raw ^ b_raw) & sign_mask) | infinity_bits, false, false);
        }
        return;
    }

    const double a = std::bit_cast<double>(a_raw);
    const double b = std::bit_cast<double>(b_raw);
    const double result = a / b;
    const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
    if (a_infinity || b_infinity)
    {
        commit(raw_result, false, false);
        return;
    }
    if (!native_binary64_normal_or_zero(a_raw) ||
        !native_binary64_normal_or_zero(b_raw) ||
        !native_binary64_normal_or_zero(raw_result) ||
        (is_zero(raw_result) && !a_zero))
    {
        record.result_raw = raw_result;
        state->fpr[destination] = raw_result;
        native_set_double_fprf(state, raw_result);
        record.wrote_result = true;
        return;
    }

    const double remainder = std::fma(-result, b, a);
    const bool inexact = remainder != 0.0;
    const bool error_negative = std::signbit(remainder) != std::signbit(b);
    const bool rounded_up = inexact && result != 0.0 &&
        ((result > 0.0 && error_negative) || (result < 0.0 && !error_negative));
    commit(raw_result, inexact, rounded_up);
}

extern "C" void native_ps_sum(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t sum0,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_sum_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsSumRecord& record = context->ps_sum_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const double a_ps0 = std::bit_cast<double>(record.source_a_ps0);
    const double c_ps0 = std::bit_cast<double>(record.source_c_ps0);
    const double b_ps1 = std::bit_cast<double>(record.source_b_ps1);
    const double c_ps1 = std::bit_cast<double>(record.source_c_ps1);
    const float result_ps0 = static_cast<float>(sum0 != 0U ? a_ps0 + b_ps1 : c_ps0);
    const float result_ps1 = static_cast<float>(sum0 != 0U ? c_ps1 : a_ps0 + b_ps1);
    record.destination_ps0 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps0));
    record.destination_ps1 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps1));
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_select(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_select_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsSelectRecord& record = context->ps_select_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto choose_c = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
        const bool is_nan = (raw & exponent_mask) == exponent_mask &&
            (raw & fraction_mask) != 0U;
        const bool is_zero = (raw & ~sign_mask) == 0U;
        return !is_nan && ((raw & sign_mask) == 0U || is_zero);
    };
    record.destination_ps0 = choose_c(record.source_a_ps0)
        ? record.source_c_ps0 : record.source_b_ps0;
    record.destination_ps1 = choose_c(record.source_a_ps1)
        ? record.source_c_ps1 : record.source_b_ps1;
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_res(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source >= state->fpr.size() ||
        instruction_index >= context->ps_res_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMergeRecord& record = context->ps_res_records[instruction_index];
    record.source_a_ps0 = state->fpr[source];
    record.source_a_ps1 = state->fpr_ps1[source];
    const auto estimate = [](std::uint64_t raw) {
        const double value = std::bit_cast<double>(raw);
        const float reciprocal = static_cast<float>(1.0 / value);
        return std::bit_cast<std::uint64_t>(static_cast<double>(reciprocal));
    };
    record.destination_ps0 = estimate(record.source_a_ps0);
    record.destination_ps1 = estimate(record.source_a_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_rsqrte(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source >= state->fpr.size() ||
        instruction_index >= context->ps_rsqrte_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMergeRecord& record = context->ps_rsqrte_records[instruction_index];
    record.source_a_ps0 = state->fpr[source];
    record.source_a_ps1 = state->fpr_ps1[source];
    const auto estimate = [](std::uint64_t raw) {
        const double value = std::bit_cast<double>(raw);
        const float reciprocal_sqrt = static_cast<float>(1.0 / std::sqrt(value));
        return std::bit_cast<std::uint64_t>(static_cast<double>(reciprocal_sqrt));
    };
    record.destination_ps0 = estimate(record.source_a_ps0);
    record.destination_ps1 = estimate(record.source_a_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_sub(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->ps_sub_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsBinaryRecord& record = context->ps_sub_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    const double a_ps0 = std::bit_cast<double>(record.source_a_ps0);
    const double a_ps1 = std::bit_cast<double>(record.source_a_ps1);
    const double b_ps0 = std::bit_cast<double>(record.source_b_ps0);
    const double b_ps1 = std::bit_cast<double>(record.source_b_ps1);
    const float result_ps0 = static_cast<float>(a_ps0 - b_ps0);
    const float result_ps1 = static_cast<float>(a_ps1 - b_ps1);
    record.destination_ps0 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps0));
    record.destination_ps1 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps1));
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_add(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->ps_add_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsBinaryRecord& record = context->ps_add_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    const double a_ps0 = std::bit_cast<double>(record.source_a_ps0);
    const double a_ps1 = std::bit_cast<double>(record.source_a_ps1);
    const double b_ps0 = std::bit_cast<double>(record.source_b_ps0);
    const double b_ps1 = std::bit_cast<double>(record.source_b_ps1);
    const float result_ps0 = static_cast<float>(a_ps0 + b_ps0);
    const float result_ps1 = static_cast<float>(a_ps1 + b_ps1);
    record.destination_ps0 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps0));
    record.destination_ps1 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps1));
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_div(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->ps_div_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsBinaryRecord& record = context->ps_div_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    const double a_ps0 = std::bit_cast<double>(record.source_a_ps0);
    const double a_ps1 = std::bit_cast<double>(record.source_a_ps1);
    const double b_ps0 = std::bit_cast<double>(record.source_b_ps0);
    const double b_ps1 = std::bit_cast<double>(record.source_b_ps1);
    const float result_ps0 = static_cast<float>(a_ps0 / b_ps0);
    const float result_ps1 = static_cast<float>(a_ps1 / b_ps1);
    record.destination_ps0 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps0));
    record.destination_ps1 = std::bit_cast<std::uint64_t>(
        static_cast<double>(result_ps1));
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_mul(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_c >= state->fpr.size() || instruction_index >= context->ps_mul_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMulRecord& record = context->ps_mul_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto round_operand = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        const std::uint64_t exponent = raw & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_keep_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep_mask =
                static_cast<std::uint64_t>(signed_keep_mask >> shift);
            return (raw & keep_mask) + (raw & (0x0000000008000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x0000000008000000ULL);
    };
    const auto multiply_lane = [&](std::uint64_t a_raw, std::uint64_t c_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const double c = std::bit_cast<double>(round_operand(c_raw));
        float result = static_cast<float>(a * c);
        std::uint32_t bits = std::bit_cast<std::uint32_t>(result);
        if ((bits & 0x7F800000U) == 0U && (bits & 0x007FFFFFU) != 0U)
        {
            bits &= 0x80000000U;
            result = std::bit_cast<float>(bits);
        }
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = multiply_lane(record.source_a_ps0, record.source_c_ps0);
    record.destination_ps1 = multiply_lane(record.source_a_ps1, record.source_c_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_muls(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_c,
    std::uint32_t scalar_ps1,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_c >= state->fpr.size() || instruction_index >= context->ps_muls_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMulRecord& record = context->ps_muls_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto round_operand = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        const std::uint64_t exponent = raw & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_keep_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep_mask =
                static_cast<std::uint64_t>(signed_keep_mask >> shift);
            return (raw & keep_mask) + (raw & (0x0000000008000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x0000000008000000ULL);
    };
    const std::uint64_t scalar_raw = scalar_ps1 != 0U
        ? record.source_c_ps1 : record.source_c_ps0;
    const double scalar = std::bit_cast<double>(round_operand(scalar_raw));
    const auto multiply_lane = [scalar](std::uint64_t a_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const float result = static_cast<float>(a * scalar);
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = multiply_lane(record.source_a_ps0);
    record.destination_ps1 = multiply_lane(record.source_a_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_madd(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_madd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMaddRecord& record = context->ps_madd_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto round_operand = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        const std::uint64_t exponent = raw & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_keep_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep_mask =
                static_cast<std::uint64_t>(signed_keep_mask >> shift);
            return (raw & keep_mask) + (raw & (0x0000000008000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x0000000008000000ULL);
    };
    const auto multiply_add_lane = [&](std::uint64_t a_raw,
                                       std::uint64_t c_raw,
                                       std::uint64_t b_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const double c = std::bit_cast<double>(round_operand(c_raw));
        const double b = std::bit_cast<double>(b_raw);
        const float rounded_product = static_cast<float>(a * c);
        float result = static_cast<float>(static_cast<double>(rounded_product) + b);
        std::uint32_t bits = std::bit_cast<std::uint32_t>(result);
        if ((bits & 0x7F800000U) == 0U && (bits & 0x007FFFFFU) != 0U)
        {
            bits &= 0x80000000U;
            result = std::bit_cast<float>(bits);
        }
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = multiply_add_lane(
        record.source_a_ps0, record.source_c_ps0, record.source_b_ps0);
    record.destination_ps1 = multiply_add_lane(
        record.source_a_ps1, record.source_c_ps1, record.source_b_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_madds(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t scalar_ps1,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_madd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMaddRecord& record = context->ps_madd_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];

    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    const std::uint64_t scalar_raw = scalar_ps1 != 0U
        ? record.source_c_ps1 : record.source_c_ps0;
    const std::uint64_t exponent = scalar_raw & exponent_mask;
    const std::uint64_t fraction = scalar_raw & fraction_mask;
    std::uint64_t rounded_scalar_raw{};
    if (exponent == 0U && fraction != 0U)
    {
        const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
        const std::int64_t signed_keep_mask =
            std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
        const std::uint64_t keep_mask =
            static_cast<std::uint64_t>(signed_keep_mask >> shift);
        rounded_scalar_raw = (scalar_raw & keep_mask) +
            (scalar_raw & (0x0000000008000000ULL >> shift));
    }
    else
    {
        rounded_scalar_raw =
            (scalar_raw & 0xFFFFFFFFF8000000ULL) +
            (scalar_raw & 0x0000000008000000ULL);
    }

    const double scalar = std::bit_cast<double>(rounded_scalar_raw);
    const auto multiply_add_lane = [scalar](std::uint64_t a_raw,
                                             std::uint64_t b_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const float result = static_cast<float>(a * scalar + b);
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = multiply_add_lane(record.source_a_ps0,
        record.source_b_ps0);
    record.destination_ps1 = multiply_add_lane(record.source_a_ps1,
        record.source_b_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_msub(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_madd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMaddRecord& record = context->ps_madd_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto round_operand = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        const std::uint64_t exponent = raw & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_keep_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep_mask =
                static_cast<std::uint64_t>(signed_keep_mask >> shift);
            return (raw & keep_mask) + (raw & (0x0000000008000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) + (raw & 0x0000000008000000ULL);
    };
    const auto multiply_subtract_lane = [&](std::uint64_t a_raw,
                                            std::uint64_t b_raw,
                                            std::uint64_t c_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(round_operand(c_raw));
        const float result = static_cast<float>(a * c - b);
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = multiply_subtract_lane(
        record.source_a_ps0, record.source_b_ps0, record.source_c_ps0);
    record.destination_ps1 = multiply_subtract_lane(
        record.source_a_ps1, record.source_b_ps1, record.source_c_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
}

extern "C" void native_ps_nmadd(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t subtract,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->ps_madd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativePsMaddRecord& record = context->ps_madd_records[instruction_index];
    record.source_a_ps0 = state->fpr[source_a];
    record.source_a_ps1 = state->fpr_ps1[source_a];
    record.source_b_ps0 = state->fpr[source_b];
    record.source_b_ps1 = state->fpr_ps1[source_b];
    record.source_c_ps0 = state->fpr[source_c];
    record.source_c_ps1 = state->fpr_ps1[source_c];
    const auto round_operand = [](std::uint64_t raw) {
        constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
        constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
        const std::uint64_t exponent = raw & exponent_mask;
        const std::uint64_t fraction = raw & fraction_mask;
        if (exponent == 0U && fraction != 0U)
        {
            const unsigned shift = static_cast<unsigned>(std::countl_zero(fraction) - 11);
            const std::int64_t signed_keep_mask =
                std::bit_cast<std::int64_t>(0xFFFFFFFFF8000000ULL);
            const std::uint64_t keep_mask =
                static_cast<std::uint64_t>(signed_keep_mask >> shift);
            return (raw & keep_mask) +
                (raw & (0x0000000008000000ULL >> shift));
        }
        return (raw & 0xFFFFFFFFF8000000ULL) +
            (raw & 0x0000000008000000ULL);
    };
    const auto negative_multiply_add_lane = [&](std::uint64_t a_raw,
                                                std::uint64_t b_raw,
                                                std::uint64_t c_raw) {
        const double a = std::bit_cast<double>(a_raw);
        const double b = std::bit_cast<double>(b_raw);
        const double c = std::bit_cast<double>(round_operand(c_raw));
        const double inner = subtract != 0U ? a * c - b : a * c + b;
        const float result = static_cast<float>(-inner);
        return std::bit_cast<std::uint64_t>(static_cast<double>(result));
    };
    record.destination_ps0 = negative_multiply_add_lane(
        record.source_a_ps0, record.source_b_ps0, record.source_c_ps0);
    record.destination_ps1 = negative_multiply_add_lane(
        record.source_a_ps1, record.source_b_ps1, record.source_c_ps1);
    state->fpr[destination] = record.destination_ps0;
    state->fpr_ps1[destination] = record.destination_ps1;
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

extern "C" void native_fmsub(
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
            // Match the interpreter's scoped binary64 FMA status policy.
            native_set_double_fprf(state, raw_result);
        }
        record.result_raw = raw_result;
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        commit(is_nan(source_raw)
                ? source_raw | quiet_bit
                : 0x7FF8000000000000ULL,
            true);
    };
    const auto signal_invalid = [&](std::uint32_t exception,
                                    std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw) ||
        is_signaling_nan(c_raw))
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
    if (product_is_infinite && b_infinity && product_is_negative == b_is_negative)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double result = std::fma(std::bit_cast<double>(a_raw),
        std::bit_cast<double>(c_raw), -std::bit_cast<double>(b_raw));
    const std::uint64_t raw_result = std::bit_cast<std::uint64_t>(result);
    const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
    commit(raw_result, known_exact);
}

extern "C" void native_fnmadd(
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
            native_set_double_fprf(state, raw_result);
        }
        record.result_raw = raw_result;
        record.wrote_result = true;
    };
    const auto commit_nan = [&](std::uint64_t source_raw) {
        commit(is_nan(source_raw)
                ? source_raw | quiet_bit
                : 0x7FF8000000000000ULL,
            true);
    };
    const auto signal_invalid = [&](std::uint32_t exception,
                                    std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, exception);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit_nan(source_raw);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw) ||
        is_signaling_nan(c_raw))
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

    const double temporary = std::fma(std::bit_cast<double>(a_raw),
        std::bit_cast<double>(c_raw), std::bit_cast<double>(b_raw));
    const std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
    const std::uint64_t raw_result = is_nan(temporary_raw)
        ? temporary_raw
        : temporary_raw ^ sign_mask;
    const bool known_exact = a_zero || c_zero || a_infinity || c_infinity || b_infinity;
    commit(raw_result, known_exact);
}

extern "C" void native_fnmsub(
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
    const auto commit = [&](std::uint64_t raw_result, bool known_exact) {
        state->fpr[destination] = raw_result;
        if (known_exact)
        {
            native_set_double_result_status(state, raw_result, false, false);
        }
        else
        {
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

    if (is_signaling_nan(a_raw) || is_signaling_nan(c_raw) || is_signaling_nan(b_raw))
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
    const bool product_infinity = a_infinity || c_infinity;
    const bool product_negative = ((a_raw ^ c_raw) & sign_mask) != 0U;
    const bool b_negative = (b_raw & sign_mask) != 0U;
    if (product_infinity && b_infinity && product_negative == b_negative)
    {
        signal_invalid(fpscr::vxisi_mask, 0U);
        return;
    }

    const double temporary = std::fma(std::bit_cast<double>(a_raw),
        std::bit_cast<double>(c_raw), -std::bit_cast<double>(b_raw));
    const std::uint64_t temporary_raw = std::bit_cast<std::uint64_t>(temporary);
    const std::uint64_t raw_result = is_nan(temporary_raw)
        ? temporary_raw : temporary_raw ^ sign_mask;
    const bool known_exact = c_zero || a_infinity || c_infinity || b_infinity;
    commit(raw_result, known_exact);
}

extern "C" void native_divwu(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t record_form,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->gpr.size() || source_a >= state->gpr.size() ||
        source_b >= state->gpr.size() || instruction_index >= context->divwu_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeDivwuRecord& record = context->divwu_records[instruction_index];
    record.dividend = state->gpr[source_a];
    record.divisor = state->gpr[source_b];
    record.result = record.divisor == 0U ? 0U : record.dividend / record.divisor;
    state->gpr[destination] = record.result;
    if (record_form != 0U)
    {
        std::uint32_t cr0 = (state->xer >> 31U) & 1U;
        if ((record.result & 0x80000000U) != 0U)
        {
            cr0 |= 0x8U;
        }
        else if (record.result != 0U)
        {
            cr0 |= 0x4U;
        }
        else
        {
            cr0 |= 0x2U;
        }
        state->cr = (state->cr & 0x0FFFFFFFU) | (cr0 << 28U);
    }
}

extern "C" void native_divw(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t record_form,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->gpr.size() || source_a >= state->gpr.size() ||
        source_b >= state->gpr.size() || instruction_index >= context->divw_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeDivwRecord& record = context->divw_records[instruction_index];
    record.dividend = state->gpr[source_a];
    record.divisor = state->gpr[source_b];
    const std::int32_t dividend = std::bit_cast<std::int32_t>(record.dividend);
    const std::int32_t divisor = std::bit_cast<std::int32_t>(record.divisor);
    if (divisor == 0)
    {
        // Match Affogato's deterministic non-OE exceptional-division behavior.
        record.result = dividend < 0 ? 0xFFFFFFFFU : 0U;
    }
    else if (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1)
    {
        record.result = 0xFFFFFFFFU;
    }
    else
    {
        record.result = static_cast<std::uint32_t>(dividend / divisor);
    }
    state->gpr[destination] = record.result;
    if (record_form != 0U)
    {
        std::uint32_t cr0 = (state->xer >> 31U) & 1U;
        if ((record.result & 0x80000000U) != 0U)
        {
            cr0 |= 0x8U;
        }
        else if (record.result != 0U)
        {
            cr0 |= 0x4U;
        }
        else
        {
            cr0 |= 0x2U;
        }
        state->cr = (state->cr & 0x0FFFFFFFU) | (cr0 << 28U);
    }
}

extern "C" void native_fsel(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t destination,
    std::uint32_t source_a,
    std::uint32_t source_b,
    std::uint32_t source_c,
    std::uint32_t instruction_index) noexcept
{
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || source_c >= state->fpr.size() ||
        instruction_index >= context->fsel_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFselRecord& record = context->fsel_records[instruction_index];
    record.source_a_raw = state->fpr[source_a];
    record.source_b_raw = state->fpr[source_b];
    record.source_c_raw = state->fpr[source_c];
    constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
    constexpr std::uint64_t exponent_mask = 0x7FF0000000000000ULL;
    constexpr std::uint64_t fraction_mask = 0x000FFFFFFFFFFFFFULL;
    const std::uint64_t magnitude = record.source_a_raw & ~sign_mask;
    const bool is_nan = (magnitude & exponent_mask) == exponent_mask &&
        (magnitude & fraction_mask) != 0U;
    const bool selects_c = !is_nan &&
        ((record.source_a_raw & sign_mask) == 0U || magnitude == 0U);
    record.result_raw = selects_c ? record.source_c_raw : record.source_b_raw;
    state->fpr[destination] = record.result_raw;
}

extern "C" void native_fadd(
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
    constexpr std::uint64_t infinity_bits = 0x7FF0000000000000ULL;
    if (destination >= state->fpr.size() || source_a >= state->fpr.size() ||
        source_b >= state->fpr.size() || instruction_index >= context->fadd_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeFsubRecord& record = context->fadd_records[instruction_index];
    record.a_raw = state->fpr[source_a];
    record.b_raw = state->fpr[source_b];
    const std::uint64_t a_raw = record.a_raw;
    const std::uint64_t b_raw = record.b_raw;
    const auto is_nan = [&](std::uint64_t raw) {
        return (raw & exponent_mask) == exponent_mask && (raw & fraction_mask) != 0U;
    };
    const auto is_signaling_nan = [&](std::uint64_t raw) {
        return is_nan(raw) && (raw & quiet_bit) == 0U;
    };
    const auto is_infinity = [&](std::uint64_t raw) {
        return (raw & ~sign_mask) == infinity_bits;
    };
    const auto commit = [&](std::uint64_t raw_result, bool inexact, bool rounded_up) {
        record.result_raw = raw_result;
        state->fpr[destination] = raw_result;
        native_set_double_result_status(state, raw_result, inexact, rounded_up);
        record.wrote_result = true;
    };
    const auto quiet_nan = [&](std::uint64_t raw) {
        return is_nan(raw) ? raw | quiet_bit : 0x7FF8000000000000ULL;
    };
    const auto signal_invalid = [&](std::uint64_t source_raw) {
        native_raise_fpscr_exception(state, fpscr::vxsnan_mask);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit(quiet_nan(source_raw), false, false);
        }
    };

    if (is_signaling_nan(a_raw) || is_signaling_nan(b_raw))
    {
        signal_invalid(is_signaling_nan(a_raw) ? a_raw : b_raw);
        return;
    }
    if (is_nan(a_raw) || is_nan(b_raw))
    {
        commit(quiet_nan(is_nan(a_raw) ? a_raw : b_raw), false, false);
        return;
    }
    const bool a_infinity = is_infinity(a_raw);
    const bool b_infinity = is_infinity(b_raw);
    if (a_infinity && b_infinity && ((a_raw ^ b_raw) & sign_mask) != 0U)
    {
        native_raise_fpscr_exception(state, fpscr::vxisi_mask);
        if ((state->fpscr & fpscr::ve_mask) == 0U)
        {
            commit(0x7FF8000000000000ULL, false, false);
        }
        return;
    }

    const double a = std::bit_cast<double>(a_raw);
    const double b = std::bit_cast<double>(b_raw);
    const double result = a + b;
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
        const double virtual_b = result - a;
        const double virtual_a = result - virtual_b;
        const double roundoff_b = b - virtual_b;
        const double roundoff_a = a - virtual_a;
        const double error = roundoff_a + roundoff_b;
        const bool inexact = error != 0.0;
        const bool rounded_up = inexact && result != 0.0 &&
            ((result > 0.0 && error < 0.0) || (result < 0.0 && error > 0.0));
        commit(raw_result, inexact, rounded_up);
    }
    else
    {
        record.result_raw = raw_result;
        state->fpr[destination] = raw_result;
        native_set_double_fprf(state, raw_result);
        record.wrote_result = true;
    }
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

extern "C" void native_data_cache_block_zero(
    NativeExecutionContext* context,
    std::uint32_t address,
    std::uint32_t base_value,
    std::uint32_t index_value,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint32_t block_size = 32U;
    if (instruction_index >= context->memory_records.size())
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = block_size;
        context->failed_flags = 1U;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeMemoryRecord& record = context->memory_records[instruction_index];
    record.address = address;
    record.value = 0U;
    record.base_value = base_value;
    record.index_value = index_value;
    record.indexed = true;

    constexpr std::array<std::uint8_t, block_size> zero_block{};
    try
    {
        ++context->memory_access_calls;
        if (context->memory->has_write_watchpoint())
        {
            context->memory->set_current_guest_cia(
                context->block_cia + instruction_index * 4U);
        }
        context->memory->write_bytes(address, zero_block);
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = block_size;
        context->failed_flags = 1U;
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
    }
}

extern "C" void native_load_word_reserve(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t address,
    std::uint32_t destination,
    std::uint32_t instruction_index) noexcept
{
    const std::uint32_t flags = 4U; // indexed read
    constexpr std::uint32_t width = sizeof(std::uint32_t);
    try
    {
        ++context->memory_access_calls;
        if (destination >= state->gpr.size() ||
            instruction_index >= context->memory_records.size())
        {
            context->failed = true;
            context->failed_address = address;
            context->failed_width = width;
            context->failed_flags = flags;
            context->failed_instruction_index = instruction_index;
            return;
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.indexed = true;
        if ((address & 3U) != 0U)
        {
            throw GuestMemoryFault(address, width, GuestMemoryAccess::read);
        }
        const std::uint32_t value = context->memory->read32_be(address);
        record.value = value;
        state->gpr[destination] = value;
        state->reservation_address = context->memory->reservation_address(address);
        state->reservation_valid = true;
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = width;
        context->failed_flags = flags;
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
    }
}

extern "C" void native_store_word_conditional(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t address,
    std::uint32_t value,
    std::uint32_t instruction_index) noexcept
{
    constexpr std::uint32_t width = sizeof(std::uint32_t);
    constexpr std::uint32_t flags = 5U; // indexed write
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
            return;
        }
        NativeMemoryRecord& record = context->memory_records[instruction_index];
        record.address = address;
        record.value = value;
        record.indexed = true;
        if ((address & 3U) != 0U)
        {
            throw GuestMemoryFault(address, width, GuestMemoryAccess::write);
        }

        const std::uint32_t canonical_address = context->memory->reservation_address(address);
        const bool success = state->reservation_valid &&
            state->reservation_address == canonical_address;
        if (success)
        {
            if (context->memory->has_write_watchpoint())
            {
                context->memory->set_current_guest_cia(
                    context->block_cia + instruction_index * 4U);
            }
            context->memory->write32_be(address, value);
        }

        // Match the interpreter: a completed conditional-store attempt consumes
        // the reservation whether or not its address matched.
        state->reservation_valid = false;
        const std::uint32_t cr0 =
            (success ? 0x2U : 0U) | ((state->xer & 0x80000000U) != 0U ? 1U : 0U);
        state->cr = (state->cr & 0x0FFFFFFFU) | (cr0 << 28U);
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = width;
        context->failed_flags = flags;
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

extern "C" void native_load_string_word_immediate(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t first_register,
    std::uint32_t base_register,
    std::uint32_t byte_count,
    std::uint32_t instruction_index) noexcept
{
    if (first_register >= state->gpr.size() || base_register >= state->gpr.size() ||
        byte_count == 0U || byte_count > 32U ||
        instruction_index >= context->lswi_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeStringLoadRecord& record = context->lswi_records[instruction_index];
    record = {};
    record.address = base_register == 0U ? 0U : state->gpr[base_register];
    record.base_value = record.address;
    record.byte_count = static_cast<std::uint8_t>(byte_count);
    record.register_count = static_cast<std::uint8_t>((byte_count + 3U) / 4U);
    try
    {
        ++context->memory_access_calls;
        context->memory->read_bytes(record.address,
            std::span<std::uint8_t>(record.bytes).first(byte_count));
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = record.address;
        context->failed_width = byte_count;
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
        return;
    }

    for (std::uint32_t i = 0U; i < byte_count; ++i)
    {
        const std::uint32_t register_index = i / 4U;
        const unsigned shift = 24U - (i % 4U) * 8U;
        record.values[register_index] |=
            static_cast<std::uint32_t>(record.bytes[i]) << shift;
    }
    for (std::uint32_t i = 0U; i < record.register_count; ++i)
    {
        const std::uint8_t reg = static_cast<std::uint8_t>(
            (first_register + i) & 31U);
        record.registers[i] = reg;
        state->gpr[reg] = record.values[i];
    }
}

extern "C" void native_load_string_word_indexed(
    NativeExecutionContext* context,
    CpuState* state,
    std::uint32_t first_register,
    std::uint32_t base_register,
    std::uint32_t index_register,
    std::uint32_t instruction_index) noexcept
{
    if (first_register >= state->gpr.size() || base_register >= state->gpr.size() ||
        index_register >= state->gpr.size() ||
        instruction_index >= context->lswi_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeStringLoadRecord& record = context->lswi_records[instruction_index];
    record = {};
    record.indexed = true;
    record.base_value = base_register == 0U ? 0U : state->gpr[base_register];
    record.index_value = state->gpr[index_register];
    record.address = record.base_value + record.index_value;
    const std::uint32_t byte_count = state->xer & 0x7FU;
    record.byte_count = static_cast<std::uint8_t>(byte_count);
    record.register_count = static_cast<std::uint8_t>((byte_count + 3U) / 4U);
    for (std::uint32_t i = 0U; i < record.register_count; ++i)
    {
        record.registers[i] = static_cast<std::uint8_t>(
            (first_register + i) & 31U);
    }
    if (byte_count == 0U)
    {
        return;
    }

    try
    {
        ++context->memory_access_calls;
        context->memory->read_bytes(record.address,
            std::span<std::uint8_t>(record.bytes).first(byte_count));
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = record.address;
        context->failed_width = byte_count;
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
        return;
    }

    for (std::uint32_t i = 0U; i < byte_count; ++i)
    {
        const std::uint32_t register_index = i / 4U;
        const unsigned shift = 24U - (i % 4U) * 8U;
        record.values[register_index] |=
            static_cast<std::uint32_t>(record.bytes[i]) << shift;
    }
    for (std::uint32_t i = 0U; i < record.register_count; ++i)
    {
        state->gpr[record.registers[i]] = record.values[i];
    }
}

extern "C" void native_store_string_word_immediate(
    NativeExecutionContext* context,
    const CpuState* state,
    std::uint32_t first_register,
    std::uint32_t address,
    std::uint32_t byte_count,
    std::uint32_t instruction_index) noexcept
{
    if (first_register >= state->gpr.size() || byte_count == 0U ||
        byte_count > 32U || instruction_index >= context->stswi_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeStringStoreRecord& record = context->stswi_records[instruction_index];
    record = {};
    record.address = address;
    record.base_value = address;
    record.byte_count = static_cast<std::uint8_t>(byte_count);
    record.register_count = static_cast<std::uint8_t>((byte_count + 3U) / 4U);
    for (std::uint32_t i = 0U; i < byte_count; ++i)
    {
        const std::uint32_t register_index = i / 4U;
        if ((i & 3U) == 0U)
        {
            const std::uint8_t reg = static_cast<std::uint8_t>(
                (first_register + register_index) & 31U);
            record.registers[register_index] = reg;
            record.values[register_index] = state->gpr[reg];
        }
        const unsigned shift = 24U - (i % 4U) * 8U;
        record.bytes[i] = static_cast<std::uint8_t>(
            record.values[register_index] >> shift);
    }

    try
    {
        ++context->memory_access_calls;
        context->memory->write_bytes(address,
            std::span<const std::uint8_t>(record.bytes).first(byte_count));
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = address;
        context->failed_width = byte_count;
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
    }
}

extern "C" void native_store_string_word_indexed(
    NativeExecutionContext* context,
    const CpuState* state,
    std::uint32_t first_register,
    std::uint32_t base_register,
    std::uint32_t index_register,
    std::uint32_t instruction_index) noexcept
{
    if (first_register >= state->gpr.size() || base_register >= state->gpr.size() ||
        index_register >= state->gpr.size() ||
        instruction_index >= context->stswi_records.size())
    {
        context->failed = true;
        context->failed_instruction_index = instruction_index;
        return;
    }

    NativeStringStoreRecord& record = context->stswi_records[instruction_index];
    record = {};
    record.indexed = true;
    record.base_value = base_register == 0U ? 0U : state->gpr[base_register];
    record.index_value = state->gpr[index_register];
    record.address = record.base_value + record.index_value;
    const std::uint32_t byte_count = state->xer & 0x7FU;
    record.byte_count = static_cast<std::uint8_t>(byte_count);
    record.register_count = static_cast<std::uint8_t>((byte_count + 3U) / 4U);
    if (byte_count == 0U)
    {
        return;
    }

    for (std::uint32_t i = 0U; i < record.register_count; ++i)
    {
        const std::uint32_t register_index = i;
        const std::uint8_t reg = static_cast<std::uint8_t>(
            (first_register + register_index) & 31U);
        record.registers[register_index] = reg;
        record.values[register_index] = state->gpr[reg];
    }
    for (std::uint32_t i = 0U; i < byte_count; ++i)
    {
        const std::uint32_t register_index = i / 4U;
        const unsigned shift = 24U - (i % 4U) * 8U;
        record.bytes[i] = static_cast<std::uint8_t>(
            record.values[register_index] >> shift);
    }

    try
    {
        ++context->memory_access_calls;
        context->memory->write_bytes(record.address,
            std::span<const std::uint8_t>(record.bytes).first(byte_count));
    }
    catch (...)
    {
        context->failed = true;
        context->failed_address = record.address;
        context->failed_width = byte_count;
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
    }
}

extern "C" std::uint64_t native_load_double(
    NativeExecutionContext* context,
    std::uint32_t address,
    std::uint32_t indexed,
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
        record.indexed = indexed != 0U;
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
    std::uint32_t indexed,
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
        record.indexed = indexed != 0U;
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
    std::uint32_t indexed,
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
        record.indexed = indexed != 0U;
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
    std::uint32_t indexed,
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
        record.indexed = indexed != 0U;
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
    if (insn.record && (insn.opcode == Opcode::add ||
        insn.opcode == Opcode::subtract_from ||
        insn.opcode == Opcode::bitwise_or ||
        insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor ||
        insn.opcode == Opcode::bitwise_nor || insn.opcode == Opcode::bitwise_equivalence ||
        insn.opcode == Opcode::bitwise_nand ||
        insn.opcode == Opcode::bitwise_and_complement ||
        insn.opcode == Opcode::bitwise_or_complement))
    {
        return true;
    }
    if (insn.opcode == Opcode::add_immediate_carry && insn.record)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::negate_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::multiply_low_word_overflow)
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
    if (insn.opcode == Opcode::multiply_high_word_unsigned ||
        insn.opcode == Opcode::multiply_high_word_signed)
    {
        return true;
    }
    if (insn.opcode == Opcode::divide_word_signed)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_carrying)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_carrying)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_carrying_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_carrying_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_extended_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_to_minus_one_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_to_minus_one_extended_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_minus_one_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_to_zero_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::add_to_zero_extended_overflow)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_zero_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::subtract_from_extended)
    {
        return true;
    }
    if (insn.opcode == Opcode::arithmetic_shift_right_word)
    {
        return true;
    }
    if (insn.opcode == Opcode::extend_sign_byte ||
        insn.opcode == Opcode::extend_sign_halfword)
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
    case Opcode::divide_word_unsigned:
    case Opcode::divide_word_signed:
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
    case Opcode::bitwise_and_complement:
    case Opcode::bitwise_or_complement:
    case Opcode::bitwise_xor:
    case Opcode::bitwise_equivalence:
    case Opcode::bitwise_nor:
    case Opcode::bitwise_nand:
    case Opcode::move_condition_register_field:
    case Opcode::move_condition_register_from_fpscr:
    case Opcode::move_from_condition_register:
    case Opcode::move_to_condition_register_fields:
    case Opcode::move_condition_register_from_xer:
    case Opcode::compare_signed_immediate:
    case Opcode::compare_unsigned_immediate:
    case Opcode::compare_signed_register:
    case Opcode::compare_unsigned_register:
    case Opcode::floating_compare_unordered:
    case Opcode::floating_compare_ordered:
    case Opcode::paired_single_compare_unordered0:
    case Opcode::paired_single_compare_unordered1:
    case Opcode::paired_single_compare_ordered0:
    case Opcode::paired_single_compare_ordered1:
    case Opcode::floating_convert_to_integer_word:
    case Opcode::floating_convert_to_integer_word_zero:
    case Opcode::floating_reciprocal_estimate_single:
    case Opcode::floating_reciprocal_square_root_estimate:
    case Opcode::floating_round_to_single:
    case Opcode::floating_add_single:
    case Opcode::floating_subtract_single:
    case Opcode::floating_divide_single:
    case Opcode::floating_multiply_single:
    case Opcode::floating_multiply_subtract_single:
    case Opcode::floating_multiply_add_single:
    case Opcode::floating_negative_multiply_subtract_single:
    case Opcode::floating_negative_multiply_add_single:
    case Opcode::floating_multiply_double:
    case Opcode::floating_multiply_add_double:
    case Opcode::floating_multiply_subtract_double:
    case Opcode::floating_negative_multiply_add_double:
    case Opcode::floating_negative_multiply_subtract_double:
    case Opcode::floating_select:
    case Opcode::floating_subtract_double:
    case Opcode::floating_divide_double:
    case Opcode::floating_add_double:
    case Opcode::paired_single_sum0:
    case Opcode::paired_single_sum1:
    case Opcode::paired_single_select:
    case Opcode::paired_single_add:
    case Opcode::paired_single_subtract:
    case Opcode::paired_single_divide:
    case Opcode::paired_single_multiply:
    case Opcode::paired_single_multiply_scalar0:
    case Opcode::paired_single_multiply_scalar1:
    case Opcode::paired_single_multiply_add:
    case Opcode::paired_single_multiply_subtract:
    case Opcode::paired_single_multiply_add_scalar0:
    case Opcode::paired_single_multiply_add_scalar1:
    case Opcode::paired_single_negative_multiply_add:
    case Opcode::paired_single_negative_multiply_subtract:
    case Opcode::branch:
    case Opcode::conditional_branch:
    case Opcode::conditional_branch_to_link_register:
    case Opcode::conditional_branch_to_count_register:
    case Opcode::move_from_link_register:
    case Opcode::move_to_link_register:
    case Opcode::move_from_count_register:
    case Opcode::move_to_count_register:
    case Opcode::move_from_xer:
    case Opcode::move_from_time_base_lower:
    case Opcode::move_from_time_base_upper:
    case Opcode::move_to_xer:
    case Opcode::move_from_gqr:
    case Opcode::move_to_gqr:
    case Opcode::floating_move_register:
    case Opcode::floating_negate:
    case Opcode::floating_absolute_value:
    case Opcode::floating_negative_absolute_value:
    case Opcode::floating_move_from_fpscr:
    case Opcode::floating_move_to_fpscr_bit_zero:
    case Opcode::floating_move_to_fpscr_bit_one:
    case Opcode::floating_move_to_fpscr_field_immediate:
    case Opcode::floating_move_to_fpscr_fields:
    case Opcode::paired_single_merge00:
    case Opcode::paired_single_merge01:
    case Opcode::paired_single_merge10:
    case Opcode::paired_single_merge11:
    case Opcode::paired_single_move_register:
    case Opcode::paired_single_negate:
    case Opcode::paired_single_absolute_value:
    case Opcode::paired_single_negative_absolute_value:
    case Opcode::paired_single_reciprocal_estimate:
    case Opcode::paired_single_reciprocal_square_root_estimate:
    case Opcode::paired_single_quantized_load:
    case Opcode::paired_single_quantized_store:
    case Opcode::instruction_sync:
    case Opcode::enforce_in_order_io:
    case Opcode::memory_synchronize:
    case Opcode::memory_synchronize_lightweight:
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::arithmetic_shift_right_immediate:
    case Opcode::count_leading_zeros:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::load_word_zero:
    case Opcode::load_multiple_word:
    case Opcode::load_string_word_immediate:
    case Opcode::load_string_word_indexed:
    case Opcode::store_string_word_immediate:
    case Opcode::store_string_word_indexed:
    case Opcode::store_multiple_word:
    case Opcode::load_word_update:
    case Opcode::load_byte_zero:
    case Opcode::load_byte_update:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_zero_update:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_single_update:
    case Opcode::load_single_indexed:
    case Opcode::load_single_indexed_update:
    case Opcode::load_double:
    case Opcode::load_double_update:
    case Opcode::load_double_indexed:
    case Opcode::load_double_indexed_update:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::store_halfword:
    case Opcode::store_halfword_update:
    case Opcode::store_single:
    case Opcode::store_single_update:
    case Opcode::store_single_indexed:
    case Opcode::store_single_indexed_update:
    case Opcode::store_double:
    case Opcode::store_double_update:
    case Opcode::store_double_indexed:
    case Opcode::store_double_indexed_update:
    case Opcode::store_floating_as_integer_word_indexed:
    case Opcode::load_word_indexed:
    case Opcode::load_word_indexed_update:
    case Opcode::load_word_reserve_indexed:
    case Opcode::load_word_byte_reverse_indexed:
    case Opcode::load_halfword_byte_reverse_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_byte_zero_indexed_update:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_zero_indexed_update:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::load_halfword_algebraic_indexed_update:
    case Opcode::store_word_indexed:
    case Opcode::store_word_indexed_update:
    case Opcode::store_word_byte_reverse_indexed:
    case Opcode::store_word_conditional_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_byte_indexed_update:
    case Opcode::store_halfword_indexed:
    case Opcode::store_halfword_indexed_update:
    case Opcode::store_halfword_byte_reverse_indexed:
    case Opcode::instruction_cache_block_invalidate:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
    case Opcode::data_cache_block_touch:
    case Opcode::data_cache_block_touch_for_store:
    case Opcode::data_cache_block_zero:
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
    case Opcode::rotate_left_word_and_mask_register:
        return true;
    case Opcode::and_immediate_record:
        return true;
    case Opcode::and_immediate_shifted_record:
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
    case Opcode::multiply_low_word_overflow: return "mullwo";
    case Opcode::multiply_high_word_unsigned: return "mulhwu";
    case Opcode::multiply_high_word_signed: return "mulhw";
    case Opcode::divide_word_unsigned: return "divwu";
    case Opcode::divide_word_signed: return "divw";
    case Opcode::extend_sign_byte: return "extsb";
    case Opcode::extend_sign_halfword: return "extsh";
    case Opcode::negate: return "neg";
    case Opcode::negate_overflow: return "nego";
    case Opcode::add_immediate_carry: return "addic";
    case Opcode::subtract_from_immediate_carry: return "subfic";
    case Opcode::ori: return "ori";
    case Opcode::or_immediate_shifted: return "oris";
    case Opcode::and_immediate_record: return "andi";
    case Opcode::and_immediate_shifted_record: return "andis";
    case Opcode::xor_immediate: return "xori";
    case Opcode::xor_immediate_shifted: return "xoris";
    case Opcode::add: return "add";
    case Opcode::add_overflow: return "addo";
    case Opcode::subtract_from: return "subf";
    case Opcode::subtract_from_overflow: return "subfo";
    case Opcode::subtract_from_carrying: return "subfc";
    case Opcode::subtract_from_carrying_overflow: return "subfco";
    case Opcode::add_carrying: return "addc";
    case Opcode::add_carrying_overflow: return "addco";
    case Opcode::add_extended: return "adde";
    case Opcode::add_extended_overflow: return "addeo";
    case Opcode::add_to_minus_one_extended: return "addme";
    case Opcode::add_to_minus_one_extended_overflow: return "addmeo";
    case Opcode::subtract_from_minus_one_extended: return "subfme";
    case Opcode::add_to_zero_extended: return "addze";
    case Opcode::add_to_zero_extended_overflow: return "addzeo";
    case Opcode::subtract_from_zero_extended: return "subfze";
    case Opcode::subtract_from_extended: return "subfe";
    case Opcode::rotate_left_word_and_mask_insert: return "rlwimi";
    case Opcode::rotate_left_word_and_mask_register: return "rlwnm";
    case Opcode::paired_single_merge00: return "ps_merge00";
    case Opcode::paired_single_merge01: return "ps_merge01";
    case Opcode::paired_single_merge10: return "ps_merge10";
    case Opcode::paired_single_merge11: return "ps_merge11";
    case Opcode::paired_single_move_register: return "ps_mr";
    case Opcode::paired_single_negate: return "ps_neg";
    case Opcode::paired_single_absolute_value: return "ps_abs";
    case Opcode::paired_single_negative_absolute_value: return "ps_nabs";
    case Opcode::paired_single_reciprocal_estimate: return "ps_res";
    case Opcode::paired_single_reciprocal_square_root_estimate: return "ps_rsqrte";
    case Opcode::paired_single_select: return "ps_sel";
    case Opcode::paired_single_quantized_store: return "psq_st";
    case Opcode::paired_single_quantized_load: return "psq_l";
    case Opcode::instruction_sync: return "isync";
    case Opcode::enforce_in_order_io: return "eieio";
    case Opcode::memory_synchronize: return "sync";
    case Opcode::memory_synchronize_lightweight: return "lwsync";
    case Opcode::bitwise_or: return "or";
    case Opcode::bitwise_and: return "and";
    case Opcode::bitwise_and_complement: return "andc";
    case Opcode::bitwise_or_complement: return "orc";
    case Opcode::bitwise_xor: return "xor";
    case Opcode::bitwise_equivalence: return "eqv";
    case Opcode::bitwise_nor: return "nor";
    case Opcode::bitwise_nand: return "nand";
    case Opcode::compare_signed_immediate: return "cmpwi";
    case Opcode::compare_unsigned_immediate: return "cmplwi";
    case Opcode::compare_signed_register: return "cmpw";
    case Opcode::compare_unsigned_register: return "cmplw";
    case Opcode::floating_compare_unordered: return "fcmpu";
    case Opcode::floating_compare_ordered: return "fcmpo";
    case Opcode::paired_single_compare_unordered0: return "ps_cmpu0";
    case Opcode::paired_single_compare_unordered1: return "ps_cmpu1";
    case Opcode::paired_single_compare_ordered0: return "ps_cmpo0";
    case Opcode::paired_single_compare_ordered1: return "ps_cmpo1";
    case Opcode::move_condition_register_field: return "mcrf";
    case Opcode::move_condition_register_from_fpscr: return "mcrfs";
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
    case Opcode::move_from_time_base_lower: return "mftb";
    case Opcode::move_from_time_base_upper: return "mftbu";
    case Opcode::move_to_xer: return "mtxer";
    case Opcode::move_from_gqr: return "mfgqr";
    case Opcode::move_to_gqr: return "mtgqr";
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
    case Opcode::floating_reciprocal_square_root_estimate: return "frsqrte";
    case Opcode::floating_round_to_single: return "frsp";
    case Opcode::floating_add_single: return "fadds";
    case Opcode::floating_subtract_single: return "fsubs";
    case Opcode::floating_divide_single: return "fdivs";
    case Opcode::floating_multiply_single: return "fmuls";
    case Opcode::floating_multiply_add_single: return "fmadds";
    case Opcode::floating_multiply_subtract_single: return "fmsubs";
    case Opcode::floating_negative_multiply_add_single: return "fnmadds";
    case Opcode::floating_negative_multiply_subtract_single: return "fnmsubs";
    case Opcode::floating_multiply_double: return "fmul";
    case Opcode::floating_multiply_add_double: return "fmadd";
    case Opcode::floating_multiply_subtract_double: return "fmsub";
    case Opcode::floating_negative_multiply_add_double: return "fnmadd";
    case Opcode::floating_negative_multiply_subtract_double: return "fnmsub";
    case Opcode::floating_reciprocal_estimate_single: return "fres";
    case Opcode::floating_select: return "fsel";
    case Opcode::floating_subtract_double: return "fsub";
    case Opcode::floating_divide_double: return "fdiv";
    case Opcode::floating_add_double: return "fadd";
    case Opcode::paired_single_sum0: return "ps_sum0";
    case Opcode::paired_single_sum1: return "ps_sum1";
    case Opcode::paired_single_add: return "ps_add";
    case Opcode::paired_single_subtract: return "ps_sub";
    case Opcode::paired_single_divide: return "ps_div";
    case Opcode::paired_single_multiply: return "ps_mul";
    case Opcode::paired_single_multiply_scalar0: return "ps_muls0";
    case Opcode::paired_single_multiply_scalar1: return "ps_muls1";
    case Opcode::paired_single_multiply_add: return "ps_madd";
    case Opcode::paired_single_multiply_subtract: return "ps_msub";
    case Opcode::paired_single_multiply_add_scalar0: return "ps_madds0";
    case Opcode::paired_single_multiply_add_scalar1: return "ps_madds1";
    case Opcode::paired_single_negative_multiply_add: return "ps_nmadd";
    case Opcode::paired_single_negative_multiply_subtract: return "ps_nmsub";
    case Opcode::shift_left_word: return "slw";
    case Opcode::shift_right_word: return "srw";
    case Opcode::arithmetic_shift_right_word: return "sraw";
    case Opcode::arithmetic_shift_right_immediate: return "srawi";
    case Opcode::count_leading_zeros: return "cntlzw";
    case Opcode::load_word_zero: return "lwz";
    case Opcode::load_word_reserve_indexed: return "lwarx";
    case Opcode::load_word_byte_reverse_indexed: return "lwbrx";
    case Opcode::load_halfword_byte_reverse_indexed: return "lhbrx";
    case Opcode::load_multiple_word: return "lmw";
    case Opcode::load_string_word_immediate: return "lswi";
    case Opcode::load_string_word_indexed: return "lswx";
    case Opcode::store_string_word_immediate: return "stswi";
    case Opcode::store_string_word_indexed: return "stswx";
    case Opcode::store_multiple_word: return "stmw";
    case Opcode::load_word_update: return "lwzu";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_byte_update: return "lbzu";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_zero_update: return "lhzu";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::load_single: return "lfs";
    case Opcode::load_single_update: return "lfsu";
    case Opcode::load_single_indexed: return "lfsx";
    case Opcode::load_single_indexed_update: return "lfsux";
    case Opcode::load_double: return "lfd";
    case Opcode::load_double_update: return "lfdu";
    case Opcode::load_double_indexed: return "lfdx";
    case Opcode::load_double_indexed_update: return "lfdux";
    case Opcode::store_word: return "stw";
    case Opcode::store_word_conditional_indexed: return "stwcx.";
    case Opcode::store_word_update: return "stwu";
    case Opcode::store_byte: return "stb";
    case Opcode::store_byte_update: return "stbu";
    case Opcode::store_halfword: return "sth";
    case Opcode::store_halfword_update: return "sthu";
    case Opcode::store_single: return "stfs";
    case Opcode::store_single_update: return "stfsu";
    case Opcode::store_single_indexed: return "stfsx";
    case Opcode::store_single_indexed_update: return "stfsux";
    case Opcode::store_double: return "stfd";
    case Opcode::store_double_update: return "stfdu";
    case Opcode::store_double_indexed: return "stfdx";
    case Opcode::store_double_indexed_update: return "stfdux";
    case Opcode::store_floating_as_integer_word_indexed: return "stfiwx";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::load_word_indexed_update: return "lwzux";
    case Opcode::load_byte_zero_indexed: return "lbzx";
    case Opcode::load_byte_zero_indexed_update: return "lbzux";
    case Opcode::load_halfword_zero_indexed: return "lhzx";
    case Opcode::load_halfword_zero_indexed_update: return "lhzux";
    case Opcode::load_halfword_algebraic_indexed: return "lhax";
    case Opcode::load_halfword_algebraic_indexed_update: return "lhaux";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_word_indexed_update: return "stwux";
    case Opcode::store_word_byte_reverse_indexed: return "stwbrx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::store_byte_indexed_update: return "stbux";
    case Opcode::store_halfword_indexed: return "sthx";
    case Opcode::store_halfword_indexed_update: return "sthux";
    case Opcode::store_halfword_byte_reverse_indexed: return "sthbrx";
    case Opcode::instruction_cache_block_invalidate: return "icbi";
    case Opcode::data_cache_block_store: return "dcbst";
    case Opcode::data_cache_block_flush: return "dcbf";
    case Opcode::data_cache_block_touch: return "dcbt";
    case Opcode::data_cache_block_touch_for_store: return "dcbtst";
    case Opcode::data_cache_block_zero: return "dcbz";
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
    case Opcode::load_halfword_zero_update:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_single_update:
    case Opcode::load_single_indexed:
    case Opcode::load_single_indexed_update:
    case Opcode::load_double:
    case Opcode::load_double_update:
    case Opcode::load_double_indexed:
    case Opcode::load_double_indexed_update:
    case Opcode::store_single:
    case Opcode::store_single_update:
    case Opcode::store_single_indexed:
    case Opcode::store_single_indexed_update:
    case Opcode::store_double:
    case Opcode::store_double_update:
    case Opcode::store_double_indexed:
    case Opcode::store_double_indexed_update:
    case Opcode::store_floating_as_integer_word_indexed:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_byte_update:
    case Opcode::store_halfword:
    case Opcode::store_halfword_update:
    case Opcode::load_word_indexed:
    case Opcode::load_word_indexed_update:
    case Opcode::load_word_reserve_indexed:
    case Opcode::load_word_byte_reverse_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_byte_zero_indexed_update:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_zero_indexed_update:
    case Opcode::load_halfword_byte_reverse_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::load_halfword_algebraic_indexed_update:
    case Opcode::store_word_indexed:
    case Opcode::store_word_indexed_update:
    case Opcode::store_word_byte_reverse_indexed:
    case Opcode::store_word_conditional_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_byte_indexed_update:
    case Opcode::store_halfword_indexed:
    case Opcode::store_halfword_indexed_update:
    case Opcode::store_halfword_byte_reverse_indexed:
    case Opcode::paired_single_quantized_store:
    case Opcode::paired_single_quantized_load:
    case Opcode::instruction_cache_block_invalidate:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
    case Opcode::data_cache_block_touch:
    case Opcode::data_cache_block_touch_for_store:
    case Opcode::data_cache_block_zero:
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
    case Opcode::load_byte_zero_indexed_update:
    case Opcode::store_byte_indexed:
    case Opcode::store_byte_indexed_update:
    case Opcode::data_cache_block_store:
    case Opcode::data_cache_block_flush:
        return 1U;
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_zero_update:
    case Opcode::load_halfword_algebraic:
    case Opcode::store_halfword:
    case Opcode::store_halfword_update:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_zero_indexed_update:
    case Opcode::load_halfword_byte_reverse_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::load_halfword_algebraic_indexed_update:
    case Opcode::store_halfword_indexed:
    case Opcode::store_halfword_indexed_update:
    case Opcode::store_halfword_byte_reverse_indexed:
        return 2U;
    case Opcode::store_double:
    case Opcode::store_double_update:
    case Opcode::store_double_indexed:
    case Opcode::store_double_indexed_update:
        return 8U;
    case Opcode::data_cache_block_zero:
        return 32U;
    default:
        return 4U;
    }
}

[[nodiscard]] bool is_memory_store(Opcode opcode) noexcept
{
    return opcode == Opcode::store_word || opcode == Opcode::store_byte ||
        opcode == Opcode::store_halfword || opcode == Opcode::store_halfword_update ||
        opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_word_indexed_update ||
        opcode == Opcode::store_word_byte_reverse_indexed ||
        opcode == Opcode::store_word_conditional_indexed ||
        opcode == Opcode::store_byte_update || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_byte_indexed_update ||
        opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::store_halfword_indexed_update ||
        opcode == Opcode::store_halfword_byte_reverse_indexed ||
        opcode == Opcode::store_single || opcode == Opcode::store_single_indexed ||
        opcode == Opcode::store_single_indexed_update ||
        opcode == Opcode::store_double || opcode == Opcode::store_double_update ||
        opcode == Opcode::store_double_indexed ||
        opcode == Opcode::store_double_indexed_update ||
        opcode == Opcode::store_word_update || opcode == Opcode::store_single_update ||
        opcode == Opcode::store_floating_as_integer_word_indexed ||
        opcode == Opcode::data_cache_block_zero;
}

[[nodiscard]] bool is_indexed_memory(Opcode opcode) noexcept
{
    return opcode == Opcode::load_word_indexed ||
        opcode == Opcode::load_word_indexed_update ||
        opcode == Opcode::load_word_reserve_indexed ||
        opcode == Opcode::load_word_byte_reverse_indexed ||
        opcode == Opcode::load_byte_zero_indexed ||
        opcode == Opcode::load_byte_zero_indexed_update ||
        opcode == Opcode::load_single_indexed ||
        opcode == Opcode::load_single_indexed_update ||
        opcode == Opcode::load_double_indexed ||
        opcode == Opcode::load_double_indexed_update ||
        opcode == Opcode::load_halfword_zero_indexed ||
        opcode == Opcode::load_halfword_zero_indexed_update ||
        opcode == Opcode::load_halfword_byte_reverse_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed_update ||
        opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_word_indexed_update ||
        opcode == Opcode::store_word_byte_reverse_indexed ||
        opcode == Opcode::store_word_conditional_indexed ||
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
        opcode == Opcode::data_cache_block_zero;
}

[[nodiscard]] std::uint32_t immediate_value(const DecodedInstruction& insn) noexcept
{
    switch (insn.opcode)
    {
    case Opcode::addis:
        return static_cast<std::uint32_t>(insn.immediate) << 16U;
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate_shifted:
    case Opcode::and_immediate_shifted_record:
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
        insn.opcode == Opcode::and_immediate_shifted_record ||
        insn.opcode == Opcode::ori || insn.opcode == Opcode::or_immediate_shifted ||
        insn.opcode == Opcode::xor_immediate ||
        insn.opcode == Opcode::xor_immediate_shifted)
    {
        entry.has_immediate = true;
        entry.immediate = insn.immediate;
        entry.immediate_hex = insn.opcode == Opcode::ori ||
            insn.opcode == Opcode::or_immediate_shifted ||
            insn.opcode == Opcode::xor_immediate ||
            insn.opcode == Opcode::xor_immediate_shifted ||
            insn.opcode == Opcode::and_immediate_shifted_record;
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
    NativeExecutionContext execution_context{};

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
            case Opcode::arithmetic_shift_right_word:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    sraw_records) + index * sizeof(NativeSrawRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp source = read_guest_gpr(insn.source);
                x86::Gp shift = compiler.new_gp32("sraw_shift");
                compiler.mov(shift, read_guest_gpr(insn.base));
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSrawRecord, source))), source);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSrawRecord, shift))), shift);
                compiler.and_(shift, imm(0x3FU));

                x86::Gp result = compiler.new_gp32("sraw_result");
                compiler.mov(result, source);
                Label large_result_shift = compiler.new_label();
                Label result_shift_done = compiler.new_label();
                compiler.test(shift, imm(0x20U));
                compiler.jnz(large_result_shift);
                compiler.sar(result, shift);
                compiler.jmp(result_shift_done);
                compiler.bind(large_result_shift);
                compiler.sar(result, imm(31U));
                compiler.bind(result_shift_done);

                x86::Gp carry_out = compiler.new_gp32("sraw_carry_out");
                compiler.mov(carry_out, imm(0U));
                Label large_carry_shift = compiler.new_label();
                Label carry_done = compiler.new_label();
                compiler.test(shift, shift);
                compiler.jz(carry_done);
                compiler.test(shift, imm(0x20U));
                compiler.jnz(large_carry_shift);
                x86::Gp dropped = compiler.new_gp32("sraw_dropped_bits");
                compiler.mov(dropped, source);
                x86::Gp left_shift = compiler.new_gp32("sraw_drop_shift");
                compiler.mov(left_shift, imm(32U));
                compiler.sub(left_shift, shift);
                compiler.shl(dropped, left_shift);
                compiler.test(dropped, dropped);
                compiler.setne(carry_out.r8());
                x86::Gp sign = compiler.new_gp32("sraw_sign");
                compiler.mov(sign, source);
                compiler.shr(sign, imm(31U));
                compiler.and_(carry_out, sign);
                compiler.jmp(carry_done);
                compiler.bind(large_carry_shift);
                compiler.mov(carry_out, source);
                compiler.shr(carry_out, imm(31U));
                compiler.bind(carry_done);

                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSrawRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSrawRecord, carry_out))), carry_out);
                x86::Gp xer = compiler.new_gp32("sraw_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("sraw_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("sraw_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("sraw_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("sraw_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("sraw_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("sraw_cr0");
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
            case Opcode::extend_sign_byte:
            case Opcode::extend_sign_halfword:
            {
                const bool halfword = insn.opcode == Opcode::extend_sign_halfword;
                const std::size_t record_offset = halfword
                    ? offsetof(NativeExecutionContext, extsh_records) +
                        index * sizeof(NativeExtshRecord)
                    : offsetof(NativeExecutionContext, extsb_records) +
                        index * sizeof(NativeExtsbRecord);
                x86::Gp source = read_guest_gpr(insn.source);
                x86::Gp result = compiler.new_gp32(
                    halfword ? "extsh_result" : "extsb_result");
                compiler.mov(result, source);
                const std::uint32_t shift = halfword ? 16U : 24U;
                compiler.shl(result, imm(shift));
                compiler.sar(result, imm(shift));
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeExtsbRecord, source))), source);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeExtsbRecord, result))), result);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32(
                        halfword ? "extsh_cr0_lt" : "extsb_cr0_lt");
                    x86::Gp greater = compiler.new_gp32(
                        halfword ? "extsh_cr0_gt" : "extsb_cr0_gt");
                    x86::Gp equal = compiler.new_gp32(
                        halfword ? "extsh_cr0_eq" : "extsb_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32(
                        halfword ? "extsh_cr0_so" : "extsb_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32(
                        halfword ? "extsh_cr0" : "extsb_cr0");
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
            case Opcode::multiply_low_word_overflow:
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
                if (insn.opcode == Opcode::multiply_low_word_overflow)
                {
                    x86::Gp overflow = compiler.new_gp32("mullwo_overflow");
                    compiler.mov(overflow, imm(0U));
                    compiler.seto(overflow.r8());
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMullwRecord, overflow))), overflow);
                    constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                    x86::Gp xer = compiler.new_gp32("mullwo_xer");
                    compiler.mov(xer, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    x86::Gp summary_overflow =
                        compiler.new_gp32("mullwo_summary_overflow");
                    compiler.mov(summary_overflow, xer);
                    compiler.shr(summary_overflow, imm(31U));
                    compiler.or_(summary_overflow, overflow);
                    compiler.shl(summary_overflow, imm(31U));
                    x86::Gp overflow_bit = compiler.new_gp32("mullwo_overflow_bit");
                    compiler.mov(overflow_bit, overflow);
                    compiler.shl(overflow_bit, imm(30U));
                    compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                        xer_overflow_mask)));
                    compiler.or_(xer, summary_overflow);
                    compiler.or_(xer, overflow_bit);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)), xer);
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMullwRecord, summary_overflow))),
                        summary_overflow);
                }
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
            case Opcode::multiply_high_word_unsigned:
            case Opcode::multiply_high_word_signed:
            {
                const std::size_t record_base =
                    offsetof(NativeExecutionContext, mulhwu_records) +
                    index * sizeof(NativeMulhwuRecord);
                const x86::Gp source_a = read_guest_gpr(insn.base);
                const x86::Gp source_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMulhwuRecord, source_a))), source_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMulhwuRecord, source_b))), source_b);
                x86::Gp product = compiler.new_gp64("mulhwu_product");
                x86::Gp rhs = compiler.new_gp64("mulhwu_rhs");
                if (insn.opcode == Opcode::multiply_high_word_signed)
                {
                    compiler.movsxd(product, source_a);
                    compiler.movsxd(rhs, source_b);
                }
                else
                {
                    compiler.mov(product.r32(), source_a);
                    compiler.mov(rhs.r32(), source_b);
                }
                compiler.imul(product, rhs);
                compiler.shr(product, imm(32U));
                x86::Gp result = compiler.new_gp32("mulhwu_result");
                compiler.mov(result, product.r32());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMulhwuRecord, result))), result);
                write_guest_gpr(insn.destination, result);
                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("mulhwu_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("mulhwu_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("mulhwu_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("mulhwu_xer_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("mulhwu_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::divide_word_unsigned:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_divwu));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.destination));
                invoke->set_arg(3, imm(insn.base));
                invoke->set_arg(4, imm(insn.source));
                invoke->set_arg(5, imm(insn.record ? 1U : 0U));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::divide_word_signed:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_divw));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.destination));
                invoke->set_arg(3, imm(insn.base));
                invoke->set_arg(4, imm(insn.source));
                invoke->set_arg(5, imm(insn.record ? 1U : 0U));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::negate_overflow:
            {
                const std::size_t record_offset =
                    offsetof(NativeExecutionContext, negate_overflow_records) +
                    index * sizeof(NativeNegateOverflowRecord);
                x86::Gp source = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeNegateOverflowRecord, source))), source);
                x86::Gp result = compiler.new_gp32("nego_result");
                compiler.mov(result, source);
                compiler.neg(result);
                x86::Gp overflow = compiler.new_gp32("nego_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeNegateOverflowRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeNegateOverflowRecord, overflow))), overflow);

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("nego_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("nego_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("nego_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask | xer_overflow_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeNegateOverflowRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("nego_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("nego_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("nego_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("nego_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("nego_cr0");
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
            case Opcode::and_immediate_record:
            case Opcode::and_immediate_shifted_record:
            case Opcode::xor_immediate:
            case Opcode::xor_immediate_shifted:
            {
                x86::Gp value = read_guest_gpr(insn.source);
                if (insn.destination != insn.source)
                {
                    value = compiler.new_gp32("value");
                    compiler.mov(value, read_guest_gpr(insn.source));
                }
                const bool and_immediate =
                    insn.opcode == Opcode::and_immediate_record ||
                    insn.opcode == Opcode::and_immediate_shifted_record;
                if (and_immediate)
                {
                    compiler.and_(value, imm(immediate_value(insn)));
                }
                else if (insn.opcode == Opcode::ori ||
                    insn.opcode == Opcode::or_immediate_shifted)
                    compiler.or_(value, imm(immediate_value(insn)));
                else
                    compiler.xor_(value, imm(immediate_value(insn)));
                write_guest_gpr(insn.destination, value);
                if (and_immediate)
                {
                    x86::Gp field = compiler.new_gp32("andi_cr0_field");
                    compiler.mov(field, value);
                    compiler.shr(field, imm(28U));
                    compiler.and_(field, imm(8U));

                    x86::Gp equal = compiler.new_gp32("andi_cr0_eq");
                    x86::Gp greater = compiler.new_gp32("andi_cr0_gt");
                    compiler.test(value, value);
                    compiler.setz(equal.r8());
                    compiler.setg(greater.r8());
                    compiler.shl(equal, imm(1U));
                    compiler.or_(field, equal);
                    compiler.shl(greater, imm(2U));
                    compiler.or_(field, greater);

                    x86::Gp summary_overflow = compiler.new_gp32("andi_cr0_so");
                    compiler.mov(summary_overflow, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                    compiler.shr(summary_overflow, imm(31U));
                    compiler.or_(field, summary_overflow);
                    compiler.shl(field, imm(28U));

                    x86::Gp cr = compiler.new_gp32("andi_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, field);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_overflow:
            case Opcode::subtract_from_overflow:
            {
                const bool subtract = insn.opcode == Opcode::subtract_from_overflow;
                const std::size_t record_offset = (subtract
                    ? offsetof(NativeExecutionContext, subf_overflow_records)
                    : offsetof(NativeExecutionContext, add_overflow_records)) +
                    index * sizeof(NativeAddOverflowRecord);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddOverflowRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddOverflowRecord, operand_b))), operand_b);

                x86::Gp result = compiler.new_gp32("addo_result");
                compiler.mov(result, subtract ? operand_b : operand_a);
                if (subtract)
                {
                    compiler.sub(result, operand_a);
                }
                else
                {
                    compiler.add(result, operand_b);
                }
                x86::Gp overflow = compiler.new_gp32("addo_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddOverflowRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddOverflowRecord, overflow))), overflow);

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("addo_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("addo_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("addo_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask | xer_overflow_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddOverflowRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp field = compiler.new_gp32("addo_cr0_field");
                    x86::Gp less = compiler.new_gp32("addo_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addo_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addo_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addo_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    compiler.mov(field, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(field, imm(0x0FFFFFFFU));
                    compiler.or_(field, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), field);
                }
                break;
            }
            case Opcode::add:
            case Opcode::subtract_from:
            case Opcode::bitwise_or:
            case Opcode::bitwise_and:
            case Opcode::bitwise_and_complement:
            case Opcode::bitwise_or_complement:
            case Opcode::bitwise_xor:
            case Opcode::bitwise_equivalence:
            case Opcode::bitwise_nor:
            case Opcode::bitwise_nand:
            {
                x86::Gp lhs = compiler.new_gp32("lhs");
                x86::Gp source_a = read_guest_gpr(insn.base);
                x86::Gp source_b = read_guest_gpr(insn.source);
                if (insn.opcode == Opcode::bitwise_and_complement ||
                    insn.opcode == Opcode::bitwise_or_complement)
                {
                    compiler.mov(lhs, source_b);
                    x86::Gp complemented = compiler.new_gp32("andc_complemented_rhs");
                    compiler.mov(complemented, source_a);
                    compiler.not_(complemented);
                    if (insn.opcode == Opcode::bitwise_and_complement)
                    {
                        compiler.and_(lhs, complemented);
                    }
                    else
                    {
                        compiler.or_(lhs, complemented);
                    }
                }
                else if (insn.opcode == Opcode::add || insn.opcode == Opcode::bitwise_or ||
                    insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor ||
                    insn.opcode == Opcode::bitwise_nor || insn.opcode == Opcode::bitwise_nand ||
                    insn.opcode == Opcode::bitwise_equivalence)
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
                    else if (insn.opcode == Opcode::bitwise_equivalence)
                    {
                        compiler.xor_(lhs, rhs);
                        compiler.not_(lhs);
                    }
                    else if (insn.opcode == Opcode::bitwise_nor)
                    {
                        compiler.or_(lhs, rhs);
                        compiler.not_(lhs);
                    }
                    else
                    {
                        compiler.and_(lhs, rhs);
                        compiler.not_(lhs);
                    }
                }
                if (insn.opcode == Opcode::bitwise_and_complement ||
                    insn.opcode == Opcode::bitwise_or_complement ||
                    insn.opcode == Opcode::add ||
                    insn.opcode == Opcode::bitwise_or ||
                    insn.opcode == Opcode::bitwise_and ||
                    insn.opcode == Opcode::bitwise_xor ||
                    insn.opcode == Opcode::bitwise_nor ||
                    insn.opcode == Opcode::bitwise_nand ||
                    insn.opcode == Opcode::bitwise_equivalence)
                {
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
                    const std::size_t record_offset =
                        offsetof(NativeExecutionContext, subf_records) +
                        index * sizeof(NativeSubfRecord);
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_offset + offsetof(NativeSubfRecord, source_a))), source_a);
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_offset + offsetof(NativeSubfRecord, source_b))), source_b);
                    x86::Gp result = compiler.new_gp32("subf_result");
                    compiler.mov(result, source_b);
                    compiler.sub(result, source_a);
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_offset + offsetof(NativeSubfRecord, result))), result);
                    write_guest_gpr(insn.destination, result);
                    if (insn.record)
                    {
                        compiler.test(result, result);
                        x86::Gp field = compiler.new_gp32("subf_cr0_field");
                        x86::Gp less = compiler.new_gp32("subf_cr0_lt");
                        x86::Gp greater = compiler.new_gp32("subf_cr0_gt");
                        x86::Gp equal = compiler.new_gp32("subf_cr0_eq");
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

                        x86::Gp xer_so = compiler.new_gp32("subf_xer_so");
                        compiler.mov(xer_so, x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, xer))));
                        compiler.shr(xer_so, imm(31U));
                        compiler.or_(less, xer_so);
                        compiler.mov(field, less);
                        compiler.shl(field, imm(28U));
                        x86::Gp cr = compiler.new_gp32("subf_cr0");
                        compiler.mov(cr, x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                        compiler.and_(cr, imm(0x0FFFFFFFU));
                        compiler.or_(cr, field);
                        compiler.mov(x86::dword_ptr(
                            state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
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
            case Opcode::subtract_from_carrying_overflow:
            {
                const std::size_t record_offset =
                    offsetof(NativeExecutionContext, subfco_records) +
                    index * sizeof(NativeSubfcoRecord);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, operand_b))), operand_b);

                x86::Gp result = compiler.new_gp32("subfco_result");
                compiler.mov(result, operand_b);
                compiler.sub(result, operand_a);
                x86::Gp carry_out = compiler.new_gp32("subfco_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setae(carry_out.r8());
                x86::Gp overflow = compiler.new_gp32("subfco_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, carry_out))), carry_out);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, overflow))), overflow);

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("subfco_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("subfco_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("subfco_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                x86::Gp carry_bit = compiler.new_gp32("subfco_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                    xer_overflow_mask | xer_carry_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeSubfcoRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("subfco_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("subfco_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("subfco_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("subfco_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("subfco_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_carrying_overflow:
            {
                const std::size_t record_offset =
                    offsetof(NativeExecutionContext, addco_records) +
                    index * sizeof(NativeAddcoRecord);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, operand_b))), operand_b);

                x86::Gp result = compiler.new_gp32("addco_result");
                compiler.mov(result, operand_a);
                compiler.add(result, operand_b);
                x86::Gp carry_out = compiler.new_gp32("addco_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                x86::Gp overflow = compiler.new_gp32("addco_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, carry_out))), carry_out);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, overflow))), overflow);

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("addco_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("addco_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("addco_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                x86::Gp carry_bit = compiler.new_gp32("addco_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                    xer_overflow_mask | xer_carry_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddcoRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addco_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addco_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addco_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addco_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addco_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_carrying:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext, addc_records) +
                    index * sizeof(NativeAddcRecord);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddcRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddcRecord, operand_b))), operand_b);

                x86::Gp result = compiler.new_gp32("addc_result");
                compiler.mov(result, operand_a);
                compiler.add(result, operand_b);
                x86::Gp carry_out = compiler.new_gp32("addc_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddcRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddcRecord, carry_out))), carry_out);

                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp xer = compiler.new_gp32("addc_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("addc_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp field = compiler.new_gp32("addc_cr0_field");
                    x86::Gp less = compiler.new_gp32("addc_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addc_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addc_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addc_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    compiler.mov(field, less);
                    x86::Gp cr = compiler.new_gp32("addc_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, field);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_extended:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    adde_records) + index * sizeof(NativeAddeRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                x86::Gp carry_in = compiler.new_gp32("adde_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));

                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddeRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddeRecord, operand_b))), operand_b);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddeRecord, carry_in))), carry_in);

                x86::Gp result = compiler.new_gp32("adde_result");
                compiler.mov(result, operand_a);
                compiler.add(result, operand_b);
                x86::Gp carry_ab = compiler.new_gp32("adde_carry_ab");
                compiler.mov(carry_ab, imm(0U));
                compiler.setb(carry_ab.r8());
                compiler.add(result, carry_in);
                x86::Gp carry_cin = compiler.new_gp32("adde_carry_cin");
                compiler.mov(carry_cin, imm(0U));
                compiler.setb(carry_cin.r8());
                compiler.or_(carry_ab, carry_cin);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddeRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddeRecord, carry_out))), carry_ab);

                x86::Gp xer = compiler.new_gp32("adde_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("adde_carry_bit");
                compiler.mov(carry_bit, carry_ab);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("adde_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("adde_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("adde_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("adde_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("adde_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_extended_overflow:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    addeo_records) + index * sizeof(NativeAddeoRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp operand_a = read_guest_gpr(insn.base);
                x86::Gp operand_b = read_guest_gpr(insn.source);
                x86::Gp carry_in = compiler.new_gp32("addeo_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));

                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, operand_a))), operand_a);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, operand_b))), operand_b);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, carry_in))), carry_in);

                // Let x86 ADC compute both the wrapped result and the exact
                // carry/overflow flags for A+B+old_CA in one operation.
                x86::Gp result = compiler.new_gp32("addeo_result");
                compiler.mov(result, operand_a);
                compiler.bt(carry_in, imm(0U));
                compiler.adc(result, operand_b);
                x86::Gp carry_out = compiler.new_gp32("addeo_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                x86::Gp overflow = compiler.new_gp32("addeo_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, carry_out))), carry_out);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, overflow))), overflow);

                x86::Gp xer = compiler.new_gp32("addeo_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("addeo_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("addeo_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                x86::Gp carry_bit = compiler.new_gp32("addeo_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                    xer_overflow_mask | xer_carry_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddeoRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addeo_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addeo_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addeo_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addeo_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addeo_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
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
            case Opcode::subtract_from_minus_one_extended:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    subfme_records) + index * sizeof(NativeAddzeRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp carry_in = compiler.new_gp32("subfme_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));

                x86::Gp operand = read_guest_gpr(insn.base);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_in))), carry_in);

                x86::Gp result = compiler.new_gp32("subfme_result");
                compiler.mov(result, operand);
                compiler.not_(result);
                compiler.sub(result, imm(1U));
                compiler.add(result, carry_in);
                x86::Gp carry_out = compiler.new_gp32("subfme_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.cmp(operand, imm(0xFFFFFFFFU));
                compiler.setne(carry_out.r8());
                compiler.or_(carry_out, carry_in);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_out))), carry_out);

                x86::Gp xer = compiler.new_gp32("subfme_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("subfme_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("subfme_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("subfme_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("subfme_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("subfme_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("subfme_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_to_minus_one_extended:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    addme_records) + index * sizeof(NativeAddzeRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp carry_in = compiler.new_gp32("addme_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));

                x86::Gp operand = read_guest_gpr(insn.base);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_in))), carry_in);

                // addme computes rA + 0xFFFFFFFF + CAin, equivalently
                // rA - 1 + CAin. Its carry-out is CAin || (rA != 0).
                x86::Gp result = compiler.new_gp32("addme_result");
                compiler.mov(result, operand);
                compiler.sub(result, imm(1U));
                compiler.add(result, carry_in);
                x86::Gp carry_out = compiler.new_gp32("addme_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.test(operand, operand);
                compiler.setne(carry_out.r8());
                compiler.or_(carry_out, carry_in);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_out))), carry_out);

                x86::Gp xer = compiler.new_gp32("addme_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("addme_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addme_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addme_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addme_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addme_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addme_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_to_minus_one_extended_overflow:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    addmeo_records) + index * sizeof(NativeAddmeoRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp operand = read_guest_gpr(insn.base);
                x86::Gp carry_in = compiler.new_gp32("addmeo_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, carry_in))), carry_in);

                // addmeo computes rA + 0xFFFFFFFF + old_CA. ADC supplies the
                // exact carry-out and signed-overflow flags for that sum.
                x86::Gp result = compiler.new_gp32("addmeo_result");
                compiler.mov(result, operand);
                compiler.bt(carry_in, imm(0U));
                compiler.adc(result, imm(0xFFFFFFFFU));
                x86::Gp carry_out = compiler.new_gp32("addmeo_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                x86::Gp overflow = compiler.new_gp32("addmeo_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, carry_out))), carry_out);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, overflow))), overflow);

                x86::Gp xer = compiler.new_gp32("addmeo_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("addmeo_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("addmeo_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                x86::Gp carry_bit = compiler.new_gp32("addmeo_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                    xer_overflow_mask | xer_carry_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddmeoRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addmeo_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addmeo_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addmeo_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addmeo_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addmeo_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_to_zero_extended:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext, addze_records) +
                    index * sizeof(NativeAddzeRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp old_carry = compiler.new_gp32("addze_old_ca");
                compiler.mov(old_carry, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(old_carry, imm(29U));
                compiler.and_(old_carry, imm(1U));

                x86::Gp operand = read_guest_gpr(insn.base);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_in))), old_carry);

                x86::Gp result = compiler.new_gp32("addze_result");
                compiler.mov(result, operand);
                compiler.add(result, old_carry);
                x86::Gp carry_out = compiler.new_gp32("addze_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeAddzeRecord, carry_out))), carry_out);

                x86::Gp xer = compiler.new_gp32("addze_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("addze_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addze_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addze_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addze_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addze_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addze_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::add_to_zero_extended_overflow:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    addzeo_records) + index * sizeof(NativeAddzeoRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp operand = read_guest_gpr(insn.base);
                x86::Gp carry_in = compiler.new_gp32("addzeo_carry_in");
                compiler.mov(carry_in, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(carry_in, imm(29U));
                compiler.and_(carry_in, imm(1U));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, carry_in))), carry_in);

                x86::Gp result = compiler.new_gp32("addzeo_result");
                compiler.mov(result, operand);
                compiler.bt(carry_in, imm(0U));
                compiler.adc(result, imm(0U));
                x86::Gp carry_out = compiler.new_gp32("addzeo_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                x86::Gp overflow = compiler.new_gp32("addzeo_overflow");
                compiler.mov(overflow, imm(0U));
                compiler.seto(overflow.r8());
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, result))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, carry_out))), carry_out);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, overflow))), overflow);

                x86::Gp xer = compiler.new_gp32("addzeo_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                x86::Gp summary_overflow = compiler.new_gp32("addzeo_summary_overflow");
                compiler.mov(summary_overflow, xer);
                compiler.shr(summary_overflow, imm(31U));
                compiler.or_(summary_overflow, overflow);
                compiler.shl(summary_overflow, imm(31U));
                x86::Gp overflow_bit = compiler.new_gp32("addzeo_overflow_bit");
                compiler.mov(overflow_bit, overflow);
                compiler.shl(overflow_bit, imm(30U));
                x86::Gp carry_bit = compiler.new_gp32("addzeo_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.and_(xer, imm(~(xer_summary_overflow_mask |
                    xer_overflow_mask | xer_carry_mask)));
                compiler.or_(xer, summary_overflow);
                compiler.or_(xer, overflow_bit);
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeAddzeoRecord, summary_overflow))),
                    summary_overflow);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp less = compiler.new_gp32("addzeo_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("addzeo_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("addzeo_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("addzeo_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    x86::Gp cr = compiler.new_gp32("addzeo_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, less);
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);
                }
                break;
            }
            case Opcode::subtract_from_zero_extended:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    subfze_records) + index * sizeof(NativeSubfzeRecord);
                constexpr std::size_t xer_offset = offsetof(CpuState, xer);
                x86::Gp old_carry = compiler.new_gp32("subfze_old_ca");
                compiler.mov(old_carry, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.shr(old_carry, imm(29U));
                compiler.and_(old_carry, imm(1U));

                x86::Gp operand = read_guest_gpr(insn.base);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSubfzeRecord, operand))), operand);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSubfzeRecord, carry_in))), old_carry);

                // subfze is ~rA + CAin. The addition's unsigned carry is the
                // architectural no-borrow result and replaces only XER.CA.
                x86::Gp result = compiler.new_gp32("subfze_result");
                compiler.mov(result, operand);
                compiler.not_(result);
                compiler.add(result, old_carry);
                x86::Gp carry_out = compiler.new_gp32("subfze_carry_out");
                compiler.mov(carry_out, imm(0U));
                compiler.setb(carry_out.r8());
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSubfzeRecord, result))), result);
                compiler.mov(x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_offset +
                        offsetof(NativeSubfzeRecord, carry_out))), carry_out);

                x86::Gp xer = compiler.new_gp32("subfze_xer");
                compiler.mov(xer, x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)));
                compiler.and_(xer, imm(~xer_carry_mask));
                x86::Gp carry_bit = compiler.new_gp32("subfze_carry_bit");
                compiler.mov(carry_bit, carry_out);
                compiler.shl(carry_bit, imm(29U));
                compiler.or_(xer, carry_bit);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(xer_offset)), xer);
                write_guest_gpr(insn.destination, result);

                if (insn.record)
                {
                    compiler.test(result, result);
                    x86::Gp field = compiler.new_gp32("subfze_cr0_field");
                    x86::Gp less = compiler.new_gp32("subfze_cr0_lt");
                    x86::Gp greater = compiler.new_gp32("subfze_cr0_gt");
                    x86::Gp equal = compiler.new_gp32("subfze_cr0_eq");
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
                    x86::Gp xer_so = compiler.new_gp32("subfze_cr0_so");
                    compiler.mov(xer_so, x86::dword_ptr(
                        state, static_cast<std::int32_t>(xer_offset)));
                    compiler.shr(xer_so, imm(31U));
                    compiler.or_(less, xer_so);
                    compiler.shl(less, imm(28U));
                    compiler.mov(field, less);
                    x86::Gp cr = compiler.new_gp32("subfze_cr0");
                    compiler.mov(cr, x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cr))));
                    compiler.and_(cr, imm(0x0FFFFFFFU));
                    compiler.or_(cr, field);
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
            case Opcode::floating_compare_ordered:
            case Opcode::paired_single_compare_unordered0:
            case Opcode::paired_single_compare_unordered1:
            case Opcode::paired_single_compare_ordered0:
            case Opcode::paired_single_compare_ordered1:
            {
                const std::uint32_t compare_record_offset = static_cast<std::uint32_t>(
                    offsetof(NativeExecutionContext, fp_compare_records) +
                    index * sizeof(NativeFpCompareRecord));
                x86::Gp a_raw = compiler.new_gp64("fcmpu_a_raw");
                x86::Gp b_raw = compiler.new_gp64("fcmpu_b_raw");
                const std::int32_t a_offset = insn.fp_compare_uses_ps1
                    ? static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        insn.fp_compare_a * sizeof(std::uint64_t))
                    : fpr_offset(insn.fp_compare_a);
                const std::int32_t b_offset = insn.fp_compare_uses_ps1
                    ? static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        insn.fp_compare_b * sizeof(std::uint64_t))
                    : fpr_offset(insn.fp_compare_b);
                compiler.mov(a_raw, x86::qword_ptr(
                    state, a_offset));
                compiler.mov(b_raw, x86::qword_ptr(
                    state, b_offset));

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
                auto* invoke_invalid = static_cast<InvokeNode*>(nullptr);
                if (insn.opcode == Opcode::floating_compare_ordered ||
                    insn.opcode == Opcode::paired_single_compare_ordered0 ||
                    insn.opcode == Opcode::paired_single_compare_ordered1)
                {
                    compiler.invoke(Out(invoke_invalid),
                        reinterpret_cast<std::uintptr_t>(
                            &native_raise_fpscr_compare_invalid_if_present),
                        FuncSignature::build<void, CpuState*, std::uint64_t,
                            std::uint64_t, std::uint32_t>(CallConvId::kCDecl));
                    invoke_invalid->set_arg(3, imm(1U));
                }
                else
                {
                    compiler.invoke(Out(invoke_invalid),
                        reinterpret_cast<std::uintptr_t>(
                            &native_raise_fpscr_snan_if_present),
                        FuncSignature::build<void, CpuState*, std::uint64_t,
                            std::uint64_t>(CallConvId::kCDecl));
                }
                invoke_invalid->set_arg(0, state);
                invoke_invalid->set_arg(1, a_raw);
                invoke_invalid->set_arg(2, b_raw);
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
            case Opcode::move_condition_register_from_fpscr:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    mcrfs_records) + index * sizeof(NativeMcrfsRecord);
                const std::uint32_t fpscr_shift =
                    (7U - insn.fpscr_field) * 4U;
                const std::uint32_t valid_field_mask =
                    insn.fpscr_field == 5U ? 0x7U : 0xFU;
                const std::uint32_t cr_shift =
                    (7U - insn.cr_field) * 4U;
                const std::uint32_t cr_mask = 0xFU << cr_shift;
                x86::Gp fpscr_value = compiler.new_gp32("mcrfs_fpscr");
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMcrfsRecord, before))), fpscr_value);

                x86::Gp field_value = compiler.new_gp32("mcrfs_value");
                compiler.mov(field_value, fpscr_value);
                compiler.shr(field_value, imm(fpscr_shift));
                compiler.and_(field_value, imm(valid_field_mask));
                compiler.mov(x86::byte_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMcrfsRecord, value))), field_value.r8());
                x86::Gp cr = compiler.new_gp32("mcrfs_cr");
                compiler.mov(cr, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, cr))));
                compiler.and_(cr, imm(~cr_mask));
                x86::Gp cr_field_value = compiler.new_gp32("mcrfs_cr_field");
                compiler.mov(cr_field_value, field_value);
                compiler.shl(cr_field_value, imm(cr_shift));
                compiler.or_(cr, cr_field_value);
                compiler.mov(x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, cr))), cr);

                std::uint32_t clear_mask = 0U;
                switch (insn.fpscr_field)
                {
                case 0U: clear_mask = fpscr::fx_mask | fpscr::ox_mask; break;
                case 1U: clear_mask = fpscr::ux_mask | fpscr::zx_mask |
                    fpscr::xx_mask | fpscr::vxsnan_mask; break;
                case 2U: clear_mask = fpscr::vxisi_mask | fpscr::vxidi_mask |
                    fpscr::vxzdz_mask | fpscr::vximz_mask; break;
                case 3U: clear_mask = fpscr::vxvc_mask; break;
                case 5U: clear_mask = fpscr::vxsoft_mask | fpscr::vxsqrt_mask |
                    fpscr::vxcvi_mask; break;
                default: break;
                }
                compiler.and_(fpscr_value, imm(~clear_mask));
                compiler.mov(x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))), fpscr_value);
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                compiler.invoke(Out(invoke),
                    reinterpret_cast<std::uintptr_t>(&native_update_fpscr_summaries),
                    FuncSignature::build<void, CpuState*>(CallConvId::kCDecl));
                invoke->set_arg(0, state);
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMcrfsRecord, after))), fpscr_value);
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
            case Opcode::rotate_left_word_and_mask_register:
            case Opcode::rotate_left_word_and_mask_insert:
            {
                const bool insert =
                    insn.opcode == Opcode::rotate_left_word_and_mask_insert;
                const bool register_shift =
                    insn.opcode == Opcode::rotate_left_word_and_mask_register;
                x86::Gp old_destination{};
                x86::Gp value = compiler.new_gp32("rotate_mask_value");
                if (insert)
                {
                    old_destination = read_guest_gpr(insn.destination);
                }
                compiler.mov(value, read_guest_gpr(insn.source));
                if (register_shift)
                {
                    x86::Gp shift = compiler.new_gp32("rotate_mask_shift");
                    compiler.mov(shift, read_guest_gpr(insn.base));
                    compiler.and_(shift, imm(0x1FU));
                    compiler.rol(value, shift);
                }
                else if (insn.shift != 0U)
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
            case Opcode::enforce_in_order_io:
                // Guest RAM accesses are synchronous and there is no separate
                // cache-inhibited I/O queue in the current memory model.
                break;
            case Opcode::memory_synchronize:
            case Opcode::memory_synchronize_lightweight:
                // x86 MFENCE provides the full ordering required by sync and
                // is a safe stronger ordering for lwsync in this host model.
                compiler.mfence();
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
                if (target != cia + 4U || index + 1U == block.instructions.size())
                {
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(offsetof(CpuState, cia))),
                        imm(target));
                }
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
            case Opcode::move_from_time_base_lower:
            case Opcode::move_from_time_base_upper:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_read_time_base));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, imm(insn.opcode == Opcode::move_from_time_base_upper
                    ? 1U : 0U));
                x86::Gp value = compiler.new_gp32("time_base");
                invoke->set_ret(0, value);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, time_base_records) +
                    index * sizeof(NativeTimeBaseRecord) +
                    offsetof(NativeTimeBaseRecord, result))), value);
                write_guest_gpr(insn.destination, value);
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
            case Opcode::floating_negative_absolute_value:
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
                else if (insn.opcode == Opcode::floating_negative_absolute_value)
                {
                    compiler.bts(value, imm(63U));
                }
                compiler.mov(x86::qword_ptr(
                    state, static_cast<std::int32_t>(fpr_offset(insn.fp_register))), value);
                break;
            }
            case Opcode::floating_move_from_fpscr:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    mffs_records) + index * sizeof(NativeMffsRecord);
                x86::Gp fpscr_value = compiler.new_gp32("mffs_fpscr");
                x86::Gp result = compiler.new_gp64("mffs_result");
                x86::Gp fpscr_low = compiler.new_gp64("mffs_fpscr_low");
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(result, imm(0xFFFFFFFF00000000ULL));
                compiler.mov(fpscr_low.r32(), fpscr_value);
                compiler.or_(result, fpscr_low);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))), result);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMffsRecord, fpscr))), fpscr_value);
                compiler.mov(x86::qword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMffsRecord, result))), result);
                break;
            }
            case Opcode::floating_move_to_fpscr_bit_zero:
            case Opcode::floating_move_to_fpscr_bit_one:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    mtfsb_records) + index * sizeof(NativeMtfsbRecord);
                x86::Gp fpscr_value = compiler.new_gp32("mtfsb_fpscr");
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsbRecord, before))), fpscr_value);
                if (insn.fpscr_bit != 1U && insn.fpscr_bit != 2U)
                {
                    const std::uint32_t mask = 1U << (31U - insn.fpscr_bit);
                    if (insn.opcode == Opcode::floating_move_to_fpscr_bit_zero)
                    {
                        compiler.and_(fpscr_value, imm(~mask));
                    }
                    else
                    {
                        compiler.or_(fpscr_value, imm(mask));
                    }
                    compiler.mov(x86::dword_ptr(state,
                        static_cast<std::int32_t>(offsetof(CpuState, fpscr))), fpscr_value);
                    auto* invoke = static_cast<InvokeNode*>(nullptr);
                    compiler.invoke(Out(invoke),
                        reinterpret_cast<std::uintptr_t>(&native_update_fpscr_summaries),
                        FuncSignature::build<void, CpuState*>(CallConvId::kCDecl));
                    invoke->set_arg(0, state);
                }
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsbRecord, after))), fpscr_value);
                break;
            }
            case Opcode::floating_move_to_fpscr_field_immediate:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    mtfsfi_records) + index * sizeof(NativeMtfsfiRecord);
                const std::uint32_t shift = 28U -
                    (static_cast<std::uint32_t>(insn.fpscr_field) * 4U);
                std::uint32_t writable_mask = 0xFU << shift;
                if (insn.fpscr_field == 0U)
                {
                    writable_mask &= ~(fpscr::fex_mask | fpscr::vx_mask);
                }
                const std::uint32_t value =
                    static_cast<std::uint32_t>(insn.fpscr_immediate) << shift;
                x86::Gp fpscr_value = compiler.new_gp32("mtfsfi_fpscr");
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsfiRecord, before))), fpscr_value);
                compiler.and_(fpscr_value, imm(~writable_mask));
                compiler.or_(fpscr_value, imm(value & writable_mask));
                compiler.mov(x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))), fpscr_value);
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                compiler.invoke(Out(invoke),
                    reinterpret_cast<std::uintptr_t>(&native_update_fpscr_summaries),
                    FuncSignature::build<void, CpuState*>(CallConvId::kCDecl));
                invoke->set_arg(0, state);
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsfiRecord, after))), fpscr_value);
                break;
            }
            case Opcode::floating_move_to_fpscr_fields:
            {
                const std::size_t record_offset = offsetof(NativeExecutionContext,
                    mtfsf_records) + index * sizeof(NativeMtfsfRecord);
                std::uint32_t writable_mask = 0U;
                for (std::uint32_t field = 0U; field < 8U; ++field)
                {
                    if ((insn.fpscr_field_mask & (1U << (7U - field))) != 0U)
                    {
                        writable_mask |= 0xFU << (28U - field * 4U);
                    }
                }
                writable_mask &= ~(fpscr::fex_mask | fpscr::vx_mask);
                x86::Gp source_raw = compiler.new_gp64("mtfsf_source_raw");
                x86::Gp source = compiler.new_gp32("mtfsf_source");
                x86::Gp fpscr_value = compiler.new_gp32("mtfsf_fpscr");
                compiler.mov(source_raw, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source, source_raw.r32());
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::qword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsfRecord, source_raw))), source_raw);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsfRecord, before))), fpscr_value);
                compiler.and_(fpscr_value, imm(~writable_mask));
                compiler.and_(source, imm(writable_mask));
                compiler.or_(fpscr_value, source);
                compiler.mov(x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))), fpscr_value);
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                compiler.invoke(Out(invoke),
                    reinterpret_cast<std::uintptr_t>(&native_update_fpscr_summaries),
                    FuncSignature::build<void, CpuState*>(CallConvId::kCDecl));
                invoke->set_arg(0, state);
                compiler.mov(fpscr_value, x86::dword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpscr))));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_offset + offsetof(NativeMtfsfRecord, after))), fpscr_value);
                break;
            }
            case Opcode::floating_convert_to_integer_word:
            case Opcode::floating_convert_to_integer_word_zero:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fctiw));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_b));
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));
                invoke->set_arg(5, imm(insn.opcode ==
                        Opcode::floating_convert_to_integer_word_zero
                    ? 1U
                    : 0U));
                break;
            }
            case Opcode::floating_select:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fsel));
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
            case Opcode::floating_reciprocal_square_root_estimate:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_frsqrte));
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
            case Opcode::floating_reciprocal_estimate_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fres));
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
            case Opcode::floating_add_single:
            case Opcode::floating_subtract_single:
            case Opcode::floating_divide_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(
                        insn.opcode == Opcode::floating_add_single ? &native_fadds :
                        insn.opcode == Opcode::floating_subtract_single ? &native_fsubs :
                            &native_fdivs));
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
            case Opcode::floating_multiply_add_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fmadds));
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
            case Opcode::floating_multiply_subtract_single:
            case Opcode::floating_negative_multiply_subtract_single:
            case Opcode::floating_negative_multiply_add_single:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fnmsubs));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(insn.fp_source_c));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                invoke->set_arg(7, imm(insn.opcode ==
                        Opcode::floating_multiply_subtract_single ||
                        insn.opcode == Opcode::floating_negative_multiply_subtract_single
                    ? 1U
                    : 0U));
                invoke->set_arg(8, imm(insn.opcode ==
                        Opcode::floating_negative_multiply_subtract_single ||
                        insn.opcode == Opcode::floating_negative_multiply_add_single
                    ? 1U
                    : 0U));
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
            case Opcode::floating_multiply_subtract_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fmsub));
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
            case Opcode::floating_negative_multiply_add_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fnmadd));
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
            case Opcode::floating_negative_multiply_subtract_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fnmsub));
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
            case Opcode::floating_add_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fadd));
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
            case Opcode::floating_divide_double:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_fdiv));
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
            case Opcode::paired_single_sum0:
            case Opcode::paired_single_sum1:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_sum));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(insn.fp_source_c));
                invoke->set_arg(6, imm(insn.opcode == Opcode::paired_single_sum0 ? 1U : 0U));
                invoke->set_arg(7, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::paired_single_select:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_select));
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
            case Opcode::paired_single_reciprocal_estimate:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_res));
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
            case Opcode::paired_single_reciprocal_square_root_estimate:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_rsqrte));
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
            case Opcode::paired_single_subtract:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_sub));
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
            case Opcode::paired_single_divide:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                compiler.invoke(Out(invoke),
                    reinterpret_cast<std::uintptr_t>(&native_ps_div),
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
            case Opcode::paired_single_add:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_add));
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
            case Opcode::paired_single_multiply:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_mul));
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
            case Opcode::paired_single_multiply_scalar0:
            case Opcode::paired_single_multiply_scalar1:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_muls));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_c));
                invoke->set_arg(5, imm(insn.opcode ==
                    Opcode::paired_single_multiply_scalar1 ? 1U : 0U));
                invoke->set_arg(6, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::paired_single_multiply_add_scalar0:
            case Opcode::paired_single_multiply_add_scalar1:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_ps_madds));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(insn.fp_source_c));
                invoke->set_arg(6, imm(insn.opcode ==
                    Opcode::paired_single_multiply_add_scalar1 ? 1U : 0U));
                invoke->set_arg(7, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::paired_single_multiply_add:
            case Opcode::paired_single_multiply_subtract:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const std::uintptr_t helper_address =
                    insn.opcode == Opcode::paired_single_multiply_add
                        ? reinterpret_cast<std::uintptr_t>(&native_ps_madd)
                        : reinterpret_cast<std::uintptr_t>(&native_ps_msub);
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
            case Opcode::paired_single_negative_multiply_add:
            case Opcode::paired_single_negative_multiply_subtract:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                compiler.invoke(Out(invoke),
                    reinterpret_cast<std::uintptr_t>(&native_ps_nmadd),
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.fp_register));
                invoke->set_arg(3, imm(insn.fp_source_a));
                invoke->set_arg(4, imm(insn.fp_source_b));
                invoke->set_arg(5, imm(insn.fp_source_c));
                invoke->set_arg(6, imm(insn.opcode ==
                    Opcode::paired_single_negative_multiply_subtract ? 1U : 0U));
                invoke->set_arg(7, imm(static_cast<std::uint32_t>(index)));
                break;
            }
            case Opcode::paired_single_merge00:
            case Opcode::paired_single_merge01:
            case Opcode::paired_single_merge10:
            case Opcode::paired_single_merge11:
            {
                // Snapshot both lanes before either destination lane is
                // written; frD may alias either or both sources.
                x86::Gp source_a_ps0 = compiler.new_gp64("ps_merge_a_ps0");
                x86::Gp source_a_ps1 = compiler.new_gp64("ps_merge_a_ps1");
                x86::Gp source_b_ps0 = compiler.new_gp64("ps_merge_b_ps0");
                x86::Gp source_b_ps1 = compiler.new_gp64("ps_merge_b_ps1");
                const x86::Gp source_a_lane =
                    insn.opcode == Opcode::paired_single_merge10 ||
                    insn.opcode == Opcode::paired_single_merge11
                        ? source_a_ps1 : source_a_ps0;
                const x86::Gp source_b_lane =
                    insn.opcode == Opcode::paired_single_merge01 ||
                    insn.opcode == Opcode::paired_single_merge11
                        ? source_b_ps1 : source_b_ps0;
                x86::Gp record_ptr = compiler.new_gp64("ps_merge_record");
                compiler.mov(source_a_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_a))));
                compiler.mov(source_a_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_a)));
                compiler.mov(source_b_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source_b_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_b)));
                compiler.lea(record_ptr, x86::ptr(context,
                    static_cast<std::int32_t>(offsetof(NativeExecutionContext,
                        ps_merge_records) + index * sizeof(NativePsMergeRecord))));
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps0))),
                    source_a_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps1))),
                    source_a_ps1);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_b_ps0))),
                    source_b_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_b_ps1))),
                    source_b_ps1);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))),
                    source_a_lane);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)),
                    source_b_lane);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps0))),
                    source_a_lane);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps1))),
                    source_b_lane);
                break;
            }
            case Opcode::paired_single_move_register:
            {
                // Snapshot both lanes before writing either one so frD may
                // safely alias the source register.
                x86::Gp source_ps0 = compiler.new_gp64("ps_mr_source_ps0");
                x86::Gp source_ps1 = compiler.new_gp64("ps_mr_source_ps1");
                x86::Gp record_ptr = compiler.new_gp64("ps_mr_record");
                const std::int32_t source_ps1_offset = static_cast<std::int32_t>(
                    offsetof(CpuState, fpr_ps1) +
                    sizeof(std::uint64_t) * insn.fp_source_b);
                const std::int32_t destination_ps1_offset = static_cast<std::int32_t>(
                    offsetof(CpuState, fpr_ps1) +
                    sizeof(std::uint64_t) * insn.fp_register);
                compiler.mov(source_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source_ps1, x86::qword_ptr(state, source_ps1_offset));
                compiler.lea(record_ptr, x86::ptr(context,
                    static_cast<std::int32_t>(offsetof(NativeExecutionContext,
                        ps_move_records) + index * sizeof(NativePsMergeRecord))));
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps1))),
                    source_ps1);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))), source_ps0);
                compiler.mov(x86::qword_ptr(state, destination_ps1_offset), source_ps1);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps1))),
                    source_ps1);
                break;
            }
            case Opcode::paired_single_absolute_value:
            {
                x86::Gp source_ps0 = compiler.new_gp64("ps_abs_source_ps0");
                x86::Gp source_ps1 = compiler.new_gp64("ps_abs_source_ps1");
                x86::Gp record_ptr = compiler.new_gp64("ps_abs_record");
                compiler.mov(source_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_b)));
                compiler.lea(record_ptr, x86::ptr(context,
                    static_cast<std::int32_t>(offsetof(NativeExecutionContext,
                        ps_abs_records) + index * sizeof(NativePsMergeRecord))));
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps1))),
                    source_ps1);
                compiler.btr(source_ps0, imm(63U));
                compiler.btr(source_ps1, imm(63U));
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))), source_ps0);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)), source_ps1);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps1))),
                    source_ps1);
                break;
            }
            case Opcode::paired_single_negative_absolute_value:
            {
                x86::Gp source_ps0 = compiler.new_gp64("ps_nabs_source_ps0");
                x86::Gp source_ps1 = compiler.new_gp64("ps_nabs_source_ps1");
                x86::Gp record_ptr = compiler.new_gp64("ps_nabs_record");
                compiler.mov(source_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_b)));
                compiler.lea(record_ptr, x86::ptr(context,
                    static_cast<std::int32_t>(offsetof(NativeExecutionContext,
                        ps_nabs_records) + index * sizeof(NativePsMergeRecord))));
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps1))),
                    source_ps1);
                compiler.bts(source_ps0, imm(63U));
                compiler.bts(source_ps1, imm(63U));
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))), source_ps0);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)), source_ps1);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps1))),
                    source_ps1);
                break;
            }
            case Opcode::paired_single_negate:
            {
                constexpr std::uint64_t sign_mask = 0x8000000000000000ULL;
                x86::Gp source_ps0 = compiler.new_gp64("ps_neg_source_ps0");
                x86::Gp source_ps1 = compiler.new_gp64("ps_neg_source_ps1");
                x86::Gp sign = compiler.new_gp64("ps_neg_sign");
                x86::Gp record_ptr = compiler.new_gp64("ps_neg_record");
                compiler.mov(source_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(source_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_b)));
                compiler.lea(record_ptr, x86::ptr(context,
                    static_cast<std::int32_t>(offsetof(NativeExecutionContext,
                        ps_neg_records) + index * sizeof(NativePsMergeRecord))));
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, source_a_ps1))),
                    source_ps1);
                compiler.mov(sign, imm(sign_mask));
                compiler.xor_(source_ps0, sign);
                compiler.xor_(source_ps1, sign);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))), source_ps0);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)), source_ps1);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps0))),
                    source_ps0);
                compiler.mov(x86::qword_ptr(record_ptr,
                    static_cast<std::int32_t>(offsetof(NativePsMergeRecord, destination_ps1))),
                    source_ps1);
                break;
            }
            case Opcode::load_string_word_immediate:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(
                        &native_load_string_word_immediate));
                const std::uint32_t byte_count = insn.string_byte_count == 0U
                    ? 32U : insn.string_byte_count;
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // The helper reads and writes a wrapped GPR range directly
                // through CpuState. Synchronize and invalidate cached GPRs
                // around that architectural state boundary.
                flush_guest_gprs();
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.destination));
                invoke->set_arg(3, imm(insn.base));
                invoke->set_arg(4, imm(byte_count));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));

                guest_gpr_initialized.fill(false);
                x86::Gp failed = compiler.new_gp32("lswi_failed");
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
                x86::Gp executed = compiler.new_gp32("lswi_completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
                break;
            }
            case Opcode::load_string_word_indexed:
            {
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(
                        &native_load_string_word_indexed));
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // The helper reads the complete XER-selected byte range before
                // writing its wrapped destination GPRs, so faults are atomic.
                flush_guest_gprs();
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.destination));
                invoke->set_arg(3, imm(insn.base));
                invoke->set_arg(4, imm(insn.source));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));

                guest_gpr_initialized.fill(false);
                x86::Gp failed = compiler.new_gp32("lswx_failed");
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
                x86::Gp executed = compiler.new_gp32("lswx_completed_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                compiler.bind(completed);
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
            case Opcode::store_string_word_immediate:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::uint32_t byte_count = insn.string_byte_count == 0U
                    ? 32U : insn.string_byte_count;
                x86::Gp address = compiler.new_gp32("stswi_address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // The helper snapshots source GPRs from CpuState and performs
                // one validated guest-memory write, so publish cached values.
                flush_guest_gprs();
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(
                        &native_store_string_word_immediate));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*,
                        const CpuState*, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.source));
                invoke->set_arg(3, address);
                invoke->set_arg(4, imm(byte_count));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));

                x86::Gp failed = compiler.new_gp32("stswi_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                compiler.jnz(fault);

                // A string store may modify translated code. Leave the block
                // when its byte range overlaps this translation.
                Label store_is_safe = compiler.new_label();
                Label self_modifying = compiler.new_label();
                compiler.cmp(address, imm(0xFFFFFFFFU - byte_count + 1U));
                compiler.ja(self_modifying);
                const std::uint32_t block_end = block.address +
                    static_cast<std::uint32_t>(block.words.size() * sizeof(std::uint32_t));
                compiler.cmp(address, imm(block_end));
                compiler.jae(store_is_safe);
                x86::Gp store_end = compiler.new_gp32("stswi_end");
                compiler.mov(store_end, address);
                compiler.add(store_end, imm(byte_count));
                compiler.cmp(store_end, imm(block.address));
                compiler.jbe(store_is_safe);
                compiler.jmp(self_modifying);

                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                x86::Gp failed_count = compiler.new_gp32("stswi_fault_count");
                compiler.mov(failed_count, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(failed_count);

                compiler.bind(self_modifying);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia + 4U));
                x86::Gp completed_count = compiler.new_gp32("stswi_completed_count");
                compiler.mov(completed_count, imm(static_cast<std::uint32_t>(index + 1U)));
                compiler.ret(completed_count);
                compiler.bind(store_is_safe);
                break;
            }
            case Opcode::store_string_word_indexed:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                // Publish cached source registers; the helper snapshots the
                // whole string before validating and committing its byte span.
                flush_guest_gprs();
                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(
                        &native_store_string_word_indexed));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*,
                        const CpuState*, std::uint32_t, std::uint32_t,
                        std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, state);
                invoke->set_arg(2, imm(insn.source));
                invoke->set_arg(3, imm(insn.base));
                invoke->set_arg(4, imm(insn.destination));
                invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));

                x86::Gp failed = compiler.new_gp32("stswx_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                compiler.jnz(fault);

                const std::size_t record_base = offsetof(NativeExecutionContext,
                    stswi_records) + index * sizeof(NativeStringStoreRecord);
                x86::Gp address = compiler.new_gp32("stswx_address");
                x86::Gp byte_count = compiler.new_gp32("stswx_byte_count");
                compiler.mov(address, x86::dword_ptr(context,
                    static_cast<std::int32_t>(record_base +
                        offsetof(NativeStringStoreRecord, address))));
                compiler.mov(byte_count, x86::byte_ptr(context,
                    static_cast<std::int32_t>(record_base +
                        offsetof(NativeStringStoreRecord, byte_count))));
                Label store_is_safe = compiler.new_label();
                Label self_modifying = compiler.new_label();
                compiler.test(byte_count, byte_count);
                compiler.jz(store_is_safe);

                x86::Gp max_start = compiler.new_gp32("stswx_max_start");
                compiler.mov(max_start, imm(0xFFFFFFFFU));
                compiler.sub(max_start, byte_count);
                compiler.inc(max_start);
                compiler.cmp(address, max_start);
                compiler.jae(self_modifying);

                const std::uint32_t block_end = block.address +
                    static_cast<std::uint32_t>(block.words.size() * sizeof(std::uint32_t));
                compiler.cmp(address, imm(block_end));
                compiler.jae(store_is_safe);
                x86::Gp store_end = compiler.new_gp32("stswx_end");
                compiler.mov(store_end, address);
                compiler.add(store_end, byte_count);
                compiler.cmp(store_end, imm(block.address));
                compiler.jbe(store_is_safe);
                compiler.jmp(self_modifying);

                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                x86::Gp failed_count = compiler.new_gp32("stswx_fault_count");
                compiler.mov(failed_count, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(failed_count);

                compiler.bind(self_modifying);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia + 4U));
                x86::Gp completed_count = compiler.new_gp32("stswx_completed_count");
                compiler.mov(completed_count, imm(static_cast<std::uint32_t>(index + 1U)));
                compiler.ret(completed_count);
                compiler.bind(store_is_safe);
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
            case Opcode::load_single_update:
            case Opcode::load_single_indexed:
            case Opcode::load_single_indexed_update:
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
                const bool indexed = insn.opcode == Opcode::load_single_indexed ||
                    insn.opcode == Opcode::load_single_indexed_update;
                x86::Gp index_value;
                if (indexed)
                {
                    index_value = compiler.new_gp32("lfs_index");
                    compiler.mov(index_value, read_guest_gpr(insn.source));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, index_value))), index_value);
                    compiler.add(address, index_value);
                }
                else
                {
                    compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_load_single));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint64_t, NativeExecutionContext*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, address);
                invoke->set_arg(2, imm(indexed ? 1U : 0U));
                invoke->set_arg(3, imm(static_cast<std::uint32_t>(index)));
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
                if (insn.opcode == Opcode::load_single_update ||
                    insn.opcode == Opcode::load_single_indexed_update)
                {
                    write_guest_gpr(insn.base, address);
                }
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
            case Opcode::load_double_update:
            case Opcode::load_double_indexed:
            case Opcode::load_double_indexed_update:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                x86::Gp address = compiler.new_gp32("lfd_address");
                const bool indexed = insn.opcode == Opcode::load_double_indexed ||
                    insn.opcode == Opcode::load_double_indexed_update;
                if (insn.opcode == Opcode::load_double_update ||
                    insn.opcode == Opcode::load_double_indexed_update)
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                else if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0U));
                }
                else
                {
                    compiler.mov(address, read_guest_gpr(insn.base));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                if (indexed)
                {
                    x86::Gp index_value = compiler.new_gp32("lfd_index");
                    compiler.mov(index_value, read_guest_gpr(insn.source));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, index_value))), index_value);
                    compiler.add(address, index_value);
                }
                else
                {
                    compiler.add(address, imm(static_cast<std::uint32_t>(insn.immediate)));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_load_double));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<std::uint64_t, NativeExecutionContext*,
                        std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, address);
                invoke->set_arg(2, imm(indexed ? 1U : 0U));
                invoke->set_arg(3, imm(static_cast<std::uint32_t>(index)));
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
                if (insn.opcode == Opcode::load_double_update ||
                    insn.opcode == Opcode::load_double_indexed_update)
                {
                    write_guest_gpr(insn.base, address);
                }
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
            case Opcode::instruction_cache_block_invalidate:
            case Opcode::data_cache_block_touch:
            case Opcode::data_cache_block_touch_for_store:
            {
                const std::size_t record_base = offsetof(NativeExecutionContext,
                    memory_records) + index * sizeof(NativeMemoryRecord);
                x86::Gp base_value = compiler.new_gp32("cache_hint_base");
                if (insn.base == 0U)
                {
                    compiler.mov(base_value, imm(0U));
                }
                else
                {
                    compiler.mov(base_value, read_guest_gpr(insn.base));
                }
                x86::Gp index_value = compiler.new_gp32("cache_hint_index");
                compiler.mov(index_value, read_guest_gpr(insn.source));
                x86::Gp address = compiler.new_gp32("cache_hint_address");
                compiler.mov(address, base_value);
                compiler.add(address, index_value);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, address))), address);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, base_value))), base_value);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, index_value))), index_value);
                compiler.mov(x86::byte_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, indexed))), imm(1U));
                break;
            }
            case Opcode::data_cache_block_zero:
            {
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);
                const std::size_t record_base = offsetof(NativeExecutionContext,
                    memory_records) + index * sizeof(NativeMemoryRecord);
                x86::Gp base_value = compiler.new_gp32("dcbz_base");
                if (insn.base == 0U)
                {
                    compiler.mov(base_value, imm(0U));
                }
                else
                {
                    compiler.mov(base_value, read_guest_gpr(insn.base));
                }
                x86::Gp index_value = compiler.new_gp32("dcbz_index");
                compiler.mov(index_value, read_guest_gpr(insn.source));
                x86::Gp address = compiler.new_gp32("dcbz_address");
                compiler.mov(address, base_value);
                compiler.add(address, index_value);
                compiler.and_(address, imm(0xFFFFFFE0U));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, address))), address);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, base_value))), base_value);
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, index_value))), index_value);
                compiler.mov(x86::byte_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, indexed))), imm(1U));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const auto helper_address = static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(&native_data_cache_block_zero));
                compiler.invoke(Out(invoke), helper_address,
                    FuncSignature::build<void, NativeExecutionContext*,
                        std::uint32_t, std::uint32_t, std::uint32_t,
                        std::uint32_t>(CallConvId::kCDecl));
                invoke->set_arg(0, context);
                invoke->set_arg(1, address);
                invoke->set_arg(2, base_value);
                invoke->set_arg(3, index_value);
                invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));

                x86::Gp failed = compiler.new_gp32("dcbz_failed");
                compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, failed))));
                compiler.test(failed, failed);
                Label fault = compiler.new_label();
                compiler.jnz(fault);

                // dcbz writes a full cache line. Leave the current native block
                // after every successful operation so self-modifying code is
                // revalidated before any already-decoded instruction can run.
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia + 4U));
                flush_guest_gprs();
                x86::Gp completed = compiler.new_gp32("dcbz_completed_count");
                compiler.mov(completed, imm(static_cast<std::uint32_t>(index + 1U)));
                compiler.ret(completed);

                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
                flush_guest_gprs();
                x86::Gp executed = compiler.new_gp32("dcbz_fault_count");
                compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                compiler.ret(executed);
                break;
            }
            case Opcode::load_word_zero:
            case Opcode::load_word_update:
            case Opcode::load_byte_zero:
            case Opcode::load_byte_update:
            case Opcode::load_halfword_zero:
            case Opcode::load_halfword_algebraic:
    case Opcode::load_halfword_zero_update:
            case Opcode::store_word:
            case Opcode::store_word_update:
            case Opcode::store_byte:
            case Opcode::store_byte_update:
            case Opcode::store_halfword:
            case Opcode::store_halfword_update:
            case Opcode::store_single:
            case Opcode::store_single_update:
            case Opcode::store_single_indexed:
            case Opcode::store_single_indexed_update:
            case Opcode::store_double:
            case Opcode::store_double_update:
            case Opcode::store_double_indexed:
            case Opcode::store_double_indexed_update:
            case Opcode::store_floating_as_integer_word_indexed:
            case Opcode::load_word_indexed:
            case Opcode::load_word_indexed_update:
            case Opcode::load_word_reserve_indexed:
            case Opcode::load_word_byte_reverse_indexed:
            case Opcode::load_byte_zero_indexed:
            case Opcode::load_byte_zero_indexed_update:
            case Opcode::load_halfword_zero_indexed:
            case Opcode::load_halfword_zero_indexed_update:
            case Opcode::load_halfword_byte_reverse_indexed:
            case Opcode::load_halfword_algebraic_indexed:
            case Opcode::load_halfword_algebraic_indexed_update:
            case Opcode::store_word_indexed:
            case Opcode::store_word_indexed_update:
            case Opcode::store_word_byte_reverse_indexed:
            case Opcode::store_word_conditional_indexed:
            case Opcode::store_byte_indexed:
            case Opcode::store_byte_indexed_update:
            case Opcode::store_halfword_indexed:
            case Opcode::store_halfword_indexed_update:
            case Opcode::store_halfword_byte_reverse_indexed:
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
                    insn.opcode == Opcode::store_single_update ||
                    insn.opcode == Opcode::store_single_indexed ||
                    insn.opcode == Opcode::store_single_indexed_update;
                const bool double_store = insn.opcode == Opcode::store_double ||
                    insn.opcode == Opcode::store_double_update ||
                    insn.opcode == Opcode::store_double_indexed ||
                    insn.opcode == Opcode::store_double_indexed_update;
                const bool fp_integer_store = insn.opcode ==
                    Opcode::store_floating_as_integer_word_indexed;
                const bool update_base = insn.opcode == Opcode::load_byte_update ||
                    insn.opcode == Opcode::load_word_update ||
                    insn.opcode == Opcode::load_word_indexed_update ||
                    insn.opcode == Opcode::load_byte_zero_indexed_update ||
                    insn.opcode == Opcode::load_halfword_zero_update ||
                    insn.opcode == Opcode::load_halfword_zero_indexed_update ||
                    insn.opcode == Opcode::load_halfword_algebraic_indexed_update ||
                    insn.opcode == Opcode::store_byte_update ||
                    insn.opcode == Opcode::store_word_update ||
                    insn.opcode == Opcode::store_word_indexed_update ||
                    insn.opcode == Opcode::store_halfword_update ||
                    insn.opcode == Opcode::store_byte_indexed_update ||
                    insn.opcode == Opcode::store_halfword_indexed_update ||
                    insn.opcode == Opcode::store_single_update ||
                    insn.opcode == Opcode::store_single_indexed_update ||
                    insn.opcode == Opcode::store_double_update ||
                    insn.opcode == Opcode::store_double_indexed_update;
                const bool sign_extend =
                    insn.opcode == Opcode::load_halfword_algebraic ||
                    insn.opcode == Opcode::load_halfword_algebraic_indexed ||
                    insn.opcode == Opcode::load_halfword_algebraic_indexed_update;
                const std::uint32_t width = memory_width(insn.opcode);
                const bool direct_flat_load = !store && !validation_only &&
                    !single_store && !double_store && !fp_integer_store &&
                    !sign_extend && (width == 1U || width == 2U || width == 4U);
                const std::size_t record_base = offsetof(NativeExecutionContext, memory_records) +
                    index * sizeof(NativeMemoryRecord);
                const std::uint32_t cia = block.address +
                    static_cast<std::uint32_t>(index * 4U);

                x86::Gp address = compiler.new_gp32("address");
                if (insn.base == 0U)
                {
                    compiler.mov(address, imm(0));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), imm(0U));
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
                    else if (fp_integer_store)
                    {
                        compiler.mov(stored_value, x86::dword_ptr(state,
                            static_cast<std::int32_t>(fpr_offset(insn.fp_register))));
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
                if (insn.opcode == Opcode::store_word_byte_reverse_indexed)
                {
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, source_value))),
                        stored_value);
                    compiler.bswap(stored_value);
                }
                else if (insn.opcode == Opcode::store_halfword_byte_reverse_indexed)
                {
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, source_value))),
                        stored_value);
                    compiler.rol(stored_value.r16(), imm(8U));
                }
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    offsetof(NativeExecutionContext, fault_cia))), imm(cia));
                compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, address))), address);
                compiler.mov(x86::byte_ptr(context, static_cast<std::int32_t>(
                    record_base + offsetof(NativeMemoryRecord, indexed))),
                    imm(indexed ? 1U : 0U));

                if (insn.opcode == Opcode::load_word_reserve_indexed)
                {
                    // Flush before the helper updates the architectural RT. The
                    // reservation is established only after its aligned read succeeds.
                    flush_guest_gprs();
                    auto* invoke = static_cast<InvokeNode*>(nullptr);
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_load_word_reserve));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                            std::uint32_t, std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, imm(insn.destination));
                    invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));

                    x86::Gp failed = compiler.new_gp32("lwarx_failed");
                    compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                        offsetof(NativeExecutionContext, failed))));
                    compiler.test(failed, failed);
                    Label fault = compiler.new_label();
                    Label completed = compiler.new_label();
                    compiler.jnz(fault);
                    if (guest_gpr_initialized[insn.destination])
                    {
                        compiler.mov(guest_gprs[insn.destination], x86::dword_ptr(state,
                            static_cast<std::int32_t>(gpr_offset(insn.destination))));
                    }
                    compiler.jmp(completed);
                    compiler.bind(fault);
                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                        offsetof(CpuState, cia))), imm(cia));
                    x86::Gp executed = compiler.new_gp32("lwarx_failed_count");
                    compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                    compiler.ret(executed);
                    compiler.bind(completed);
                    break;
                }
                if (insn.opcode == Opcode::store_word_conditional_indexed)
                {
                    // The block builder ends a block at stwcx., so any write to
                    // translated code is revalidated before another native block runs.
                    flush_guest_gprs();
                    auto* invoke = static_cast<InvokeNode*>(nullptr);
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_store_word_conditional));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<void, NativeExecutionContext*, CpuState*,
                            std::uint32_t, std::uint32_t,
                            std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, stored_value);
                    invoke->set_arg(4, imm(static_cast<std::uint32_t>(index)));

                    x86::Gp failed = compiler.new_gp32("stwcx_failed");
                    compiler.mov(failed, x86::byte_ptr(context, static_cast<std::int32_t>(
                        offsetof(NativeExecutionContext, failed))));
                    compiler.test(failed, failed);
                    Label fault = compiler.new_label();
                    Label completed = compiler.new_label();
                    compiler.jnz(fault);
                    compiler.jmp(completed);
                    compiler.bind(fault);
                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                        offsetof(CpuState, cia))), imm(cia));
                    x86::Gp executed = compiler.new_gp32("stwcx_failed_count");
                    compiler.mov(executed, imm(static_cast<std::uint32_t>(index)));
                    compiler.ret(executed);
                    compiler.bind(completed);
                    break;
                }

                auto* invoke = static_cast<InvokeNode*>(nullptr);
                const std::uint32_t flags = (store ? 1U : 0U) |
                    (sign_extend ? 2U : 0U) | (indexed ? 4U : 0U);
                x86::Gp value = compiler.new_gp32("loaded");
                Label direct_load_slow = compiler.new_label();
                Label direct_load_done = compiler.new_label();
                if (direct_flat_load)
                {
                    x86::Gp access_end = compiler.new_gp64("flat_load_end");
                    compiler.mov(access_end.r32(), address);
                    compiler.add(access_end, imm(width));
                    x86::Gp flat_size = compiler.new_gp64("flat_memory_size");
                    compiler.mov(flat_size, x86::qword_ptr(context,
                        static_cast<std::int32_t>(offsetof(
                            NativeExecutionContext, flat_memory_size))));
                    compiler.cmp(access_end, flat_size);
                    compiler.ja(direct_load_slow);

                    x86::Gp host_address = compiler.new_gp_ptr("flat_memory_address");
                    compiler.mov(host_address, x86::qword_ptr(context,
                        static_cast<std::int32_t>(offsetof(
                            NativeExecutionContext, flat_memory_data))));
                    x86::Gp guest_offset = compiler.new_gp64("flat_guest_offset");
                    compiler.mov(guest_offset.r32(), address);
                    compiler.add(host_address, guest_offset);
                    if (width == 1U)
                    {
                        compiler.movzx(value, x86::byte_ptr(host_address));
                    }
                    else if (width == 2U)
                    {
                        compiler.movzx(value, x86::word_ptr(host_address));
                        compiler.rol(value.r16(), imm(8));
                    }
                    else
                    {
                        compiler.mov(value, x86::dword_ptr(host_address));
                        compiler.bswap(value);
                    }
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, value))), value);
                    compiler.jmp(direct_load_done);
                    compiler.bind(direct_load_slow);
                }
                if (single_store)
                {
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_store_single));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                            CpuState*, std::uint32_t, std::uint32_t,
                            std::uint32_t, std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, imm(insn.fp_register));
                    invoke->set_arg(4, imm(indexed ? 1U : 0U));
                    invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
                }
                else if (double_store)
                {
                    const auto helper_address = static_cast<std::uint64_t>(
                        reinterpret_cast<std::uintptr_t>(&native_store_double));
                    compiler.invoke(Out(invoke), helper_address,
                        FuncSignature::build<std::uint32_t, NativeExecutionContext*,
                            CpuState*, std::uint32_t, std::uint32_t, std::uint32_t,
                            std::uint32_t>(CallConvId::kCDecl));
                    invoke->set_arg(0, context);
                    invoke->set_arg(1, state);
                    invoke->set_arg(2, address);
                    invoke->set_arg(3, imm(insn.fp_register));
                    invoke->set_arg(4, imm(indexed ? 1U : 0U));
                    invoke->set_arg(5, imm(static_cast<std::uint32_t>(index)));
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
                invoke->set_ret(0, value);
                if (direct_flat_load)
                {
                    compiler.bind(direct_load_done);
                }
                if (insn.opcode == Opcode::load_word_byte_reverse_indexed)
                {
                    compiler.bswap(value);
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, value))), value);
                }
                else if (insn.opcode == Opcode::load_halfword_byte_reverse_indexed)
                {
                    compiler.rol(value.r16(), imm(8U));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, value))), value);
                }

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
                            decoded.opcode == Opcode::conditional_branch_to_link_register ||
                            decoded.opcode == Opcode::conditional_branch_to_count_register ||
                            decoded.opcode == Opcode::store_word_conditional_indexed)
                        {
                            break;
                        }
                        if (decoded.opcode == Opcode::branch ||
                            decoded.opcode == Opcode::conditional_branch)
                        {
                            const std::uint32_t branch_target = decoded.absolute
                                ? static_cast<std::uint32_t>(decoded.immediate)
                                : cia + static_cast<std::uint32_t>(decoded.immediate);
                            if (branch_target != cia + 4U)
                            {
                                break;
                            }
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

        NativeExecutionContext& context = impl_->execution_context;
        context.begin_block(core.memory, block->address, block->instructions);
        context.core = &core;
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
            if (context.failed && i == context.failed_instruction_index &&
                (insn.opcode == Opcode::load_string_word_immediate ||
                    insn.opcode == Opcode::load_string_word_indexed ||
                    insn.opcode == Opcode::store_string_word_immediate ||
                    insn.opcode == Opcode::store_string_word_indexed))
            {
                continue;
            }
            if (insn.opcode == Opcode::load_string_word_immediate ||
                insn.opcode == Opcode::load_string_word_indexed)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeStringLoadRecord& record = context.lswi_records[i];
                entry.has_string_transfer = true;
                entry.string_transfer_first_register = insn.destination;
                entry.string_transfer_base_register = insn.base;
                entry.string_transfer_indexed = record.indexed;
                entry.string_transfer_base_value = record.base_value;
                if (record.indexed)
                {
                    entry.string_transfer_index_register = insn.source;
                    entry.string_transfer_index_value = record.index_value;
                }
                entry.string_transfer_byte_count = record.byte_count;
                entry.string_transfer_register_count = record.register_count;
                entry.has_effective_address = true;
                entry.effective_address = record.address;
                std::copy_n(record.bytes.begin(), record.byte_count,
                    entry.string_transfer_bytes.begin());
                for (std::uint8_t transfer = 0U;
                     transfer < record.register_count; ++transfer)
                {
                    entry.string_transfer_registers[transfer] =
                        record.registers[transfer];
                    entry.string_transfer_register_values[transfer] =
                        record.values[transfer];
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::store_string_word_immediate ||
                insn.opcode == Opcode::store_string_word_indexed)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeStringStoreRecord& record = context.stswi_records[i];
                entry.has_string_transfer = true;
                entry.string_transfer_store = true;
                entry.string_transfer_first_register = insn.source;
                entry.string_transfer_base_register = insn.base;
                entry.string_transfer_indexed = record.indexed;
                entry.string_transfer_base_value = record.base_value;
                if (record.indexed)
                {
                    entry.string_transfer_index_register = insn.destination;
                    entry.string_transfer_index_value = record.index_value;
                }
                entry.string_transfer_byte_count = record.byte_count;
                entry.string_transfer_register_count = record.register_count;
                entry.has_effective_address = true;
                entry.effective_address = record.address;
                std::copy_n(record.bytes.begin(), record.byte_count,
                    entry.string_transfer_bytes.begin());
                for (std::uint8_t transfer = 0U;
                     transfer < record.register_count; ++transfer)
                {
                    entry.string_transfer_registers[transfer] =
                        record.registers[transfer];
                    entry.string_transfer_register_values[transfer] =
                        record.values[transfer];
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_merge01 ||
                insn.opcode == Opcode::paired_single_merge11)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_merge_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_b_ps0;
                entry.paired_fp_source_b_ps1 = record.source_b_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_negate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_neg_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_move_register)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_move_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_absolute_value)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_abs_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_negative_absolute_value)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_nabs_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_reciprocal_estimate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_res_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_reciprocal_square_root_estimate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMergeRecord& record = context.ps_rsqrte_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_unary = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_b;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_a_ps0;
                entry.paired_fp_source_b_ps1 = record.source_a_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_reciprocal_square_root_estimate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFrsqrteRecord& record = context.frsqrte_records[i];
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_source_b;
                entry.fp_source_value = record.source_raw;
                entry.has_fp_destination = true;
                entry.fp_destination_register = insn.fp_register;
                entry.fp_destination_value = record.wrote_result
                    ? record.result_raw : core.state.fpr[insn.fp_register];
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_reciprocal_estimate_single)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFrspRecord& record = context.fres_records[i];
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_source_b;
                entry.fp_source_value = record.source_raw;
                entry.has_fp_destination = true;
                entry.fp_destination_register = insn.fp_register;
                entry.fp_destination_value = record.wrote_result
                    ? record.result_raw : core.state.fpr[insn.fp_register];
                core.append_instruction_history(std::move(entry));
                continue;
            }
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
            if (insn.opcode == Opcode::floating_add_single ||
                insn.opcode == Opcode::floating_subtract_single ||
                insn.opcode == Opcode::floating_divide_single)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFaddsRecord& record =
                    insn.opcode == Opcode::floating_add_single ? context.fadds_records[i] :
                    insn.opcode == Opcode::floating_subtract_single ? context.fsubs_records[i] :
                        context.fdivs_records[i];
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
                    entry.has_fp_arithmetic_single_bits = true;
                    entry.fp_arithmetic_single_bits = record.single_bits;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_multiply_add_single ||
                insn.opcode == Opcode::floating_multiply_subtract_single ||
                insn.opcode == Opcode::floating_negative_multiply_add_single ||
                insn.opcode == Opcode::floating_negative_multiply_subtract_single)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFmaddsRecord& record =
                    insn.opcode == Opcode::floating_multiply_add_single
                        ? context.fmadds_records[i]
                        : insn.opcode == Opcode::floating_multiply_subtract_single
                            ? context.fmsubs_records[i] : context.fnmsubs_records[i];
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
            if (insn.opcode == Opcode::floating_multiply_subtract_double)
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
            if (insn.opcode == Opcode::floating_negative_multiply_add_double)
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
            if (insn.opcode == Opcode::floating_negative_multiply_subtract_double)
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
            if (insn.opcode == Opcode::floating_add_double)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFsubRecord& record = context.fadd_records[i];
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
            if (insn.opcode == Opcode::floating_divide_double)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFsubRecord& record = context.fdiv_records[i];
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
            if (insn.opcode == Opcode::paired_single_select)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsSelectRecord& record = context.ps_select_records[i];
                entry.has_paired_fp_select = true;
                entry.paired_fp_select_destination = insn.fp_register;
                entry.paired_fp_select_a = insn.fp_source_a;
                entry.paired_fp_select_b = insn.fp_source_b;
                entry.paired_fp_select_c = insn.fp_source_c;
                entry.paired_fp_select_a_ps0 = record.source_a_ps0;
                entry.paired_fp_select_a_ps1 = record.source_a_ps1;
                entry.paired_fp_select_b_ps0 = record.source_b_ps0;
                entry.paired_fp_select_b_ps1 = record.source_b_ps1;
                entry.paired_fp_select_c_ps0 = record.source_c_ps0;
                entry.paired_fp_select_c_ps1 = record.source_c_ps1;
                entry.paired_fp_select_result_ps0 = record.destination_ps0;
                entry.paired_fp_select_result_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_sum0 ||
                insn.opcode == Opcode::paired_single_sum1)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsSumRecord& record = context.ps_sum_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_sum0 = insn.opcode == Opcode::paired_single_sum0;
                entry.paired_fp_sum1 = insn.opcode == Opcode::paired_single_sum1;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_c_register = insn.fp_source_c;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_b_ps0;
                entry.paired_fp_source_b_ps1 = record.source_b_ps1;
                entry.paired_fp_source_c_ps0 = record.source_c_ps0;
                entry.paired_fp_source_c_ps1 = record.source_c_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_add ||
                insn.opcode == Opcode::paired_single_subtract ||
                insn.opcode == Opcode::paired_single_divide)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsBinaryRecord& record = [&]() -> const NativePsBinaryRecord& {
                    if (insn.opcode == Opcode::paired_single_add)
                    {
                        return context.ps_add_records[i];
                    }
                    if (insn.opcode == Opcode::paired_single_subtract)
                    {
                        return context.ps_sub_records[i];
                    }
                    return context.ps_div_records[i];
                }();
                entry.has_paired_fp_state = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_b_ps0;
                entry.paired_fp_source_b_ps1 = record.source_b_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_multiply)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMulRecord& record = context.ps_mul_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_vector_multiply = true;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_c;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_c_ps0;
                entry.paired_fp_source_b_ps1 = record.source_c_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_multiply_scalar0 ||
                insn.opcode == Opcode::paired_single_multiply_scalar1)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMulRecord& record = context.ps_muls_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_scalar_multiply = true;
                entry.paired_fp_scalar_source_ps1 = insn.opcode ==
                    Opcode::paired_single_multiply_scalar1;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_c;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_c_ps0;
                entry.paired_fp_source_b_ps1 = record.source_c_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::paired_single_multiply_add ||
                insn.opcode == Opcode::paired_single_multiply_subtract ||
                insn.opcode == Opcode::paired_single_multiply_add_scalar0 ||
                insn.opcode == Opcode::paired_single_multiply_add_scalar1 ||
                insn.opcode == Opcode::paired_single_negative_multiply_add ||
                insn.opcode == Opcode::paired_single_negative_multiply_subtract)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativePsMaddRecord& record = context.ps_madd_records[i];
                entry.has_paired_fp_state = true;
                entry.paired_fp_multiply_add =
                    insn.opcode == Opcode::paired_single_multiply_add;
                entry.paired_fp_multiply_subtract =
                    insn.opcode == Opcode::paired_single_multiply_subtract;
                entry.paired_fp_multiply_add_scalar0 =
                    insn.opcode == Opcode::paired_single_multiply_add_scalar0;
                entry.paired_fp_multiply_add_scalar1 =
                    insn.opcode == Opcode::paired_single_multiply_add_scalar1;
                entry.paired_fp_negative_multiply_add =
                    insn.opcode == Opcode::paired_single_negative_multiply_add;
                entry.paired_fp_negative_multiply_subtract =
                    insn.opcode == Opcode::paired_single_negative_multiply_subtract;
                entry.paired_fp_destination_register = insn.fp_register;
                entry.paired_fp_source_a_register = insn.fp_source_a;
                entry.paired_fp_source_b_register = insn.fp_source_b;
                entry.paired_fp_source_c_register = insn.fp_source_c;
                entry.paired_fp_source_a_ps0 = record.source_a_ps0;
                entry.paired_fp_source_a_ps1 = record.source_a_ps1;
                entry.paired_fp_source_b_ps0 = record.source_b_ps0;
                entry.paired_fp_source_b_ps1 = record.source_b_ps1;
                entry.paired_fp_source_c_ps0 = record.source_c_ps0;
                entry.paired_fp_source_c_ps1 = record.source_c_ps1;
                entry.paired_fp_destination_ps0 = record.destination_ps0;
                entry.paired_fp_destination_ps1 = record.destination_ps1;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_move_from_fpscr)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMffsRecord& record = context.mffs_records[i];
                entry.has_fp_status_move = true;
                entry.fp_status_move_destination = insn.fp_register;
                entry.fp_status_move_source = record.fpscr;
                entry.fp_status_move_result = record.result;
                entry.fp_status_move_destination_ps1 =
                    core.state.fpr_ps1[insn.fp_register];
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::move_condition_register_from_fpscr)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMcrfsRecord& record = context.mcrfs_records[i];
                entry.has_fpscr_cr_move = true;
                entry.fpscr_cr_destination = insn.cr_field;
                entry.fpscr_cr_source = insn.fpscr_field;
                entry.fpscr_cr_value = record.value;
                entry.fpscr_before_cr_move = record.before;
                entry.fpscr_after_cr_move = record.after;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_move_to_fpscr_bit_zero ||
                insn.opcode == Opcode::floating_move_to_fpscr_bit_one)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMtfsbRecord& record = context.mtfsb_records[i];
                entry.has_fpscr_bit_operation = true;
                entry.fpscr_bit_set = insn.opcode ==
                    Opcode::floating_move_to_fpscr_bit_one;
                entry.fpscr_bit_number = insn.fpscr_bit;
                entry.fpscr_before_bit_operation = record.before;
                entry.fpscr_after_bit_operation = record.after;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_move_to_fpscr_field_immediate)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMtfsfiRecord& record = context.mtfsfi_records[i];
                entry.has_fpscr_field_write = true;
                entry.fpscr_field_number = insn.fpscr_field;
                entry.fpscr_field_immediate = insn.fpscr_immediate;
                entry.fpscr_before_field_write = record.before;
                entry.fpscr_after_field_write = record.after;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_move_to_fpscr_fields)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMtfsfRecord& record = context.mtfsf_records[i];
                entry.has_fpscr_masked_write = true;
                entry.fpscr_masked_write_source = insn.fp_source_b;
                entry.fpscr_masked_write_mask = insn.fpscr_field_mask;
                entry.fpscr_masked_write_source_raw = record.source_raw;
                entry.fpscr_before_masked_write = record.before;
                entry.fpscr_after_masked_write = record.after;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_convert_to_integer_word ||
                insn.opcode == Opcode::floating_convert_to_integer_word_zero)
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
            if (insn.opcode == Opcode::floating_compare_unordered ||
                insn.opcode == Opcode::floating_compare_ordered ||
                insn.opcode == Opcode::paired_single_compare_unordered0 ||
                insn.opcode == Opcode::paired_single_compare_unordered1 ||
                insn.opcode == Opcode::paired_single_compare_ordered0 ||
                insn.opcode == Opcode::paired_single_compare_ordered1)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFpCompareRecord& record = context.fp_compare_records[i];
                entry.has_fp_compare = true;
                entry.fp_compare_cr_field = insn.cr_field;
                entry.fp_compare_a_register = insn.fp_compare_a;
                entry.fp_compare_b_register = insn.fp_compare_b;
                entry.fp_compare_uses_ps1 = insn.fp_compare_uses_ps1;
                entry.fp_compare_a_raw = record.a_raw;
                entry.fp_compare_b_raw = record.b_raw;
                entry.fp_compare_result = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::extend_sign_byte ||
                insn.opcode == Opcode::extend_sign_halfword)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeExtsbRecord& record =
                    insn.opcode == Opcode::extend_sign_halfword
                    ? context.extsh_records[i] : context.extsb_records[i];
                entry.source_registers[0] = insn.source;
                entry.source_values[0] = record.source;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
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
            if (insn.opcode == Opcode::multiply_low_word ||
                insn.opcode == Opcode::multiply_low_word_overflow)
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
                if (insn.opcode == Opcode::multiply_low_word_overflow)
                {
                    entry.has_overflow_result = true;
                    entry.overflow_result = record.overflow != 0U;
                    entry.summary_overflow_result = record.summary_overflow != 0U;
                }
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_overflow ||
                insn.opcode == Opcode::subtract_from_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddOverflowRecord& record =
                    insn.opcode == Opcode::add_overflow
                        ? context.add_overflow_records[i]
                        : context.subf_overflow_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.operand_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::negate_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeNegateOverflowRecord& record =
                    context.negate_overflow_records[i];
                entry.source_registers[0] = insn.source;
                entry.source_values[0] = record.source;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::subtract_from_carrying_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeSubfcoRecord& record = context.subfco_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.operand_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_carrying_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddcoRecord& record = context.addco_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.operand_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_carrying)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddcRecord& record = context.addc_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.operand_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_extended)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddeRecord& record = context.adde_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_registers[1] = insn.source;
                entry.source_values[1] = record.operand_b;
                entry.source_count = insn.source == insn.base ? 1U : 2U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_extended_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddeoRecord& record = context.addeo_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand_a;
                entry.source_registers[1] = insn.source;
                entry.source_values[1] = record.operand_b;
                entry.source_count = insn.source == insn.base ? 1U : 2U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::subtract_from_minus_one_extended)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddzeRecord& record = context.subfme_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::arithmetic_shift_right_word)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeSrawRecord& record = context.sraw_records[i];
                entry.source_registers[0] = insn.source;
                entry.source_values[0] = record.source;
                entry.source_count = 1U;
                if (insn.base != insn.source)
                {
                    entry.source_registers[1] = insn.base;
                    entry.source_values[1] = record.shift;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_to_minus_one_extended)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddzeRecord& record = context.addme_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_to_minus_one_extended_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddmeoRecord& record = context.addmeo_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_to_zero_extended)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddzeRecord& record = context.addze_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::add_to_zero_extended_overflow)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeAddzeoRecord& record = context.addzeo_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                entry.has_overflow_result = true;
                entry.overflow_result = record.overflow != 0U;
                entry.summary_overflow_result = record.summary_overflow != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::subtract_from_zero_extended)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeSubfzeRecord& record = context.subfze_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.operand;
                entry.source_count = 1U;
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                entry.has_carry_input = true;
                entry.carry_input = record.carry_in != 0U;
                entry.has_carry_result = true;
                entry.carry_result = record.carry_out != 0U;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::multiply_high_word_unsigned ||
                insn.opcode == Opcode::multiply_high_word_signed)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeMulhwuRecord& record = context.mulhwu_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.source_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.source_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::floating_select)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeFselRecord& record = context.fsel_records[i];
                entry.has_fp_select = true;
                entry.fp_select_destination = insn.fp_register;
                entry.fp_select_a = insn.fp_source_a;
                entry.fp_select_b = insn.fp_source_b;
                entry.fp_select_c = insn.fp_source_c;
                entry.fp_select_a_raw = record.source_a_raw;
                entry.fp_select_b_raw = record.source_b_raw;
                entry.fp_select_c_raw = record.source_c_raw;
                entry.fp_select_result_raw = record.result_raw;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::subtract_from)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const NativeSubfRecord& record = context.subf_records[i];
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = record.source_a;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = record.source_b;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = record.result;
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::divide_word_unsigned ||
                insn.opcode == Opcode::divide_word_signed)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                const std::uint32_t dividend =
                    insn.opcode == Opcode::divide_word_unsigned
                        ? context.divwu_records[i].dividend
                        : context.divw_records[i].dividend;
                const std::uint32_t divisor =
                    insn.opcode == Opcode::divide_word_unsigned
                        ? context.divwu_records[i].divisor
                        : context.divw_records[i].divisor;
                const std::uint32_t result =
                    insn.opcode == Opcode::divide_word_unsigned
                        ? context.divwu_records[i].result
                        : context.divw_records[i].result;
                entry.source_registers[0] = insn.base;
                entry.source_values[0] = dividend;
                entry.source_count = 1U;
                if (insn.source != insn.base)
                {
                    entry.source_registers[1] = insn.source;
                    entry.source_values[1] = divisor;
                    entry.source_count = 2U;
                }
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = result;
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
            if (insn.opcode == Opcode::move_from_time_base_lower ||
                insn.opcode == Opcode::move_from_time_base_upper)
            {
                InstructionHistoryEntry entry = native_history(insn, cia);
                entry.has_destination = true;
                entry.destination_register = insn.destination;
                entry.destination_value = context.time_base_records[i].result;
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
                if (insn.opcode != Opcode::store_halfword_update &&
                    insn.opcode != Opcode::load_single_update &&
                    insn.opcode != Opcode::load_halfword_zero_update &&
                    insn.opcode != Opcode::load_double_update)
                {
                    entry.has_immediate = true;
                    entry.immediate = insn.immediate;
                }
            }
            if (insn.opcode == Opcode::instruction_cache_block_invalidate ||
                insn.opcode == Opcode::data_cache_block_touch ||
                insn.opcode == Opcode::data_cache_block_touch_for_store ||
                insn.opcode == Opcode::data_cache_block_store ||
                insn.opcode == Opcode::data_cache_block_flush ||
                insn.opcode == Opcode::data_cache_block_zero)
            {
                // Cache hints/maintenance instructions have no architectural
                // destination. icbi/dcbt/dcbtst have no separate cache state
                // in the coherent-memory model; dcbz writes its line while
                // dcbst/dcbf validate.
                core.append_instruction_history(std::move(entry));
                continue;
            }
            if (insn.opcode == Opcode::load_single ||
                insn.opcode == Opcode::load_single_update ||
                insn.opcode == Opcode::load_single_indexed ||
                insn.opcode == Opcode::load_single_indexed_update ||
                insn.opcode == Opcode::load_double ||
                insn.opcode == Opcode::load_double_update ||
                insn.opcode == Opcode::load_double_indexed ||
                insn.opcode == Opcode::load_double_indexed_update)
            {
                entry.has_fp_destination = true;
                entry.fp_destination_register = insn.fp_register;
                entry.fp_destination_value = record.fp_value;
                if (insn.opcode == Opcode::load_single_update ||
                    insn.opcode == Opcode::load_single_indexed_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
                else if (insn.opcode == Opcode::load_double_update ||
                    insn.opcode == Opcode::load_double_indexed_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
            }
            else if (insn.opcode == Opcode::store_single ||
                insn.opcode == Opcode::store_single_update ||
                insn.opcode == Opcode::store_single_indexed ||
                insn.opcode == Opcode::store_single_indexed_update)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = record.fp_value;
                entry.has_stored_single_value = true;
                entry.stored_single_value = record.value;
                if (insn.opcode == Opcode::store_single_update ||
                    insn.opcode == Opcode::store_single_indexed_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
            }
            else if (insn.opcode == Opcode::store_floating_as_integer_word_indexed)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = core.state.fpr[insn.fp_register];
                entry.has_effective_address = true;
                entry.effective_address = record.address;
                if (insn.base != 0U)
                {
                    entry.source_registers[entry.source_count] = insn.base;
                    entry.source_values[entry.source_count++] = record.base_value;
                }
                entry.source_registers[entry.source_count] = insn.source;
                entry.source_values[entry.source_count++] = record.index_value;
                entry.has_stored_integer_word_value = true;
                entry.stored_integer_word_value = record.value;
            }
            else if (insn.opcode == Opcode::store_double ||
                insn.opcode == Opcode::store_double_update ||
                insn.opcode == Opcode::store_double_indexed ||
                insn.opcode == Opcode::store_double_indexed_update)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = record.fp_value;
                entry.has_stored_double_value = true;
                entry.stored_double_value = record.fp_value;
                if (insn.opcode == Opcode::store_double_update ||
                    insn.opcode == Opcode::store_double_indexed_update)
                {
                    entry.has_destination = true;
                    entry.destination_register = insn.base;
                    entry.destination_value = record.address;
                }
            }
            else if (is_memory_store(insn.opcode))
            {
                if (insn.opcode != Opcode::store_halfword_update ||
                    insn.destination != insn.base)
                {
                    entry.source_registers[entry.source_count] = insn.destination;
                    entry.source_values[entry.source_count++] =
                        (insn.opcode == Opcode::store_word_byte_reverse_indexed ||
                         insn.opcode == Opcode::store_halfword_byte_reverse_indexed)
                        ? record.source_value : record.value;
                }
                if (insn.opcode == Opcode::store_halfword ||
                    insn.opcode == Opcode::store_halfword_update ||
                    insn.opcode == Opcode::store_halfword_indexed ||
                    insn.opcode == Opcode::store_halfword_byte_reverse_indexed)
                {
                    entry.has_stored_halfword_value = true;
                    entry.stored_halfword_value = static_cast<std::uint16_t>(record.value);
                }
                if (insn.opcode == Opcode::store_word ||
                    insn.opcode == Opcode::store_word_indexed ||
                    insn.opcode == Opcode::store_word_conditional_indexed ||
                    insn.opcode == Opcode::store_word_update)
                {
                    entry.has_stored_integer_word_value = true;
                    entry.stored_integer_word_value = record.value;
                }
                if (insn.opcode == Opcode::store_word_update ||
                    insn.opcode == Opcode::store_byte_update ||
                    insn.opcode == Opcode::store_halfword_update ||
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
                    if (faulting_instruction.opcode ==
                            Opcode::load_string_word_immediate ||
                        faulting_instruction.opcode ==
                            Opcode::load_string_word_indexed)
                    {
                        const NativeStringLoadRecord& record =
                            context.lswi_records[context.failed_instruction_index];
                        InstructionHistoryEntry entry = native_history(
                            faulting_instruction, context.fault_cia);
                        entry.completed = false;
                        entry.has_string_transfer = true;
                        entry.string_transfer_indexed = record.indexed;
                        entry.string_transfer_first_register =
                            faulting_instruction.destination;
                        entry.string_transfer_base_register =
                            faulting_instruction.base;
                        entry.string_transfer_base_value = record.base_value;
                        entry.string_transfer_byte_count = record.byte_count;
                        entry.string_transfer_register_count = record.register_count;
                        if (record.indexed)
                        {
                            entry.string_transfer_index_register =
                                faulting_instruction.source;
                            entry.string_transfer_index_value = record.index_value;
                        }
                        entry.has_effective_address = true;
                        entry.effective_address = record.address;
                        std::copy_n(record.bytes.begin(), record.byte_count,
                            entry.string_transfer_bytes.begin());
                        std::copy_n(record.registers.begin(),
                            record.register_count,
                            entry.string_transfer_registers.begin());
                        std::copy_n(record.values.begin(),
                            record.register_count,
                            entry.string_transfer_register_values.begin());
                        core.state.cia = context.fault_cia;
                        core.append_instruction_history(std::move(entry));
                        std::rethrow_exception(context.exception);
                    }
                    if (faulting_instruction.opcode ==
                            Opcode::store_string_word_immediate ||
                        faulting_instruction.opcode ==
                            Opcode::store_string_word_indexed)
                    {
                        const NativeStringStoreRecord& record =
                            context.stswi_records[context.failed_instruction_index];
                        InstructionHistoryEntry entry = native_history(
                            faulting_instruction, context.fault_cia);
                        entry.completed = false;
                        entry.has_string_transfer = true;
                        entry.string_transfer_store = true;
                        entry.string_transfer_indexed = record.indexed;
                        entry.string_transfer_first_register =
                            faulting_instruction.source;
                        entry.string_transfer_base_register =
                            faulting_instruction.base;
                        entry.string_transfer_base_value = record.base_value;
                        entry.string_transfer_byte_count = record.byte_count;
                        entry.string_transfer_register_count = record.register_count;
                        if (record.indexed)
                        {
                            entry.string_transfer_index_register =
                                faulting_instruction.destination;
                            entry.string_transfer_index_value = record.index_value;
                        }
                        entry.has_effective_address = true;
                        entry.effective_address = record.address;
                        std::copy_n(record.bytes.begin(), record.byte_count,
                            entry.string_transfer_bytes.begin());
                        std::copy_n(record.registers.begin(), record.register_count,
                            entry.string_transfer_registers.begin());
                        std::copy_n(record.values.begin(), record.register_count,
                            entry.string_transfer_register_values.begin());
                        core.state.cia = context.fault_cia;
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
