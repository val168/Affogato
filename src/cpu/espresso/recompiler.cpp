#include "cpu/espresso/recompiler.hpp"

#include <asmjit/core.h>
#include <asmjit/x86.h>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <limits>
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

struct NativeExecutionContext
{
    GuestMemory* memory{};
    std::uint32_t block_cia{};
    std::array<NativeMemoryRecord, 64> memory_records{};
    std::array<NativeFpCompareRecord, 64> fp_compare_records{};
    std::uint64_t memory_access_calls{};
    bool failed{};
    bool guest_memory_fault{};
    std::exception_ptr exception{};
    std::uint32_t fault_cia{};
    std::uint32_t failed_address{};
    std::uint32_t failed_width{};
    std::uint32_t failed_flags{};
    std::uint32_t failed_instruction_index{};
};

static_assert(std::is_standard_layout_v<NativeExecutionContext>);

using NativeBlock = std::uint32_t (*)(CpuState*, NativeExecutionContext*);

extern "C" void native_raise_fpscr_snan(CpuState* state) noexcept
{
    constexpr std::uint32_t implemented_invalid_masks =
        fpscr::vxsnan_mask | fpscr::vxzdz_mask | fpscr::vximz_mask |
        fpscr::vxidi_mask | fpscr::vxisi_mask | fpscr::vxcvi_mask |
        fpscr::vxvc_mask | fpscr::vxsqrt_mask | fpscr::vxsoft_mask;
    state->fpscr |= fpscr::vxsnan_mask | fpscr::fx_mask;
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
        insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor))
    {
        return true;
    }
    if (insn.opcode == Opcode::add_immediate_carry && insn.record)
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
    case Opcode::move_condition_register_field:
    case Opcode::move_from_condition_register:
    case Opcode::move_to_condition_register_fields:
    case Opcode::move_condition_register_from_xer:
    case Opcode::compare_signed_immediate:
    case Opcode::compare_signed_register:
    case Opcode::compare_unsigned_register:
    case Opcode::floating_compare_unordered:
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
    case Opcode::paired_single_merge10:
    case Opcode::instruction_sync:
    case Opcode::shift_left_word:
    case Opcode::shift_right_word:
    case Opcode::arithmetic_shift_right_immediate:
    case Opcode::count_leading_zeros:
    case Opcode::rotate_left_word_and_mask_insert:
    case Opcode::load_word_zero:
    case Opcode::load_byte_zero:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_double:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_halfword:
    case Opcode::store_single:
    case Opcode::store_double:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_halfword_indexed:
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
    case Opcode::paired_single_merge10: return "ps_merge10";
    case Opcode::instruction_sync: return "isync";
    case Opcode::bitwise_or: return "or";
    case Opcode::bitwise_and: return "and";
    case Opcode::bitwise_xor: return "xor";
    case Opcode::compare_signed_immediate: return "cmpwi";
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
    case Opcode::shift_left_word: return "slw";
    case Opcode::shift_right_word: return "srw";
    case Opcode::arithmetic_shift_right_immediate: return "srawi";
    case Opcode::count_leading_zeros: return "cntlzw";
    case Opcode::load_word_zero: return "lwz";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::load_single: return "lfs";
    case Opcode::load_double: return "lfd";
    case Opcode::store_word: return "stw";
    case Opcode::store_word_update: return "stwu";
    case Opcode::store_byte: return "stb";
    case Opcode::store_halfword: return "sth";
    case Opcode::store_single: return "stfs";
    case Opcode::store_double: return "stfd";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::load_byte_zero_indexed: return "lbzx";
    case Opcode::load_halfword_zero_indexed: return "lhzx";
    case Opcode::load_halfword_algebraic_indexed: return "lhax";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::store_halfword_indexed: return "sthx";
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
    case Opcode::load_byte_zero:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::load_single:
    case Opcode::load_double:
    case Opcode::store_single:
    case Opcode::store_double:
    case Opcode::store_word:
    case Opcode::store_word_update:
    case Opcode::store_byte:
    case Opcode::store_halfword:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_halfword_indexed:
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
    case Opcode::store_byte:
    case Opcode::load_byte_zero_indexed:
    case Opcode::store_byte_indexed:
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
        opcode == Opcode::store_byte_indexed || opcode == Opcode::store_halfword_indexed ||
        opcode == Opcode::store_single || opcode == Opcode::store_double ||
        opcode == Opcode::store_word_update;
}

[[nodiscard]] bool is_indexed_memory(Opcode opcode) noexcept
{
    return opcode == Opcode::load_word_indexed || opcode == Opcode::load_byte_zero_indexed ||
        opcode == Opcode::load_halfword_zero_indexed ||
        opcode == Opcode::load_halfword_algebraic_indexed ||
        opcode == Opcode::store_word_indexed || opcode == Opcode::store_byte_indexed ||
        opcode == Opcode::store_halfword_indexed;
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
            {
                x86::Gp lhs = compiler.new_gp32("lhs");
                x86::Gp source_a = read_guest_gpr(insn.base);
                x86::Gp source_b = read_guest_gpr(insn.source);
                if (insn.opcode == Opcode::add || insn.opcode == Opcode::bitwise_or ||
                    insn.opcode == Opcode::bitwise_and || insn.opcode == Opcode::bitwise_xor)
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
                    else compiler.xor_(lhs, rhs);
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
            case Opcode::compare_signed_register:
            case Opcode::compare_unsigned_register:
            {
                x86::Gp lhs = read_guest_gpr(insn.base);
                if (insn.opcode == Opcode::compare_signed_immediate)
                {
                    compiler.cmp(lhs, imm(static_cast<std::uint32_t>(insn.immediate)));
                }
                else
                {
                    compiler.cmp(lhs, read_guest_gpr(insn.source));
                }
                const bool signed_compare =
                    insn.opcode != Opcode::compare_unsigned_register;

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
            case Opcode::paired_single_merge10:
            {
                // Snapshot both lanes before either destination lane is
                // written; frD may alias either or both sources.
                x86::Gp source_a_ps1 = compiler.new_gp64("ps_merge10_a_ps1");
                x86::Gp source_b_ps0 = compiler.new_gp64("ps_merge10_b_ps0");
                compiler.mov(source_a_ps1, x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_source_a)));
                compiler.mov(source_b_ps0, x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_source_b))));
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(fpr_offset(insn.fp_register))),
                    source_a_ps1);
                compiler.mov(x86::qword_ptr(state,
                    static_cast<std::int32_t>(offsetof(CpuState, fpr_ps1) +
                        sizeof(std::uint64_t) * insn.fp_register)),
                    source_b_ps0);
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
            case Opcode::load_byte_zero:
            case Opcode::load_halfword_zero:
            case Opcode::load_halfword_algebraic:
            case Opcode::store_word:
            case Opcode::store_word_update:
            case Opcode::store_byte:
            case Opcode::store_halfword:
            case Opcode::store_single:
            case Opcode::store_double:
            case Opcode::load_word_indexed:
            case Opcode::load_byte_zero_indexed:
            case Opcode::load_halfword_zero_indexed:
            case Opcode::load_halfword_algebraic_indexed:
            case Opcode::store_word_indexed:
            case Opcode::store_byte_indexed:
            case Opcode::store_halfword_indexed:
            {
                const bool indexed = is_indexed_memory(insn.opcode);
                const bool store = is_memory_store(insn.opcode);
                const bool single_store = insn.opcode == Opcode::store_single;
                const bool double_store = insn.opcode == Opcode::store_double;
                const bool update_base = insn.opcode == Opcode::store_word_update;
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
                if (!store)
                {
                    write_guest_gpr(insn.destination, value);
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
            if (insn.opcode == Opcode::load_single || insn.opcode == Opcode::load_double)
            {
                entry.has_fp_destination = true;
                entry.fp_destination_register = insn.fp_register;
                entry.fp_destination_value = record.fp_value;
            }
            else if (insn.opcode == Opcode::store_single)
            {
                entry.has_fp_source = true;
                entry.fp_source_register = insn.fp_register;
                entry.fp_source_value = record.fp_value;
                entry.has_stored_single_value = true;
                entry.stored_single_value = record.value;
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
                if (insn.opcode == Opcode::store_word_update)
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
        if (context.failed)
        {
            try
            {
                if (context.guest_memory_fault)
                    static_cast<void>(core.step());
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
