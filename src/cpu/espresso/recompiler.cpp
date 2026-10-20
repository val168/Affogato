#include "cpu/espresso/recompiler.hpp"

#include <asmjit/core.h>
#include <asmjit/x86.h>

#include <algorithm>
#include <array>
#include <bit>
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

struct NativeMemoryRecord
{
    std::uint32_t address{};
    std::uint32_t value{};
    std::uint32_t base_value{};
    std::uint32_t index_value{};
    bool indexed{};
};

struct NativeExecutionContext
{
    GuestMemory* memory{};
    std::array<NativeMemoryRecord, 64> memory_records{};
    bool failed{};
    bool guest_memory_fault{};
    std::exception_ptr exception{};
    std::uint32_t fault_cia{};
};

static_assert(std::is_standard_layout_v<NativeExecutionContext>);

using NativeBlock = std::uint32_t (*)(CpuState*, NativeExecutionContext*);

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
        NativeMemoryRecord& record = context->memory_records[instruction_index];
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
            default: context->failed = true; return 0;
            }
            return 0;
        }

        std::uint32_t loaded{};
        switch (width)
        {
        case 1: loaded = context->memory->read8(address); break;
        case 2: loaded = context->memory->read16_be(address); break;
        case 4: loaded = context->memory->read32_be(address); break;
        default: context->failed = true; return 0;
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

[[nodiscard]] bool is_native_instruction(const DecodedInstruction& insn) noexcept
{
    switch (insn.opcode)
    {
    case Opcode::addi:
    case Opcode::addis:
    case Opcode::ori:
    case Opcode::or_immediate_shifted:
    case Opcode::xor_immediate:
    case Opcode::xor_immediate_shifted:
    case Opcode::add:
    case Opcode::subtract_from:
    case Opcode::bitwise_or:
    case Opcode::bitwise_and:
    case Opcode::bitwise_xor:
    case Opcode::branch:
    case Opcode::conditional_branch:
    case Opcode::conditional_branch_to_link_register:
    case Opcode::conditional_branch_to_count_register:
    case Opcode::move_from_link_register:
    case Opcode::move_to_link_register:
    case Opcode::move_from_count_register:
    case Opcode::move_to_count_register:
    case Opcode::load_word_zero:
    case Opcode::load_byte_zero:
    case Opcode::load_halfword_zero:
    case Opcode::load_halfword_algebraic:
    case Opcode::store_word:
    case Opcode::store_byte:
    case Opcode::store_halfword:
    case Opcode::load_word_indexed:
    case Opcode::load_byte_zero_indexed:
    case Opcode::load_halfword_zero_indexed:
    case Opcode::load_halfword_algebraic_indexed:
    case Opcode::store_word_indexed:
    case Opcode::store_byte_indexed:
    case Opcode::store_halfword_indexed:
        return !insn.record;
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
    case Opcode::ori: return "ori";
    case Opcode::or_immediate_shifted: return "oris";
    case Opcode::xor_immediate: return "xori";
    case Opcode::xor_immediate_shifted: return "xoris";
    case Opcode::add: return "add";
    case Opcode::subtract_from: return "subf";
    case Opcode::bitwise_or: return "or";
    case Opcode::bitwise_and: return "and";
    case Opcode::bitwise_xor: return "xor";
    case Opcode::branch: return "b";
    case Opcode::conditional_branch: return "bc";
    case Opcode::conditional_branch_to_link_register: return "bclr";
    case Opcode::conditional_branch_to_count_register: return "bcctr";
    case Opcode::move_from_link_register: return "mflr";
    case Opcode::move_to_link_register: return "mtlr";
    case Opcode::move_from_count_register: return "mfctr";
    case Opcode::move_to_count_register: return "mtctr";
    case Opcode::load_word_zero: return "lwz";
    case Opcode::load_byte_zero: return "lbz";
    case Opcode::load_halfword_zero: return "lhz";
    case Opcode::load_halfword_algebraic: return "lha";
    case Opcode::store_word: return "stw";
    case Opcode::store_byte: return "stb";
    case Opcode::store_halfword: return "sth";
    case Opcode::load_word_indexed: return "lwzx";
    case Opcode::load_byte_zero_indexed: return "lbzx";
    case Opcode::load_halfword_zero_indexed: return "lhzx";
    case Opcode::load_halfword_algebraic_indexed: return "lhax";
    case Opcode::store_word_indexed: return "stwx";
    case Opcode::store_byte_indexed: return "stbx";
    case Opcode::store_halfword_indexed: return "sthx";
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
    case Opcode::store_word:
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
    default:
        return 4U;
    }
}

[[nodiscard]] bool is_memory_store(Opcode opcode) noexcept
{
    return opcode == Opcode::store_word || opcode == Opcode::store_byte ||
        opcode == Opcode::store_halfword || opcode == Opcode::store_word_indexed ||
        opcode == Opcode::store_byte_indexed || opcode == Opcode::store_halfword_indexed;
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
    std::uint32_t cia,
    const CpuState& before,
    const CpuState& after)
{
    InstructionHistoryEntry entry{};
    entry.cia = cia;
    entry.instruction_word = insn.raw;
    entry.has_instruction_word = true;
    entry.opcode_name = name(insn.opcode);
    entry.completed = true;

    // Native blocks may contain dependencies between guest instructions. Until
    // per-instruction snapshots are emitted into the JIT context, retain the
    // exact native trace identity without reporting potentially stale values.
    static_cast<void>(before);
    static_cast<void>(after);
    if (insn.opcode == Opcode::addi || insn.opcode == Opcode::addis ||
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
    struct CacheEntry
    {
        BlockCode block;
        const GuestMemory* memory_owner{};
        std::uint64_t memory_write_generation{};
    };

    asmjit::JitRuntime runtime;
    std::unordered_map<std::uint32_t, CacheEntry> cache;
    RecompilerStatistics stats{};

    ~Impl()
    {
        clear();
    }

    void clear() noexcept
    {
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

        for (std::size_t index = 0; index < block.instructions.size(); ++index)
        {
            const DecodedInstruction& insn = block.instructions[index];
            using namespace asmjit;
            switch (insn.opcode)
            {
            case Opcode::addi:
            case Opcode::addis:
            {
                x86::Gp value = compiler.new_gp32("value");
                if (insn.base == 0U)
                {
                    compiler.mov(value, imm(0));
                }
                else
                {
                    compiler.mov(value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.base))));
                }
                compiler.add(value, imm(immediate_value(insn)));
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.destination))), value);
                break;
            }
            case Opcode::ori:
            case Opcode::or_immediate_shifted:
            case Opcode::xor_immediate:
            case Opcode::xor_immediate_shifted:
            {
                x86::Gp value = compiler.new_gp32("value");
                compiler.mov(value, x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.source))));
                if (insn.opcode == Opcode::ori || insn.opcode == Opcode::or_immediate_shifted)
                    compiler.or_(value, imm(immediate_value(insn)));
                else
                    compiler.xor_(value, imm(immediate_value(insn)));
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.destination))), value);
                break;
            }
            case Opcode::add:
            case Opcode::subtract_from:
            case Opcode::bitwise_or:
            case Opcode::bitwise_and:
            case Opcode::bitwise_xor:
            {
                x86::Gp lhs = compiler.new_gp32("lhs");
                x86::Gp rhs = compiler.new_gp32("rhs");
                compiler.mov(lhs, x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.base))));
                compiler.mov(rhs, x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.source))));
                if (insn.opcode == Opcode::add) compiler.add(lhs, rhs);
                else if (insn.opcode == Opcode::subtract_from)
                {
                    compiler.sub(rhs, lhs);
                    compiler.mov(lhs, rhs);
                }
                else if (insn.opcode == Opcode::bitwise_or) compiler.or_(lhs, rhs);
                else if (insn.opcode == Opcode::bitwise_and) compiler.and_(lhs, rhs);
                else compiler.xor_(lhs, rhs);
                compiler.mov(x86::dword_ptr(
                    state, static_cast<std::int32_t>(gpr_offset(insn.destination))), lhs);
                break;
            }
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
            case Opcode::move_to_link_register:
            case Opcode::move_to_count_register:
            {
                const std::size_t special_offset =
                    insn.opcode == Opcode::move_from_link_register ||
                    insn.opcode == Opcode::move_to_link_register
                    ? offsetof(CpuState, lr)
                    : offsetof(CpuState, ctr);
                if (insn.opcode == Opcode::move_from_link_register ||
                    insn.opcode == Opcode::move_from_count_register)
                {
                    x86::Gp value = compiler.new_gp32("special");
                    compiler.mov(value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(special_offset)));
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.destination))), value);
                }
                else
                {
                    x86::Gp value = compiler.new_gp32("gpr");
                    compiler.mov(value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.destination))));
                    compiler.mov(x86::dword_ptr(
                        state, static_cast<std::int32_t>(special_offset)), value);
                }
                break;
            }
            case Opcode::load_word_zero:
            case Opcode::load_byte_zero:
            case Opcode::load_halfword_zero:
            case Opcode::load_halfword_algebraic:
            case Opcode::store_word:
            case Opcode::store_byte:
            case Opcode::store_halfword:
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
                    compiler.mov(address, x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.base))));
                    compiler.mov(x86::dword_ptr(context, static_cast<std::int32_t>(
                        record_base + offsetof(NativeMemoryRecord, base_value))), address);
                }
                if (indexed)
                {
                    x86::Gp index_value = compiler.new_gp32("index");
                    compiler.mov(index_value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.source))));
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
                    compiler.mov(stored_value, x86::dword_ptr(
                        state, static_cast<std::int32_t>(gpr_offset(insn.destination))));
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
                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                        gpr_offset(insn.destination))), value);
                }
                else
                {
                    // Any store may modify translated code. Exit to the C++
                    // dispatcher so the next block validates its cached words.
                    compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                        offsetof(CpuState, cia))), imm(cia + 4U));
                    x86::Gp executed = compiler.new_gp32("completed_store");
                    compiler.mov(executed, imm(static_cast<std::uint32_t>(index + 1U)));
                    compiler.ret(executed);
                }
                compiler.jmp(completed);
                compiler.bind(fault);
                compiler.mov(x86::dword_ptr(state, static_cast<std::int32_t>(
                    offsetof(CpuState, cia))), imm(cia));
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
        const std::uint32_t start = core.state.cia;
        if (start >= HleDispatcher::first_import_address &&
            start < HleDispatcher::import_address_limit)
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
            auto found = impl_->cache.find(start);
            bool valid = found != impl_->cache.end();
            if (valid && (found->second.memory_owner != &core.memory ||
                    found->second.memory_write_generation != core.memory.write_generation()))
            {
                const auto& words = found->second.block.words;
                for (std::size_t i = 0; i < words.size(); ++i)
                {
                    if (core.memory.read32_be(start + static_cast<std::uint32_t>(i * 4U)) != words[i])
                    {
                        valid = false;
                        break;
                    }
                }
                if (!valid)
                {
                    if (found->second.block.function != nullptr)
                        static_cast<void>(impl_->runtime.release(found->second.block.function));
                    impl_->cache.erase(found);
                    found = impl_->cache.end();
                }
                else
                {
                    found->second.memory_owner = &core.memory;
                    found->second.memory_write_generation =
                        core.memory.write_generation();
                }
            }

            if (valid)
            {
                ++impl_->stats.cache_hits;
                block = &found->second.block;
            }
            else
            {
                BlockCode candidate{};
                candidate.address = start;
                constexpr std::size_t max_block_instructions = 64U;
                for (std::size_t i = 0; i < max_block_instructions; ++i)
                {
                    const std::uint32_t cia = start + static_cast<std::uint32_t>(i * 4U);
                    if (cia >= HleDispatcher::first_import_address &&
                        cia < HleDispatcher::import_address_limit)
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
                    if (decoded.opcode == Opcode::branch ||
                        decoded.opcode == Opcode::conditional_branch ||
                        decoded.opcode == Opcode::conditional_branch_to_link_register ||
                        decoded.opcode == Opcode::conditional_branch_to_count_register)
                    {
                        break;
                    }
                }
                if (!candidate.instructions.empty() && impl_->compile(candidate))
                {
                    auto [inserted, did_insert] = impl_->cache.emplace(
                        start, Impl::CacheEntry{
                            std::move(candidate), &core.memory,
                            core.memory.write_generation()});
                    static_cast<void>(did_insert);
                    ++impl_->stats.compiled_blocks;
                    block = &inserted->second.block;
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

        const CpuState before = core.state;
        NativeExecutionContext context{};
        context.memory = &core.memory;
        const std::uint32_t executed = block->function(&core.state, &context);
        impl_->stats.native_instructions += executed;
        for (std::size_t i = 0; i < executed; ++i)
        {
            const std::uint32_t cia = block->address + static_cast<std::uint32_t>(i * 4U);
            const DecodedInstruction& insn = block->instructions[i];
            if (!is_memory_instruction(insn.opcode))
            {
                core.append_instruction_history(native_history(
                    insn, cia, before, core.state));
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
            if (is_memory_store(insn.opcode))
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
                    insn.opcode == Opcode::store_word_indexed)
                {
                    entry.has_stored_integer_word_value = true;
                    entry.stored_integer_word_value = record.value;
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
                result.detail = "native memory helper failed without an exception";
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
