#pragma once

#include "cpu/espresso/cpu_state.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"

#include <cstddef>
#include <string>
#include <unordered_map>

namespace affogato::cpu::espresso
{

enum class StepResult
{
    executed,
    unsupported_instruction,
    unimplemented_hle_call,
};

enum class StopReason
{
    instruction_limit,
    unsupported_instruction,
    unimplemented_hle_call,
    memory_fault,
};

struct RunResult
{
    std::size_t steps{};
    StopReason reason{StopReason::instruction_limit};
    std::uint32_t cia{};
    std::uint32_t instruction_word{};
    std::string hle_call;
    std::string detail;
};

class EspressoCore
{
public:
    explicit EspressoCore(std::size_t memory_size)
        : memory(memory_size)
    {
    }

    void reset() noexcept
    {
        state.reset();
        thread_specific.clear();
        guest_heap_cursor = 0;
        guest_heap_limit = 0;
    }

    void configure_guest_heap(std::uint32_t begin, std::uint32_t end) noexcept
    {
        guest_heap_cursor = begin;
        guest_heap_limit = end;
    }

    [[nodiscard]] std::uint32_t allocate_guest_memory(
        std::uint32_t size,
        std::uint32_t alignment = 16) noexcept
    {
        if (size == 0 || alignment == 0 || (alignment & (alignment - 1U)) != 0)
        {
            return 0;
        }
        const std::uint64_t aligned =
            (static_cast<std::uint64_t>(guest_heap_cursor) + alignment - 1U) &
            ~static_cast<std::uint64_t>(alignment - 1U);
        const std::uint64_t allocation_end = aligned + size;
        if (aligned > guest_heap_limit || allocation_end > guest_heap_limit)
        {
            return 0;
        }
        guest_heap_cursor = static_cast<std::uint32_t>(allocation_end);
        return static_cast<std::uint32_t>(aligned);
    }

    [[nodiscard]] StepResult step();
    [[nodiscard]] RunResult run(std::size_t max_steps);

    CpuState state{};
    GuestMemory memory;
    HleDispatcher hle;
    std::unordered_map<std::uint32_t, std::uint32_t> thread_specific;
    std::uint32_t guest_heap_cursor{};
    std::uint32_t guest_heap_limit{};
};

}
