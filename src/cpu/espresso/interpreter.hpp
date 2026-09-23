#pragma once

#include "cpu/espresso/cpu_state.hpp"
#include "cpu/espresso/guest_memory.hpp"

#include <cstddef>

namespace affogato::cpu::espresso
{

enum class StepResult
{
    executed,
    unsupported_instruction,
};

enum class StopReason
{
    instruction_limit,
    unsupported_instruction,
};

struct RunResult
{
    std::size_t steps{};
    StopReason reason{StopReason::instruction_limit};
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
    }

    [[nodiscard]] StepResult step();
    [[nodiscard]] RunResult run(std::size_t max_steps);

    CpuState state{};
    GuestMemory memory;
};

}
