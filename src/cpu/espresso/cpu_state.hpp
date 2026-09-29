#pragma once

#include <array>
#include <cstdint>

namespace affogato::cpu::espresso
{

struct CpuState
{
    std::array<std::uint32_t, 32> gpr{};
    std::array<std::uint64_t, 32> fpr{};

    std::uint32_t cr{};
    std::uint32_t xer{};
    std::uint32_t lr{};
    std::uint32_t ctr{};
    std::uint32_t cia{};

    void reset() noexcept
    {
        *this = {};
    }
};

}
