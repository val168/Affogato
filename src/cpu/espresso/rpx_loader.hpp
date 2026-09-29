#pragma once

#include "cpu/espresso/interpreter.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace affogato::cpu::espresso
{

struct RpxLoadResult
{
    std::uint32_t entry_point{};
    std::size_t loaded_sections{};
    std::uint32_t sda_base{};
    std::uint32_t sda2_base{};
};

// Loads allocatable sections of a static Wii U RPX through GuestMemory,
// retaining flat backing where possible and using sparse mappings elsewhere.
// Sections using Cafe's deflate flag are inflated first.
// Modules with unresolved library imports or unsupported relocations fail
// explicitly through named HLE trampolines; function and data import tables
// receive those trampoline addresses so unresolved calls stop diagnostically.
[[nodiscard]] RpxLoadResult load_rpx32_powerpc(
    EspressoCore& core,
    std::span<const std::uint8_t> file);

}
