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
};

// Loads the allocatable sections of a static Wii U RPX into the current flat
// guest address space. Sections using Cafe's deflate flag are inflated first.
// Modules with unresolved library imports or unsupported relocations fail
// explicitly through named HLE trampolines; function and data import tables
// receive those trampoline addresses so unresolved calls stop diagnostically.
[[nodiscard]] RpxLoadResult load_rpx32_powerpc(
    EspressoCore& core,
    std::span<const std::uint8_t> file);

}
