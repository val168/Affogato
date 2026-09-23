#pragma once

#include "cpu/espresso/interpreter.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace affogato::cpu::espresso
{

struct ElfLoadResult
{
    std::uint32_t entry_point{};
    std::size_t loaded_segments{};
};

// Loads a flat-address-space ELF32, big-endian PowerPC executable into guest
// memory and sets the core's CIA to its entry point.
[[nodiscard]] ElfLoadResult load_elf32_powerpc(
    EspressoCore& core,
    std::span<const std::uint8_t> file);

}
