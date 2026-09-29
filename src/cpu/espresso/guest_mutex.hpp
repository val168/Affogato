#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

class EspressoCore;

inline constexpr std::uint32_t os_mutex_size = 0x2CU;
inline constexpr std::uint32_t os_mutex_tag = 0x6D557458U;

void initialize_guest_os_mutex(
    EspressoCore& core,
    std::uint32_t address,
    std::uint32_t name = 0);

[[nodiscard]] std::uint32_t allocate_guest_os_mutex(
    EspressoCore& core,
    std::uint32_t name = 0);

// Returns false when another guest thread owns the mutex. The caller must
// handle that case; Affogato does not have a scheduler to wait for the owner.
[[nodiscard]] bool lock_guest_os_mutex(EspressoCore& core, std::uint32_t address);
void unlock_guest_os_mutex(EspressoCore& core, std::uint32_t address);

}
