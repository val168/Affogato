#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

class HleDispatcher;
class EspressoCore;

void register_coreinit_hle(HleDispatcher& dispatcher);
[[nodiscard]] std::uint32_t initialize_default_guest_thread(
    EspressoCore& core,
    std::uint32_t stack_start,
    std::uint32_t stack_end);
[[nodiscard]] bool initialize_default_guest_heaps(EspressoCore& core);

}
