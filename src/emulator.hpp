#pragma once

#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/rpx_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <span>

namespace affogato
{

struct EmulatorRunResult
{
    cpu::espresso::RpxLoadResult image;
    cpu::espresso::RunResult execution;
    std::uint32_t gpr3{};
};

class Emulator
{
public:
    static constexpr std::size_t default_guest_memory_size = 0x10100000U;

    explicit Emulator(std::size_t guest_memory_size = default_guest_memory_size);

    [[nodiscard]] cpu::espresso::RpxLoadResult load_rpx(
        std::span<const std::uint8_t> file);
    [[nodiscard]] EmulatorRunResult run(std::size_t max_steps = 1'000'000);

    [[nodiscard]] const cpu::espresso::EspressoCore& core() const noexcept
    {
        return core_;
    }

private:
    cpu::espresso::EspressoCore core_;
    cpu::espresso::RpxLoadResult loaded_image_{};
    bool has_loaded_image_{};
};

}
