#include "emulator.hpp"

#include "cpu/espresso/cafe_os_hle.hpp"
#include "cpu/espresso/nn_olv_hle.hpp"

#include <stdexcept>
#include <utility>

namespace affogato
{

Emulator::Emulator(std::size_t guest_memory_size)
    : core_(guest_memory_size)
{
    cpu::espresso::register_coreinit_hle(core_.hle);
    cpu::espresso::register_nn_olv_hle(core_.hle);
}

cpu::espresso::RpxLoadResult Emulator::load_rpx(std::span<const std::uint8_t> file)
{
    if (has_loaded_image_)
    {
        throw std::logic_error("this Emulator session already has an RPX loaded");
    }
    if (core_.memory.size() <= guest_stack_size + guest_stack_top_reserve ||
        core_.memory.size() > UINT32_MAX)
    {
        throw std::invalid_argument("guest memory cannot safely reserve the RPX stack");
    }

    loaded_image_ = cpu::espresso::load_rpx32_powerpc(core_, file);
    core_.state.gpr[2] = loaded_image_.sda2_base;
    core_.state.gpr[13] = loaded_image_.sda_base;
    const std::uint32_t stack_start = static_cast<std::uint32_t>(
        (core_.memory.size() - guest_stack_top_reserve) & ~std::size_t{0xFU});
    const std::uint32_t stack_end = stack_start - guest_stack_size;
    core_.state.gpr[1] = stack_start;
    static_cast<void>(cpu::espresso::initialize_default_guest_thread(
        core_, stack_start, stack_end));
    has_loaded_image_ = true;
    return loaded_image_;
}

EmulatorRunResult Emulator::run(std::size_t max_steps)
{
    if (!has_loaded_image_)
    {
        throw std::logic_error("load an RPX before running the Emulator session");
    }

    cpu::espresso::RunResult execution = core_.run(max_steps);
    return {loaded_image_, std::move(execution), core_.state.gpr[3]};
}

}
