#include "cpu/espresso/nn_save_hle.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t save_status_ok = 0U;

}

void register_nn_save_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "nn_save",
        "SAVEInit",
        [](EspressoCore& core) {
            // Cafe normally initializes ACT/ACP, private FS state, account
            // metadata, and save mounting here. Those services are not modeled
            // yet, so retain only this per-session lifecycle state until a
            // guest-visible SAVE operation requires the dependencies.
            core.nn_save_initialized = true;
            core.state.gpr[3] = save_status_ok;
        });
}

}
