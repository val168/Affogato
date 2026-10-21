#include "cpu/espresso/nn_act_hle.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <limits>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t act_result_success = 0U;

}

void register_nn_act_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "nn_act",
        "GetSlotNo__Q2_2nn3actFv",
        [](EspressoCore& core) {
            // Cemu models the selected console account as slot 1 (slot zero
            // means no account). Affogato currently has no account selector.
            core.state.gpr[3] = core.nn_act_current_slot;
        });

    dispatcher.register_function(
        "nn_act",
        "Initialize__Q2_2nn3actFv",
        [](EspressoCore& core) {
            // ACT initialization is reference-counted by the guest API. The
            // console account backend is not modeled; this records only the
            // library lifecycle and does not fabricate account data.
            if (core.nn_act_initialize_count !=
                std::numeric_limits<std::uint32_t>::max())
            {
                ++core.nn_act_initialize_count;
            }
            core.state.gpr[3] = act_result_success;
        });
}

}
