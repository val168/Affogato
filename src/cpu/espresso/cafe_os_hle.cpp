#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

namespace affogato::cpu::espresso
{

void register_coreinit_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "coreinit",
        "OSIsDebuggerInitialized",
        [](EspressoCore& core) {
            // Affogato has no Cafe debugger/GDB transport yet.
            core.state.gpr[3] = 0;
        });
}

}
