#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

namespace affogato::cpu::espresso
{

void register_coreinit_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "coreinit",
        "FSAInit",
        [](EspressoCore& core) {
            // The startup path only needs the filesystem facade's successful
            // initialization status; file and device operations remain HLE stubs.
            core.state.gpr[3] = 0;
        });
    dispatcher.register_function(
        "coreinit",
        "FSAAddClient",
        [](EspressoCore& core) {
            // A synthetic nonzero handle is sufficient for WUT's startup
            // registration; no filesystem requests are performed by this test.
            core.state.gpr[3] = 1;
        });
    dispatcher.register_function(
        "coreinit",
        "FSAMount",
        [](EspressoCore& core) {
            // The minimal sample registers a startup device but performs no
            // file operations; report the mount operation as successful.
            core.state.gpr[3] = 0;
        });
    dispatcher.register_function(
        "coreinit",
        "FSAGetDeviceInfo",
        [](EspressoCore& core) {
            // The minimal startup probe does not access mounted files. Its
            // devoptab entry is initialized by WUT before this query, so only
            // report that the synthetic device is present.
            core.state.gpr[3] = 0;
        });
    dispatcher.register_function(
        "coreinit",
        "MEMAllocFromDefaultHeap",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.allocate_guest_memory(core.state.gpr[3]);
        });
    dispatcher.register_function(
        "coreinit",
        "MEMAllocFromDefaultHeapEx",
        [](EspressoCore& core) {
            const std::uint32_t requested_alignment = core.state.gpr[4];
            const std::uint32_t alignment = requested_alignment == 0 ? 16U : requested_alignment;
            core.state.gpr[3] = core.allocate_guest_memory(core.state.gpr[3], alignment);
        });
    dispatcher.register_function(
        "coreinit",
        "MEMFreeToDefaultHeap",
        [](EspressoCore&) {
            // The startup probe only frees transient allocations; bump-heap
            // reclamation is deferred until the allocator needs reuse semantics.
        });
    dispatcher.register_function(
        "coreinit",
        "OSFastMutex_Init",
        [](EspressoCore& core) {
            constexpr std::uint32_t fast_mutex_size = 0x2CU;
            constexpr std::uint32_t fast_mutex_tag = 0x664D7458U;
            const std::uint32_t mutex = core.state.gpr[3];
            core.memory.zero_fill(mutex, fast_mutex_size);
            core.memory.write32_be(mutex, fast_mutex_tag);
            core.memory.write32_be(mutex + 4U, core.state.gpr[4]);
        });
    dispatcher.register_function(
        "coreinit",
        "OSFastMutex_Lock",
        [](EspressoCore&) {
            // The current core executes one guest thread, so this lock cannot
            // contend; initialization still writes the guest-visible layout.
        });
    dispatcher.register_function(
        "coreinit",
        "OSFastMutex_Unlock",
        [](EspressoCore&) {
            // Single-threaded counterpart to OSFastMutex_Lock.
        });
    dispatcher.register_function(
        "coreinit",
        "OSGetThreadSpecific",
        [](EspressoCore& core) {
            const auto value = core.thread_specific.find(core.state.gpr[3]);
            core.state.gpr[3] = value == core.thread_specific.end() ? 0U : value->second;
        });
    dispatcher.register_function(
        "coreinit",
        "OSSetThreadSpecific",
        [](EspressoCore& core) {
            core.thread_specific[core.state.gpr[3]] = core.state.gpr[4];
        });
    dispatcher.register_function(
        "coreinit",
        "OSUninterruptibleSpinLock_Acquire",
        [](EspressoCore&) {
            // The current execution session is single-threaded, so this lock
            // cannot contend with another emulated Espresso thread.
        });
    dispatcher.register_function(
        "coreinit",
        "OSUninterruptibleSpinLock_Release",
        [](EspressoCore&) {
            // Matching single-threaded no-op for the lock acquire above.
        });
    dispatcher.register_function(
        "coreinit",
        "OSIsDebuggerInitialized",
        [](EspressoCore& core) {
            // Affogato has no Cafe debugger/GDB transport yet.
            core.state.gpr[3] = 0;
        });
}

}
