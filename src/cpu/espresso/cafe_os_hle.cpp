#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <stdexcept>

namespace affogato::cpu::espresso
{

std::uint32_t initialize_default_guest_thread(
    EspressoCore& core,
    std::uint32_t stack_start,
    std::uint32_t stack_end)
{
    constexpr std::uint32_t thread_size = 0x6A0U;
    constexpr std::uint32_t thread_tag_offset = 0x320U;
    constexpr std::uint32_t thread_state_offset = 0x324U;
    constexpr std::uint32_t thread_attributes_offset = 0x325U;
    constexpr std::uint32_t thread_id_offset = 0x326U;
    constexpr std::uint32_t thread_stack_start_offset = 0x394U;
    constexpr std::uint32_t thread_stack_end_offset = 0x398U;
    constexpr std::uint32_t thread_type_offset = 0x5BCU;
    constexpr std::uint32_t thread_tag = 0x74487244U;
    constexpr std::uint8_t thread_state_running = 1U << 1U;
    constexpr std::uint8_t thread_affinity_cpu1 = 1U << 1U;
    constexpr std::uint16_t default_thread_id = 1U;
    constexpr std::uint32_t thread_type_application = 2U;

    if (core.current_thread_address != 0)
    {
        throw std::logic_error("the default guest thread is already initialized");
    }
    if (stack_end >= stack_start)
    {
        throw std::invalid_argument("default guest thread stack bounds are invalid");
    }

    const std::uint32_t address = core.allocate_guest_memory(thread_size, 8U);
    if (address == 0)
    {
        throw std::runtime_error(
            "guest heap cannot fit the default Cafe OSThread object (cursor " +
            std::to_string(core.guest_heap_cursor) + ", limit " +
            std::to_string(core.guest_heap_limit) + ")");
    }

    core.memory.zero_fill(address, thread_size);
    core.memory.write32_be(address + thread_tag_offset, thread_tag);
    core.memory.write8(address + thread_state_offset, thread_state_running);
    core.memory.write8(address + thread_attributes_offset, thread_affinity_cpu1);
    core.memory.write16_be(address + thread_id_offset, default_thread_id);
    core.memory.write32_be(address + thread_stack_start_offset, stack_start);
    core.memory.write32_be(address + thread_stack_end_offset, stack_end);
    core.memory.write32_be(address + thread_type_offset, thread_type_application);
    core.current_thread_address = address;
    return address;
}

void register_coreinit_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "coreinit",
        "OSGetCurrentThread",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.current_thread_address;
        });
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
        "__ghsLock",
        [](EspressoCore&) {
            // Cafe conceptually acquires a global Green Hills runtime lock.
            // This paired no-op is only valid while Affogato runs one guest
            // thread; replace it with guest synchronization if threading lands.
        });
    dispatcher.register_function(
        "coreinit",
        "__ghsUnlock",
        [](EspressoCore&) {
            // Temporary single-thread counterpart to __ghsLock. It does not
            // provide mutual exclusion once multiple guest threads exist.
        });
    dispatcher.register_function(
        "coreinit",
        "OSGetThreadSpecific",
        [](EspressoCore& core) {
            constexpr std::uint32_t specific_slot_count = 16U;
            constexpr std::uint32_t specific_offset = 0x57CU;
            const std::uint32_t id = core.state.gpr[3];
            if (core.current_thread_address == 0 || id >= specific_slot_count)
            {
                core.state.gpr[3] = 0;
                return;
            }
            core.state.gpr[3] = core.memory.read32_be(
                core.current_thread_address + specific_offset + id * sizeof(std::uint32_t));
        });
    dispatcher.register_function(
        "coreinit",
        "OSSetThreadSpecific",
        [](EspressoCore& core) {
            constexpr std::uint32_t specific_slot_count = 16U;
            constexpr std::uint32_t specific_offset = 0x57CU;
            const std::uint32_t id = core.state.gpr[3];
            if (core.current_thread_address != 0 && id < specific_slot_count)
            {
                core.memory.write32_be(
                    core.current_thread_address + specific_offset + id * sizeof(std::uint32_t),
                    core.state.gpr[4]);
            }
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
