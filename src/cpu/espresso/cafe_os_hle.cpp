#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/guest_mutex.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <optional>
#include <stdexcept>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t os_system_info_size = 0x20U;
constexpr std::uint32_t os_system_info_bus_clock_speed = 248625000U;
constexpr std::uint32_t os_system_info_core_clock_speed = 1243125000U;
constexpr std::uint32_t os_system_info_timer_clock_speed =
    os_system_info_bus_clock_speed / 4U;
constexpr std::uint32_t os_system_info_l2_core0_size = 512U * 1024U;
constexpr std::uint32_t os_system_info_l2_core1_size = 2U * 1024U * 1024U;
constexpr std::uint32_t os_system_info_l2_core2_size = 512U * 1024U;
constexpr std::uint32_t os_system_info_cpu_ratio = 5U;
constexpr std::int64_t os_system_info_unix_epoch_offset = 946684800;
constexpr std::uint32_t current_core_index = 1U;
constexpr std::uint32_t mem2_base_heap_index = 1U;
constexpr std::uint32_t mem2_expanded_heap_tag = 0x45585048U; // EXPH
constexpr std::uint32_t mem2_expanded_heap_header_size = 0x54U;
constexpr std::uint32_t mem2_expanded_heap_block_header_size = 0x14U;
constexpr std::uint32_t mem2_minimum_heap_size = 0x6CU;
constexpr std::uint32_t default_bump_heap_minimum_remaining = 0x1000U;
constexpr std::uint16_t mem2_free_block_tag = 0x4652U; // 'FR'

static_assert(os_system_info_timer_clock_speed == 62156250U);
static_assert(os_system_info_core_clock_speed / os_system_info_bus_clock_speed ==
              os_system_info_cpu_ratio);

[[nodiscard]] std::optional<std::uint32_t> exception_callback_array_offset(
    std::uint32_t exception_type) noexcept
{
    switch (exception_type)
    {
    case 2U: return 0x630U; // DSI
    case 3U: return 0x63CU; // ISI
    case 5U: return 0x680U; // Alignment
    case 6U: return 0x648U; // Program
    case 11U: return 0x654U; // Performance monitor
    default: return std::nullopt;
    }
}

[[nodiscard]] std::uint64_t initial_os_system_base_time() noexcept
{
    using namespace std::chrono;

    const auto elapsed = system_clock::now().time_since_epoch();
    const auto unix_seconds = duration_cast<seconds>(elapsed).count();
    if (unix_seconds < os_system_info_unix_epoch_offset)
    {
        return 0;
    }

    const auto fractional_nanoseconds =
        duration_cast<nanoseconds>(elapsed - seconds(unix_seconds)).count();
    const auto seconds_since_epoch = static_cast<std::uint64_t>(
        unix_seconds - os_system_info_unix_epoch_offset);
    constexpr std::uint64_t timer_clock = os_system_info_timer_clock_speed;
    constexpr std::uint64_t max_base_time =
        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());

    const std::uint64_t fractional_ticks =
        static_cast<std::uint64_t>(fractional_nanoseconds) * timer_clock /
        1000000000ULL;
    if (seconds_since_epoch > max_base_time / timer_clock)
    {
        return max_base_time;
    }
    const std::uint64_t whole_second_ticks = seconds_since_epoch * timer_clock;
    if (whole_second_ticks > max_base_time - fractional_ticks)
    {
        return max_base_time;
    }
    return whole_second_ticks + fractional_ticks;
}

} // namespace

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

bool initialize_default_guest_heaps(EspressoCore& core)
{
    if (core.base_heap_handles[mem2_base_heap_index] != 0U ||
        core.mem2_heap_region_begin != 0U || core.mem2_heap_region_end != 0U)
    {
        return false;
    }

    const std::uint32_t allocation_begin = core.guest_heap_cursor;
    const std::uint32_t allocation_end = core.guest_heap_limit & ~0xFU;
    if (allocation_begin == 0U || allocation_end <= allocation_begin)
    {
        return false;
    }

    const std::uint64_t available =
        static_cast<std::uint64_t>(allocation_end) - allocation_begin;
    // Reserve one quarter of the loader-selected gap for MEM2 while retaining
    // most of it for existing bump-allocated Cafe OS objects.
    const std::uint64_t desired_mem2_size = std::max<std::uint64_t>(
        mem2_minimum_heap_size,
        (available + 3U) / 4U);
    const std::uint64_t aligned_mem2_size =
        (desired_mem2_size + 0xFU) & ~std::uint64_t{0xFU};
    if (aligned_mem2_size > available)
    {
        return false;
    }

    const std::uint64_t mem2_begin_unaligned =
        static_cast<std::uint64_t>(allocation_end) - aligned_mem2_size;
    const std::uint64_t mem2_begin = mem2_begin_unaligned & ~std::uint64_t{0xFU};
    const std::uint64_t mem2_end = allocation_end;
    if (mem2_begin <= allocation_begin ||
        mem2_begin - allocation_begin < default_bump_heap_minimum_remaining ||
        mem2_end - mem2_begin < mem2_minimum_heap_size ||
        mem2_end > std::numeric_limits<std::uint32_t>::max())
    {
        return false;
    }

    const std::uint64_t data_start =
        mem2_begin + mem2_expanded_heap_header_size;
    const std::uint64_t block_data_start =
        data_start + mem2_expanded_heap_block_header_size;
    if (data_start >= mem2_end || block_data_start >= mem2_end ||
        mem2_end - mem2_begin < mem2_minimum_heap_size)
    {
        return false;
    }

    const auto handle = static_cast<std::uint32_t>(mem2_begin);
    const auto heap_end = static_cast<std::uint32_t>(mem2_end);
    const auto free_block = static_cast<std::uint32_t>(data_start);
    try
    {
        // Validate the whole reserved range before touching guest bytes. The
        // loader selected this interval from a mapped, unoccupied gap.
        core.memory.validate_write_range(
            handle, static_cast<std::size_t>(mem2_end - mem2_begin));
    }
    catch (const GuestMemoryFault&)
    {
        return false;
    }

    // This is a guest-visible EXPH header plus its genuine initial free block.
    // The block occupies only the upper partition; the legacy bump allocator
    // is capped at handle, so the two allocation models cannot overlap.
    core.memory.zero_fill(
        handle,
        mem2_expanded_heap_header_size + mem2_expanded_heap_block_header_size);
    core.memory.write32_be(handle + 0x00U, mem2_expanded_heap_tag);
    core.memory.write32_be(handle + 0x18U, free_block);
    core.memory.write32_be(handle + 0x1CU, heap_end);
    core.memory.write32_be(handle + 0x30U, 0U); // no heap lock in single-lane mode

    core.memory.write32_be(handle + 0x40U, free_block);
    core.memory.write32_be(handle + 0x44U, free_block);
    core.memory.write32_be(handle + 0x48U, 0U);
    core.memory.write32_be(handle + 0x4CU, 0U);
    core.memory.write16_be(handle + 0x50U, 0U); // group ID
    core.memory.write16_be(handle + 0x52U, 0U); // heap attributes

    core.memory.write32_be(free_block + 0x00U, 0U);
    core.memory.write32_be(
        free_block + 0x04U,
        static_cast<std::uint32_t>(mem2_end - block_data_start));
    core.memory.write32_be(free_block + 0x08U, 0U);
    core.memory.write32_be(free_block + 0x0CU, 0U);
    core.memory.write16_be(free_block + 0x10U, mem2_free_block_tag);

    core.guest_heap_limit = handle;
    core.mem2_heap_region_begin = handle;
    core.mem2_heap_region_end = heap_end;
    core.base_heap_handles[mem2_base_heap_index] = handle;
    return true;
}

void register_coreinit_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "coreinit",
        "OSGetSystemInfo",
        [](EspressoCore& core) {
            const auto address = core.hle.bind_data_import(
                "coreinit", "__affogato_internal_OSSystemInfo", core.memory);
            if (!address)
            {
                throw HleExecutionError(
                    "OSGetSystemInfo internal data is not registered");
            }
            core.state.gpr[3] = *address;
        });
    dispatcher.register_function(
        "coreinit",
        "OSGetCurrentThread",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.current_thread_address;
        });
    dispatcher.register_function(
        "coreinit",
        "MEMGetBaseHeapHandle",
        [](EspressoCore& core) {
            constexpr std::size_t base_heap_count = 9U;
            const std::uint32_t type = core.state.gpr[3];
            core.state.gpr[3] = type < base_heap_count
                ? core.base_heap_handles[type]
                : 0U;
        });
    dispatcher.register_function(
        "coreinit",
        "OSSetExceptionCallback",
        [](EspressoCore& core) {
            if (core.current_thread_address == 0U)
            {
                throw HleExecutionError(
                    "OSSetExceptionCallback has no current guest OSThread");
            }

            const std::uint32_t exception_type = core.state.gpr[3];
            const std::uint32_t callback = core.state.gpr[4];
            const auto callback_array_offset =
                exception_callback_array_offset(exception_type);
            if (!callback_array_offset)
            {
                core.state.gpr[3] = 0U;
                return;
            }

            // Affogato currently runs one guest execution lane, corresponding
            // to CPU1 affinity on its default thread. Replace this with real
            // per-core state when multicore scheduling is implemented.
            const std::uint32_t slot = core.current_thread_address +
                *callback_array_offset + current_core_index *
                    static_cast<std::uint32_t>(sizeof(std::uint32_t));
            // With no asynchronous guest interrupt delivery, this replacement
            // is atomic in the current single-threaded execution model.
            const std::uint32_t previous = core.memory.read32_be(slot);
            core.memory.write32_be(slot, callback);
            core.state.gpr[3] = previous;
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
        "memset",
        [](EspressoCore& core) {
            const std::uint32_t destination = core.state.gpr[3];
            const std::uint8_t value = static_cast<std::uint8_t>(core.state.gpr[4]);
            const std::uint32_t size = core.state.gpr[5];
            if (size != 0)
            {
                core.memory.fill_bytes(destination, size, value);
            }
            core.state.gpr[3] = destination;
        });
    dispatcher.register_function(
        "coreinit",
        "memcpy",
        [](EspressoCore& core) {
            const std::uint32_t destination = core.state.gpr[3];
            const std::uint32_t source = core.state.gpr[4];
            const std::uint32_t size = core.state.gpr[5];
            core.memory.copy_bytes(source, destination, size);
            core.state.gpr[3] = destination;
        });
    dispatcher.register_function(
        "coreinit",
        "OSBlockMove",
        [](EspressoCore& core) {
            const std::uint32_t destination = core.state.gpr[3];
            const std::uint32_t source = core.state.gpr[4];
            const std::uint32_t size = core.state.gpr[5];
            const std::uint32_t flush = core.state.gpr[6];

            core.memory.move_bytes(source, destination, size);
            core.state.gpr[3] = destination;

            // Cafe can flush cache state after OSBlockMove. Affogato exposes
            // coherent GuestMemory and has no D-cache/GPU cache model yet, so
            // the flush flag is accepted but has no additional effect.
            (void)flush;
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
        "__ghs_mtx_init",
        [](EspressoCore& core) {
            const std::uint32_t wrapper = core.state.gpr[3];
            if (wrapper == 0)
            {
                throw HleExecutionError("__ghs_mtx_init received a null mutex wrapper");
            }
            static_cast<void>(core.memory.read32_be(wrapper));
            const std::uint32_t mutex = allocate_guest_os_mutex(core);
            if (mutex != 0)
            {
                core.memory.write32_be(wrapper, mutex);
            }
        });
    dispatcher.register_function(
        "coreinit",
        "__ghs_mtx_lock",
        [](EspressoCore& core) {
            const std::uint32_t mutex = core.memory.read32_be(core.state.gpr[3]);
            if (!lock_guest_os_mutex(core, mutex))
            {
                throw HleExecutionError(
                    "__ghs_mtx_lock encountered another guest-thread owner; "
                    "Affogato has no scheduler to wait for it");
            }
        });
    dispatcher.register_function(
        "coreinit",
        "__ghs_mtx_unlock",
        [](EspressoCore& core) {
            const std::uint32_t mutex = core.memory.read32_be(core.state.gpr[3]);
            unlock_guest_os_mutex(core, mutex);
        });
    dispatcher.register_function(
        "coreinit",
        "__ghs_mtx_dst",
        [](EspressoCore& core) {
            const std::uint32_t wrapper = core.state.gpr[3];
            if (wrapper == 0)
            {
                throw HleExecutionError("__ghs_mtx_dst received a null mutex wrapper");
            }
            // The bump allocator cannot reclaim the OSMutex yet. Clear the
            // wrapper so guest code no longer treats the object as live.
            core.memory.write32_be(wrapper, 0);
        });
    dispatcher.register_function(
        "coreinit",
        "__ghs_flock_ptr",
        [](EspressoCore& core) {
            constexpr std::uint32_t iob_entry_size = 0x10U;
            constexpr std::uint32_t iob_count = 20U;
            constexpr std::uint32_t lock_count = 21U;
            const auto iob_base = core.hle.bind_data_import("coreinit", "_iob", core.memory);
            const auto lock_base =
                core.hle.bind_data_import("coreinit", "_iob_lock", core.memory);
            if (!iob_base || !lock_base)
            {
                throw HleExecutionError(
                    "__ghs_flock_ptr requires registered coreinit::_iob and coreinit::_iob_lock data exports");
            }

            const std::uint32_t iob = core.state.gpr[3];
            if (iob < *iob_base)
            {
                throw HleExecutionError(
                    "__ghs_flock_ptr received a pointer below the coreinit::_iob array");
            }
            const std::uint32_t delta = iob - *iob_base;
            if (delta < iob_count * iob_entry_size && delta % iob_entry_size != 0)
            {
                throw HleExecutionError(
                    "__ghs_flock_ptr received a misaligned pointer inside coreinit::_iob");
            }

            const std::uint32_t index = std::min(delta / iob_entry_size, lock_count - 1U);
            core.state.gpr[3] = *lock_base + index * sizeof(std::uint32_t);
        });
    dispatcher.register_function(
        "coreinit",
        "OSInitMutex",
        [](EspressoCore& core) {
            initialize_guest_os_mutex(core, core.state.gpr[3], core.state.gpr[4]);
        });
    dispatcher.register_function(
        "coreinit",
        "OSLockMutex",
        [](EspressoCore& core) {
            if (!lock_guest_os_mutex(core, core.state.gpr[3]))
            {
                throw HleExecutionError(
                    "OSLockMutex encountered another guest-thread owner; "
                    "Affogato has no scheduler to wait for it");
            }
        });
    dispatcher.register_function(
        "coreinit",
        "OSUnlockMutex",
        [](EspressoCore& core) {
            unlock_guest_os_mutex(core, core.state.gpr[3]);
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

    dispatcher.register_data(
        "coreinit",
        "_iob",
        20U * 0x10U,
        8U,
        [](GuestMemory& memory, std::uint32_t address) {
            constexpr std::uint32_t info_offset = 0x0CU;
            constexpr std::uint32_t readable = 1U << 2U;
            constexpr std::uint32_t writable = 1U << 1U;
            constexpr std::uint32_t channel_shift = 18U;
            memory.write32_be(address + info_offset, readable);
            memory.write32_be(address + 0x10U + info_offset,
                               writable | (1U << channel_shift));
            memory.write32_be(address + 0x20U + info_offset,
                               writable | (2U << channel_shift));
        });
    dispatcher.register_data("coreinit", "_iob_lock", 21U * sizeof(std::uint32_t), 4U);
    dispatcher.register_data(
        "coreinit",
        "__gh_FOPEN_MAX",
        sizeof(std::uint16_t),
        alignof(std::uint16_t),
        [](GuestMemory& memory, std::uint32_t address) {
            memory.write16_be(address, 20U);
        });
    dispatcher.register_data(
        "coreinit",
        "__affogato_internal_OSSystemInfo",
        os_system_info_size,
        8U,
        [](GuestMemory& memory, std::uint32_t address) {
            // Cafe baseTime is timer-clock ticks since 2000-01-01 UTC. This
            // one-time host-clock snapshot is a scoped approximation until
            // Affogato models the Espresso time base and OSGetTime family.
            const std::uint64_t base_time = initial_os_system_base_time();
            memory.write32_be(address + 0x00U, os_system_info_bus_clock_speed);
            memory.write32_be(address + 0x04U, os_system_info_core_clock_speed);
            memory.write64_be(address + 0x08U, base_time);
            memory.write32_be(address + 0x10U, os_system_info_l2_core0_size);
            memory.write32_be(address + 0x14U, os_system_info_l2_core1_size);
            memory.write32_be(address + 0x18U, os_system_info_l2_core2_size);
            memory.write32_be(address + 0x1CU, os_system_info_cpu_ratio);
        });
}

}
