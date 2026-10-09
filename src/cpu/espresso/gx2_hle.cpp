#include "cpu/espresso/gx2_hle.hpp"

#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <cstdint>
#include <string>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t gx2_init_end = 0U;
constexpr std::uint32_t gx2_init_command_buffer_base = 1U;
constexpr std::uint32_t gx2_init_command_buffer_pool_size = 2U;
constexpr std::uint32_t gx2_init_argc = 7U;
constexpr std::uint32_t gx2_init_argv = 8U;
constexpr std::uint32_t gx2_init_profile_mode = 9U;
constexpr std::uint32_t gx2_init_toss_stage = 10U;
constexpr std::uint32_t gx2_init_app_io_thread_stack_size = 11U;

constexpr std::uint32_t default_command_pool_size = 0x00400000U;
constexpr std::uint32_t minimum_command_pool_size = 0x2000U;
constexpr std::uint32_t command_pool_alignment = 0x100U;
constexpr std::uint32_t command_buffer_alignment = 0x40U;
constexpr std::uint32_t default_app_io_stack_size = 0x1000U;
constexpr std::size_t maximum_attribute_pairs = 64U;
constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;

struct ParsedGx2Attributes
{
    std::uint32_t command_pool_base{};
    std::uint32_t command_pool_size{default_command_pool_size};
    bool command_pool_size_present{};
    std::uint32_t argc{};
    std::uint32_t argv{};
    std::uint32_t profile_mode{};
    std::uint32_t toss_stage{};
    std::uint32_t app_io_thread_stack_size{default_app_io_stack_size};
};

[[noreturn]] void gx2_init_error(const std::string& message)
{
    throw HleExecutionError("gx2::GX2Init: " + message);
}

ParsedGx2Attributes parse_attributes(GuestMemory& memory, std::uint32_t address)
{
    ParsedGx2Attributes parsed;
    if (address == 0U)
    {
        return parsed;
    }

    std::uint64_t cursor = address;
    std::size_t pairs = 0U;
    while (true)
    {
        if (cursor > guest_address_space_end - sizeof(std::uint32_t))
        {
            gx2_init_error("attribute stream address wraps the guest address space");
        }

        std::uint32_t id = 0U;
        try
        {
            id = memory.read32_be(static_cast<std::uint32_t>(cursor));
        }
        catch (const GuestMemoryFault&)
        {
            gx2_init_error("could not read an attribute ID from guest memory");
        }
        if (id == gx2_init_end)
        {
            return parsed;
        }
        if (pairs >= maximum_attribute_pairs)
        {
            gx2_init_error("attribute stream exceeded the 64-pair safety limit without GX2_INIT_END");
        }

        cursor += sizeof(std::uint32_t);
        if (cursor > guest_address_space_end - sizeof(std::uint32_t))
        {
            gx2_init_error("attribute value address wraps the guest address space");
        }

        std::uint32_t value = 0U;
        try
        {
            value = memory.read32_be(static_cast<std::uint32_t>(cursor));
        }
        catch (const GuestMemoryFault&)
        {
            gx2_init_error("attribute ID was readable but its value was not mapped");
        }

        switch (id)
        {
        case gx2_init_command_buffer_base:
            parsed.command_pool_base = value;
            break;
        case gx2_init_command_buffer_pool_size:
            parsed.command_pool_size = value;
            parsed.command_pool_size_present = true;
            break;
        case gx2_init_argc:
            parsed.argc = value;
            break;
        case gx2_init_argv:
            parsed.argv = value;
            break;
        case gx2_init_profile_mode:
            parsed.profile_mode = value;
            break;
        case gx2_init_toss_stage:
            parsed.toss_stage = value;
            break;
        case gx2_init_app_io_thread_stack_size:
            parsed.app_io_thread_stack_size = value;
            break;
        default:
            // Cafe/Decaf consume unknown ID/value pairs and continue.
            break;
        }

        ++pairs;
        cursor += sizeof(std::uint32_t);
    }
}

Gx2RuntimeState initialize_gx2(EspressoCore& core, std::uint32_t attribute_address)
{
    ParsedGx2Attributes parsed = parse_attributes(core.memory, attribute_address);
    parsed.command_pool_size = parsed.command_pool_size < minimum_command_pool_size
        ? minimum_command_pool_size
        : parsed.command_pool_size;

    const bool owned = parsed.command_pool_base == 0U;
    const std::uint32_t old_heap_cursor = core.guest_heap_cursor;
    std::uint32_t pool_base = parsed.command_pool_base;
    if (owned)
    {
        pool_base = core.allocate_guest_memory(
            parsed.command_pool_size, command_pool_alignment);
        if (pool_base == 0U)
        {
            core.guest_heap_cursor = old_heap_cursor;
            gx2_init_error("could not allocate the command-buffer pool from the default guest heap");
        }
    }

    const std::uint64_t pool_end =
        static_cast<std::uint64_t>(pool_base) + parsed.command_pool_size;
    if (pool_end > guest_address_space_end)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool range wraps the guest address space");
    }
    if ((pool_base & (command_buffer_alignment - 1U)) != 0U)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool base is not 0x40-byte aligned");
    }

    try
    {
        core.memory.validate_write_range(pool_base, parsed.command_pool_size);
    }
    catch (const GuestMemoryFault&)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool range is not fully mapped and writable");
    }

    // The command pool is registered, not initialized: command emission and
    // Latte/PM4 execution are intentionally deferred to later GX2 work.
    Gx2RuntimeState state;
    state.initialized = true;
    // Affogato currently has one Espresso execution lane corresponding to CPU1.
    // This becomes dynamic when multiple Espresso cores are modeled.
    state.main_core_id = 1U;
    state.command_buffer_pool_base = pool_base;
    state.command_buffer_pool_size = parsed.command_pool_size;
    state.command_buffer_pool_owned = owned;
    state.argc = parsed.argc;
    state.argv = parsed.argv;
    state.profile_mode = parsed.profile_mode;
    state.toss_stage = parsed.toss_stage;
    // The AppIO stack is recorded but not carved from the pool until its
    // thread/event machinery exists.
    state.app_io_thread_stack_size = parsed.app_io_thread_stack_size;
    state.gpu_timeout_ms = 10'000U;
    state.hang_response = 1U;
    state.hang_reset_swap_timeout = 1000U;
    state.hang_reset_swaps_outstanding = 3U;
    state.swap_interval = 1U;
    return state;
}

} // namespace

void register_gx2_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "gx2",
        "GX2Init",
        [](EspressoCore& core) {
            // Cemu treats repeated GX2Init calls as successful no-ops. Check
            // this before touching r3 so even an invalid later pointer is safe.
            if (core.gx2.initialized)
            {
                return;
            }

            const std::uint32_t attributes = core.state.gpr[3];
            Gx2RuntimeState initialized = initialize_gx2(core, attributes);
            core.gx2 = initialized;
            // GX2Init is void: preserve all guest GPRs including r3.
        });
}

} // namespace affogato::cpu::espresso
