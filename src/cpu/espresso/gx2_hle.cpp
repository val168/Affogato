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
constexpr std::uint32_t db_depth_control_register = 0x00028800U;
constexpr std::uint32_t db_stencilrefmask_register = 0x00028430U;
constexpr std::uint32_t db_stencilrefmask_bf_register = 0x00028434U;
constexpr std::uint32_t pa_su_sc_mode_cntl_register = 0x00028814U;
constexpr std::uint32_t cb_color_control_register = 0x00028808U;

std::uint32_t pack_stencil_ref_mask(
    std::uint32_t mask,
    std::uint32_t write_mask,
    std::uint32_t reference)
{
    return (reference & 0xFFU) |
           ((mask & 0xFFU) << 8U) |
           ((write_mask & 0xFFU) << 16U);
}

std::uint32_t pack_polygon_control(
    std::uint32_t front_face,
    std::uint32_t cull_front,
    std::uint32_t cull_back,
    std::uint32_t polygon_mode,
    std::uint32_t polygon_mode_front,
    std::uint32_t polygon_mode_back,
    std::uint32_t polygon_offset_front_enable,
    std::uint32_t polygon_offset_back_enable,
    std::uint32_t polygon_offset_parallel_enable)
{
    const auto boolean_bit = [](std::uint32_t value) {
        return value != 0U ? 1U : 0U;
    };

    return boolean_bit(cull_front) |
           (boolean_bit(cull_back) << 1U) |
           (boolean_bit(front_face) << 2U) |
           ((polygon_mode & 0x3U) << 3U) |
           ((polygon_mode_front & 0x7U) << 5U) |
           ((polygon_mode_back & 0x7U) << 8U) |
           (boolean_bit(polygon_offset_front_enable) << 11U) |
           (boolean_bit(polygon_offset_back_enable) << 12U) |
           (boolean_bit(polygon_offset_parallel_enable) << 13U);
}

std::uint32_t pack_color_control(
    std::uint32_t rop3,
    std::uint32_t target_blend_enable,
    std::uint32_t multi_write_enable,
    std::uint32_t color_write_enable)
{
    const std::uint32_t multi_write = multi_write_enable != 0U ? 1U : 0U;
    // Cafe maps disabled color writes to SPECIAL_OP::DISABLE; enabled writes
    // select NORMAL. CB_COLOR_CONTROL has no separate color-write bit.
    const std::uint32_t special_op = color_write_enable != 0U ? 0U : 1U;

    return (multi_write << 1U) |
           (special_op << 4U) |
           ((target_blend_enable & 0xFFU) << 8U) |
           ((rop3 & 0xFFU) << 16U);
}

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

    dispatcher.register_function(
        "gx2",
        "GX2SetDepthStencilControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthStencilControl: GX2 has not been initialized");
            }

            const std::uint32_t depth_test = core.state.gpr[3];
            const std::uint32_t depth_write = core.state.gpr[4];
            const std::uint32_t depth_compare = core.state.gpr[5];
            const std::uint32_t stencil_test = core.state.gpr[6];
            const std::uint32_t backface_stencil = core.state.gpr[7];
            const std::uint32_t front_stencil_func = core.state.gpr[8];
            const std::uint32_t front_stencil_zpass = core.state.gpr[9];
            const std::uint32_t front_stencil_zfail = core.state.gpr[10];

            constexpr std::uint32_t stack_argument_base_offset = 0x08U;
            constexpr std::uint32_t stack_argument_last_offset = 0x18U;
            const std::uint64_t stack_pointer = core.state.gpr[1];
            const std::uint64_t last_stack_argument =
                stack_pointer + stack_argument_last_offset;
            if (last_stack_argument + sizeof(std::uint32_t) > guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthStencilControl: stack arguments wrap guest address space");
            }

            // Read every stack-passed argument before changing GX2 state. A
            // failed guest read therefore leaves the register mirror/queue intact.
            const std::uint32_t front_stencil_fail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + stack_argument_base_offset));
            const std::uint32_t back_stencil_func = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x0CU));
            const std::uint32_t back_stencil_zpass = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x10U));
            const std::uint32_t back_stencil_zfail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x14U));
            const std::uint32_t back_stencil_fail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + stack_argument_last_offset));

            const auto boolean_bit = [](std::uint32_t value) {
                return value != 0U ? 1U : 0U;
            };
            const std::uint32_t packed =
                boolean_bit(stencil_test) |
                (boolean_bit(depth_test) << 1U) |
                (boolean_bit(depth_write) << 2U) |
                ((depth_compare & 0x7U) << 4U) |
                (boolean_bit(backface_stencil) << 7U) |
                ((front_stencil_func & 0x7U) << 8U) |
                ((front_stencil_fail & 0x7U) << 11U) |
                ((front_stencil_zpass & 0x7U) << 14U) |
                ((front_stencil_zfail & 0x7U) << 17U) |
                ((back_stencil_func & 0x7U) << 20U) |
                ((back_stencil_fail & 0x7U) << 23U) |
                ((back_stencil_zpass & 0x7U) << 26U) |
                ((back_stencil_zfail & 0x7U) << 29U);

            core.gx2.db_depth_control = packed;
            core.gx2.db_depth_control_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_depth_control_register,
                packed});
            // GX2SetDepthStencilControl is void. Guest arguments and all CPU
            // state are preserved; the dispatcher returns through LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetStencilMask",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetStencilMask: GX2 has not been initialized");
            }

            const std::uint32_t front_mask = core.state.gpr[3] & 0xFFU;
            const std::uint32_t front_write_mask = core.state.gpr[4] & 0xFFU;
            const std::uint32_t front_reference = core.state.gpr[5] & 0xFFU;
            const std::uint32_t back_mask = core.state.gpr[6] & 0xFFU;
            const std::uint32_t back_write_mask = core.state.gpr[7] & 0xFFU;
            const std::uint32_t back_reference = core.state.gpr[8] & 0xFFU;

            // Compute the complete pair before mutating the current register
            // mirrors or appending either command.
            const std::uint32_t front_value = pack_stencil_ref_mask(
                front_mask, front_write_mask, front_reference);
            const std::uint32_t back_value = pack_stencil_ref_mask(
                back_mask, back_write_mask, back_reference);

            core.gx2.db_stencilrefmask = front_value;
            core.gx2.db_stencilrefmask_valid = true;
            core.gx2.db_stencilrefmask_bf = back_value;
            core.gx2.db_stencilrefmask_bf_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_stencilrefmask_register,
                front_value});
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_stencilrefmask_bf_register,
                back_value});
            // This API is void: preserve all guest CPU state and return via LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPolygonControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPolygonControl: GX2 has not been initialized");
            }

            const std::uint32_t front_face = core.state.gpr[3];
            const std::uint32_t cull_front = core.state.gpr[4];
            const std::uint32_t cull_back = core.state.gpr[5];
            const std::uint32_t polygon_mode = core.state.gpr[6];
            const std::uint32_t polygon_mode_front = core.state.gpr[7];
            const std::uint32_t polygon_mode_back = core.state.gpr[8];
            const std::uint32_t polygon_offset_front_enable = core.state.gpr[9];
            const std::uint32_t polygon_offset_back_enable = core.state.gpr[10];

            constexpr std::uint32_t stack_argument_offset = 0x08U;
            const std::uint64_t stack_pointer = core.state.gpr[1];
            const std::uint64_t stack_argument_address =
                stack_pointer + stack_argument_offset;
            if (stack_argument_address + sizeof(std::uint32_t) > guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPolygonControl: stack argument wraps guest address space");
            }

            // Gather the ninth ABI argument before touching the register mirror
            // or command queue. GPR11 is not an argument register here.
            const std::uint32_t polygon_offset_parallel_enable =
                core.memory.read32_be(static_cast<std::uint32_t>(stack_argument_address));
            const std::uint32_t packed = pack_polygon_control(
                front_face,
                cull_front,
                cull_back,
                polygon_mode,
                polygon_mode_front,
                polygon_mode_back,
                polygon_offset_front_enable,
                polygon_offset_back_enable,
                polygon_offset_parallel_enable);

            core.gx2.pa_su_sc_mode_cntl = packed;
            core.gx2.pa_su_sc_mode_cntl_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_sc_mode_cntl_register,
                packed});
            // This API is void: preserve all guest CPU state and return via LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetColorControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetColorControl: GX2 has not been initialized");
            }

            const std::uint32_t rop3 = core.state.gpr[3];
            const std::uint32_t target_blend_enable = core.state.gpr[4];
            const std::uint32_t multi_write_enable = core.state.gpr[5];
            const std::uint32_t color_write_enable = core.state.gpr[6];
            const std::uint32_t packed = pack_color_control(
                rop3,
                target_blend_enable,
                multi_write_enable,
                color_write_enable);

            core.gx2.cb_color_control = packed;
            core.gx2.cb_color_control_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                cb_color_control_register,
                packed});
            // This void setter records logical Latte state only. PM4 encoding,
            // command-pool writes, and renderer translation remain deferred.
        });
}

} // namespace affogato::cpu::espresso
