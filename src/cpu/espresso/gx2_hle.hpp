#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <vector>

namespace affogato::cpu::espresso
{

class HleDispatcher;

enum class Gx2CommandType
{
    set_context_register,
    set_shader_resource,
    set_shader_sampler,
    flush,
    draw_done,
    gpu_fence,
};

struct Gx2Command
{
    Gx2CommandType type{};
    std::uint32_t register_address{};
    std::uint32_t value{};
    std::uint32_t guest_memory_address{};
    std::uint32_t mask{};
    std::uint32_t compare_op{};

    bool operator==(const Gx2Command&) const = default;
};

struct Gx2ClearColorRequest
{
    std::uint32_t color_buffer{};
    std::array<float, 4U> rgba{};

    bool operator==(const Gx2ClearColorRequest&) const = default;
};

struct Gx2EventCallback
{
    std::uint32_t function{};
    std::uint32_t user_data{};

    bool operator==(const Gx2EventCallback&) const = default;
};

struct Gx2TextureBinding
{
    std::uint32_t texture_address{};
    std::array<std::uint32_t, 7U> registers{};

    bool operator==(const Gx2TextureBinding&) const = default;
};

struct Gx2SamplerBinding
{
    std::uint32_t sampler_address{};
    std::array<std::uint32_t, 3U> registers{};

    bool operator==(const Gx2SamplerBinding&) const = default;
};

struct Gx2ColorBufferState
{
    std::uint32_t guest_address{};
    std::uint32_t dimension{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth{};
    std::uint32_t mip_levels{};
    std::uint32_t format{};
    std::uint32_t aa_mode{};
    std::uint32_t use{};
    std::uint32_t image_size{};
    std::uint32_t image_address{};
    std::uint32_t mipmap_size{};
    std::uint32_t mipmap_address{};
    std::uint32_t tile_mode{};
    std::uint32_t swizzle{};
    std::uint32_t alignment{};
    std::uint32_t pitch{};
    std::array<std::uint32_t, 13U> mip_level_offsets{};
    std::uint32_t view_mip{};
    std::uint32_t view_first_slice{};
    std::uint32_t view_num_slices{};
    std::uint32_t aa_buffer_address{};
    std::uint32_t aa_size{};
    std::array<std::uint32_t, 5U> registers{};

    bool operator==(const Gx2ColorBufferState&) const = default;
};

struct Gx2RuntimeState
{
    bool initialized{};
    bool gx2r_uses_default_allocator{};
    std::uint32_t gx2r_allocate_function{};
    std::uint32_t gx2r_free_function{};
    std::uint32_t default_allocate_function{};
    std::uint32_t default_free_function{};
    std::uint32_t main_core_id{};
    std::uint32_t command_buffer_pool_base{};
    std::uint32_t command_buffer_pool_size{};
    bool command_buffer_pool_owned{};
    std::uint32_t argc{};
    std::uint32_t argv{};
    std::uint32_t profile_mode{};
    std::uint32_t toss_stage{};
    std::uint32_t app_io_thread_stack_size{};
    std::uint32_t gpu_timeout_ms{};
    std::uint32_t hang_state{};
    std::uint32_t hang_response{};
    std::uint32_t hang_reset_swap_timeout{};
    std::uint32_t hang_reset_swaps_outstanding{};
    std::uint32_t swap_interval{};
    // Headless single-lane virtual display cadence. This models the wait
    // boundary only; it does not imply that a guest frame was rendered.
    std::chrono::steady_clock::time_point next_vsync_deadline{};
    bool vsync_deadline_initialized{};
    std::uint64_t vsync_count{};
    std::uint64_t last_vsync_time{};
    std::uint64_t last_flip_time{};
    std::uint32_t flip_request_count{};
    std::uint32_t flip_execute_count{};
    // GX2EventType values 2 (vsync), 3 (flip), and 4 (display-list overrun).
    std::array<Gx2EventCallback, 3U> event_callbacks{};
    bool tv_scan_buffer_configured{};
    std::uint32_t tv_scan_buffer_address{};
    std::uint32_t tv_scan_buffer_size{};
    std::uint32_t tv_render_mode{};
    std::uint32_t tv_surface_format{};
    std::uint32_t tv_buffering_mode{};
    std::uint32_t tv_scan_width{};
    std::uint32_t tv_scan_height{};
    // An unset value means the output service has not defined an initial
    // enable state. This request alone does not imply pixel presentation.
    std::optional<bool> tv_output_enabled;
    bool tv_scale_configured{};
    std::uint32_t tv_scale_width{};
    std::uint32_t tv_scale_height{};
    bool drc_scan_buffer_configured{};
    std::uint32_t drc_scan_buffer_address{};
    std::uint32_t drc_scan_buffer_size{};
    std::uint32_t drc_render_mode{};
    std::uint32_t drc_surface_format{};
    std::uint32_t drc_buffering_mode{};
    std::uint32_t drc_scan_width{};
    std::uint32_t drc_scan_height{};
    std::uint32_t drc_scan_pitch{};
    std::optional<bool> drc_output_enabled;
    bool drc_scale_configured{};
    std::uint32_t drc_scale_width{};
    std::uint32_t drc_scale_height{};
    bool db_depth_control_valid{};
    std::uint32_t db_depth_control{};
    bool db_stencilrefmask_valid{};
    std::uint32_t db_stencilrefmask{};
    bool db_stencilrefmask_bf_valid{};
    std::uint32_t db_stencilrefmask_bf{};
    bool pa_su_sc_mode_cntl_valid{};
    std::uint32_t pa_su_sc_mode_cntl{};
    bool cb_color_control_valid{};
    std::uint32_t cb_color_control{};
    std::array<bool, 8U> cb_blend_control_valid{};
    std::array<std::uint32_t, 8U> cb_blend_control{};
    std::array<bool, 4U> cb_blend_constant_valid{};
    std::array<std::uint32_t, 4U> cb_blend_constant{};
    std::array<std::optional<Gx2ColorBufferState>, 8U> color_buffer_descriptors{};
    std::array<std::uint32_t, 8U> bound_color_buffers{};
    std::array<std::optional<Gx2TextureBinding>, 16U> pixel_textures{};
    std::array<std::optional<Gx2TextureBinding>, 16U> vertex_textures{};
    std::array<std::optional<Gx2TextureBinding>, 16U> geometry_textures{};
    std::array<std::optional<Gx2SamplerBinding>, 16U> pixel_samplers{};
    std::array<std::optional<Gx2SamplerBinding>, 16U> vertex_samplers{};
    std::array<std::optional<Gx2SamplerBinding>, 16U> geometry_samplers{};
    std::array<std::optional<std::array<std::uint32_t, 4U>>, 12U>
        pixel_sampler_border_colors{};
    std::array<std::optional<std::array<std::uint32_t, 4U>>, 12U>
        vertex_sampler_border_colors{};
    std::array<std::optional<std::array<std::uint32_t, 4U>>, 12U>
        geometry_sampler_border_colors{};
    std::uint64_t clear_color_sequence{};
    std::optional<Gx2ClearColorRequest> latest_clear_color;
    bool sx_alpha_test_control_valid{};
    std::uint32_t sx_alpha_test_control{};
    bool sx_alpha_ref_valid{};
    std::uint32_t sx_alpha_ref{};
    bool cb_target_mask_valid{};
    std::uint32_t cb_target_mask{};
    bool db_alpha_to_mask_valid{};
    std::uint32_t db_alpha_to_mask{};
    bool context_state_shadowing_enabled{};
    std::uint32_t current_context_state{};
    std::uint32_t current_context_state_flags{};
    bool context_state_profiling_enabled{};
    bool context_state_shadow_display_list_requested{};
    std::vector<Gx2Command> pending_commands;

    bool operator==(const Gx2RuntimeState&) const = default;
};

void register_gx2_hle(HleDispatcher& dispatcher);
[[nodiscard]] std::vector<Gx2Command> take_gx2_commands(
    Gx2RuntimeState& state);

}
