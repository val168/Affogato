#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

namespace affogato::cpu::espresso
{

class HleDispatcher;

enum class Gx2CommandType
{
    set_context_register,
};

struct Gx2Command
{
    Gx2CommandType type{};
    std::uint32_t register_address{};
    std::uint32_t value{};

    bool operator==(const Gx2Command&) const = default;
};

struct Gx2ClearColorRequest
{
    std::uint32_t color_buffer{};
    std::array<float, 4U> rgba{};

    bool operator==(const Gx2ClearColorRequest&) const = default;
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
    std::uint32_t flip_request_count{};
    std::uint32_t flip_execute_count{};
    bool tv_scan_buffer_configured{};
    std::uint32_t tv_scan_buffer_address{};
    std::uint32_t tv_scan_buffer_size{};
    std::uint32_t tv_render_mode{};
    std::uint32_t tv_surface_format{};
    std::uint32_t tv_buffering_mode{};
    std::uint32_t tv_scan_width{};
    std::uint32_t tv_scan_height{};
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

}
