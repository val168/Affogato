#pragma once

#include <cstdint>
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
    std::vector<Gx2Command> pending_commands;
};

void register_gx2_hle(HleDispatcher& dispatcher);

}
