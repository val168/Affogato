#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

class HleDispatcher;

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
};

void register_gx2_hle(HleDispatcher& dispatcher);

}
