#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/guest_mutex.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <filesystem>
#include <limits>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>

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
constexpr std::uint32_t os_thread_size = 0x6A0U;
constexpr std::uint64_t os_context_tag = 0x4F53436F6E747874ULL;
constexpr std::uint32_t os_thread_tag_offset = 0x320U;
constexpr std::uint32_t os_thread_state_offset = 0x324U;
constexpr std::uint32_t os_thread_attributes_offset = 0x325U;
constexpr std::uint32_t os_thread_id_offset = 0x326U;
constexpr std::uint32_t os_thread_suspend_count_offset = 0x328U;
constexpr std::uint32_t os_thread_priority_offset = 0x32CU;
constexpr std::uint32_t os_thread_base_priority_offset = 0x330U;
constexpr std::uint32_t os_thread_exit_value_offset = 0x334U;
constexpr std::uint32_t os_thread_stack_start_offset = 0x394U;
constexpr std::uint32_t os_thread_stack_end_offset = 0x398U;
constexpr std::uint32_t os_thread_entry_point_offset = 0x39CU;
constexpr std::uint32_t os_thread_specific_offset = 0x578U;
constexpr std::uint32_t os_thread_type_offset = 0x5BCU;
constexpr std::uint32_t os_thread_tag = 0x74487244U; // tHrD
constexpr std::uint32_t os_thread_ready = 1U;
constexpr std::uint32_t os_thread_type_application = 2U;
constexpr std::uint32_t os_thread_id_unassigned = 0x8000U;
constexpr std::uint32_t os_context_gpr_offset = 0x08U;
constexpr std::uint32_t os_context_cr_offset = 0x88U;
constexpr std::uint32_t os_context_lr_offset = 0x8CU;
constexpr std::uint32_t os_context_ctr_offset = 0x90U;
constexpr std::uint32_t os_context_xer_offset = 0x94U;
constexpr std::uint32_t os_context_srr0_offset = 0x98U;
constexpr std::uint32_t os_context_srr1_offset = 0x9CU;
constexpr std::uint32_t os_context_fpscr_offset = 0xB4U;
constexpr std::uint32_t os_context_fpr_offset = 0xB8U;
constexpr std::uint32_t os_context_gqr_offset = 0x1BCU;
constexpr std::uint32_t os_context_upir_offset = 0x1DCU;
constexpr std::uint32_t os_context_psf_offset = 0x1E0U;
constexpr std::uint32_t os_context_affinity_offset = 0x304U;
constexpr std::uint32_t os_context_size = 0x320U;
constexpr std::uint8_t os_thread_running = 1U << 1U;
constexpr std::uint8_t os_thread_waiting = 1U << 2U;
constexpr std::int32_t os_thread_default_app_priority = 16;
constexpr std::int32_t os_thread_app_priority_base = 64;
constexpr std::int32_t os_thread_default_internal_priority =
    os_thread_app_priority_base + os_thread_default_app_priority;
constexpr std::uint32_t os_message_queue_size = 0x3CU;
constexpr std::uint32_t os_message_queue_tag = 0x6D536751U; // mSgQ
constexpr std::uint32_t os_message_queue_send_queue_offset = 0x0CU;
constexpr std::uint32_t os_message_queue_recv_queue_offset = 0x1CU;
constexpr std::uint32_t os_message_queue_messages_offset = 0x2CU;
constexpr std::uint32_t os_message_queue_capacity_offset = 0x30U;
constexpr std::uint32_t os_message_queue_first_offset = 0x34U;
constexpr std::uint32_t os_message_queue_used_offset = 0x38U;
constexpr std::uint32_t kpad_unified_wpad_status_size = 0x44U;
constexpr std::uint32_t kpad_mpls_work_size = 0x5FE0U;
constexpr std::uint32_t os_message_size = 0x10U;
constexpr std::uint32_t os_message_flag_blocking = 1U << 0U;
constexpr std::uint32_t os_message_flag_high_priority = 1U << 1U;
constexpr std::uint32_t os_thread_queue_head_offset = 0x00U;
constexpr std::uint32_t os_thread_queue_tail_offset = 0x04U;
constexpr std::uint32_t os_thread_queue_parent_offset = 0x08U;
constexpr std::uint32_t mem2_base_heap_index = 1U;
constexpr std::uint32_t foreground_base_heap_index = 8U;
constexpr std::uint32_t mem2_expanded_heap_tag = 0x45585048U; // EXPH
constexpr std::uint32_t mem2_expanded_heap_header_size = 0x54U;
constexpr std::uint32_t mem2_expanded_heap_block_header_size = 0x14U;
constexpr std::uint32_t foreground_frame_heap_tag = 0x46524D48U; // FRMH
constexpr std::uint32_t foreground_frame_heap_header_size = 0x4CU;
constexpr std::uint32_t mem_exp_heap_data_start_offset = 0x18U;
constexpr std::uint32_t mem_exp_heap_data_end_offset = 0x1CU;
constexpr std::uint32_t frame_heap_flags_offset = 0x30U;
constexpr std::uint32_t frame_heap_head_offset = 0x40U;
constexpr std::uint32_t frame_heap_tail_offset = 0x44U;
constexpr std::uint32_t frame_heap_previous_state_offset = 0x48U;
constexpr std::uint32_t mem_heap_flag_zero_allocated = 1U << 0U;
constexpr std::uint32_t mem_heap_flag_debug_mode = 1U << 1U;
constexpr std::uint32_t mem_exp_heap_free_list_head_offset = 0x40U;
constexpr std::uint32_t mem_exp_heap_block_size_offset = 0x04U;
constexpr std::uint32_t mem_exp_heap_block_next_offset = 0x0CU;
constexpr std::uint32_t mem_exp_heap_block_tag_offset = 0x10U;
constexpr std::uint32_t mem2_minimum_heap_size = 0x6CU;
constexpr std::uint32_t default_bump_heap_minimum_remaining = 0x1000U;
constexpr std::uint16_t mem2_free_block_tag = 0x4652U; // 'FR'
constexpr std::uint32_t fs_client_size = 0x1700U;
constexpr std::uint32_t fs_client_body_size = 0x1620U;
constexpr std::uint32_t fs_client_body_alignment_mask = 0x3FU;
constexpr std::uint32_t fs_cmd_block_size = 0xA80U;
constexpr std::uint32_t fs_cmd_block_body_alignment_mask = 0x3FU;
constexpr std::uint32_t fs_cmd_block_status_offset = 0x93CU;
constexpr std::uint32_t fs_cmd_block_priority_offset = 0x9E8U;
constexpr std::uint32_t fs_cmd_block_initialized_status = 0x0D900A21U;
constexpr std::uint32_t fs_cmd_block_available_status = 0x0D900A24U;
constexpr std::uint32_t fs_cmd_block_async_result_status_offset = 0x990U;
constexpr std::uint8_t fs_cmd_block_default_priority = 0x10U;
constexpr std::uint32_t fs_stat_size = 0x64U;
constexpr std::uint32_t fs_stat_flags_offset = 0x00U;
constexpr std::uint32_t fs_stat_mode_offset = 0x04U;
constexpr std::uint32_t fs_stat_owner_offset = 0x08U;
constexpr std::uint32_t fs_stat_group_offset = 0x0CU;
constexpr std::uint32_t fs_stat_size_offset = 0x10U;
constexpr std::uint32_t fs_stat_allocated_size_offset = 0x14U;
constexpr std::uint32_t fs_stat_quota_size_offset = 0x18U;
constexpr std::uint32_t fs_stat_entry_id_offset = 0x20U;
constexpr std::uint32_t fs_stat_created_offset = 0x24U;
constexpr std::uint32_t fs_stat_modified_offset = 0x2CU;
constexpr std::uint32_t fs_status_ok = 0U;
constexpr std::uint32_t fs_status_fatal_error = 0xFFFFFC00U;
constexpr std::uint32_t fs_status_not_found = 0xFFFFFFFAU;
constexpr std::uint32_t fs_status_not_file = 0xFFFFFFF9U;
constexpr std::uint32_t fs_status_access_error = 0xFFFFFFF7U;
constexpr std::uint32_t fs_status_permission_error = 0xFFFFFFF6U;
constexpr std::uint32_t fs_status_file_too_big = 0xFFFFFFF5U;
constexpr std::uint32_t fs_status_media_not_ready = 0xFFFFFFF1U;
constexpr std::uint32_t fs_stat_flag_directory = 0x80000000U;
constexpr std::uint32_t fs_stat_flag_file = 0x01000000U;
constexpr std::uint32_t fs_mode_read_owner = 0x400U;
constexpr std::uint32_t fs_mode_write_owner = 0x200U;
constexpr std::uint32_t fs_mode_exec_owner = 0x100U;
constexpr std::uint32_t fs_mode_read_group = 0x040U;
constexpr std::uint32_t fs_mode_write_group = 0x020U;
constexpr std::uint32_t fs_mode_exec_group = 0x010U;
constexpr std::uint32_t fs_mode_read_other = 0x004U;
constexpr std::uint32_t fs_mode_write_other = 0x002U;
constexpr std::uint32_t fs_mode_exec_other = 0x001U;
constexpr std::uint32_t fs_fast_mutex_tag = 0x664D7458U;
constexpr std::uint32_t fs_fast_mutex_size = 0x2CU;
constexpr std::uint32_t fs_alarm_tag = 0x614C724DU;
constexpr std::uint32_t fs_alarm_size = 0x58U;

constexpr std::uint32_t fs_client_handle_offset = 0x1444U;
constexpr std::uint32_t fs_client_fsm_offset = 0x1448U;
constexpr std::uint32_t fs_client_fsm_current_state_offset = 0x00U;
constexpr std::uint32_t fs_client_fsm_volume_state_offset = 0x04U;
constexpr std::uint32_t fs_client_fsm_send_state_change_notification_offset = 0x10U;
constexpr std::uint32_t fs_client_fsm_state_change_info_offset = 0x14U;
constexpr std::uint32_t fs_client_fsm_state_change_info_size = 0x24U;
constexpr std::uint32_t fs_state_change_callback_offset = 0x00U;
constexpr std::uint32_t fs_state_change_param_offset = 0x04U;
constexpr std::uint32_t fs_state_change_queue_offset = 0x08U;
constexpr std::uint32_t fs_state_change_message_offset = 0x0CU;
constexpr std::uint32_t fs_message_data_offset = 0x00U;
constexpr std::uint32_t fs_message_type_offset = 0x0CU;
constexpr std::uint32_t fs_state_change_client_offset = 0x1CU;
constexpr std::uint32_t fs_state_change_state_offset = 0x20U;
constexpr std::uint32_t fs_state_change_message_type = 11U;
constexpr std::uint32_t fs_client_cmd_queue_offset = 0x1480U;
constexpr std::uint32_t fs_client_last_dequeued_command_offset = 0x14C4U;
constexpr std::uint32_t fs_client_emulated_error_offset = 0x14C8U;
constexpr std::uint32_t fs_client_mutex_offset = 0x1560U;
constexpr std::uint32_t fs_client_fsm_alarm_offset = 0x1590U;
constexpr std::uint32_t fs_client_last_error_offset = 0x15E8U;
constexpr std::uint32_t fs_client_last_error_without_volume_offset = 0x15ECU;
constexpr std::uint32_t fs_client_mount_source_type_offset = 0x1610U;
constexpr std::uint32_t fs_client_link_offset = 0x1614U;
constexpr std::uint32_t fs_client_pointer_offset = 0x161CU;
constexpr std::uint32_t fs_client_link_next_offset = 0x00U;
constexpr std::uint32_t fs_client_link_prev_offset = 0x04U;

constexpr std::uint32_t fs_cmd_queue_head_offset = 0x00U;
constexpr std::uint32_t fs_cmd_queue_tail_offset = 0x04U;
constexpr std::uint32_t fs_cmd_queue_mutex_offset = 0x08U;
constexpr std::uint32_t fs_cmd_queue_dequeue_handler_offset = 0x34U;
constexpr std::uint32_t fs_cmd_queue_active_count_offset = 0x38U;
constexpr std::uint32_t fs_cmd_queue_max_active_count_offset = 0x3CU;
constexpr std::uint32_t fs_cmd_queue_status_offset = 0x40U;

constexpr std::uint32_t fs_volume_state_ready = 1U;

static_assert(fs_state_change_state_offset + sizeof(std::uint32_t) ==
              fs_client_fsm_state_change_info_size);
static_assert(os_system_info_timer_clock_speed == 62156250U);
static_assert(os_system_info_core_clock_speed / os_system_info_bus_clock_speed ==
              os_system_info_cpu_ratio);

void initialize_guest_thread_queue(
    GuestMemory& memory,
    std::uint32_t thread_queue,
    std::uint32_t parent)
{
    memory.write32_be(thread_queue + os_thread_queue_head_offset, 0U);
    memory.write32_be(thread_queue + os_thread_queue_tail_offset, 0U);
    memory.write32_be(thread_queue + os_thread_queue_parent_offset, parent);
    memory.write32_be(
        thread_queue + os_thread_queue_parent_offset + sizeof(std::uint32_t), 0U);
}

void initialize_guest_message_queue(
    GuestMemory& memory,
    std::uint32_t queue,
    std::uint32_t messages,
    std::uint32_t size)
{
    memory.zero_fill(queue, os_message_queue_size);
    memory.write32_be(queue, os_message_queue_tag);
    initialize_guest_thread_queue(
        memory, queue + os_message_queue_send_queue_offset, queue);
    initialize_guest_thread_queue(
        memory, queue + os_message_queue_recv_queue_offset, queue);
    memory.write32_be(queue + os_message_queue_messages_offset, messages);
    memory.write32_be(queue + os_message_queue_capacity_offset, size);
    memory.write32_be(queue + os_message_queue_first_offset, 0U);
    memory.write32_be(queue + os_message_queue_used_offset, 0U);
}

[[nodiscard]] bool send_guest_message(
    EspressoCore& core,
    std::uint32_t queue,
    std::uint32_t message,
    std::uint32_t flags)
{
    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    if (queue == 0U)
    {
        throw HleExecutionError("OSSendMessage received null OSMessageQueue pointer");
    }
    if (static_cast<std::uint64_t>(queue) + os_message_queue_size >
        guest_address_space_end)
    {
        throw HleExecutionError("OSSendMessage OSMessageQueue address wraps guest address space");
    }
    try
    {
        core.memory.validate_write_range(queue, os_message_queue_size);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError("OSSendMessage OSMessageQueue is outside mapped guest memory");
    }

    const std::uint32_t messages =
        core.memory.read32_be(queue + os_message_queue_messages_offset);
    const std::uint32_t capacity =
        core.memory.read32_be(queue + os_message_queue_capacity_offset);
    const std::uint32_t first =
        core.memory.read32_be(queue + os_message_queue_first_offset);
    const std::uint32_t used =
        core.memory.read32_be(queue + os_message_queue_used_offset);

    if (capacity != 0U && first >= capacity)
    {
        throw HleExecutionError("OSSendMessage encountered an invalid message-queue first index");
    }
    if (used >= capacity)
    {
        if ((flags & os_message_flag_blocking) != 0U)
        {
            // Correct blocking requires waking/scheduling guest threads, which
            // Affogato's current single-lane runtime does not yet implement.
            throw HleExecutionError(
                "OSSendMessage would block on a full queue, but guest thread scheduling is unavailable");
        }
        return false;
    }

    if (message == 0U)
    {
        throw HleExecutionError("OSSendMessage received null OSMessage pointer");
    }

    const std::uint64_t slot_index = (flags & os_message_flag_high_priority) != 0U
        ? (static_cast<std::uint64_t>(first) + capacity - 1U) % capacity
        : (static_cast<std::uint64_t>(first) + used) % capacity;
    const std::uint64_t slot_address =
        static_cast<std::uint64_t>(messages) + slot_index * os_message_size;
    if (slot_address + os_message_size > guest_address_space_end)
    {
        throw HleExecutionError("OSSendMessage message slot wraps guest address space");
    }

    std::array<std::uint8_t, os_message_size> message_bytes{};
    try
    {
        core.memory.read_bytes(message, message_bytes);
        core.memory.validate_write_range(
            static_cast<std::uint32_t>(slot_address), os_message_size);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError("OSSendMessage message source or queue slot is outside mapped guest memory");
    }

    // Both ranges are validated before any guest memory is changed. Snapshot
    // also makes overlapping source/destination buffers safe.
    core.memory.write_bytes(
        static_cast<std::uint32_t>(slot_address), message_bytes);
    if ((flags & os_message_flag_high_priority) != 0U)
    {
        core.memory.write32_be(queue + os_message_queue_first_offset,
                               static_cast<std::uint32_t>(slot_index));
    }
    core.memory.write32_be(queue + os_message_queue_used_offset, used + 1U);
    return true;
}

[[nodiscard]] bool receive_guest_message(
    EspressoCore& core,
    std::uint32_t queue,
    std::uint32_t destination,
    std::uint32_t flags)
{
    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    if (queue == 0U)
    {
        throw HleExecutionError("OSReceiveMessage received null OSMessageQueue pointer");
    }
    if (static_cast<std::uint64_t>(queue) + os_message_queue_size >
        guest_address_space_end)
    {
        throw HleExecutionError("OSReceiveMessage OSMessageQueue address wraps guest address space");
    }

    std::array<std::uint8_t, os_message_queue_size> queue_snapshot{};
    try
    {
        core.memory.read_bytes(queue, queue_snapshot);
        core.memory.validate_write_range(queue, os_message_queue_size);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError("OSReceiveMessage OSMessageQueue is outside mapped guest memory");
    }
    const auto queue_u32 = [&queue_snapshot](std::uint32_t offset) {
        return (static_cast<std::uint32_t>(queue_snapshot[offset]) << 24U) |
            (static_cast<std::uint32_t>(queue_snapshot[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(queue_snapshot[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(queue_snapshot[offset + 3U]);
    };

    const std::uint32_t messages = queue_u32(os_message_queue_messages_offset);
    const std::uint32_t capacity = queue_u32(os_message_queue_capacity_offset);
    const std::uint32_t first = queue_u32(os_message_queue_first_offset);
    const std::uint32_t used = queue_u32(os_message_queue_used_offset);
    if (capacity != 0U && first >= capacity)
    {
        throw HleExecutionError(
            "OSReceiveMessage encountered an invalid message-queue first index");
    }
    if (used > capacity)
    {
        throw HleExecutionError(
            "OSReceiveMessage encountered a message-queue used count above capacity");
    }
    if (used == 0U)
    {
        if ((flags & os_message_flag_blocking) != 0U)
        {
            throw HleExecutionError(
                "OSReceiveMessage would block on an empty queue; receiver scheduling is not implemented");
        }
        return false;
    }
    if (destination == 0U)
    {
        throw HleExecutionError("OSReceiveMessage received null OSMessage output pointer");
    }

    const std::uint64_t slot_address = static_cast<std::uint64_t>(messages) +
        static_cast<std::uint64_t>(first) * os_message_size;
    if (slot_address + os_message_size > guest_address_space_end)
    {
        throw HleExecutionError("OSReceiveMessage message slot wraps guest address space");
    }
    std::array<std::uint8_t, os_message_size> message{};
    try
    {
        core.memory.read_bytes(static_cast<std::uint32_t>(slot_address), message);
        core.memory.validate_write_range(destination, message.size());
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(
            "OSReceiveMessage message slot or output buffer is outside mapped guest memory");
    }

    // Read and validate both sides before committing output or ring metadata.
    core.memory.write_bytes(destination, message);
    core.memory.write32_be(
        queue + os_message_queue_first_offset, (first + 1U) % capacity);
    core.memory.write32_be(queue + os_message_queue_used_offset, used - 1U);
    return true;
}

void initialize_guest_fast_mutex(
    GuestMemory& memory,
    std::uint32_t mutex,
    std::uint32_t name)
{
    memory.zero_fill(mutex, fs_fast_mutex_size);
    memory.write32_be(mutex, fs_fast_mutex_tag);
    memory.write32_be(mutex + sizeof(std::uint32_t), name);
}

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

template <typename Visitor>
void visit_validated_exp_heap_free_blocks(
    const GuestMemory& memory,
    std::uint32_t heap,
    Visitor&& visitor)
{
    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    if (heap == 0U)
    {
        throw HleExecutionError(
            "MEMGetTotalFreeSizeForExpHeap received null heap handle");
    }
    if (static_cast<std::uint64_t>(heap) + mem2_expanded_heap_header_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            "MEMGetTotalFreeSizeForExpHeap heap header wraps guest address space");
    }

    // Validate the complete guest header before trusting any of its pointers.
    std::array<std::uint8_t, mem2_expanded_heap_header_size> header{};
    try
    {
        memory.read_bytes(heap, header);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(
            "MEMGetTotalFreeSizeForExpHeap heap header is outside mapped guest memory");
    }

    const std::uint32_t tag = memory.read32_be(heap);
    if (tag != mem2_expanded_heap_tag)
    {
        throw HleExecutionError(
            "MEMGetTotalFreeSizeForExpHeap handle is not an EXPH heap (tag " +
            std::to_string(tag) + ")");
    }

    const std::uint32_t data_start =
        memory.read32_be(heap + mem_exp_heap_data_start_offset);
    const std::uint32_t data_end =
        memory.read32_be(heap + mem_exp_heap_data_end_offset);
    if (data_start > data_end)
    {
        throw HleExecutionError(
            "MEMGetTotalFreeSizeForExpHeap has invalid heap data bounds");
    }

    std::uint32_t block =
        memory.read32_be(heap + mem_exp_heap_free_list_head_offset);
    // The current execution lane is single-threaded, so this read-only walk
    // cannot race guest heap mutations. Revisit when Cafe scheduling exists.
    std::set<std::uint32_t> visited;
    while (block != 0U)
    {
        if (!visited.insert(block).second)
        {
            throw HleExecutionError(
                "MEMGetTotalFreeSizeForExpHeap free-list cycle detected");
        }

        const std::uint64_t block_header_end =
            static_cast<std::uint64_t>(block) +
            mem2_expanded_heap_block_header_size;
        if (block < data_start || block_header_end > data_end ||
            block_header_end > guest_address_space_end)
        {
            throw HleExecutionError(
                "MEMGetTotalFreeSizeForExpHeap free block lies outside heap bounds");
        }

        std::uint32_t block_size{};
        std::uint32_t next{};
        std::uint16_t block_tag{};
        try
        {
            block_size = memory.read32_be(block + mem_exp_heap_block_size_offset);
            next = memory.read32_be(block + mem_exp_heap_block_next_offset);
            block_tag = memory.read16_be(block + mem_exp_heap_block_tag_offset);
        }
        catch (const GuestMemoryFault&)
        {
            throw HleExecutionError(
                "MEMGetTotalFreeSizeForExpHeap free block header is outside mapped guest memory");
        }
        const std::uint64_t block_data_end =
            block_header_end + static_cast<std::uint64_t>(block_size);
        if (block_data_end > data_end || block_data_end > guest_address_space_end)
        {
            throw HleExecutionError(
                "MEMGetTotalFreeSizeForExpHeap free block exceeds heap bounds");
        }
        if (block_tag != mem2_free_block_tag)
        {
            throw HleExecutionError(
                "MEMGetTotalFreeSizeForExpHeap free-list block has invalid tag");
        }

        visitor(block, block_size, block_header_end, block_data_end);
        block = next;
    }
}

[[nodiscard]] std::uint32_t total_free_size_for_exp_heap(
    const GuestMemory& memory,
    std::uint32_t heap)
{
    std::uint64_t total = 0U;
    visit_validated_exp_heap_free_blocks(
        memory,
        heap,
        [&total](std::uint32_t, std::uint32_t block_size, std::uint64_t, std::uint64_t) {
            total += block_size;
            if (total > std::numeric_limits<std::uint32_t>::max())
            {
                throw HleExecutionError(
                    "MEMGetTotalFreeSizeForExpHeap free-size overflow");
            }
        });
    return static_cast<std::uint32_t>(total);
}

[[nodiscard]] std::uint32_t allocatable_size_for_exp_heap(
    const GuestMemory& memory,
    std::uint32_t heap,
    std::uint32_t alignment)
{
    std::uint64_t largest = 0U;
    visit_validated_exp_heap_free_blocks(
        memory,
        heap,
        [alignment, &largest](
            std::uint32_t,
            std::uint32_t,
            std::uint64_t raw_start,
            std::uint64_t raw_end) {
            const std::uint64_t aligned_start =
                (raw_start + alignment - 1U) &
                ~static_cast<std::uint64_t>(alignment - 1U);
            if (aligned_start >= raw_end)
            {
                return;
            }
            largest = std::max(largest, raw_end - aligned_start);
        });
    if (largest > std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForExpHeapEx result overflow");
    }
    return static_cast<std::uint32_t>(largest);
}

struct ValidatedFrameHeap
{
    std::uint32_t flags{};
    std::uint32_t data_start{};
    std::uint32_t data_end{};
    std::uint32_t head{};
    std::uint32_t tail{};
};

[[nodiscard]] ValidatedFrameHeap validate_frame_heap(
    const GuestMemory& memory,
    std::uint32_t heap,
    const char* function_name)
{
    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    const std::string function(function_name);
    if (heap == 0U)
    {
        throw HleExecutionError(function + " received null heap handle");
    }
    if (static_cast<std::uint64_t>(heap) + foreground_frame_heap_header_size >
        guest_address_space_end)
    {
        throw HleExecutionError(function + " heap header wraps guest address space");
    }

    std::array<std::uint8_t, foreground_frame_heap_header_size> header{};
    try
    {
        memory.read_bytes(heap, header);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(function + " heap header is outside mapped guest memory");
    }

    if (memory.read32_be(heap) != foreground_frame_heap_tag)
    {
        throw HleExecutionError(function + " handle is not an FRMH heap");
    }

    const std::uint64_t minimum_data_start =
        static_cast<std::uint64_t>(heap) + foreground_frame_heap_header_size;
    const ValidatedFrameHeap result{
        .flags = memory.read32_be(heap + frame_heap_flags_offset),
        .data_start = memory.read32_be(heap + mem_exp_heap_data_start_offset),
        .data_end = memory.read32_be(heap + mem_exp_heap_data_end_offset),
        .head = memory.read32_be(heap + frame_heap_head_offset),
        .tail = memory.read32_be(heap + frame_heap_tail_offset),
    };
    if (result.data_start < minimum_data_start ||
        result.data_start > result.data_end || result.head < result.data_start ||
        result.head > result.tail || result.tail > result.data_end)
    {
        throw HleExecutionError(function + " has invalid frame-heap bounds");
    }
    return result;
}

[[nodiscard]] std::uint32_t allocatable_size_for_frame_heap(
    const GuestMemory& memory,
    std::uint32_t heap,
    std::uint32_t alignment)
{
    const ValidatedFrameHeap frame_heap = validate_frame_heap(
        memory, heap, "MEMGetAllocatableSizeForFrmHeapEx");

    const std::uint64_t aligned_head =
        (static_cast<std::uint64_t>(frame_heap.head) + alignment - 1U) &
        ~static_cast<std::uint64_t>(alignment - 1U);
    if (aligned_head >= frame_heap.tail)
    {
        return 0U;
    }
    return static_cast<std::uint32_t>(frame_heap.tail - aligned_head);
}

[[nodiscard]] std::uint32_t allocate_from_frame_heap(
    GuestMemory& memory,
    std::uint32_t heap,
    std::uint32_t requested_size,
    std::int32_t signed_alignment)
{
    if (signed_alignment == 0)
    {
        throw HleExecutionError(
            "MEMAllocFromFrmHeapEx received zero alignment");
    }
    if (signed_alignment == std::numeric_limits<std::int32_t>::min())
    {
        throw HleExecutionError(
            "MEMAllocFromFrmHeapEx alignment magnitude is not representable");
    }
    const std::uint32_t alignment = static_cast<std::uint32_t>(
        signed_alignment < 0 ? -signed_alignment : signed_alignment);
    if ((alignment & (alignment - 1U)) != 0U)
    {
        throw HleExecutionError(
            "MEMAllocFromFrmHeapEx alignment magnitude must be a power of two");
    }

    const ValidatedFrameHeap frame_heap = validate_frame_heap(
        memory, heap, "MEMAllocFromFrmHeapEx");
    if ((frame_heap.flags & mem_heap_flag_debug_mode) != 0U)
    {
        throw HleExecutionError(
            "MEMAllocFromFrmHeapEx does not support DEBUG_MODE without heap fill-value APIs");
    }

    // Decaf documents that coreinit converts a zero-byte request to one byte.
    // It does not round ordinary requested sizes up to a word boundary.
    const std::uint64_t effective_size = requested_size == 0U
        ? 1U
        : static_cast<std::uint64_t>(requested_size);
    const std::uint64_t old_head = frame_heap.head;
    const std::uint64_t old_tail = frame_heap.tail;
    std::uint64_t allocation_start{};
    std::uint64_t updated_boundary{};
    const bool allocate_from_head = signed_alignment > 0;

    if (allocate_from_head)
    {
        allocation_start =
            (old_head + alignment - 1U) &
            ~static_cast<std::uint64_t>(alignment - 1U);
        updated_boundary = allocation_start + effective_size;
        if (allocation_start < old_head || updated_boundary > old_tail ||
            updated_boundary > std::numeric_limits<std::uint32_t>::max())
        {
            return 0U;
        }
    }
    else
    {
        if (effective_size > old_tail)
        {
            return 0U;
        }
        const std::uint64_t candidate = old_tail - effective_size;
        allocation_start = candidate &
            ~static_cast<std::uint64_t>(alignment - 1U);
        updated_boundary = allocation_start;
        if (allocation_start < old_head ||
            allocation_start + effective_size > old_tail)
        {
            return 0U;
        }
    }

    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    const std::uint64_t allocation_end = allocation_start + effective_size;
    if (allocation_start >= guest_address_space_end ||
        allocation_end > guest_address_space_end)
    {
        return 0U;
    }
    try
    {
        memory.validate_write_range(
            static_cast<std::uint32_t>(allocation_start),
            static_cast<std::size_t>(effective_size));
        memory.validate_write_range(
            heap + (allocate_from_head ? frame_heap_head_offset : frame_heap_tail_offset),
            sizeof(std::uint32_t));
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(
            "MEMAllocFromFrmHeapEx allocation or heap boundary is outside writable guest memory");
    }

    // GuestMemory's complete-range validation above makes ZERO_ALLOCATED
    // writes safe before the single guest-visible head/tail commit.
    if ((frame_heap.flags & mem_heap_flag_zero_allocated) != 0U)
    {
        memory.zero_fill(
            static_cast<std::uint32_t>(allocation_start),
            static_cast<std::size_t>(effective_size));
    }
    memory.write32_be(
        heap + (allocate_from_head ? frame_heap_head_offset : frame_heap_tail_offset),
        static_cast<std::uint32_t>(updated_boundary));
    return static_cast<std::uint32_t>(allocation_start);
}

[[nodiscard]] std::uint64_t current_os_epoch_ticks() noexcept
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

void set_u64_return(CpuState& state, std::uint64_t value) noexcept
{
    state.gpr[3] = static_cast<std::uint32_t>(value >> 32U);
    state.gpr[4] = static_cast<std::uint32_t>(value);
}

GuestThreadRuntimeState& ensure_guest_thread(EspressoCore& core, std::uint32_t address)
{
    const auto found = std::find_if(
        core.guest_threads.begin(), core.guest_threads.end(),
        [address](const GuestThreadRuntimeState& thread) {
            return thread.address == address;
        });
    if (found != core.guest_threads.end())
    {
        return *found;
    }
    core.guest_threads.push_back({address, 0U, false});
    return core.guest_threads.back();
}

void save_guest_context(
    EspressoCore& core,
    std::uint32_t thread,
    std::uint32_t resume_cia)
{
    core.memory.validate_write_range(thread, os_context_size);
    for (std::uint32_t reg = 0U; reg < core.state.gpr.size(); ++reg)
    {
        core.memory.write32_be(
            thread + os_context_gpr_offset + reg * sizeof(std::uint32_t),
            core.state.gpr[reg]);
    }
    core.memory.write32_be(thread + os_context_cr_offset, core.state.cr);
    core.memory.write32_be(thread + os_context_lr_offset, core.state.lr);
    core.memory.write32_be(thread + os_context_ctr_offset, core.state.ctr);
    core.memory.write32_be(thread + os_context_xer_offset, core.state.xer);
    core.memory.write32_be(thread + os_context_srr0_offset, resume_cia);
    core.memory.write32_be(thread + os_context_fpscr_offset, core.state.fpscr);
    for (std::uint32_t reg = 0U; reg < core.state.fpr.size(); ++reg)
    {
        core.memory.write64_be(
            thread + os_context_fpr_offset + reg * sizeof(std::uint64_t),
            core.state.fpr[reg]);
        core.memory.write64_be(
            thread + os_context_psf_offset + reg * sizeof(std::uint64_t),
            core.state.fpr_ps1[reg]);
    }
}

void load_guest_context(EspressoCore& core, std::uint32_t thread)
{
    std::array<std::uint8_t, os_context_size> context{};
    core.memory.read_bytes(thread, context);
    const auto read32 = [&context](std::uint32_t offset) {
        return (static_cast<std::uint32_t>(context[offset]) << 24U) |
            (static_cast<std::uint32_t>(context[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(context[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(context[offset + 3U]);
    };
    const auto read64 = [&read32](std::uint32_t offset) {
        return (static_cast<std::uint64_t>(read32(offset)) << 32U) |
            read32(offset + sizeof(std::uint32_t));
    };
    for (std::uint32_t reg = 0U; reg < core.state.gpr.size(); ++reg)
    {
        core.state.gpr[reg] = read32(
            os_context_gpr_offset + reg * sizeof(std::uint32_t));
    }
    core.state.cr = read32(os_context_cr_offset);
    core.state.lr = read32(os_context_lr_offset);
    core.state.ctr = read32(os_context_ctr_offset);
    core.state.xer = read32(os_context_xer_offset);
    core.state.cia = read32(os_context_srr0_offset);
    core.state.fpscr = read32(os_context_fpscr_offset);
    for (std::uint32_t reg = 0U; reg < core.state.fpr.size(); ++reg)
    {
        core.state.fpr[reg] = read64(
            os_context_fpr_offset + reg * sizeof(std::uint64_t));
        core.state.fpr_ps1[reg] = read64(
            os_context_psf_offset + reg * sizeof(std::uint64_t));
    }
}

bool switch_guest_thread(EspressoCore& core, bool block_current, std::uint64_t sleep_ticks)
{
    const std::uint32_t current = core.current_thread_address;
    if (current == 0U)
    {
        return false;
    }
    const auto current_entry = std::find_if(
        core.guest_threads.begin(), core.guest_threads.end(),
        [current](const GuestThreadRuntimeState& thread) {
            return thread.address == current;
        });
    // Direct HLE unit tests and partially initialized clients may set a
    // current-thread pointer without an OSContext. Do not touch such memory.
    if (current_entry == core.guest_threads.end())
    {
        return false;
    }
    auto& current_runtime = *current_entry;
    const std::uint32_t current_priority = core.memory.read32_be(
        current + os_thread_priority_offset);

    if (block_current)
    {
        core.memory.write8(current + os_thread_state_offset, os_thread_waiting);
        current_runtime.sleeping = true;
        current_runtime.wake_tick = sleep_ticks >
                std::numeric_limits<std::uint64_t>::max() - core.guest_scheduler_ticks
            ? std::numeric_limits<std::uint64_t>::max()
            : core.guest_scheduler_ticks + sleep_ticks;
    }
    else
    {
        core.memory.write8(current + os_thread_state_offset, os_thread_ready);
        current_runtime.sleeping = false;
    }

    const auto wake_due_threads = [&core]() {
        for (auto& thread : core.guest_threads)
        {
            if (thread.sleeping && thread.wake_tick <= core.guest_scheduler_ticks)
            {
                thread.sleeping = false;
                core.memory.write8(
                    thread.address + os_thread_state_offset, os_thread_ready);
            }
        }
    };
    wake_due_threads();

    const auto select_ready_thread = [&]() -> std::uint32_t {
        std::uint32_t selected = 0U;
        std::uint32_t selected_priority = std::numeric_limits<std::uint32_t>::max();
        bool after_current = false;
        for (std::size_t pass = 0U; pass < 2U; ++pass)
        {
            after_current = pass != 0U;
            for (const auto& thread : core.guest_threads)
            {
                if (after_current && thread.address == current)
                {
                    after_current = false;
                    continue;
                }
                if (thread.address == current || thread.sleeping)
                {
                    continue;
                }
                const std::uint8_t state = core.memory.read8(
                    thread.address + os_thread_state_offset);
                const std::uint32_t suspended = core.memory.read32_be(
                    thread.address + os_thread_suspend_count_offset);
                const std::uint8_t affinity = core.memory.read8(
                    thread.address + os_thread_attributes_offset);
                // Affinity zero is Cafe's unpinned/any-core setting. A
                // nonzero mask must include the currently emulated core.
                if ((state & os_thread_ready) == 0U || suspended != 0U ||
                    (affinity != 0U &&
                     (affinity & (1U << current_core_index)) == 0U))
                {
                    continue;
                }
                const std::uint32_t priority = core.memory.read32_be(
                    thread.address + os_thread_priority_offset);
                if (priority < selected_priority)
                {
                    selected = thread.address;
                    selected_priority = priority;
                }
            }
            if (selected != 0U)
            {
                break;
            }
        }
        if (selected != 0U && !block_current && selected_priority > current_priority)
        {
            return 0U;
        }
        return selected;
    };

    std::uint32_t next = select_ready_thread();
    if (next == 0U && block_current)
    {
        std::uint64_t next_wake = std::numeric_limits<std::uint64_t>::max();
        for (const auto& thread : core.guest_threads)
        {
            if (thread.sleeping)
            {
                next_wake = std::min(next_wake, thread.wake_tick);
            }
        }
        if (next_wake != std::numeric_limits<std::uint64_t>::max())
        {
            core.guest_scheduler_ticks = std::max(core.guest_scheduler_ticks, next_wake);
            wake_due_threads();
            next = select_ready_thread();
        }
    }
    if (next == 0U)
    {
        if (block_current)
        {
            current_runtime.sleeping = false;
            core.memory.write8(current + os_thread_state_offset, os_thread_running);
        }
        else
        {
            core.memory.write8(current + os_thread_state_offset, os_thread_running);
        }
        return false;
    }

    save_guest_context(core, current, core.state.lr);
    core.memory.write8(next + os_thread_state_offset, os_thread_running);
    load_guest_context(core, next);
    core.current_thread_address = next;
    ++core.guest_thread_switches;
    return true;
}

} // namespace

std::uint32_t initialize_default_guest_thread(
    EspressoCore& core,
    std::uint32_t stack_start,
    std::uint32_t stack_end)
{
    constexpr std::uint32_t thread_state_offset = 0x324U;
    constexpr std::uint32_t thread_attributes_offset = 0x325U;
    constexpr std::uint32_t thread_id_offset = 0x326U;
    constexpr std::uint32_t thread_stack_start_offset = 0x394U;
    constexpr std::uint32_t thread_stack_end_offset = 0x398U;
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

    const std::uint32_t address = core.allocate_guest_memory(os_thread_size, 8U);
    if (address == 0)
    {
        throw std::runtime_error(
            "guest heap cannot fit the default Cafe OSThread object (cursor " +
            std::to_string(core.guest_heap_cursor) + ", limit " +
            std::to_string(core.guest_heap_limit) + ")");
    }

    core.memory.zero_fill(address, os_thread_size);
    core.memory.write32_be(address + os_thread_tag_offset, os_thread_tag);
    core.memory.write8(address + thread_state_offset, thread_state_running);
    core.memory.write8(address + thread_attributes_offset, thread_affinity_cpu1);
    core.memory.write16_be(address + thread_id_offset, default_thread_id);
    core.memory.write32_be(
        address + os_thread_priority_offset,
        static_cast<std::uint32_t>(os_thread_default_internal_priority));
    core.memory.write32_be(
        address + os_thread_base_priority_offset,
        static_cast<std::uint32_t>(os_thread_default_internal_priority));
    core.memory.write32_be(address + thread_stack_start_offset, stack_start);
    core.memory.write32_be(address + thread_stack_end_offset, stack_end);
    core.memory.write32_be(address + os_thread_type_offset, thread_type_application);
    core.current_thread_address = address;
    ensure_guest_thread(core, address);
    return address;
}

bool initialize_default_guest_heaps(EspressoCore& core)
{
    if (core.base_heap_handles[mem2_base_heap_index] != 0U ||
        core.base_heap_handles[foreground_base_heap_index] != 0U ||
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

    // The loader-selected gap does not model Cafe's physical foreground
    // bucket. Provide a synthetic FRMH below MEM2, while keeping a reserve for
    // the legacy bump allocator and leaving the MEM2 partition unchanged.
    const std::uint64_t lower_span = mem2_begin - allocation_begin;
    if (lower_span <= default_bump_heap_minimum_remaining)
    {
        return false;
    }
    const std::uint64_t max_foreground_size =
        (lower_span - default_bump_heap_minimum_remaining) & ~std::uint64_t{0xFU};
    const std::uint64_t foreground_size =
        std::min(aligned_mem2_size, max_foreground_size);
    if (foreground_size < 0x50U)
    {
        return false;
    }
    const std::uint64_t foreground_begin = mem2_begin - foreground_size;
    const std::uint64_t foreground_data_start =
        foreground_begin + foreground_frame_heap_header_size;
    if (foreground_begin < allocation_begin ||
        foreground_begin - allocation_begin < default_bump_heap_minimum_remaining ||
        foreground_data_start + 4U > mem2_begin)
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
    const auto foreground_handle = static_cast<std::uint32_t>(foreground_begin);
    const auto foreground_start = static_cast<std::uint32_t>(foreground_data_start);
    try
    {
        // Validate both complete reservations before touching guest bytes.
        core.memory.validate_write_range(
            handle, static_cast<std::size_t>(mem2_end - mem2_begin));
        core.memory.validate_write_range(
            foreground_handle, static_cast<std::size_t>(foreground_size));
    }
    catch (const GuestMemoryFault&)
    {
        return false;
    }

    core.memory.zero_fill(foreground_handle, foreground_frame_heap_header_size);
    core.memory.write32_be(foreground_handle, foreground_frame_heap_tag);
    core.memory.write32_be(foreground_handle + mem_exp_heap_data_start_offset,
                           foreground_start);
    core.memory.write32_be(foreground_handle + mem_exp_heap_data_end_offset,
                           handle);
    core.memory.write32_be(foreground_handle + frame_heap_flags_offset, 0U);
    core.memory.write32_be(foreground_handle + frame_heap_head_offset,
                           foreground_start);
    core.memory.write32_be(foreground_handle + frame_heap_tail_offset, handle);
    core.memory.write32_be(
        foreground_handle + frame_heap_previous_state_offset, 0U);

    // This is a guest-visible EXPH header plus its genuine initial free block.
    // The block occupies only the upper partition; the legacy bump allocator
    // is capped at foreground_handle, so all three regions remain disjoint.
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

    core.guest_heap_limit = foreground_handle;
    core.mem2_heap_region_begin = handle;
    core.mem2_heap_region_end = heap_end;
    core.base_heap_handles[mem2_base_heap_index] = handle;
    core.base_heap_handles[foreground_base_heap_index] = foreground_handle;
    return true;
}

void register_coreinit_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "proc_ui",
        "ProcUIInit",
        [](EspressoCore& core) {
            // ProcUIInit takes the save-ready callback in r3. Cemu likewise
            // installs/replaces this pointer before its idempotent internal
            // initialization check.
            core.proc_ui_save_callback = core.state.gpr[3];
            core.proc_ui_save_callback_ex = 0U;
            core.proc_ui_save_callback_ex_argument = 0U;

            if (core.proc_ui_initialized)
            {
                return;
            }

            // ProcUI starts foreground with status IN_FOREGROUND (0), and is
            // not shutting down. Affogato has one synchronous guest lane; it
            // does not yet model ProcUI's Cafe message queue, events, or worker
            // threads, so the save callback is retained but not invoked here.
            core.proc_ui_initialized = true;
            core.proc_ui_in_foreground = true;
            core.proc_ui_in_shutdown = false;
            core.proc_ui_status = 0U;
            core.proc_ui_draw_done_release = false;
        });

    dispatcher.register_function(
        "proc_ui",
        "ProcUIRegisterCallback",
        [](EspressoCore& core) {
            // This API registers an event callback; it does not invoke it.
            // Keep the guest pointer/parameter/priority for a future ProcUI
            // event dispatcher rather than fabricating an event here.
            core.proc_ui_callbacks.push_back({
                core.state.gpr[3],
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6]});
        });

    dispatcher.register_function(
        "proc_ui",
        "ProcUIClearCallbacks",
        [](EspressoCore& core) {
            core.proc_ui_callbacks.clear();
        });

    dispatcher.register_function(
        "coreinit",
        "IMIsDimEnabled",
        [](EspressoCore& core) {
            core.memory.write32_be(
                core.state.gpr[3], core.im_dim_enabled ? 1U : 0U);
            core.state.gpr[3] = 0U; // IM_ERROR_NONE
        });

    dispatcher.register_function(
        "coreinit",
        "IMEnableDim",
        [](EspressoCore& core) {
            core.im_dim_enabled = true;
            core.state.gpr[3] = 0U; // IM_ERROR_NONE
        });

    dispatcher.register_function(
        "coreinit",
        "IMDisableDim",
        [](EspressoCore& core) {
            core.im_dim_enabled = false;
            core.state.gpr[3] = 0U; // IM_ERROR_NONE
        });

    dispatcher.register_function(
        "coreinit",
        "OSEnableHomeButtonMenu",
        [](EspressoCore& core) {
            core.home_button_menu_enabled = core.state.gpr[3] != 0U;
        });

    dispatcher.register_function(
        "coreinit",
        "OSIsHomeButtonMenuEnabled",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.home_button_menu_enabled ? 1U : 0U;
        });

    dispatcher.register_function(
        "padscore",
        "WPADEnableURCC",
        [](EspressoCore& core) {
            // Preserve the requested controller-connect policy. Controller
            // discovery/delivery is handled by the future padscore layer.
            core.wpad_urcc_enabled = core.state.gpr[3] != 0U;
        });

    dispatcher.register_function(
        "padscore",
        "WPADEnableWiiRemote",
        [](EspressoCore& core) {
            core.wpad_wii_remote_enabled = core.state.gpr[3] != 0U;
        });
    dispatcher.register_function(
        "padscore",
        "KPADInitEx",
        [](EspressoCore& core) {
            const std::uint32_t buffer = core.state.gpr[3];
            const std::uint32_t count = core.state.gpr[4];
            const std::uint64_t buffer_size =
                static_cast<std::uint64_t>(count) * kpad_unified_wpad_status_size;
            if (count != 0U)
            {
                if (buffer == 0U ||
                    static_cast<std::uint64_t>(buffer) + buffer_size >
                        (std::uint64_t{1} << 32U))
                {
                    throw HleExecutionError(
                        "padscore::KPADInitEx received an invalid external status buffer");
                }
                core.memory.validate_write_range(
                    buffer, static_cast<std::size_t>(buffer_size));
            }

            // Keep the application's optional unified-status ring buffer for
            // subsequent KPAD APIs. Controller sampling remains a separate
            // input-backend feature; initialization does not fabricate samples.
            core.kpad_initialized = true;
            core.kpad_unified_status_buffer = buffer;
            core.kpad_unified_status_count = count;
        });
    dispatcher.register_function(
        "padscore",
        "KPADGetMplsWorkSize",
        [](EspressoCore& core) {
            core.state.gpr[3] = kpad_mpls_work_size;
        });
    dispatcher.register_function(
        "padscore",
        "KPADSetMplsWorkarea",
        [](EspressoCore& core) {
            const std::uint32_t workarea = core.state.gpr[3];
            if (workarea != 0U)
            {
                core.memory.validate_write_range(workarea, kpad_mpls_work_size);
            }
            core.kpad_mpls_workarea = workarea;
        });

    dispatcher.register_function(
        "coreinit",
        "OSCreateThread",
        [](EspressoCore& core) {
            const std::uint32_t thread = core.state.gpr[3];
            const std::uint32_t entry = core.state.gpr[4];
            const std::uint32_t argument = core.state.gpr[5];
            const std::uint32_t argument_pointer = core.state.gpr[6];
            const std::uint32_t stack_top = core.state.gpr[7];
            const std::uint32_t stack_size = core.state.gpr[8];
            const std::int32_t priority =
                static_cast<std::int32_t>(core.state.gpr[9]);
            const std::uint32_t attributes = core.state.gpr[10];

            if (thread == 0U || stack_top == 0U || stack_size == 0U ||
                priority < 0 || priority >= 32 || attributes >= 0x20U ||
                stack_top < stack_size)
            {
                core.state.gpr[3] = 0U;
                return;
            }

            constexpr std::uint32_t thread_extent = os_thread_size;
            const std::uint64_t thread_end =
                static_cast<std::uint64_t>(thread) + thread_extent;
            const std::uint32_t stack_bottom = stack_top - stack_size;
            const std::uint32_t aligned_stack_top = stack_top & ~7U;
            const std::uint32_t effective_attributes =
                (attributes & 7U) == 0U
                    ? attributes | (1U << current_core_index)
                    : attributes;
            if (thread_end > (std::uint64_t{1} << 32U) ||
                static_cast<std::uint64_t>(aligned_stack_top) <
                    static_cast<std::uint64_t>(stack_bottom) + 12U)
            {
                core.state.gpr[3] = 0U;
                return;
            }

            // Validate every range that will be written before initializing
            // either object, so an unmapped stack cannot leave a half-built
            // OSThread behind.
            core.memory.validate_write_range(thread, os_thread_size);
            core.memory.validate_write_range(stack_bottom, stack_size);
            constexpr std::array<std::uint32_t, 5> callback_offsets{
                0x630U, 0x63CU, 0x648U, 0x654U, 0x680U};
            std::array<std::uint32_t, 15> inherited_callbacks{};
            if (core.current_thread_address != 0U &&
                core.current_thread_address != thread)
            {
                std::size_t callback_index = 0U;
                for (const std::uint32_t callback_offset : callback_offsets)
                {
                    for (std::uint32_t core_index = 0U; core_index < 3U; ++core_index)
                    {
                        const std::uint32_t offset =
                            callback_offset + core_index * sizeof(std::uint32_t);
                        inherited_callbacks[callback_index++] = core.memory.read32_be(
                            core.current_thread_address + offset);
                    }
                }
            }
            const std::uint32_t context_return =
                core.hle.bind_import("coreinit", "OSExitThread");

            // Cafe initializes newly-created threads suspended. Their context
            // is guest-visible now, but Affogato does not schedule it until a
            // real resume/scheduler implementation exists.
            core.memory.zero_fill(thread, os_thread_size);
            core.memory.write64_be(thread, os_context_tag);
            core.memory.write32_be(thread + os_context_gpr_offset + 1U * 4U,
                                   aligned_stack_top - 8U);
            core.memory.write32_be(thread + os_context_gpr_offset + 3U * 4U,
                                   argument);
            core.memory.write32_be(thread + os_context_gpr_offset + 4U * 4U,
                                   argument_pointer);
            core.memory.write32_be(thread + os_context_lr_offset, context_return);
            core.memory.write32_be(thread + os_context_srr0_offset, entry);
            core.memory.write32_be(thread + os_context_srr1_offset, 0x9032U);
            core.memory.write32_be(thread + os_context_fpscr_offset, 4U);
            core.memory.write32_be(thread + os_context_gqr_offset + 2U * 4U,
                                   0x00040004U);
            core.memory.write32_be(thread + os_context_gqr_offset + 3U * 4U,
                                   0x00050005U);
            core.memory.write32_be(thread + os_context_gqr_offset + 4U * 4U,
                                   0x00060006U);
            core.memory.write32_be(thread + os_context_gqr_offset + 5U * 4U,
                                   0x00070007U);
            core.memory.write32_be(thread + os_context_upir_offset, current_core_index);
            core.memory.write32_be(thread + os_context_affinity_offset,
                                   effective_attributes & 7U);
            core.memory.write32_be(thread + os_thread_tag_offset, os_thread_tag);
            core.memory.write8(thread + os_thread_state_offset,
                               entry != 0U ? os_thread_ready : 0U);
            core.memory.write8(thread + os_thread_attributes_offset,
                               static_cast<std::uint8_t>(effective_attributes));
            core.memory.write16_be(thread + os_thread_id_offset,
                                    static_cast<std::uint16_t>(os_thread_id_unassigned));
            core.memory.write32_be(thread + os_thread_suspend_count_offset,
                                   entry != 0U ? 1U : 0U);
            const std::uint32_t effective_priority =
                static_cast<std::uint32_t>(priority) + os_thread_app_priority_base;
            core.memory.write32_be(thread + os_thread_priority_offset,
                                   effective_priority);
            core.memory.write32_be(thread + os_thread_base_priority_offset,
                                   effective_priority);
            core.memory.write32_be(thread + os_thread_exit_value_offset, 0xFFFFFFFFU);
            core.memory.write32_be(thread + os_thread_stack_start_offset, stack_top);
            core.memory.write32_be(thread + os_thread_stack_end_offset, stack_bottom);
            core.memory.write32_be(thread + os_thread_entry_point_offset, entry);
            core.memory.write32_be(thread + os_thread_type_offset,
                                   os_thread_type_application);
            ensure_guest_thread(core, thread);
            core.memory.write32_be(aligned_stack_top - 4U, 0U);
            core.memory.write32_be(aligned_stack_top - 8U, 0U);
            core.memory.write32_be(stack_bottom, 0xDEADBABEU);

            // Copy the installed exception handlers from the current thread,
            // matching OSCreateThread's guest-visible inheritance semantics.
            if (core.current_thread_address != 0U &&
                core.current_thread_address != thread)
            {
                std::size_t callback_index = 0U;
                for (const std::uint32_t callback_offset : callback_offsets)
                {
                    for (std::uint32_t core_index = 0U; core_index < 3U; ++core_index)
                    {
                        const std::uint32_t offset =
                            callback_offset + core_index * sizeof(std::uint32_t);
                        core.memory.write32_be(thread + offset,
                                               inherited_callbacks[callback_index++]);
                    }
                }
            }

            // Specific values are the final thread-specific array in the
            // public OSThread layout and are already zero from the full clear.
            static_assert(os_thread_specific_offset + 0x40U <= os_thread_size);
            core.state.gpr[3] = 1U;
        });

    dispatcher.register_function(
        "coreinit",
        "OSSetThreadName",
        [](EspressoCore& core) {
            constexpr std::uint32_t thread_name_offset = 0x5C0U;
            const std::uint32_t thread = core.state.gpr[3];
            const std::uint32_t name = core.state.gpr[4];
            core.memory.write32_be(thread + thread_name_offset, name);
        });

    dispatcher.register_function(
        "coreinit",
        "OSSetThreadAffinity",
        [](EspressoCore& core) {
            constexpr std::uint32_t attributes_offset = 0x325U;
            const std::uint32_t thread = core.state.gpr[3];
            const std::uint32_t affinity = core.state.gpr[4] & 7U;
            const std::uint64_t attributes_address =
                static_cast<std::uint64_t>(thread) + attributes_offset;
            const std::uint64_t context_affinity_address =
                static_cast<std::uint64_t>(thread) + os_context_affinity_offset;
            if (attributes_address > 0xFFFFFFFFULL ||
                context_affinity_address + sizeof(std::uint32_t) >
                    (std::uint64_t{1} << 32U))
            {
                throw GuestMemoryFault(thread, os_thread_size, GuestMemoryAccess::write);
            }

            const auto attributes = core.memory.read8(
                static_cast<std::uint32_t>(attributes_address));
            core.memory.validate_write_range(
                static_cast<std::uint32_t>(attributes_address), sizeof(std::uint8_t));
            core.memory.validate_write_range(
                static_cast<std::uint32_t>(context_affinity_address),
                sizeof(std::uint32_t));

            // This updates the guest-visible thread/context state. Affogato's
            // current single-lane model has no per-core run queues to reschedule.
            core.memory.write8(
                static_cast<std::uint32_t>(attributes_address),
                static_cast<std::uint8_t>((attributes & ~7U) | affinity));
            core.memory.write32_be(
                static_cast<std::uint32_t>(context_affinity_address), affinity);
            core.state.gpr[3] = 1U;
        });

    dispatcher.register_function(
        "coreinit",
        "OSResumeThread",
        [](EspressoCore& core) {
            const std::uint32_t thread = core.state.gpr[3];
            const std::uint64_t count_address =
                static_cast<std::uint64_t>(thread) + os_thread_suspend_count_offset;
            if (count_address + sizeof(std::uint32_t) >
                (std::uint64_t{1} << 32U))
            {
                throw GuestMemoryFault(thread, os_thread_size, GuestMemoryAccess::read);
            }

            const std::uint32_t previous_count = core.memory.read32_be(
                static_cast<std::uint32_t>(count_address));
            // Cafe returns the old suspend count. Reaching zero makes the
            // thread eligible for a run queue; Affogato currently has no
            // guest-thread scheduler, so it records the counter transition
            // without switching the active Espresso architectural state.
            if (previous_count != 0U)
            {
                core.memory.write32_be(
                    static_cast<std::uint32_t>(count_address), previous_count - 1U);
            }
            core.state.gpr[3] = previous_count;
        });

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
        "OSGetSystemTime",
        [](EspressoCore& core) {
            const auto address = core.hle.bind_data_import(
                "coreinit", "__affogato_internal_OSSystemInfo", core.memory);
            if (!address)
            {
                throw HleExecutionError(
                    "OSGetSystemTime internal OSSystemInfo data is not registered");
            }

            std::uint64_t base_time{};
            try
            {
                base_time = core.memory.read64_be(*address + 0x08U);
            }
            catch (const GuestMemoryFault&)
            {
                throw HleExecutionError(
                    "OSGetSystemTime OSSystemInfo baseTime is outside mapped guest memory");
            }

            // Affogato does not yet emulate Espresso's CPU time base. Until
            // cycle accounting exists, derive elapsed timer ticks from the
            // same host-clock epoch used for the persistent Cafe baseTime.
            const std::uint64_t current_epoch_ticks = current_os_epoch_ticks();
            const std::uint64_t elapsed_ticks = current_epoch_ticks >= base_time
                ? current_epoch_ticks - base_time
                : 0U;
            const std::uint64_t system_time =
                core.os_sleep_ticks_offset >
                    std::numeric_limits<std::uint64_t>::max() - elapsed_ticks
                    ? std::numeric_limits<std::uint64_t>::max()
                    : elapsed_ticks + core.os_sleep_ticks_offset;
            set_u64_return(core.state, system_time);
        });
    dispatcher.register_function(
        "coreinit",
        "OSSleepTicks",
        [](EspressoCore& core) {
            // OSTime is a signed 64-bit PPC ABI argument in r3:r4. A deadline
            // at or before the current time is already expired. Affogato has
            // one synchronously executing guest thread and no scheduler yet;
            // advance the emulated time exposed by OSGetSystemTime instead of
            // blocking the host or pretending to switch guest threads.
            const std::uint64_t raw_ticks =
                (static_cast<std::uint64_t>(core.state.gpr[3]) << 32U) |
                core.state.gpr[4];
            const std::int64_t ticks = std::bit_cast<std::int64_t>(raw_ticks);
            if (ticks > 0)
            {
                const std::uint64_t positive_ticks =
                    static_cast<std::uint64_t>(ticks);
                core.os_sleep_ticks_offset =
                    positive_ticks >
                            std::numeric_limits<std::uint64_t>::max() -
                                core.os_sleep_ticks_offset
                        ? std::numeric_limits<std::uint64_t>::max()
                        : core.os_sleep_ticks_offset + positive_ticks;
            }
            (void)switch_guest_thread(
                core, ticks > 0, ticks > 0 ? static_cast<std::uint64_t>(ticks) : 0U);
        });

    dispatcher.register_function(
        "coreinit",
        "OSYieldThread",
        [](EspressoCore& core) {
            (void)switch_guest_thread(core, false, 0U);
        });
    dispatcher.register_function(
        "coreinit",
        "OSGetCurrentThread",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.current_thread_address;
        });
    dispatcher.register_function(
        "coreinit",
        "OSGetThreadPriority",
        [](EspressoCore& core) {
            constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
            const std::uint32_t thread = core.state.gpr[3];
            if (thread == 0U)
            {
                throw HleExecutionError(
                    "OSGetThreadPriority received null OSThread pointer");
            }
            constexpr std::uint32_t required_thread_extent =
                os_thread_type_offset + sizeof(std::uint32_t);
            if (static_cast<std::uint64_t>(thread) + required_thread_extent >
                guest_address_space_end)
            {
                throw HleExecutionError(
                    "OSGetThreadPriority OSThread address wraps guest address space");
            }

            std::uint32_t tag{};
            std::uint32_t base_priority_bits{};
            std::uint32_t type{};
            try
            {
                tag = core.memory.read32_be(thread + os_thread_tag_offset);
                if (tag != os_thread_tag)
                {
                    throw HleExecutionError(
                        "OSGetThreadPriority received object with invalid thread tag");
                }
                base_priority_bits = core.memory.read32_be(
                    thread + os_thread_base_priority_offset);
                type = core.memory.read32_be(thread + os_thread_type_offset);
            }
            catch (const GuestMemoryFault&)
            {
                throw HleExecutionError(
                    "OSGetThreadPriority OSThread fields are outside mapped guest memory");
            }

            std::int32_t priority_namespace_base{};
            switch (type)
            {
            case 0U: priority_namespace_base = 0; break;  // Driver
            case 1U: priority_namespace_base = 32; break; // I/O
            case 2U: priority_namespace_base = 64; break; // Application
            default:
                throw HleExecutionError(
                    "OSGetThreadPriority encountered unsupported thread type");
            }

            const std::int32_t base_priority =
                std::bit_cast<std::int32_t>(base_priority_bits);
            const std::int64_t external_priority =
                static_cast<std::int64_t>(base_priority) - priority_namespace_base;
            if (external_priority < 0 || external_priority >= 32)
            {
                throw HleExecutionError(
                    "OSGetThreadPriority base priority is outside the valid range for thread type");
            }
            core.state.gpr[3] = static_cast<std::uint32_t>(external_priority);
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
        "MEMGetAllocatableSizeForFrmHeapEx",
        [](EspressoCore& core) {
            const std::uint32_t heap = core.state.gpr[3];
            const std::int32_t alignment =
                std::bit_cast<std::int32_t>(core.state.gpr[4]);
            if (alignment == 0)
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForFrmHeapEx received zero alignment");
            }
            if (alignment < 0)
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForFrmHeapEx negative alignment behavior is not modeled");
            }
            const auto unsigned_alignment = static_cast<std::uint32_t>(alignment);
            if ((unsigned_alignment & (unsigned_alignment - 1U)) != 0U)
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForFrmHeapEx alignment must be a power of two");
            }
            core.state.gpr[3] = allocatable_size_for_frame_heap(
                core.memory, heap, unsigned_alignment);
        });
    dispatcher.register_function(
        "coreinit",
        "MEMAllocFromFrmHeapEx",
        [](EspressoCore& core) {
            const std::uint32_t heap = core.state.gpr[3];
            const std::uint32_t size = core.state.gpr[4];
            const std::int32_t alignment =
                std::bit_cast<std::int32_t>(core.state.gpr[5]);
            core.state.gpr[3] = allocate_from_frame_heap(
                core.memory, heap, size, alignment);
        });
    dispatcher.register_function(
        "coreinit",
        "MEMGetTotalFreeSizeForExpHeap",
        [](EspressoCore& core) {
            const std::uint32_t heap = core.state.gpr[3];
            core.state.gpr[3] = total_free_size_for_exp_heap(core.memory, heap);
        });
    dispatcher.register_function(
        "coreinit",
        "MEMGetAllocatableSizeForExpHeapEx",
        [](EspressoCore& core) {
            const std::uint32_t heap = core.state.gpr[3];
            const std::int32_t signed_alignment =
                std::bit_cast<std::int32_t>(core.state.gpr[4]);
            if (signed_alignment == 0)
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForExpHeapEx received zero alignment");
            }
            if (signed_alignment == std::numeric_limits<std::int32_t>::min())
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForExpHeapEx alignment magnitude is not representable");
            }

            // This size query normalizes negative alignment to its magnitude;
            // it does not use negative alignment as a tail-allocation request.
            const std::uint64_t alignment_magnitude = signed_alignment < 0
                ? static_cast<std::uint64_t>(-static_cast<std::int64_t>(signed_alignment))
                : static_cast<std::uint64_t>(signed_alignment);
            if (alignment_magnitude < 4U || (alignment_magnitude & 3U) != 0U ||
                (alignment_magnitude & (alignment_magnitude - 1U)) != 0U)
            {
                throw HleExecutionError(
                    "MEMGetAllocatableSizeForExpHeapEx alignment must be a power of two and at least 4");
            }

            core.state.gpr[3] = allocatable_size_for_exp_heap(
                core.memory,
                heap,
                static_cast<std::uint32_t>(alignment_magnitude));
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
        "FSInit",
        [](EspressoCore& core) {
            // Track Cafe FS-library initialization per emulated core. Client
            // registration and IOSU-backed operations are modeled separately.
            core.fs_initialized = true;
        });
    dispatcher.register_function(
        "coreinit",
        "FSInitCmdBlock",
        [](EspressoCore& core) {
            const std::uint32_t command_block = core.state.gpr[3];
            if (command_block == 0U)
            {
                throw HleExecutionError(
                    "FSInitCmdBlock received a null FSCmdBlock");
            }

            const std::uint64_t body_address =
                (static_cast<std::uint64_t>(command_block) +
                 fs_cmd_block_body_alignment_mask) &
                ~static_cast<std::uint64_t>(fs_cmd_block_body_alignment_mask);
            const std::uint64_t command_block_end =
                static_cast<std::uint64_t>(command_block) + fs_cmd_block_size;
            if (command_block_end > (std::uint64_t{1} << 32U) ||
                body_address + fs_cmd_block_priority_offset + 1U >
                    command_block_end)
            {
                throw HleExecutionError(
                    "FSInitCmdBlock FSCmdBlock range wraps guest address space");
            }

            // The guest ABI stores the opaque 0xA80-byte command block, with
            // its internal body aligned to 64 bytes. Validate the complete
            // write before clearing anything so a bad guest pointer cannot
            // leave a partially initialized block.
            core.memory.validate_write_range(command_block, fs_cmd_block_size);
            core.memory.zero_fill(command_block, fs_cmd_block_size);
            core.memory.write32_be(
                static_cast<std::uint32_t>(body_address) +
                    fs_cmd_block_status_offset,
                fs_cmd_block_initialized_status);
            core.memory.write8(
                static_cast<std::uint32_t>(body_address) +
                    fs_cmd_block_priority_offset,
                fs_cmd_block_default_priority);
        });
    dispatcher.register_function(
        "coreinit",
        "FSGetStat",
        [](EspressoCore& core) {
            const std::uint32_t client = core.state.gpr[3];
            const std::uint32_t command_block = core.state.gpr[4];
            const std::uint32_t guest_path_address = core.state.gpr[5];
            const std::uint32_t guest_stat_address = core.state.gpr[6];
            if (!core.fs_initialized || client == 0U || command_block == 0U ||
                guest_path_address == 0U || guest_stat_address == 0U)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            const auto registration = std::find_if(
                core.fs_clients.begin(), core.fs_clients.end(),
                [client](const FsClientRegistration& candidate) {
                    return candidate.client_address == client;
                });
            if (registration == core.fs_clients.end())
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            const std::uint64_t command_body_address =
                (static_cast<std::uint64_t>(command_block) +
                 fs_cmd_block_body_alignment_mask) &
                ~static_cast<std::uint64_t>(fs_cmd_block_body_alignment_mask);
            constexpr std::uint64_t guest_address_space_end =
                std::uint64_t{1} << 32U;
            if (command_body_address + fs_cmd_block_async_result_status_offset +
                    sizeof(std::uint32_t) > guest_address_space_end)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }
            const std::uint32_t command_body =
                static_cast<std::uint32_t>(command_body_address);
            const std::uint32_t command_status = core.memory.read32_be(
                command_body + fs_cmd_block_status_offset);
            if (command_status != fs_cmd_block_initialized_status &&
                command_status != fs_cmd_block_available_status)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            core.memory.validate_write_range(
                guest_stat_address, fs_stat_size);
            core.memory.validate_write_range(
                command_body + fs_cmd_block_status_offset,
                sizeof(std::uint32_t));
            core.memory.validate_write_range(
                command_body + fs_cmd_block_async_result_status_offset,
                sizeof(std::uint32_t));

            constexpr std::size_t maximum_path_size = 0x280U;
            std::string guest_path;
            guest_path.reserve(maximum_path_size);
            bool terminated = false;
            for (std::size_t offset = 0U; offset < maximum_path_size; ++offset)
            {
                const std::uint64_t address =
                    static_cast<std::uint64_t>(guest_path_address) + offset;
                if (address >= guest_address_space_end)
                {
                    break;
                }
                const char character = static_cast<char>(core.memory.read8(
                    static_cast<std::uint32_t>(address)));
                if (character == '\0')
                {
                    terminated = true;
                    break;
                }
                guest_path.push_back(character);
            }
            if (!terminated)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            std::uint32_t status = fs_status_media_not_ready;
            std::optional<std::filesystem::path> host_path;
            constexpr std::string_view content_volume = "/vol/content";
            if (!core.fs_content_root.empty() &&
                (guest_path == content_volume ||
                 (guest_path.size() > content_volume.size() &&
                  guest_path.starts_with(content_volume) &&
                  guest_path[content_volume.size()] == '/')))
            {
                status = fs_status_not_found;
                const std::string_view relative_utf8 = guest_path.size() ==
                        content_volume.size()
                    ? std::string_view{}
                    : std::string_view(guest_path).substr(
                          content_volume.size() + 1U);
                std::u8string relative_path_utf8;
                relative_path_utf8.reserve(relative_utf8.size());
                for (const unsigned char character : relative_utf8)
                {
                    relative_path_utf8.push_back(
                        static_cast<char8_t>(character));
                }
                const std::filesystem::path relative(relative_path_utf8);
                bool safe_relative_path = !relative.is_absolute();
                for (const auto& component : relative)
                {
                    if (component == "." || component == "..")
                    {
                        safe_relative_path = false;
                        break;
                    }
                }

                if (safe_relative_path)
                {
                    std::error_code error;
                    const std::filesystem::path canonical_root =
                        std::filesystem::canonical(core.fs_content_root, error);
                    if (!error)
                    {
                        const std::filesystem::path candidate = canonical_root / relative;
                        const std::filesystem::path canonical_candidate =
                            std::filesystem::canonical(candidate, error);
                        if (!error)
                        {
                            const std::filesystem::path within_root =
                                canonical_candidate.lexically_relative(canonical_root);
                            const bool escapes_root =
                                within_root.empty() && canonical_candidate != canonical_root
                                    ? true
                                    : (!within_root.empty() &&
                                       *within_root.begin() == "..");
                            if (!escapes_root)
                            {
                                host_path = canonical_candidate;
                            }
                        }
                        else if (error == std::errc::permission_denied)
                        {
                            status = fs_status_permission_error;
                        }
                    }
                }
            }

            std::uint32_t stat_flags = 0U;
            std::uint32_t mode = 0U;
            std::uint32_t file_size = 0U;
            if (host_path)
            {
                std::error_code error;
                const std::filesystem::file_status file_status =
                    std::filesystem::status(*host_path, error);
                if (error)
                {
                    status = error == std::errc::permission_denied
                        ? fs_status_permission_error
                        : fs_status_not_found;
                }
                else if (std::filesystem::is_directory(file_status))
                {
                    status = fs_status_ok;
                    stat_flags = fs_stat_flag_directory;
                }
                else if (std::filesystem::is_regular_file(file_status))
                {
                    const std::uintmax_t host_file_size =
                        std::filesystem::file_size(*host_path, error);
                    if (error)
                    {
                        status = error == std::errc::permission_denied
                            ? fs_status_permission_error
                            : fs_status_not_found;
                    }
                    else if (host_file_size >
                             std::numeric_limits<std::uint32_t>::max())
                    {
                        status = fs_status_file_too_big;
                    }
                    else
                    {
                        status = fs_status_ok;
                        stat_flags = fs_stat_flag_file;
                        file_size = static_cast<std::uint32_t>(host_file_size);
                    }
                }
                else
                {
                    status = fs_status_not_file;
                }

                if (status == fs_status_ok || stat_flags == fs_stat_flag_directory)
                {
                    const auto permissions = file_status.permissions();
                    const auto add_mode = [&](std::filesystem::perms permission,
                                              std::uint32_t guest_mode) {
                        if ((permissions & permission) != std::filesystem::perms::none)
                        {
                            mode |= guest_mode;
                        }
                    };
                    add_mode(std::filesystem::perms::owner_read, fs_mode_read_owner);
                    add_mode(std::filesystem::perms::owner_write, fs_mode_write_owner);
                    add_mode(std::filesystem::perms::owner_exec, fs_mode_exec_owner);
                    add_mode(std::filesystem::perms::group_read, fs_mode_read_group);
                    add_mode(std::filesystem::perms::group_write, fs_mode_write_group);
                    add_mode(std::filesystem::perms::group_exec, fs_mode_exec_group);
                    add_mode(std::filesystem::perms::others_read, fs_mode_read_other);
                    add_mode(std::filesystem::perms::others_write, fs_mode_write_other);
                    add_mode(std::filesystem::perms::others_exec, fs_mode_exec_other);
                }
            }

            if (status == fs_status_ok)
            {
                core.memory.zero_fill(guest_stat_address, fs_stat_size);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_flags_offset, stat_flags);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_mode_offset, mode);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_owner_offset, 0U);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_group_offset, 0U);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_size_offset, file_size);
                // The host filesystem API does not expose Cafe allocation
                // units or quota accounting portably. Leave those fields zero
                // rather than presenting the logical byte size as allocated
                // storage or inventing title-specific values.
                core.memory.write32_be(
                    guest_stat_address + fs_stat_allocated_size_offset,
                    0U);
                core.memory.write64_be(
                    guest_stat_address + fs_stat_quota_size_offset, 0U);
                core.memory.write32_be(
                    guest_stat_address + fs_stat_entry_id_offset, 0U);
                core.memory.write64_be(
                    guest_stat_address + fs_stat_created_offset, 0U);
                core.memory.write64_be(
                    guest_stat_address + fs_stat_modified_offset, 0U);
            }

            core.memory.write32_be(
                command_body + fs_cmd_block_status_offset,
                fs_cmd_block_available_status);
            core.memory.write32_be(
                command_body + fs_cmd_block_async_result_status_offset,
                status);
            core.state.gpr[3] = status;
        });
    dispatcher.register_function(
        "coreinit",
        "FSOpenFile",
        [](EspressoCore& core) {
            const std::uint32_t client = core.state.gpr[3];
            const std::uint32_t command_block = core.state.gpr[4];
            const std::uint32_t guest_path_address = core.state.gpr[5];
            const std::uint32_t guest_mode_address = core.state.gpr[6];
            const std::uint32_t guest_handle_address = core.state.gpr[7];
            if (!core.fs_initialized || client == 0U || command_block == 0U ||
                guest_path_address == 0U || guest_mode_address == 0U ||
                guest_handle_address == 0U)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            const auto registration = std::find_if(
                core.fs_clients.begin(), core.fs_clients.end(),
                [client](const FsClientRegistration& candidate) {
                    return candidate.client_address == client;
                });
            if (registration == core.fs_clients.end())
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            constexpr std::uint64_t guest_address_space_end =
                std::uint64_t{1} << 32U;
            const std::uint64_t command_body_address =
                (static_cast<std::uint64_t>(command_block) +
                 fs_cmd_block_body_alignment_mask) &
                ~static_cast<std::uint64_t>(fs_cmd_block_body_alignment_mask);
            if (command_body_address + fs_cmd_block_async_result_status_offset +
                    sizeof(std::uint32_t) > guest_address_space_end)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }
            const std::uint32_t command_body =
                static_cast<std::uint32_t>(command_body_address);
            const std::uint32_t command_status = core.memory.read32_be(
                command_body + fs_cmd_block_status_offset);
            if (command_status != fs_cmd_block_initialized_status &&
                command_status != fs_cmd_block_available_status)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            core.memory.validate_write_range(
                guest_handle_address, sizeof(std::uint32_t));
            core.memory.validate_write_range(
                command_body + fs_cmd_block_status_offset,
                sizeof(std::uint32_t));
            core.memory.validate_write_range(
                command_body + fs_cmd_block_async_result_status_offset,
                sizeof(std::uint32_t));

            const auto read_guest_string = [&](std::uint32_t address,
                                               std::size_t maximum_length)
                -> std::optional<std::string> {
                std::string value;
                value.reserve(maximum_length);
                for (std::size_t offset = 0U; offset < maximum_length; ++offset)
                {
                    const std::uint64_t current =
                        static_cast<std::uint64_t>(address) + offset;
                    if (current >= guest_address_space_end)
                    {
                        return std::nullopt;
                    }
                    const char character = static_cast<char>(core.memory.read8(
                        static_cast<std::uint32_t>(current)));
                    if (character == '\0')
                    {
                        return value;
                    }
                    value.push_back(character);
                }
                return std::nullopt;
            };

            const auto guest_path = read_guest_string(guest_path_address, 0x280U);
            const auto guest_mode = read_guest_string(guest_mode_address, 0x10U);
            std::uint32_t status = fs_status_media_not_ready;
            std::optional<std::filesystem::path> host_path;
            constexpr std::string_view content_volume = "/vol/content";
            if (guest_path && guest_mode && !core.fs_content_root.empty() &&
                (*guest_path == content_volume ||
                 (guest_path->size() > content_volume.size() &&
                  guest_path->starts_with(content_volume) &&
                  (*guest_path)[content_volume.size()] == '/')))
            {
                status = fs_status_not_found;
                const std::string_view relative_utf8 = guest_path->size() ==
                        content_volume.size()
                    ? std::string_view{}
                    : std::string_view(*guest_path).substr(
                          content_volume.size() + 1U);
                std::u8string relative_path_utf8;
                relative_path_utf8.reserve(relative_utf8.size());
                for (const unsigned char character : relative_utf8)
                {
                    relative_path_utf8.push_back(
                        static_cast<char8_t>(character));
                }
                const std::filesystem::path relative(relative_path_utf8);
                bool safe_relative_path = !relative.is_absolute();
                for (const auto& component : relative)
                {
                    if (component == "." || component == "..")
                    {
                        safe_relative_path = false;
                        break;
                    }
                }

                if (safe_relative_path)
                {
                    std::error_code error;
                    const std::filesystem::path canonical_root =
                        std::filesystem::canonical(core.fs_content_root, error);
                    if (!error)
                    {
                        const auto candidate = canonical_root / relative;
                        const auto canonical_candidate =
                            std::filesystem::canonical(candidate, error);
                        if (!error)
                        {
                            const auto within_root =
                                canonical_candidate.lexically_relative(canonical_root);
                            const bool escapes_root =
                                within_root.empty() && canonical_candidate != canonical_root
                                    ? true
                                    : (!within_root.empty() &&
                                       *within_root.begin() == "..");
                            if (!escapes_root)
                            {
                                host_path = canonical_candidate;
                            }
                        }
                        else if (error == std::errc::permission_denied)
                        {
                            status = fs_status_permission_error;
                        }
                    }
                }
            }

            if (!guest_path || !guest_mode)
            {
                status = fs_status_fatal_error;
            }
            else if (*guest_mode != "r" && *guest_mode != "rb")
            {
                status = fs_status_permission_error;
            }
            else if (host_path)
            {
                std::error_code error;
                const auto file_status = std::filesystem::status(*host_path, error);
                if (error)
                {
                    status = error == std::errc::permission_denied
                        ? fs_status_permission_error
                        : fs_status_not_found;
                }
                else if (!std::filesystem::is_regular_file(file_status))
                {
                    status = std::filesystem::is_directory(file_status)
                        ? fs_status_not_file
                        : fs_status_access_error;
                }
                else
                {
                    status = fs_status_ok;
                }
            }

            if (status == fs_status_ok &&
                (core.next_fs_file_handle == 0U ||
                 core.next_fs_file_handle > 0x7FFFFFFFU))
            {
                status = fs_status_fatal_error;
            }
            if (status == fs_status_ok)
            {
                core.fs_open_files.reserve(core.fs_open_files.size() + 1U);
                const std::uint32_t handle = core.next_fs_file_handle;
                core.fs_open_files.push_back(
                    {handle, client, *host_path, 0U});
                core.next_fs_file_handle = handle + 1U;
                core.memory.write32_be(guest_handle_address, handle);
            }

            core.memory.write32_be(
                command_body + fs_cmd_block_status_offset,
                fs_cmd_block_available_status);
            core.memory.write32_be(
                command_body + fs_cmd_block_async_result_status_offset,
                status);
            core.state.gpr[3] = status;
        });
    dispatcher.register_function(
        "coreinit",
        "FSAddClient",
        [](EspressoCore& core) {
            constexpr std::uint64_t guest_address_space_end =
                std::uint64_t{1} << 32U;

            const std::uint32_t client = core.state.gpr[3];
            const std::uint32_t error_mask = core.state.gpr[4];
            (void)error_mask;

            if (!core.fs_initialized || client == 0U)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            const std::uint64_t client_address = client;
            const std::uint64_t client_end = client_address + fs_client_size;
            if (client_end > guest_address_space_end)
            {
                throw HleExecutionError(
                    "FSAddClient client storage wraps the guest address space");
            }

            const std::uint64_t body_address =
                (client_address + fs_client_body_alignment_mask) &
                ~static_cast<std::uint64_t>(fs_client_body_alignment_mask);
            const std::uint64_t body_end = body_address + fs_client_body_size;
            if (body_address > std::numeric_limits<std::uint32_t>::max() ||
                body_end > client_end)
            {
                throw HleExecutionError(
                    "FSAddClient cannot derive a valid aligned FSClientBody");
            }

            const std::uint32_t body = static_cast<std::uint32_t>(body_address);
            core.memory.validate_write_range(client, fs_client_size);

            const bool duplicate = std::any_of(
                core.fs_clients.begin(), core.fs_clients.end(),
                [client, body, client_end](const FsClientRegistration& registration) {
                    const std::uint64_t registered_begin =
                        registration.client_address;
                    const std::uint64_t registered_end =
                        registered_begin + fs_client_size;
                    return registration.client_address == client ||
                           registration.body_address == body ||
                           (static_cast<std::uint64_t>(client) < registered_end &&
                            registered_begin < client_end);
                });
            if (duplicate || core.next_fs_client_handle == 0U ||
                core.next_fs_client_handle > 0x7FFFFFFFU)
            {
                core.state.gpr[3] = fs_status_fatal_error;
                return;
            }

            // Reserve bookkeeping before touching guest memory, so host-side
            // allocation failure cannot leave an initialized but unregistered client.
            core.fs_clients.reserve(core.fs_clients.size() + 1U);
            const std::uint32_t handle = core.next_fs_client_handle;

            core.memory.zero_fill(client, fs_client_size);

            const std::uint32_t fsm = body + fs_client_fsm_offset;
            const std::uint32_t command_queue = body + fs_client_cmd_queue_offset;
            core.memory.write32_be(
                body + fs_client_handle_offset, handle);
            core.memory.write32_be(
                fsm + fs_client_fsm_current_state_offset,
                fs_volume_state_ready);
            core.memory.write32_be(
                fsm + fs_client_fsm_volume_state_offset,
                fs_volume_state_ready);

            core.memory.write32_be(
                command_queue + fs_cmd_queue_head_offset, 0U);
            core.memory.write32_be(
                command_queue + fs_cmd_queue_tail_offset, 0U);
            initialize_guest_fast_mutex(
                core.memory,
                command_queue + fs_cmd_queue_mutex_offset,
                0U);
            core.memory.write32_be(
                command_queue + fs_cmd_queue_dequeue_handler_offset, 0U);
            core.memory.write32_be(
                command_queue + fs_cmd_queue_active_count_offset, 0U);
            core.memory.write32_be(
                command_queue + fs_cmd_queue_max_active_count_offset, 1U);
            core.memory.write32_be(
                command_queue + fs_cmd_queue_status_offset, 0U);

            core.memory.write32_be(
                body + fs_client_last_dequeued_command_offset, 0U);
            core.memory.write32_be(
                body + fs_client_emulated_error_offset, 0U);
            initialize_guest_fast_mutex(
                core.memory, body + fs_client_mutex_offset, 0U);

            const std::uint32_t alarm = body + fs_client_fsm_alarm_offset;
            core.memory.zero_fill(alarm, fs_alarm_size);
            core.memory.write32_be(alarm, fs_alarm_tag);
            core.memory.write32_be(
                body + fs_client_last_error_offset, 0U);
            core.memory.write32_be(
                body + fs_client_last_error_without_volume_offset, 0U);
            core.memory.write32_be(
                body + fs_client_mount_source_type_offset, 0U);
            core.memory.write32_be(
                body + fs_client_pointer_offset, client);

            if (core.fs_clients.empty())
            {
                const std::uint32_t link = body + fs_client_link_offset;
                core.memory.write32_be(
                    link + fs_client_link_next_offset, body);
                core.memory.write32_be(
                    link + fs_client_link_prev_offset, body);
            }
            else
            {
                const std::uint32_t head =
                    core.fs_clients.front().body_address;
                const std::uint32_t tail =
                    core.fs_clients.back().body_address;
                const std::uint32_t link = body + fs_client_link_offset;
                core.memory.write32_be(
                    link + fs_client_link_next_offset, head);
                core.memory.write32_be(
                    link + fs_client_link_prev_offset, tail);
                core.memory.write32_be(
                    tail + fs_client_link_offset + fs_client_link_next_offset,
                    body);
                core.memory.write32_be(
                    head + fs_client_link_offset + fs_client_link_prev_offset,
                    body);
            }

            // IOS_Open("/dev/fsa") is not modeled; this stable positive guest
            // handle represents that per-client connection for now.
            core.fs_clients.push_back({client, body, handle});
            core.next_fs_client_handle = handle + 1U;
            core.state.gpr[3] = 0U;
        });
    dispatcher.register_function(
        "coreinit",
        "FSSetStateChangeNotification",
        [](EspressoCore& core) {
            const std::uint32_t client = core.state.gpr[3];
            const std::uint32_t params_address = core.state.gpr[4];
            if (!core.fs_initialized)
            {
                throw HleExecutionError(
                    "FSSetStateChangeNotification called before FSInit");
            }
            if (client == 0U)
            {
                throw HleExecutionError(
                    "FSSetStateChangeNotification received a null FSClient");
            }

            const auto registration = std::find_if(
                core.fs_clients.begin(), core.fs_clients.end(),
                [client](const FsClientRegistration& candidate) {
                    return candidate.client_address == client;
                });
            if (registration == core.fs_clients.end())
            {
                throw HleExecutionError(
                    "FSSetStateChangeNotification received an unregistered FSClient");
            }

            constexpr std::uint64_t guest_address_space_end =
                std::uint64_t{1} << 32U;
            const std::uint64_t expected_body =
                (static_cast<std::uint64_t>(client) +
                 fs_client_body_alignment_mask) &
                ~static_cast<std::uint64_t>(fs_client_body_alignment_mask);
            if (expected_body != registration->body_address ||
                expected_body + fs_client_body_size > guest_address_space_end)
            {
                throw HleExecutionError(
                    "FSSetStateChangeNotification has inconsistent FSClient bookkeeping");
            }

            const std::uint32_t fsm =
                registration->body_address + fs_client_fsm_offset;
            const std::uint32_t state_change_info =
                fsm + fs_client_fsm_state_change_info_offset;
            if (params_address == 0U)
            {
                // A null parameter block disables notifications without
                // disturbing the previously registered callback metadata.
                core.memory.write32_be(
                    fsm + fs_client_fsm_send_state_change_notification_offset,
                    0U);
                return;
            }

            // Snapshot the whole ABI parameter block before changing client
            // state, so an invalid or partially mapped input cannot leave a
            // half-registered notification behind.
            std::array<std::uint8_t, 3U * sizeof(std::uint32_t)> params{};
            core.memory.read_bytes(params_address, params);
            const auto read_be32 = [&params](std::size_t offset) {
                return (static_cast<std::uint32_t>(params[offset]) << 24U) |
                    (static_cast<std::uint32_t>(params[offset + 1U]) << 16U) |
                    (static_cast<std::uint32_t>(params[offset + 2U]) << 8U) |
                    static_cast<std::uint32_t>(params[offset + 3U]);
            };
            const std::uint32_t callback =
                read_be32(fs_state_change_callback_offset);
            const std::uint32_t param = read_be32(fs_state_change_param_offset);
            const std::uint32_t queue = read_be32(fs_state_change_queue_offset);

            // Affogato has no AppIO-thread or state-change delivery model yet.
            // Preserve the caller's queue pointer verbatim; do not fabricate a
            // default queue or enqueue/invoke anything until that is modeled.
            core.memory.write32_be(
                state_change_info + fs_state_change_callback_offset, callback);
            core.memory.write32_be(
                state_change_info + fs_state_change_param_offset, param);
            core.memory.write32_be(
                state_change_info + fs_state_change_queue_offset, queue);
            core.memory.write32_be(
                state_change_info + fs_state_change_message_offset +
                    fs_message_data_offset,
                state_change_info);
            core.memory.write32_be(
                state_change_info + fs_state_change_message_offset +
                    fs_message_type_offset,
                fs_state_change_message_type);
            core.memory.write32_be(
                state_change_info + fs_state_change_client_offset, client);
            core.memory.write32_be(
                fsm + fs_client_fsm_send_state_change_notification_offset,
                1U);
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
        "OSBlockSet",
        [](EspressoCore& core) {
            const std::uint32_t destination = core.state.gpr[3];
            const std::uint8_t value = static_cast<std::uint8_t>(core.state.gpr[4]);
            const std::uint32_t size = core.state.gpr[5];
            if (size != 0U)
            {
                core.memory.fill_bytes(destination, size, value);
            }
            core.state.gpr[3] = destination;
        });
    dispatcher.register_function(
        "coreinit",
        "OSFastMutex_Init",
        [](EspressoCore& core) {
            const std::uint32_t mutex = core.state.gpr[3];
            initialize_guest_fast_mutex(core.memory, mutex, core.state.gpr[4]);
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
        "OSInitMessageQueue",
        [](EspressoCore& core) {
            constexpr std::uint64_t guest_address_space_end =
                std::uint64_t{1} << 32U;
            const std::uint32_t queue = core.state.gpr[3];
            const std::uint32_t messages = core.state.gpr[4];
            const std::uint32_t size = core.state.gpr[5];
            if (queue == 0U)
            {
                throw HleExecutionError(
                    "OSInitMessageQueue received null OSMessageQueue pointer");
            }
            if (static_cast<std::uint64_t>(queue) + os_message_queue_size >
                guest_address_space_end)
            {
                throw HleExecutionError(
                    "OSInitMessageQueue OSMessageQueue address wraps guest address space");
            }
            try
            {
                // Validate the entire object before zeroing or populating it,
                // so an invalid queue cannot be left partially initialized.
                core.memory.validate_write_range(queue, os_message_queue_size);
            }
            catch (const GuestMemoryFault&)
            {
                throw HleExecutionError(
                    "OSInitMessageQueue OSMessageQueue is outside writable guest memory");
            }

            initialize_guest_message_queue(core.memory, queue, messages, size);
        });
    dispatcher.register_function(
        "coreinit",
        "OSSendMessage",
        [](EspressoCore& core) {
            const bool sent = send_guest_message(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                core.state.gpr[5]);
            core.state.gpr[3] = sent ? 1U : 0U;
        });
    dispatcher.register_function(
        "coreinit",
        "OSReceiveMessage",
        [](EspressoCore& core) {
            const bool received = receive_guest_message(
                core, core.state.gpr[3], core.state.gpr[4], core.state.gpr[5]);
            core.state.gpr[3] = received ? 1U : 0U;
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
        "OSTryLockMutex",
        [](EspressoCore& core) {
            core.state.gpr[3] =
                lock_guest_os_mutex(core, core.state.gpr[3]) ? 1U : 0U;
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
            const std::uint64_t base_time = current_os_epoch_ticks();
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
