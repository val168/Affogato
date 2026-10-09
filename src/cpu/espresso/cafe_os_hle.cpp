#include "cpu/espresso/cafe_os_hle.hpp"

#include "cpu/espresso/guest_mutex.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <limits>
#include <optional>
#include <set>
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
constexpr std::uint32_t os_thread_size = 0x6A0U;
constexpr std::uint32_t os_thread_tag_offset = 0x320U;
constexpr std::uint32_t os_thread_priority_offset = 0x32CU;
constexpr std::uint32_t os_thread_base_priority_offset = 0x330U;
constexpr std::uint32_t os_thread_type_offset = 0x5BCU;
constexpr std::uint32_t os_thread_tag = 0x74487244U; // tHrD
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
constexpr std::uint32_t fs_status_fatal_error = 0xFFFFFC00U;
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

[[nodiscard]] std::uint32_t allocatable_size_for_frame_heap(
    const GuestMemory& memory,
    std::uint32_t heap,
    std::uint32_t alignment)
{
    constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
    if (heap == 0U)
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForFrmHeapEx received null heap handle");
    }
    if (static_cast<std::uint64_t>(heap) + foreground_frame_heap_header_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForFrmHeapEx heap header wraps guest address space");
    }

    std::array<std::uint8_t, foreground_frame_heap_header_size> header{};
    try
    {
        memory.read_bytes(heap, header);
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForFrmHeapEx heap header is outside mapped guest memory");
    }

    if (memory.read32_be(heap) != foreground_frame_heap_tag)
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForFrmHeapEx handle is not an FRMH heap");
    }

    const std::uint64_t minimum_data_start =
        static_cast<std::uint64_t>(heap) + foreground_frame_heap_header_size;
    const std::uint32_t data_start =
        memory.read32_be(heap + mem_exp_heap_data_start_offset);
    const std::uint32_t data_end =
        memory.read32_be(heap + mem_exp_heap_data_end_offset);
    const std::uint32_t head = memory.read32_be(heap + frame_heap_head_offset);
    const std::uint32_t tail = memory.read32_be(heap + frame_heap_tail_offset);
    if (data_start < minimum_data_start || data_start > data_end ||
        head < data_start || head > tail || tail > data_end)
    {
        throw HleExecutionError(
            "MEMGetAllocatableSizeForFrmHeapEx has invalid frame-heap bounds");
    }

    const std::uint64_t aligned_head =
        (static_cast<std::uint64_t>(head) + alignment - 1U) &
        ~static_cast<std::uint64_t>(alignment - 1U);
    if (aligned_head >= tail)
    {
        return 0U;
    }
    return static_cast<std::uint32_t>(tail - aligned_head);
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
            const std::uint64_t system_time = current_epoch_ticks >= base_time
                ? current_epoch_ticks - base_time
                : 0U;
            set_u64_return(core.state, system_time);
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
