#pragma once

#include "cpu/espresso/cpu_state.hpp"
#include "cpu/espresso/gx2_hle.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"

#include <cstddef>
#include <array>
#include <chrono>
#include <filesystem>
#include <string>
#include <vector>

namespace affogato::cpu::espresso
{

enum class StepResult
{
    executed,
    unsupported_instruction,
    unimplemented_hle_call,
};

enum class StopReason
{
    instruction_limit,
    unsupported_instruction,
    unimplemented_hle_call,
    hle_error,
    memory_fault,
};

inline constexpr std::size_t instruction_history_capacity = 32;

struct InstructionHistoryEntry
{
    std::uint32_t cia{};
    std::uint32_t instruction_word{};
    bool has_instruction_word{};
    std::string opcode_name;
    std::array<std::uint8_t, 3> source_registers{};
    std::array<std::uint32_t, 3> source_values{};
    std::uint8_t source_count{};
    bool has_immediate{};
    bool immediate_hex{};
    std::int32_t immediate{};
    bool has_destination{};
    std::uint8_t destination_register{};
    std::uint32_t destination_value{};
    bool has_carry_result{};
    bool carry_result{};
    bool has_carry_input{};
    bool carry_input{};
    bool has_overflow_result{};
    bool overflow_result{};
    bool summary_overflow_result{};
    bool has_old_destination{};
    std::uint8_t old_destination_register{};
    std::uint32_t old_destination_value{};
    bool has_rotate_fields{};
    std::uint8_t rotate_shift{};
    std::uint8_t rotate_mask_begin{};
    std::uint8_t rotate_mask_end{};
    bool has_effective_address{};
    std::uint32_t effective_address{};
    bool has_string_transfer{};
    bool string_transfer_store{};
    std::uint8_t string_transfer_first_register{};
    std::uint8_t string_transfer_base_register{};
    std::uint32_t string_transfer_base_value{};
    std::uint8_t string_transfer_byte_count{};
    std::uint8_t string_transfer_register_count{};
    std::array<std::uint8_t, 32> string_transfer_registers{};
    std::array<std::uint32_t, 32> string_transfer_register_values{};
    std::array<std::uint8_t, 127> string_transfer_bytes{};
    bool string_transfer_indexed{};
    std::uint8_t string_transfer_index_register{};
    std::uint32_t string_transfer_index_value{};
    bool has_register_range{};
    std::uint8_t range_first_register{};
    std::uint8_t range_last_register{};
    std::uint8_t memory_base_register{};
    std::int32_t memory_displacement{};
    bool has_fp_destination{};
    std::uint8_t fp_destination_register{};
    std::uint64_t fp_destination_value{};
    bool has_fp_source{};
    std::uint8_t fp_source_register{};
    std::uint64_t fp_source_value{};
    bool has_stored_single_value{};
    std::uint32_t stored_single_value{};
    bool has_stored_double_value{};
    std::uint64_t stored_double_value{};
    bool has_stored_halfword_value{};
    std::uint16_t stored_halfword_value{};
    bool has_stored_integer_word_value{};
    std::uint32_t stored_integer_word_value{};
    bool has_paired_fp_state{};
    bool paired_fp_unary{};
    bool paired_fp_scalar_multiply{};
    bool paired_fp_vector_multiply{};
    bool paired_fp_quantized_store{};
    std::uint8_t paired_fp_store_gqr{};
    std::uint8_t paired_fp_store_type{};
    bool paired_fp_store_w{};
    std::uint32_t paired_fp_store_value0{};
    std::uint32_t paired_fp_store_value1{};
    std::uint8_t paired_fp_destination_register{};
    std::uint8_t paired_fp_source_a_register{};
    std::uint8_t paired_fp_source_b_register{};
    std::uint64_t paired_fp_source_a_ps0{};
    std::uint64_t paired_fp_source_a_ps1{};
    std::uint64_t paired_fp_source_b_ps0{};
    std::uint64_t paired_fp_source_b_ps1{};
    std::uint64_t paired_fp_destination_ps0{};
    std::uint64_t paired_fp_destination_ps1{};
    bool has_paired_fp_select{};
    std::uint8_t paired_fp_select_destination{};
    std::uint8_t paired_fp_select_a{};
    std::uint8_t paired_fp_select_b{};
    std::uint8_t paired_fp_select_c{};
    std::uint64_t paired_fp_select_a_ps0{};
    std::uint64_t paired_fp_select_a_ps1{};
    std::uint64_t paired_fp_select_b_ps0{};
    std::uint64_t paired_fp_select_b_ps1{};
    std::uint64_t paired_fp_select_c_ps0{};
    std::uint64_t paired_fp_select_c_ps1{};
    std::uint64_t paired_fp_select_result_ps0{};
    std::uint64_t paired_fp_select_result_ps1{};
    bool has_fp_compare{};
    std::uint8_t fp_compare_cr_field{};
    std::uint8_t fp_compare_a_register{};
    std::uint8_t fp_compare_b_register{};
    std::uint64_t fp_compare_a_raw{};
    std::uint64_t fp_compare_b_raw{};
    std::uint8_t fp_compare_result{};
    bool has_fp_arithmetic{};
    bool has_fp_arithmetic_result{};
    bool has_fp_arithmetic_single_bits{};
    std::uint8_t fp_arithmetic_source_count{};
    std::uint8_t fp_arithmetic_destination{};
    std::uint8_t fp_arithmetic_source_a{};
    std::uint8_t fp_arithmetic_source_b{};
    bool has_fp_arithmetic_source_c{};
    std::uint8_t fp_arithmetic_source_c{};
    std::uint64_t fp_arithmetic_a_raw{};
    std::uint64_t fp_arithmetic_b_raw{};
    std::uint64_t fp_arithmetic_c_raw{};
    std::uint64_t fp_arithmetic_result_raw{};
    std::uint32_t fp_arithmetic_single_bits{};
    bool has_fp_move{};
    std::uint8_t fp_move_destination{};
    std::uint8_t fp_move_source{};
    std::uint64_t fp_move_source_value{};
    std::uint64_t fp_move_destination_value{};
    std::uint64_t fp_move_destination_ps1{};
    bool has_fp_status_move{};
    std::uint8_t fp_status_move_destination{};
    std::uint32_t fp_status_move_source{};
    std::uint64_t fp_status_move_result{};
    std::uint64_t fp_status_move_destination_ps1{};
    bool has_fpscr_bit_operation{};
    bool fpscr_bit_set{};
    std::uint8_t fpscr_bit_number{};
    std::uint32_t fpscr_before_bit_operation{};
    std::uint32_t fpscr_after_bit_operation{};
    bool has_fpscr_field_write{};
    std::uint8_t fpscr_field_number{};
    std::uint8_t fpscr_field_immediate{};
    std::uint32_t fpscr_before_field_write{};
    std::uint32_t fpscr_after_field_write{};
    bool has_fpscr_masked_write{};
    std::uint8_t fpscr_masked_write_source{};
    std::uint8_t fpscr_masked_write_mask{};
    std::uint64_t fpscr_masked_write_source_raw{};
    std::uint32_t fpscr_before_masked_write{};
    std::uint32_t fpscr_after_masked_write{};
    bool has_fp_select{};
    std::uint8_t fp_select_destination{};
    std::uint8_t fp_select_a{};
    std::uint8_t fp_select_b{};
    std::uint8_t fp_select_c{};
    std::uint64_t fp_select_a_raw{};
    std::uint64_t fp_select_b_raw{};
    std::uint64_t fp_select_c_raw{};
    std::uint64_t fp_select_result_raw{};
    bool has_fp_integer_conversion{};
    std::uint8_t fp_integer_conversion_source{};
    std::uint64_t fp_integer_conversion_source_raw{};
    std::uint8_t fp_integer_conversion_destination{};
    std::uint32_t fp_integer_conversion_result{};
    std::uint64_t fp_integer_conversion_encoded_raw{};
    bool fp_integer_conversion_invalid{};
    bool fp_integer_conversion_write_suppressed{};
    bool has_cr_bit_operation{};
    std::uint8_t cr_bit_destination{};
    std::uint8_t cr_bit_source_a{};
    std::uint8_t cr_bit_source_b{};
    bool cr_bit_source_a_value{};
    bool cr_bit_source_b_value{};
    bool cr_bit_result{};
    bool has_cr_field_move{};
    std::uint8_t cr_field_destination{};
    std::uint8_t cr_field_source{};
    std::uint8_t cr_field_value{};
    bool has_fpscr_cr_move{};
    std::uint8_t fpscr_cr_destination{};
    std::uint8_t fpscr_cr_source{};
    std::uint8_t fpscr_cr_value{};
    std::uint32_t fpscr_before_cr_move{};
    std::uint32_t fpscr_after_cr_move{};
    bool has_xer_cr_move{};
    std::uint8_t xer_cr_field{};
    std::uint8_t xer_cr_value{};
    bool completed{};
};

struct RunResult
{
    std::size_t steps{};
    StopReason reason{StopReason::instruction_limit};
    std::uint32_t cia{};
    std::uint32_t instruction_word{};
    bool has_instruction_word{};
    std::string hle_call;
    std::string detail;
    std::vector<InstructionHistoryEntry> instruction_history;
};

struct FsClientRegistration
{
    std::uint32_t client_address{};
    std::uint32_t body_address{};
    std::uint32_t synthetic_handle{};
};

struct FsOpenFileState
{
    std::uint32_t handle{};
    std::uint32_t client_address{};
    std::filesystem::path host_path;
    std::uint64_t position{};
};

struct GuestThreadRuntimeState
{
    std::uint32_t address{};
    std::uint64_t wake_tick{};
    bool sleeping{};
};

struct ProcUiCallbackRegistration
{
    std::uint32_t type{};
    std::uint32_t callback{};
    std::uint32_t parameter{};
    std::uint32_t priority{};
};

class EspressoCore
{
public:
    explicit EspressoCore(std::size_t memory_size)
        : memory(memory_size)
    {
    }

    void reset() noexcept
    {
        state.reset();
        timebase_origin_ = std::chrono::steady_clock::now();
        current_thread_address = 0;
        os_sleep_ticks_offset = 0;
        guest_scheduler_ticks = 0U;
        guest_thread_switches = 0U;
        guest_threads.clear();
        fs_initialized = false;
        fs_content_root.clear();
        nn_act_initialize_count = 0U;
        nn_act_current_slot = 1U;
        nn_save_initialized = false;
        proc_ui_initialized = false;
        proc_ui_save_callback = 0U;
        proc_ui_save_callback_ex = 0U;
        proc_ui_save_callback_ex_argument = 0U;
        proc_ui_in_foreground = false;
        proc_ui_in_shutdown = false;
        proc_ui_status = 0U;
        proc_ui_draw_done_release = false;
        proc_ui_callbacks.clear();
        im_dim_enabled = true;
        home_button_menu_enabled = true;
        wpad_urcc_enabled = false;
        wpad_wii_remote_enabled = false;
        kpad_initialized = false;
        kpad_unified_status_buffer = 0U;
        kpad_unified_status_count = 0U;
        kpad_mpls_workarea = 0U;
        gx2 = {};
        fs_clients.clear();
        next_fs_client_handle = 1U;
        fs_open_files.clear();
        next_fs_file_handle = 1U;
        guest_heap_cursor = 0;
        guest_heap_limit = 0;
        base_heap_handles.fill(0U);
        mem2_heap_region_begin = 0U;
        mem2_heap_region_end = 0U;
        instruction_history_ = {};
        instruction_history_next_ = 0;
        instruction_history_count_ = 0;
        pending_history_entry_ = {};
        has_pending_history_entry_ = false;
    }

    void configure_guest_heap(std::uint32_t begin, std::uint32_t end) noexcept
    {
        guest_heap_cursor = begin;
        guest_heap_limit = end;
        base_heap_handles.fill(0U);
        mem2_heap_region_begin = 0U;
        mem2_heap_region_end = 0U;
    }

    [[nodiscard]] std::uint32_t allocate_guest_memory(
        std::uint32_t size,
        std::uint32_t alignment = 16) noexcept
    {
        if (size == 0 || alignment == 0 || (alignment & (alignment - 1U)) != 0)
        {
            return 0;
        }
        const std::uint64_t aligned =
            (static_cast<std::uint64_t>(guest_heap_cursor) + alignment - 1U) &
            ~static_cast<std::uint64_t>(alignment - 1U);
        const std::uint64_t allocation_end = aligned + size;
        if (aligned > guest_heap_limit || allocation_end > guest_heap_limit)
        {
            return 0;
        }
        guest_heap_cursor = static_cast<std::uint32_t>(allocation_end);
        return static_cast<std::uint32_t>(aligned);
    }

    [[nodiscard]] StepResult step();
    [[nodiscard]] RunResult run(std::size_t max_steps);

    CpuState state{};
    GuestMemory memory;
    HleDispatcher hle;
    std::uint32_t current_thread_address{};
    std::uint64_t os_sleep_ticks_offset{};
    std::uint64_t guest_scheduler_ticks{};
    std::uint64_t guest_thread_switches{};
    std::vector<GuestThreadRuntimeState> guest_threads;
    bool fs_initialized{};
    std::filesystem::path fs_content_root;
    std::uint32_t nn_act_initialize_count{};
    std::uint8_t nn_act_current_slot{1U};
    bool nn_save_initialized{};
    bool proc_ui_initialized{};
    std::uint32_t proc_ui_save_callback{};
    std::uint32_t proc_ui_save_callback_ex{};
    std::uint32_t proc_ui_save_callback_ex_argument{};
    bool proc_ui_in_foreground{};
    bool proc_ui_in_shutdown{};
    std::uint32_t proc_ui_status{};
    bool proc_ui_draw_done_release{};
    std::vector<ProcUiCallbackRegistration> proc_ui_callbacks;
    bool im_dim_enabled{true};
    bool home_button_menu_enabled{true};
    bool wpad_urcc_enabled{};
    bool wpad_wii_remote_enabled{};
    bool kpad_initialized{};
    std::uint32_t kpad_unified_status_buffer{};
    std::uint32_t kpad_unified_status_count{};
    std::uint32_t kpad_mpls_workarea{};
    Gx2RuntimeState gx2{};
    std::vector<FsClientRegistration> fs_clients;
    std::uint32_t next_fs_client_handle{1U};
    std::vector<FsOpenFileState> fs_open_files;
    std::uint32_t next_fs_file_handle{1U};
    std::uint32_t guest_heap_cursor{};
    std::uint32_t guest_heap_limit{};
    std::array<std::uint32_t, 9> base_heap_handles{};
    std::uint32_t mem2_heap_region_begin{};
    std::uint32_t mem2_heap_region_end{};

private:
    friend class Recompiler;

    // Scoped approximation until Espresso instruction-cycle accounting exists.
    std::chrono::steady_clock::time_point timebase_origin_{
        std::chrono::steady_clock::now()};
    std::uint32_t current_instruction_word_{};
    bool current_instruction_word_fetched_{};
    std::array<InstructionHistoryEntry, instruction_history_capacity> instruction_history_{};
    std::size_t instruction_history_next_{};
    std::size_t instruction_history_count_{};
    InstructionHistoryEntry pending_history_entry_{};
    bool has_pending_history_entry_{};

    void append_instruction_history(InstructionHistoryEntry entry) noexcept;
    [[nodiscard]] std::vector<InstructionHistoryEntry> instruction_history_snapshot() const;
};

[[nodiscard]] std::string format_instruction_history(const RunResult& result);

}
