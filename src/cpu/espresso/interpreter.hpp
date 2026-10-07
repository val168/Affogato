#pragma once

#include "cpu/espresso/cpu_state.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"

#include <cstddef>
#include <array>
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
    bool has_old_destination{};
    std::uint8_t old_destination_register{};
    std::uint32_t old_destination_value{};
    bool has_rotate_fields{};
    std::uint8_t rotate_shift{};
    std::uint8_t rotate_mask_begin{};
    std::uint8_t rotate_mask_end{};
    bool has_effective_address{};
    std::uint32_t effective_address{};
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
    bool has_paired_fp_state{};
    std::uint8_t paired_fp_destination_register{};
    std::uint8_t paired_fp_source_a_register{};
    std::uint8_t paired_fp_source_b_register{};
    std::uint64_t paired_fp_source_a_ps0{};
    std::uint64_t paired_fp_source_a_ps1{};
    std::uint64_t paired_fp_source_b_ps0{};
    std::uint64_t paired_fp_source_b_ps1{};
    std::uint64_t paired_fp_destination_ps0{};
    std::uint64_t paired_fp_destination_ps1{};
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
        current_thread_address = 0;
        guest_heap_cursor = 0;
        guest_heap_limit = 0;
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
    std::uint32_t guest_heap_cursor{};
    std::uint32_t guest_heap_limit{};

private:
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
