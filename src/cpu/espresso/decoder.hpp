#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class Opcode
{
    unsupported,
    multiply_low_immediate,
    multiply_low_word,
    xor_immediate,
    xor_immediate_shifted,
    count_leading_zeros,
    addi,
    addis,
    add_immediate_carry,
    subtract_from_immediate_carry,
    add,
    subtract_from,
    subtract_from_carrying,
    negate,
    ori,
    bitwise_or,
    bitwise_and,
    bitwise_and_complement,
    bitwise_xor,
    bitwise_equivalence,
    and_immediate_record,
    rotate_left_word_and_mask,
    rotate_left_word_and_mask_insert,
    arithmetic_shift_right_immediate,
    shift_left_word,
    branch,
    instruction_sync,
    compare_signed_immediate,
    compare_signed_register,
    compare_unsigned_immediate,
    compare_unsigned_register,
    floating_compare_unordered,
    floating_divide_single,
    floating_add_single,
    floating_multiply_add_single,
    conditional_branch,
    load_word_zero,
    load_multiple_word,
    load_word_update,
    load_word_indexed,
    store_word,
    store_multiple_word,
    store_word_indexed,
    store_byte_indexed,
    load_byte_zero,
    load_byte_update,
    store_byte,
    store_byte_update,
    load_halfword_zero,
    load_halfword_algebraic,
    store_halfword,
    store_word_update,
    move_from_link_register,
    move_to_link_register,
    conditional_branch_to_link_register,
    move_from_count_register,
    move_to_count_register,
    conditional_branch_to_count_register,
    load_single,
    load_single_update,
    load_double,
    store_single,
    store_single_update,
    store_double,
    paired_single_merge10,
};

struct DecodedInstruction
{
    Opcode opcode{Opcode::unsupported};
    std::uint32_t raw{};

    // D-form instructions use destination/base. Logical immediates use source/destination.
    std::uint8_t destination{};
    std::uint8_t source{};
    std::uint8_t fp_register{};
    std::uint8_t fp_source_a{};
    std::uint8_t fp_source_b{};
    std::uint8_t base{};
    std::uint8_t cr_field{};
    std::uint8_t fp_compare_a{};
    std::uint8_t fp_compare_b{};
    std::uint8_t fp_source_c{};
    std::uint8_t branch_options{};
    std::uint8_t condition_bit{};
    std::uint8_t shift{};
    std::uint8_t mask_begin{};
    std::uint8_t mask_end{};

    // Signed for immediate arithmetic and branch displacement; unsigned for logical ops.
    std::int32_t immediate{};
    bool absolute{};
    bool link{};
    bool record{};
};

[[nodiscard]] DecodedInstruction decode(std::uint32_t raw) noexcept;

}
