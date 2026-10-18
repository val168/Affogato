#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class Opcode
{
    unsupported,
    multiply_low_immediate,
    multiply_low_word,
    multiply_high_word_unsigned,
    multiply_high_word_signed,
    divide_word_unsigned,
    divide_word_signed,
    xor_immediate,
    xor_immediate_shifted,
    count_leading_zeros,
    addi,
    addis,
    add_immediate_carry,
    add_carrying,
    add_to_zero_extended,
    add_extended,
    add_to_minus_one_extended,
    subtract_from_immediate_carry,
    add,
    subtract_from,
    subtract_from_carrying,
    subtract_from_extended,
    subtract_from_zero_extended,
    subtract_from_minus_one_extended,
    negate,
    extend_sign_byte,
    extend_sign_halfword,
    ori,
    or_immediate_shifted,
    bitwise_or,
    bitwise_and,
    bitwise_and_complement,
    bitwise_xor,
    bitwise_equivalence,
    bitwise_nand,
    bitwise_nor,
    bitwise_or_complement,
    and_immediate_record,
    and_immediate_shifted_record,
    rotate_left_word_and_mask_register,
    rotate_left_word_and_mask,
    rotate_left_word_and_mask_insert,
    arithmetic_shift_right_immediate,
    arithmetic_shift_right_word,
    shift_left_word,
    shift_right_word,
    branch,
    instruction_sync,
    compare_signed_immediate,
    compare_signed_register,
    compare_unsigned_immediate,
    compare_unsigned_register,
    floating_compare_unordered,
    floating_compare_ordered,
    floating_divide_single,
    floating_divide_double,
    floating_add_single,
    floating_subtract_single,
    floating_add_double,
    floating_multiply_single,
    floating_multiply_double,
    floating_multiply_add_single,
    floating_multiply_add_double,
    floating_negative_multiply_subtract_double,
    floating_move_register,
    floating_negate,
    floating_absolute_value,
    floating_negative_absolute_value,
    floating_convert_to_integer_word_zero,
    floating_subtract_double,
    floating_round_to_single,
    conditional_branch,
    load_word_zero,
    load_multiple_word,
    load_string_word_immediate,
    store_string_word_immediate,
    load_word_update,
    load_word_indexed,
    load_word_indexed_update,
    load_word_byte_reverse_indexed,
    load_byte_zero_indexed,
    load_byte_zero_indexed_update,
    load_halfword_zero_indexed,
    load_halfword_zero_indexed_update,
    load_halfword_byte_reverse_indexed,
    load_halfword_algebraic_indexed,
    load_halfword_algebraic_indexed_update,
    store_word,
    store_multiple_word,
    store_word_indexed,
    store_word_indexed_update,
    store_word_byte_reverse_indexed,
    store_byte_indexed,
    store_byte_indexed_update,
    store_halfword_indexed,
    store_halfword_indexed_update,
    store_halfword_byte_reverse_indexed,
    store_floating_as_integer_word_indexed,
    load_byte_zero,
    load_byte_update,
    store_byte,
    store_byte_update,
    load_halfword_zero,
    load_halfword_algebraic,
    store_halfword,
    store_halfword_update,
    store_word_update,
    move_from_link_register,
    move_to_link_register,
    conditional_branch_to_link_register,
    move_from_count_register,
    move_to_count_register,
    conditional_branch_to_count_register,
    load_single,
    load_single_update,
    load_single_indexed,
    load_single_indexed_update,
    load_double,
    load_double_indexed,
    load_double_indexed_update,
    store_single,
    store_single_update,
    store_single_indexed,
    store_single_indexed_update,
    store_double,
    store_double_indexed,
    store_double_indexed_update,
    store_double_update,
    paired_single_merge10,
    condition_bit_and,
    condition_bit_and_complement,
    condition_bit_or,
    condition_bit_or_complement,
    condition_bit_nor,
    condition_bit_xor,
    condition_bit_nand,
    condition_bit_equivalence,
    move_condition_register_field,
    move_from_condition_register,
    move_to_condition_register_fields,
    move_condition_register_from_xer,
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
    std::uint8_t cr_field_source{};
    std::uint8_t cr_field_mask{};
    std::uint8_t cr_destination_bit{};
    std::uint8_t cr_source_a_bit{};
    std::uint8_t cr_source_b_bit{};
    std::uint8_t fp_compare_a{};
    std::uint8_t fp_compare_b{};
    std::uint8_t fp_source_c{};
    std::uint8_t branch_options{};
    std::uint8_t condition_bit{};
    std::uint8_t shift{};
    std::uint8_t mask_begin{};
    std::uint8_t mask_end{};
    std::uint8_t string_byte_count{};

    // Signed for immediate arithmetic and branch displacement; unsigned for logical ops.
    std::int32_t immediate{};
    bool absolute{};
    bool link{};
    bool record{};
};

[[nodiscard]] DecodedInstruction decode(std::uint32_t raw) noexcept;

}
