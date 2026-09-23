#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class Opcode
{
    unsupported,
    addi,
    addis,
    add_immediate_carry,
    subtract_from_immediate_carry,
    add,
    subtract_from,
    ori,
    bitwise_or,
    bitwise_and,
    bitwise_and_complement,
    bitwise_xor,
    bitwise_equivalence,
    and_immediate_record,
    rotate_left_word_and_mask,
    arithmetic_shift_right_immediate,
    shift_left_word,
    branch,
    compare_signed_immediate,
    compare_signed_register,
    compare_unsigned_immediate,
    compare_unsigned_register,
    conditional_branch,
    load_word_zero,
    load_word_update,
    load_word_indexed,
    store_word,
    store_word_indexed,
    store_byte_indexed,
    load_byte_zero,
    load_byte_update,
    store_byte,
    store_byte_update,
    load_halfword_zero,
    store_halfword,
    store_word_update,
    move_from_link_register,
    move_to_link_register,
    conditional_branch_to_link_register,
    move_from_count_register,
    move_to_count_register,
    conditional_branch_to_count_register,
};

struct DecodedInstruction
{
    Opcode opcode{Opcode::unsupported};
    std::uint32_t raw{};

    // D-form instructions use destination/base. ori uses source/destination.
    std::uint8_t destination{};
    std::uint8_t source{};
    std::uint8_t base{};
    std::uint8_t cr_field{};
    std::uint8_t branch_options{};
    std::uint8_t condition_bit{};
    std::uint8_t shift{};
    std::uint8_t mask_begin{};
    std::uint8_t mask_end{};

    // Signed for addi/addis and branch displacement; unsigned for logical ops.
    std::int32_t immediate{};
    bool absolute{};
    bool link{};
    bool record{};
};

[[nodiscard]] DecodedInstruction decode(std::uint32_t raw) noexcept;

}
