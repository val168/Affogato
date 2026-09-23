#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class Opcode
{
    unsupported,
    addi,
    addis,
    add,
    subtract_from,
    ori,
    bitwise_or,
    bitwise_and,
    bitwise_xor,
    and_immediate_record,
    rotate_left_word_and_mask,
    branch,
    compare_signed_immediate,
    compare_signed_register,
    conditional_branch,
    load_word_zero,
    store_word,
    load_byte_zero,
    store_byte,
    load_halfword_zero,
    store_halfword,
    store_word_update,
    move_from_link_register,
    move_to_link_register,
    conditional_branch_to_link_register,
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
