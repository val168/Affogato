#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class Opcode
{
    unsupported,
    addi,
    addis,
    ori,
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

    // Signed for addi/addis and branch displacement; non-negative for ori.
    std::int32_t immediate{};
    bool absolute{};
    bool link{};
};

[[nodiscard]] DecodedInstruction decode(std::uint32_t raw) noexcept;

}
