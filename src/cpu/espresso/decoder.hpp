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
};

struct DecodedInstruction
{
    Opcode opcode{Opcode::unsupported};
    std::uint32_t raw{};

    // D-form instructions use destination/base. ori uses source/destination.
    std::uint8_t destination{};
    std::uint8_t source{};
    std::uint8_t base{};

    // Signed for addi/addis and branch displacement; non-negative for ori.
    std::int32_t immediate{};
    bool absolute{};
    bool link{};
};

[[nodiscard]] DecodedInstruction decode(std::uint32_t raw) noexcept;

}
