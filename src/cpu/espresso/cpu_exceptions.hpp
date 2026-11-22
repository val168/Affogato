#pragma once

#include "cpu/espresso/cpu_state.hpp"

#include <cstdint>

namespace affogato::cpu::espresso
{

enum class ProgramExceptionCause
{
    illegal_instruction,
    privileged_instruction,
    trap,
};

inline constexpr std::uint32_t program_exception_vector = 0x00000700U;
inline constexpr std::uint32_t restorable_msr_mask =
    msr::ee_mask | msr::pr_mask | msr::fp_mask | msr::me_mask |
    msr::ip_mask | msr::ir_mask | msr::dr_mask | msr::ri_mask;
inline constexpr std::uint32_t unsupported_msr_mode_mask =
    msr::pow_mask | msr::ile_mask | msr::fe0_mask | msr::se_mask |
    msr::be_mask | msr::fe1_mask | msr::le_mask | msr::pm_mask;

[[nodiscard]] constexpr bool is_supported_msr_mode(std::uint32_t value) noexcept
{
    return (value & (unsupported_msr_mode_mask | msr::ir_mask | msr::dr_mask)) == 0U;
}

[[nodiscard]] inline std::uint32_t program_exception_syndrome(
    ProgramExceptionCause cause) noexcept
{
    switch (cause)
    {
    case ProgramExceptionCause::illegal_instruction:
        return 0x00080000U;
    case ProgramExceptionCause::privileged_instruction:
        return 0x00040000U;
    case ProgramExceptionCause::trap:
        return 0x00020000U;
    }

    return 0U;
}

inline void enter_program_exception(CpuState& state, ProgramExceptionCause cause) noexcept
{
    constexpr std::uint32_t saved_msr_mask = 0x07C0FFFFU;
    constexpr std::uint32_t program_cause_mask = 0x000E0000U;
    constexpr std::uint32_t exception_disabled_msr_mask =
        msr::pow_mask | msr::ee_mask | msr::pr_mask | msr::fp_mask |
        msr::se_mask | msr::be_mask | msr::ir_mask | msr::dr_mask | msr::ri_mask;
    constexpr std::uint32_t exception_prefix_mask = 0xFFF00000U;

    const std::uint32_t interrupted_msr = state.msr;
    const std::uint32_t saved_status =
        interrupted_msr & saved_msr_mask & ~program_cause_mask;

    state.srr0 = state.cia;
    state.srr1 = saved_status | program_exception_syndrome(cause);
    state.msr = interrupted_msr & ~exception_disabled_msr_mask;

    if ((interrupted_msr & msr::ile_mask) != 0U)
    {
        state.msr |= msr::le_mask;
    }
    else
    {
        state.msr &= ~msr::le_mask;
    }

    const std::uint32_t exception_prefix =
        (interrupted_msr & msr::ip_mask) != 0U ? exception_prefix_mask : 0U;
    state.cia = exception_prefix | program_exception_vector;
}

}
