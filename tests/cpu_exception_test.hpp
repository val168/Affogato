#pragma once

#include "cpu/espresso/cpu_exceptions.hpp"

#include <cassert>
#include <cstdint>

namespace affogato::tests
{

inline void cpu_program_exception_tests()
{
    using affogato::cpu::espresso::CpuState;
    using affogato::cpu::espresso::ProgramExceptionCause;
    using affogato::cpu::espresso::enter_program_exception;
    using namespace affogato::cpu::espresso::msr;

    CpuState state{};
    state.cia = 0x12345678U;
    state.msr = pow_mask | ile_mask | ee_mask | pr_mask | fp_mask | me_mask |
        se_mask | be_mask | ip_mask | ir_mask | dr_mask | ri_mask | le_mask;

    enter_program_exception(state, ProgramExceptionCause::privileged_instruction);

    assert(state.srr0 == 0x12345678U);
    assert(state.srr1 == 0x0004F673U);
    assert(state.cia == 0xFFF00700U);
    assert((state.msr & (pow_mask | ee_mask | pr_mask | fp_mask | se_mask | be_mask |
        ir_mask | dr_mask | ri_mask)) == 0U);
    assert((state.msr & me_mask) != 0U);
    assert((state.msr & ile_mask) != 0U);
    assert((state.msr & le_mask) != 0U);
    assert(state.msr == (me_mask | ile_mask | ip_mask | le_mask));

    state = {};
    state.cia = 0x2000U;
    enter_program_exception(state, ProgramExceptionCause::illegal_instruction);

    assert(state.srr0 == 0x2000U);
    assert(state.srr1 == 0x00080000U);
    assert(state.cia == affogato::cpu::espresso::program_exception_vector);
    assert(state.msr == 0U);
}

}
