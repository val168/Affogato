#pragma once

#include "cpu/espresso/decoder.hpp"
#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/recompiler.hpp"

#include <cassert>
#include <cstdint>

namespace affogato::tests
{

inline void move_from_msr_tests()
{
    using namespace affogato::cpu::espresso;
    using namespace affogato::cpu::espresso::msr;

    constexpr std::uint32_t instruction_word = 0x7C6000A6U;
    const DecodedInstruction decoded = decode(instruction_word);
    assert(decoded.opcode == Opcode::move_from_msr);
    assert(decoded.destination == 3U);

    EspressoCore supervisor(0x2000U);
    supervisor.memory.write32_be(0x1000U, instruction_word);
    supervisor.state.cia = 0x1000U;
    supervisor.state.msr = fp_mask | me_mask;

    assert(supervisor.step() == StepResult::executed);
    assert(supervisor.state.gpr[3] == (fp_mask | me_mask));
    assert(supervisor.state.cia == 0x1004U);

    EspressoCore user(0x2000U);
    user.memory.write32_be(0x1000U, instruction_word);
    user.state.cia = 0x1000U;
    user.state.msr = pr_mask | fp_mask | me_mask;

    assert(user.step() == StepResult::executed);
    assert(user.state.srr0 == 0x1000U);
    assert(user.state.srr1 == (pr_mask | fp_mask | me_mask | 0x00040000U));
    assert(user.state.cia == program_exception_vector);
    assert((user.state.msr & pr_mask) == 0U);

    EspressoCore jit_user(0x2000U);
    jit_user.memory.write32_be(0x1000U, instruction_word);
    jit_user.state.cia = 0x1000U;
    jit_user.state.msr = pr_mask | fp_mask | me_mask;
    Recompiler recompiler;

    const RunResult result = recompiler.run(jit_user, 1U);
    assert(result.reason == StopReason::instruction_limit);
    assert(result.steps == 1U);
    assert(jit_user.state.srr0 == user.state.srr0);
    assert(jit_user.state.srr1 == user.state.srr1);
    assert(jit_user.state.msr == user.state.msr);
    assert(jit_user.state.cia == user.state.cia);
}

}
