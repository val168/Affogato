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

inline void move_to_msr_tests()
{
    using namespace affogato::cpu::espresso;
    using namespace affogato::cpu::espresso::msr;

    constexpr std::uint32_t instruction_word = 0x7C800124U;
    const DecodedInstruction decoded = decode(instruction_word);
    assert(decoded.opcode == Opcode::move_to_msr);
    assert(decoded.destination == 4U);

    EspressoCore supervisor(0x2000U);
    supervisor.memory.write32_be(0x1000U, instruction_word);
    supervisor.state.cia = 0x1000U;
    supervisor.state.gpr[4] = pr_mask | fp_mask | me_mask;

    assert(supervisor.step() == StepResult::executed);
    assert(supervisor.state.msr == (pr_mask | fp_mask | me_mask));
    assert(supervisor.state.cia == 0x1004U);

    EspressoCore jit_supervisor(0x2000U);
    jit_supervisor.memory.write32_be(0x1000U, instruction_word);
    jit_supervisor.state.cia = 0x1000U;
    jit_supervisor.state.gpr[4] = pr_mask | fp_mask | me_mask;
    Recompiler recompiler;

    const RunResult jit_result = recompiler.run(jit_supervisor, 1U);
    assert(jit_result.reason == StopReason::instruction_limit);
    assert(jit_supervisor.state.msr == supervisor.state.msr);
    assert(jit_supervisor.state.cia == supervisor.state.cia);

    EspressoCore user(0x2000U);
    user.memory.write32_be(0x1000U, instruction_word);
    user.state.cia = 0x1000U;
    user.state.msr = pr_mask;
    user.state.gpr[4] = 0U;

    assert(user.step() == StepResult::executed);
    assert(user.state.srr0 == 0x1000U);
    assert(user.state.srr1 == (pr_mask | 0x00040000U));
    assert(user.state.cia == program_exception_vector);

    EspressoCore translated(0x2000U);
    translated.memory.write32_be(0x1000U, instruction_word);
    translated.state.cia = 0x1000U;
    translated.state.gpr[4] = pr_mask | ir_mask;

    const RunResult result = translated.run(1U);
    assert(result.reason == StopReason::unsupported_architectural_state);
    assert(result.cia == 0x1000U);
    assert(translated.state.msr == 0U);
    assert(translated.state.cia == 0x1000U);
    assert(!result.detail.empty());

    EspressoCore jit_translated(0x2000U);
    jit_translated.memory.write32_be(0x1000U, instruction_word);
    jit_translated.state.cia = 0x1000U;
    jit_translated.state.gpr[4] = pr_mask | ir_mask;
    Recompiler jit_recompiler;

    const RunResult jit_translated_result = jit_recompiler.run(jit_translated, 1U);
    assert(jit_translated_result.reason == StopReason::unsupported_architectural_state);
    assert(jit_translated.state.msr == translated.state.msr);
    assert(jit_translated.state.cia == translated.state.cia);
}

inline void save_restore_spr_tests()
{
    using namespace affogato::cpu::espresso;
    using namespace affogato::cpu::espresso::msr;

    constexpr std::uint32_t move_from_srr0 = 0x7C7A02A6U;
    constexpr std::uint32_t move_to_srr1 = 0x7C9B03A6U;
    const DecodedInstruction read_srr0 = decode(move_from_srr0);
    const DecodedInstruction write_srr1 = decode(move_to_srr1);

    assert(read_srr0.opcode == Opcode::move_from_spr);
    assert(read_srr0.destination == 3U);
    assert(read_srr0.special_register == 26U);
    assert(write_srr1.opcode == Opcode::move_to_spr);
    assert(write_srr1.destination == 4U);
    assert(write_srr1.special_register == 27U);

    EspressoCore core(0x2000U);
    core.memory.write32_be(0x1000U, move_from_srr0);
    core.memory.write32_be(0x1004U, move_to_srr1);
    core.state.cia = 0x1000U;
    core.state.srr0 = 0x12345678U;
    core.state.srr1 = 0x00008002U;
    core.state.gpr[4] = 0xABCDEF01U;

    assert(core.step() == StepResult::executed);
    assert(core.state.gpr[3] == 0x12345678U);
    assert(core.step() == StepResult::executed);
    assert(core.state.srr1 == 0xABCDEF01U);

    EspressoCore user(0x2000U);
    user.memory.write32_be(0x1000U, move_from_srr0);
    user.state.cia = 0x1000U;
    user.state.msr = pr_mask;
    user.state.srr0 = 0x12345678U;

    assert(user.step() == StepResult::executed);
    assert(user.state.srr0 == 0x1000U);
    assert(user.state.srr1 == (pr_mask | 0x00040000U));
    assert(user.state.cia == program_exception_vector);
}

}
