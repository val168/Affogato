#include "cpu/espresso/decoder.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/interpreter.hpp"
#include "cpu_state_test.hpp"

#include <cassert>
#include <cstdint>

namespace
{

using namespace affogato::cpu::espresso;

void decoder_tests()
{
    const DecodedInstruction addi = decode(0x3860002AU); // addi r3, r0, 42
    assert(addi.opcode == Opcode::addi);
    assert(addi.destination == 3);
    assert(addi.base == 0);
    assert(addi.immediate == 42);

    const DecodedInstruction addis = decode(0x3C83FFFFU); // addis r4, r3, -1
    assert(addis.opcode == Opcode::addis);
    assert(addis.destination == 4);
    assert(addis.base == 3);
    assert(addis.immediate == -1);

    const DecodedInstruction ori = decode(0x60631234U); // ori r3, r3, 0x1234
    assert(ori.opcode == Opcode::ori);
    assert(ori.source == 3);
    assert(ori.destination == 3);
    assert(ori.immediate == 0x1234);

    const DecodedInstruction branch = decode(0x48000009U); // bl +8
    assert(branch.opcode == Opcode::branch);
    assert(branch.immediate == 8);
    assert(!branch.absolute);
    assert(branch.link);

    const DecodedInstruction cmpwi = decode(0x2E830007U); // cmpwi cr5, r3, 7
    assert(cmpwi.opcode == Opcode::compare_signed_immediate);
    assert(cmpwi.cr_field == 5);
    assert(cmpwi.base == 3);
    assert(cmpwi.immediate == 7);

    const DecodedInstruction cmpw = decode(0x7E832000U); // cmpw cr5, r3, r4
    assert(cmpw.opcode == Opcode::compare_signed_register);
    assert(cmpw.cr_field == 5);
    assert(cmpw.base == 3);
    assert(cmpw.source == 4);

    const DecodedInstruction bc = decode(0x4182000CU); // beq +12
    assert(bc.opcode == Opcode::conditional_branch);
    assert(bc.branch_options == 12);
    assert(bc.condition_bit == 2);
    assert(bc.immediate == 12);

    assert(decode(0U).opcode == Opcode::unsupported);
}

void guest_memory_tests()
{
    GuestMemory memory(8);

    memory.write32_be(0, 0x12345678U);

    assert(memory.read8(0) == 0x12);
    assert(memory.read8(1) == 0x34);
    assert(memory.read8(2) == 0x56);
    assert(memory.read8(3) == 0x78);
    assert(memory.read32_be(0) == 0x12345678U);
}

void interpreter_tests()
{
    EspressoCore core(0x20);

    // addi r3, r0, 5
    core.memory.write32_be(0x00, 0x38600005U);
    // ori r3, r3, 7
    core.memory.write32_be(0x04, 0x60630007U);
    // b +8: skip the unsupported word at 0x0c and continue at 0x10
    core.memory.write32_be(0x08, 0x48000008U);
    // addi r4, r3, 1
    core.memory.write32_be(0x10, 0x38830001U);

    const RunResult result = core.run(8);

    assert(result.steps == 4);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.gpr[3] == 7);
    assert(core.state.gpr[4] == 8);
    assert(core.state.cia == 0x14);

    EspressoCore link_core(0x10);
    // bl +8
    link_core.memory.write32_be(0, 0x48000009U);

    assert(link_core.step() == StepResult::executed);
    assert(link_core.state.cia == 8);
    assert(link_core.state.lr == 4);
}

void compare_and_conditional_branch_tests()
{
    const auto run_if_else = [](std::uint32_t value) {
        EspressoCore core(0x20);
        // if (r3 == 7) r4 = 1; else r4 = 0;
        core.memory.write32_be(0x00, 0x38600000U | value); // addi r3, r0, value
        core.memory.write32_be(0x04, 0x2C030007U);         // cmpwi r0, r3, 7
        core.memory.write32_be(0x08, 0x4182000CU);         // beq +12 -> then
        core.memory.write32_be(0x0C, 0x38800000U);         // else: addi r4, r0, 0
        core.memory.write32_be(0x10, 0x48000008U);         // b +8 -> end
        core.memory.write32_be(0x14, 0x38800001U);         // then: addi r4, r0, 1

        const RunResult result = core.run(8);
        assert(result.reason == StopReason::unsupported_instruction);
        assert(core.state.cia == 0x18);
        return core.state.gpr[4];
    };

    assert(run_if_else(7) == 1);
    assert(run_if_else(6) == 0);

    // Compare writes only the selected CR field and copies XER[SO].
    EspressoCore compare_core(0x10);
    compare_core.state.gpr[3] = 0xFFFFFFFFU; // signed -1
    compare_core.state.xer = 0x80000000U;
    compare_core.state.cr = 0x12345678U;
    compare_core.memory.write32_be(0, 0x2E830001U); // cmpwi cr5, r3, 1
    assert(compare_core.step() == StepResult::executed);
    assert(compare_core.state.cr == 0x12345978U);

    // The CTR form decrements before testing; BO=16 is bdnz.
    EspressoCore count_core(0x10);
    count_core.state.ctr = 2;
    count_core.memory.write32_be(0, 0x42000004U); // bc 16, 0, +4
    assert(count_core.step() == StepResult::executed);
    assert(count_core.state.ctr == 1);
    assert(count_core.state.cia == 4);
}

}

int main()
{
    affogato::tests::cpu_state_tests();
    decoder_tests();
    guest_memory_tests();
    interpreter_tests();
    compare_and_conditional_branch_tests();
    return 0;
}
