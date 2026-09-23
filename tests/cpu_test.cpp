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

}

int main()
{
    affogato::tests::cpu_state_tests();
    decoder_tests();
    guest_memory_tests();
    interpreter_tests();
    return 0;
}
