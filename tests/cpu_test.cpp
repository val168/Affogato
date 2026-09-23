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

    const DecodedInstruction add = decode(0x7C642A14U); // add r3, r4, r5
    assert(add.opcode == Opcode::add);
    assert(add.destination == 3);
    assert(add.base == 4);
    assert(add.source == 5);

    const DecodedInstruction subf = decode(0x7CC42850U); // subf r6, r4, r5
    assert(subf.opcode == Opcode::subtract_from);
    assert(subf.destination == 6);
    assert(subf.base == 4);
    assert(subf.source == 5);

    const DecodedInstruction ori = decode(0x60631234U); // ori r3, r3, 0x1234
    assert(ori.opcode == Opcode::ori);
    assert(ori.source == 3);
    assert(ori.destination == 3);
    assert(ori.immediate == 0x1234);

    const DecodedInstruction bit_or = decode(0x7C872B78U); // or r7, r4, r5
    assert(bit_or.opcode == Opcode::bitwise_or);
    assert(bit_or.source == 4);
    assert(bit_or.destination == 7);
    assert(bit_or.base == 5);

    const DecodedInstruction bit_and = decode(0x7C882838U); // and r8, r4, r5
    assert(bit_and.opcode == Opcode::bitwise_and);
    assert(bit_and.source == 4);
    assert(bit_and.destination == 8);
    assert(bit_and.base == 5);

    const DecodedInstruction bit_xor = decode(0x7C892A78U); // xor r9, r4, r5
    assert(bit_xor.opcode == Opcode::bitwise_xor);
    assert(bit_xor.source == 4);
    assert(bit_xor.destination == 9);
    assert(bit_xor.base == 5);

    const DecodedInstruction andi = decode(0x708AFF00U); // andi. r10, r4, 0xFF00
    assert(andi.opcode == Opcode::and_immediate_record);
    assert(andi.source == 4);
    assert(andi.destination == 10);
    assert(andi.immediate == 0xFF00);
    assert(andi.record);

    const DecodedInstruction rlwinm = decode(0x548B2834U); // rlwinm r11, r4, 5, 0, 26
    assert(rlwinm.opcode == Opcode::rotate_left_word_and_mask);
    assert(rlwinm.source == 4);
    assert(rlwinm.destination == 11);
    assert(rlwinm.shift == 5);
    assert(rlwinm.mask_begin == 0);
    assert(rlwinm.mask_end == 26);

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

    const DecodedInstruction lwz = decode(0x8041FFFCU); // lwz r2, -4(r1)
    assert(lwz.opcode == Opcode::load_word_zero);
    assert(lwz.destination == 2);
    assert(lwz.base == 1);
    assert(lwz.immediate == -4);

    const DecodedInstruction stw = decode(0x90610000U); // stw r3, 0(r1)
    assert(stw.opcode == Opcode::store_word);
    assert(stw.destination == 3);

    const DecodedInstruction lbz = decode(0x88810002U); // lbz r4, 2(r1)
    assert(lbz.opcode == Opcode::load_byte_zero);
    assert(lbz.destination == 4);
    assert(lbz.base == 1);
    assert(lbz.immediate == 2);

    const DecodedInstruction stb = decode(0x98610004U); // stb r3, 4(r1)
    assert(stb.opcode == Opcode::store_byte);
    assert(stb.destination == 3);

    const DecodedInstruction lhz = decode(0xA0A10002U); // lhz r5, 2(r1)
    assert(lhz.opcode == Opcode::load_halfword_zero);
    assert(lhz.destination == 5);

    const DecodedInstruction sth = decode(0xB0610006U); // sth r3, 6(r1)
    assert(sth.opcode == Opcode::store_halfword);
    assert(sth.destination == 3);

    const DecodedInstruction mflr = decode(0x7C0802A6U); // mflr r0
    assert(mflr.opcode == Opcode::move_from_link_register);
    assert(mflr.destination == 0);

    const DecodedInstruction mtlr = decode(0x7C0803A6U); // mtlr r0
    assert(mtlr.opcode == Opcode::move_to_link_register);
    assert(mtlr.destination == 0);

    const DecodedInstruction blr = decode(0x4E800020U); // blr
    assert(blr.opcode == Opcode::conditional_branch_to_link_register);
    assert(blr.branch_options == 20);
    assert(!blr.link);

    const DecodedInstruction beqlr = decode(0x4D820020U); // beqlr
    assert(beqlr.opcode == Opcode::conditional_branch_to_link_register);
    assert(beqlr.branch_options == 12);
    assert(beqlr.condition_bit == 2);

    const DecodedInstruction blrl = decode(0x4E800021U); // blrl
    assert(blrl.opcode == Opcode::conditional_branch_to_link_register);
    assert(blrl.link);

    const DecodedInstruction stwu = decode(0x9421FFF0U); // stwu r1, -16(r1)
    assert(stwu.opcode == Opcode::store_word_update);
    assert(stwu.destination == 1);
    assert(stwu.base == 1);
    assert(stwu.immediate == -16);

    assert(decode(0U).opcode == Opcode::unsupported);
    assert(decode(0x9400FFF0U).opcode == Opcode::unsupported); // stwu with rA=0
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

    memory.write16_be(4, 0xABCDU);
    assert(memory.read8(4) == 0xAB);
    assert(memory.read8(5) == 0xCD);
    assert(memory.read16_be(4) == 0xABCDU);
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

void integer_alu_tests()
{
    EspressoCore core(0x40);
    core.state.gpr[4] = 0xF0F00F0FU;
    core.state.gpr[5] = 0x0FF0FF00U;
    core.state.cr = 0x12345678U;
    core.state.xer = 0x80000000U;

    core.memory.write32_be(0x00, 0x7C642A14U); // add r3, r4, r5
    core.memory.write32_be(0x04, 0x7CC42850U); // subf r6, r4, r5
    core.memory.write32_be(0x08, 0x7C872B78U); // or r7, r4, r5
    core.memory.write32_be(0x0C, 0x7C882839U); // and. r8, r4, r5
    core.memory.write32_be(0x10, 0x7C892A78U); // xor r9, r4, r5
    core.memory.write32_be(0x14, 0x708AFF00U); // andi. r10, r4, 0xFF00
    core.memory.write32_be(0x18, 0x548B2834U); // rlwinm r11, r4, 5, 0, 26

    const RunResult result = core.run(8);

    assert(result.steps == 7);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.gpr[3] == 0x00E10E0FU);
    assert(core.state.gpr[6] == 0x1F00EFF1U);
    assert(core.state.gpr[7] == 0xFFF0FF0FU);
    assert(core.state.gpr[8] == 0x00F00F00U);
    assert(core.state.gpr[9] == 0xFF00F00FU);
    assert(core.state.gpr[10] == 0x00000F00U);
    assert(core.state.gpr[11] == 0x1E01E1E0U);
    // and. and andi. update only CR0, preserving the lower CR fields and XER.SO.
    assert(core.state.cr == 0x52345678U);

    // rlwinm masks may wrap across the word boundary: MB=28 through ME=3.
    EspressoCore wrapping_rotate_core(8);
    wrapping_rotate_core.state.gpr[4] = 0xF000000FU;
    // rlwinm r3, r4, 0, 28, 3
    wrapping_rotate_core.memory.write32_be(0, 0x54830706U);
    assert(wrapping_rotate_core.step() == StepResult::executed);
    assert(wrapping_rotate_core.state.gpr[3] == 0xF000000FU);
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

void load_store_tests()
{
    EspressoCore core(0x100);

    // Set up a base pointer and a word value, then exercise each requested
    // load/store width against the same big-endian guest memory.
    core.memory.write32_be(0x00, 0x38200080U); // addi r1, r0, 0x80
    core.memory.write32_be(0x04, 0x38601234U); // addi r3, r0, 0x1234
    core.memory.write32_be(0x08, 0x90610000U); // stw r3, 0(r1)
    core.memory.write32_be(0x0C, 0x88810002U); // lbz r4, 2(r1)
    core.memory.write32_be(0x10, 0xA0A10002U); // lhz r5, 2(r1)
    core.memory.write32_be(0x14, 0x80C10000U); // lwz r6, 0(r1)
    core.memory.write32_be(0x18, 0x98610004U); // stb r3, 4(r1)
    core.memory.write32_be(0x1C, 0xB0610006U); // sth r3, 6(r1)
    core.memory.write32_be(0x20, 0x88E10004U); // lbz r7, 4(r1)
    core.memory.write32_be(0x24, 0xA1010006U); // lhz r8, 6(r1)

    const RunResult result = core.run(16);

    assert(result.reason == StopReason::unsupported_instruction);
    assert(result.steps == 10);
    assert(core.state.gpr[1] == 0x80U);
    assert(core.state.gpr[3] == 0x1234U);
    assert(core.memory.read8(0x82) == 0x12U);
    assert(core.state.gpr[4] == 0x12U);
    assert(core.state.gpr[5] == 0x1234U);
    assert(core.state.gpr[6] == 0x1234U);
    assert(core.state.gpr[7] == 0x34U);
    assert(core.state.gpr[8] == 0x1234U);
    assert(core.memory.read32_be(0x80) == 0x00001234U);
    assert(core.memory.read8(0x84) == 0x34U);
    assert(core.memory.read8(0x86) == 0x12U);
    assert(core.memory.read8(0x87) == 0x34U);

    // D-form displacement is signed, and rA=0 supplies a zero base even if
    // the stored contents of GPR0 are nonzero.
    EspressoCore address_core(0x100);
    address_core.state.gpr[1] = 0x54U;
    address_core.memory.write32_be(0x00, 0x8041FFFCU); // lwz r2, -4(r1)
    address_core.memory.write32_be(0x50, 0x89ABCDEFU);
    assert(address_core.step() == StepResult::executed);
    assert(address_core.state.gpr[2] == 0x89ABCDEFU);

    EspressoCore zero_base_core(0x100);
    zero_base_core.state.gpr[0] = 0x20U;
    zero_base_core.memory.write32_be(0x00, 0x81200080U); // lwz r9, 0x80(r0)
    zero_base_core.memory.write32_be(0x80, 0xCAFEBABEU);
    zero_base_core.memory.write32_be(0xA0, 0xDEADBEEFU);
    assert(zero_base_core.step() == StepResult::executed);
    assert(zero_base_core.state.gpr[9] == 0xCAFEBABEU);
}

void function_call_and_stack_tests()
{
    EspressoCore link_branch_core(0x20);
    link_branch_core.state.lr = 0x12U;
    link_branch_core.memory.write32_be(0, 0x4E800021U); // blrl
    assert(link_branch_core.step() == StepResult::executed);
    assert(link_branch_core.state.cia == 0x10U); // old LR target, aligned
    assert(link_branch_core.state.lr == 0x04U); // new link address

    EspressoCore conditional_link_branch_core(0x20);
    conditional_link_branch_core.state.lr = 0x10U;
    conditional_link_branch_core.state.cr = 0x20000000U; // CR0.EQ
    conditional_link_branch_core.memory.write32_be(0, 0x4D820020U); // beqlr
    assert(conditional_link_branch_core.step() == StepResult::executed);
    assert(conditional_link_branch_core.state.cia == 0x10U);

    EspressoCore core(0x100);

    core.memory.write32_be(0x00, 0x382000F0U); // addi r1, r0, 0xF0 (initial SP)
    core.memory.write32_be(0x04, 0x4800001DU); // bl +0x1C -> function at 0x20
    core.memory.write32_be(0x08, 0x38830000U); // after return: addi r4, r3, 0

    core.memory.write32_be(0x20, 0x7C0802A6U); // mflr r0
    core.memory.write32_be(0x24, 0x9421FFF0U); // stwu r1, -16(r1)
    core.memory.write32_be(0x28, 0x9001000CU); // stw r0, 12(r1): save LR
    core.memory.write32_be(0x2C, 0x3860002AU); // addi r3, r0, 42 (return value)
    core.memory.write32_be(0x30, 0x8001000CU); // lwz r0, 12(r1)
    core.memory.write32_be(0x34, 0x7C0803A6U); // mtlr r0
    core.memory.write32_be(0x38, 0x38210010U); // addi r1, r1, 16 (restore SP)
    core.memory.write32_be(0x3C, 0x4E800020U); // blr

    const RunResult result = core.run(16);

    assert(result.steps == 11);
    assert(result.reason == StopReason::unsupported_instruction);
    assert(core.state.cia == 0x0CU);
    assert(core.state.gpr[1] == 0xF0U);
    assert(core.state.gpr[3] == 42U);
    assert(core.state.gpr[4] == 42U);
    assert(core.state.lr == 0x08U);
    assert(core.memory.read32_be(0xE0) == 0xF0U); // stwu saved the old SP
    assert(core.memory.read32_be(0xEC) == 0x08U); // function saved return LR
}

}

int main()
{
    affogato::tests::cpu_state_tests();
    decoder_tests();
    guest_memory_tests();
    interpreter_tests();
    integer_alu_tests();
    compare_and_conditional_branch_tests();
    load_store_tests();
    function_call_and_stack_tests();
    return 0;
}
