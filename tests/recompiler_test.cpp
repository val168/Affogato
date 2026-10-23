#include "cpu/espresso/recompiler.hpp"

#include <cassert>
#include <array>
#include <cstdint>
#include <string>

using namespace affogato::cpu::espresso;

namespace
{

[[nodiscard]] constexpr std::uint32_t x_form(
    std::uint8_t rd, std::uint8_t ra, std::uint8_t rb, std::uint16_t xo)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(rd) << 21U) |
           (static_cast<std::uint32_t>(ra) << 16U) |
           (static_cast<std::uint32_t>(rb) << 11U) |
           (static_cast<std::uint32_t>(xo) << 1U);
}

[[nodiscard]] constexpr std::uint32_t cr_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint16_t xo)
{
    return (19U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(source_a) << 16U) |
           (static_cast<std::uint32_t>(source_b) << 11U) |
           (static_cast<std::uint32_t>(xo) << 1U);
}

void recompiler_differential_arithmetic_test()
{
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    const std::uint32_t add = x_form(5, 3, 4, 266);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(0x100U, 0x38600005U); // addi r3,r0,5
        core->memory.write32_be(0x104U, 0x38830007U); // addi r4,r3,7
        core->memory.write32_be(0x108U, add);         // add r5,r3,r4
        core->state.cia = 0x100U;
        core->state.cr = 0x12345678U;
        core->state.xer = 0xE0000000U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0x76543210U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const RunResult oracle = interpreted.run(3U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 3U);
    assert(oracle.reason == StopReason::instruction_limit);
    assert(actual.reason == oracle.reason);
    assert(actual.steps == 3U);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.cr == 0x12345678U);
    assert(compiled.state.xer == 0xE0000000U);
    assert(compiled.state.lr == 0xABCDEF00U);
    assert(compiled.state.ctr == 0x76543210U);
    assert(compiled.state.fpscr == 0x89ABCDEFU);
    assert(compiled.state.gpr[3] == 5U);
    assert(compiled.state.gpr[4] == 12U);
    assert(compiled.state.gpr[5] == 17U);
    assert(recompiler.statistics().native_instructions == 3U);
    assert(recompiler.statistics().compiled_blocks == 1U);
    assert(!actual.instruction_history.empty());

    compiled.state.cia = 0x100U;
    compiled.state.gpr.fill(0U);
    const RunResult cached = recompiler.run(compiled, 3U);
    assert(cached.steps == 3U);
    assert(compiled.state.gpr[5] == 17U);
    assert(recompiler.statistics().cache_hits >= 1U);
}

void recompiler_destination_aliases_second_source_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 6> program{
        0x3C000274U, // addis r0,r0,0x274
        0x30006DDCU, // addic r0,r0,0x6DDC (interpreter fallback)
        0x39800000U, // addi r12,r0,0
        0x7C0C0214U, // add r0,r12,r0
        0x7C0903A6U, // mtctr r0
        0x4E800420U, // bctr
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(oracle.steps == program.size());
    assert(actual.steps == oracle.steps);
    assert(oracle.cia == 0x02746DDCU);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.ctr == 0x02746DDCU);
}

void recompiler_logical_differential_test()
{
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    const std::array<std::uint32_t, 5> program{
        0x38601234U,                 // addi r3,r0,0x1234
        0x6064ABCDU,                 // ori r4,r3,0xABCD
        0x688500FFU,                 // xori r5,r4,0x00FF
        x_form(6, 5, 3, 28),         // and r6,r5,r3
        x_form(7, 6, 4, 316),        // xor r7,r6,r4
    };
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t i = 0; i < program.size(); ++i)
            core->memory.write32_be(0x100U + static_cast<std::uint32_t>(i * 4U),
                                    program[i]);
        core->state.cia = 0x100U;
    }
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.steps == oracle.steps);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[4] == 0x0000BBFD);
    assert(recompiler.statistics().native_instructions == program.size());
}

void recompiler_condition_bit_differential_tests()
{
    constexpr std::array<std::uint16_t, 8U> xos{
        257U, 129U, 449U, 417U, 33U, 193U, 225U, 289U};
    constexpr std::uint32_t start = 0x100U;
    for (const std::uint16_t xo : xos)
    {
        for (std::uint8_t destination : {std::uint8_t{2U}, std::uint8_t{0U}})
        {
            for (std::uint32_t input_bits = 0U; input_bits < 4U; ++input_bits)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint8_t source_a = 0U;
                const std::uint8_t source_b = 1U;
                const std::uint32_t instruction = cr_form(
                    destination, source_a, source_b, xo);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.cr = 0xA5A5A5A5U &
                        ~(1U << 31U) & ~(1U << 30U) & ~(1U << (31U - destination));
                    if ((input_bits & 1U) != 0U)
                    {
                        core->state.cr |= 1U << 31U;
                    }
                    if ((input_bits & 2U) != 0U)
                    {
                        core->state.cr |= 1U << 30U;
                    }
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.steps == oracle.steps);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.cr == interpreted.state.cr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
            }
        }
    }
}

void recompiler_step_budget_test()
{
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, 0x38600001U);
    core.memory.write32_be(0x104U, 0x38800002U);
    core.memory.write32_be(0x108U, 0x38A00003U);
    core.state.cia = 0x100U;
    Recompiler recompiler;

    const RunResult result = recompiler.run(core, 2U);
    assert(result.steps == 2U);
    assert(result.reason == StopReason::instruction_limit);
    assert(result.cia == 0x108U);
    assert(core.state.gpr[3] == 1U);
    assert(core.state.gpr[4] == 2U);
    assert(core.state.gpr[5] == 0U);
    assert(recompiler.statistics().native_instructions == 0U);
    assert(recompiler.statistics().interpreter_fallbacks == 2U);
}

void recompiler_recent_history_window_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::size_t instruction_count = 80U;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (std::size_t i = 0; i < instruction_count; ++i)
    {
        interpreted.memory.write32_be(
            start + static_cast<std::uint32_t>(i * 4U), 0x38630001U);
        compiled.memory.write32_be(
            start + static_cast<std::uint32_t>(i * 4U), 0x38630001U);
    }
    interpreted.state.cia = start;
    compiled.state.cia = start;

    const RunResult oracle = interpreted.run(instruction_count);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, instruction_count);

    assert(actual.steps == oracle.steps);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[3] == instruction_count);
    assert(actual.instruction_history.size() == instruction_history_capacity);
    for (std::size_t i = 0; i < instruction_history_capacity; ++i)
    {
        const std::uint32_t expected_cia = start + static_cast<std::uint32_t>(
            (instruction_count - instruction_history_capacity + i) * 4U);
        assert(actual.instruction_history[i].cia == expected_cia);
        assert(actual.instruction_history[i].opcode_name == "addi");
        assert(actual.instruction_history[i].completed);
    }
}

void recompiler_branch_test()
{
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, 0x48000021U); // bl 0x120
    core.state.cia = 0x100U;
    Recompiler recompiler;
    const RunResult result = recompiler.run(core, 1U);
    assert(result.steps == 1U);
    assert(result.cia == 0x120U);
    assert(core.state.cia == 0x120U);
    assert(core.state.lr == 0x104U);
}

void recompiler_conditional_branch_tests()
{
    const auto compare_run = [](std::uint32_t word, std::uint32_t cr,
                                std::uint32_t ctr, std::uint32_t lr,
                                std::uint32_t expected_cia,
                                std::uint32_t expected_ctr,
                                std::uint32_t expected_lr)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, word);
            core->state.cia = 0x100U;
            core->state.cr = cr;
            core->state.ctr = ctr;
            core->state.lr = lr;
        }
        Recompiler recompiler;
        const RunResult oracle = interpreted.run(1U);
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(oracle.steps == actual.steps && actual.steps == 1U);
        assert(oracle.cia == expected_cia);
        assert(actual.cia == expected_cia);
        assert(compiled.state.ctr == expected_ctr);
        assert(compiled.state.lr == expected_lr);
        assert(compiled.state.cr == interpreted.state.cr);
    };

    compare_run(0x41820008U, 0U, 0U, 0U,
                0x104U, 0U, 0U); // beq not taken: CR bit 2 is clear
    compare_run(0x41820008U, 1U << 29U, 0U, 0U,
                0x108U, 0U, 0U); // beq taken
    compare_run(0x41820009U, 0U, 0U, 0U,
                0x104U, 0U, 0x104U); // beql updates LR even when not taken
    compare_run(0x42000008U, 0U, 1U, 0U,
                0x104U, 0U, 0U); // bdnz, decrements CTR then falls through
    compare_run(0x4E800020U, 0U, 0U, 0x203U,
                0x200U, 0U, 0x203U); // blr masks low target bits
    compare_run(0x4E800420U, 0U, 0x303U, 0U,
                0x300U, 0x303U, 0U); // bctr
}

void recompiler_special_register_move_tests()
{
    constexpr std::uint32_t mflr_r3 = 0x7C6802A6U;
    constexpr std::uint32_t mtctr_r4 = 0x7C8903A6U;
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, mflr_r3);
    core.memory.write32_be(0x104U, mtctr_r4);
    core.state.cia = 0x100U;
    core.state.lr = 0x12345678U;
    core.state.gpr[4] = 0xABCDEF00U;
    Recompiler recompiler;
    const RunResult result = recompiler.run(core, 2U);
    assert(result.steps == 2U);
    assert(core.state.gpr[3] == 0x12345678U);
    assert(core.state.ctr == 0xABCDEF00U);
}

void recompiler_interpreter_fallback_test()
{
    EspressoCore interpreted(0x1000U);
    EspressoCore core(0x1000U);
    for (EspressoCore* machine : {&interpreted, &core})
    {
        machine->memory.write32_be(0x100U, 0x38600005U); // addi r3,r0,5
        machine->memory.write32_be(0x104U, 0x84650000U); // lwzu r3,0(r5), fallback
        machine->memory.write32_be(0x108U, 0x38C30002U); // addi r6,r3,2
        machine->memory.write32_be(0x200U, 0x89ABCDEFU);
        machine->state.cia = 0x100U;
        machine->state.gpr[5] = 0x200U;
    }
    const RunResult oracle = interpreted.run(3U);
    Recompiler recompiler;
    const RunResult result = recompiler.run(core, 3U);
    assert(result.steps == oracle.steps && result.steps == 3U);
    assert(result.reason == StopReason::instruction_limit);
    assert(result.cia == oracle.cia);
    assert(core.state.gpr == interpreted.state.gpr);
    assert(core.state.gpr[3] == 0x89ABCDEFU);
    assert(core.state.gpr[6] == 0x89ABCDF1U);
    assert(recompiler.statistics().interpreter_fallbacks == 1U);
    assert(recompiler.statistics().native_instructions == 2U);
    assert(recompiler.statistics().fallbacks_by_opcode.at(
        static_cast<std::uint16_t>(Opcode::load_word_update)) == 1U);
}

void recompiler_memory_tests()
{
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    const std::array<std::uint32_t, 7> program{
        0x80A40000U, // lwz r5,0(r4)
        0x90A40004U, // stw r5,4(r4)
        0x88C40004U, // lbz r6,4(r4)
        0xA0E40006U, // lhz r7,6(r4)
        0xA9240006U, // lha r9,6(r4)
        0x99240008U, // stb r9,8(r4)
        0xB124000AU, // sth r9,10(r4)
    };
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t i = 0; i < program.size(); ++i)
            core->memory.write32_be(0x100U + static_cast<std::uint32_t>(i * 4U), program[i]);
        core->memory.write32_be(0x200U, 0xFEDCBA98U);
        core->state.cia = 0x100U;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[8] = 0x12345678U;
        core->state.gpr[9] = 0xCAFEBABEU;
    }
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.memory.read32_be(0x200U) == interpreted.memory.read32_be(0x200U));
    assert(compiled.memory.read32_be(0x204U) == 0xFEDCBA98U);
    assert(compiled.memory.read8(0x208U) == 0x98U);
    assert(compiled.memory.read16_be(0x20AU) == 0xBA98U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const std::string history = format_instruction_history(actual);
    assert(history.find("lwz") != std::string::npos);
    assert(history.find("0x00000200") != std::string::npos);

    // Indexed memory operations use GPR[RB] even when RB is r0, while RA=0 is
    // the architectural zero base. Check both BE data and sign extension.
    EspressoCore indexed_interpreted(0x1000U);
    EspressoCore indexed_compiled(0x1000U);
    const std::array<std::uint32_t, 4> indexed_program{
        x_form(3, 4, 0, 23),  // lwzx r3,r4,r0 (r0 is an ordinary index)
        x_form(3, 4, 0, 151), // stwx r3,r4,r0
        x_form(6, 4, 0, 87),  // lbzx r6,r4,r0
        x_form(7, 4, 5, 343),// lhax r7,r4,r5
    };
    for (EspressoCore* core : {&indexed_interpreted, &indexed_compiled})
    {
        for (std::size_t i = 0; i < indexed_program.size(); ++i)
            core->memory.write32_be(0x100U + static_cast<std::uint32_t>(i * 4U),
                                    indexed_program[i]);
        core->memory.write32_be(0x220U, 0xFEDCBA98U);
        core->memory.write16_be(0x204U, 0x8001U);
        core->memory.write16_be(0x224U, 0x8001U);
        core->state.cia = 0x100U;
        core->state.gpr[0] = 0x20U;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[5] = 4U;
    }
    const RunResult indexed_oracle = indexed_interpreted.run(indexed_program.size());
    Recompiler indexed_recompiler;
    const RunResult indexed_actual = indexed_recompiler.run(
        indexed_compiled, indexed_program.size());
    assert(indexed_actual.steps == indexed_oracle.steps);
    assert(indexed_compiled.state.gpr == indexed_interpreted.state.gpr);
    assert(indexed_compiled.state.gpr[3] == 0xFEDCBA98U);
    assert(indexed_compiled.state.gpr[6] == 0x000000FEU);
    assert(indexed_compiled.state.gpr[7] == 0xFFFF8001U);

    // Memory instructions must observe values still resident in the block-local
    // guest-register cache, including a freshly computed base and store source.
    EspressoCore memory_dependencies_interpreted(0x1000U);
    EspressoCore memory_dependencies_compiled(0x1000U);
    const std::array<std::uint32_t, 4> memory_dependency_program{
        0x38800200U, // addi r4,r0,0x200
        0x80C40000U, // lwz r6,0(r4)
        0x38A01234U, // addi r5,r0,0x1234
        0x90A40004U, // stw r5,4(r4)
    };
    for (EspressoCore* core : {&memory_dependencies_interpreted,
                               &memory_dependencies_compiled})
    {
        for (std::size_t i = 0; i < memory_dependency_program.size(); ++i)
        {
            core->memory.write32_be(
                0x100U + static_cast<std::uint32_t>(i * 4U),
                memory_dependency_program[i]);
        }
        core->memory.write32_be(0x200U, 0xABCDU);
        core->state.cia = 0x100U;
    }
    const RunResult memory_dependency_oracle =
        memory_dependencies_interpreted.run(memory_dependency_program.size());
    Recompiler memory_dependency_recompiler;
    const RunResult memory_dependency_actual = memory_dependency_recompiler.run(
        memory_dependencies_compiled, memory_dependency_program.size());
    assert(memory_dependency_actual.steps == memory_dependency_oracle.steps);
    assert(memory_dependencies_compiled.state.gpr ==
           memory_dependencies_interpreted.state.gpr);
    assert(memory_dependencies_compiled.memory.read32_be(0x204U) == 0x1234U);
    assert(memory_dependencies_compiled.state.gpr[6] == 0xABCDU);
}

void recompiler_memory_fault_test()
{
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(0x100U, 0x3860002AU); // addi r3,r0,42
        core->memory.write32_be(0x104U, 0x80640000U); // lwz r3,0(r4), unmapped
        core->state.cia = 0x100U;
        core->state.gpr[3] = 0xDEADBEEFU;
        core->state.gpr[4] = 0x3000U;
    }
    const RunResult oracle = interpreted.run(4U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 4U);
    assert(actual.reason == StopReason::memory_fault);
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == 1U);
    assert(actual.cia == oracle.cia && actual.cia == 0x104U);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[3] == 42U);
    assert(actual.detail.find("read 4 byte(s)") != std::string::npos);
    assert(!actual.instruction_history.empty());
    assert(!actual.instruction_history.back().completed);
}

void recompiler_self_modifying_code_test()
{
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, 0x38600001U); // addi r3,r0,1
    core.state.cia = 0x100U;
    Recompiler recompiler;

    assert(recompiler.run(core, 1U).steps == 1U);
    assert(core.state.gpr[3] == 1U);
    core.memory.write32_be(0x100U, 0x38600002U); // addi r3,r0,2
    core.state.cia = 0x100U;
    core.state.gpr[3] = 0U;
    assert(recompiler.run(core, 1U).steps == 1U);
    assert(core.state.gpr[3] == 2U);
    assert(recompiler.statistics().compiled_blocks == 2U);

    // A translated guest store can rewrite a block which was already cached.
    // The dispatcher must validate the changed code before it is entered again.
    EspressoCore guest_modified(0x1000U);
    guest_modified.memory.write32_be(0x100U, 0x90A40000U); // stw r5,0(r4)
    guest_modified.memory.write32_be(0x104U, 0x38600001U); // addi r3,r0,1
    guest_modified.state.cia = 0x104U;
    Recompiler guest_writer;
    assert(guest_writer.run(guest_modified, 1U).steps == 1U); // cache old block at 104
    guest_modified.state.gpr[4] = 0x104U;
    guest_modified.state.gpr[5] = 0x38600002U;
    guest_modified.state.cia = 0x100U;
    assert(guest_writer.run(guest_modified, 1U).steps == 1U); // store and return
    assert(guest_modified.state.cia == 0x104U);
    assert(guest_writer.run(guest_modified, 1U).steps == 1U);
    assert(guest_modified.state.gpr[3] == 2U);

    // If a store rewrites a later instruction in the currently executing
    // native block, it must exit before stale decoded instructions run.
    EspressoCore intra_block_modified(0x1000U);
    intra_block_modified.memory.write32_be(0x100U, 0x90A40000U); // stw r5,0(r4)
    intra_block_modified.memory.write32_be(0x104U, 0x38600001U); // addi r3,r0,1
    intra_block_modified.memory.write32_be(0x108U, 0x38C00001U); // addi r6,r0,1
    intra_block_modified.state.gpr[4] = 0x108U;
    intra_block_modified.state.gpr[5] = 0x38C00002U; // replace with addi r6,r0,2
    intra_block_modified.state.cia = 0x100U;
    Recompiler intra_block_writer;
    const RunResult intra_block_result = intra_block_writer.run(
        intra_block_modified, 3U);
    assert(intra_block_result.steps == 3U);
    assert(intra_block_modified.state.gpr[3] == 1U);
    assert(intra_block_modified.state.gpr[6] == 2U);
    assert(intra_block_modified.memory.read32_be(0x108U) == 0x38C00002U);

    guest_writer.clear_cache();
    assert(guest_writer.statistics().compiled_blocks == 0U);
    assert(guest_writer.statistics().native_instructions == 0U);

    EspressoCore first(0x1000U);
    EspressoCore second(0x1000U);
    first.memory.write32_be(0x100U, 0x38600001U);
    second.memory.write32_be(0x100U, 0x38600002U);
    first.state.cia = second.state.cia = 0x100U;
    Recompiler shared_cache;
    assert(shared_cache.run(first, 1U).steps == 1U);
    assert(shared_cache.run(second, 1U).steps == 1U);
    assert(first.state.gpr[3] == 1U);
    assert(second.state.gpr[3] == 2U);
    assert(shared_cache.statistics().compiled_blocks == 2U);
}

void recompiler_stop_reason_test()
{
    {
        EspressoCore core(0x1000U);
        core.memory.write32_be(0x100U, 0xFFFFFFFFU);
        core.state.cia = 0x100U;
        Recompiler recompiler;
        const RunResult result = recompiler.run(core, 1U);
        assert(result.reason == StopReason::unsupported_instruction);
        assert(result.steps == 0U);
        assert(result.instruction_word == 0xFFFFFFFFU);
        assert(result.has_instruction_word);
    }
    {
        EspressoCore core(0x1000U);
        core.memory.write32_be(0x100U, 0x3860002AU); // addi r3,r0,42
        core.memory.write32_be(0x104U, 0xFFFFFFFFU); // unsupported
        core.state.cia = 0x100U;
        Recompiler recompiler;
        const RunResult result = recompiler.run(core, 10U);
        assert(result.reason == StopReason::unsupported_instruction);
        assert(result.steps == 1U);
        assert(result.cia == 0x104U);
        assert(core.state.gpr[3] == 42U);
        assert(result.instruction_word == 0xFFFFFFFFU);
    }
    {
        EspressoCore core(0x1000U);
        core.hle.register_function("test", "missing", [](EspressoCore&) {});
        const std::uint32_t import = core.hle.bind_import("test", "missing");
        core.state.cia = import;
        Recompiler recompiler;
        const RunResult result = recompiler.run(core, 1U);
        assert(result.reason == StopReason::instruction_limit);
        assert(result.steps == 1U);
        assert(core.state.cia == core.state.lr);
    }
    {
        EspressoCore core(0x1000U);
        std::uint32_t handler_calls = 0U;
        core.hle.register_function("test", "jump_target", [&handler_calls](EspressoCore& machine)
        {
            ++handler_calls;
            machine.state.gpr[4] = 0x44U;
        });
        const std::uint32_t import = core.hle.bind_import("test", "jump_target");
        core.memory.write32_be(0x100U, 0x4E800420U); // bctr into the HLE range
        core.memory.write32_be(0x104U, 0x3860002AU); // resume at LR
        core.state.cia = 0x100U;
        core.state.lr = 0x104U;
        core.state.ctr = import;
        Recompiler recompiler;
        const RunResult result = recompiler.run(core, 3U);
        assert(result.steps == 3U);
        assert(result.reason == StopReason::instruction_limit);
        assert(handler_calls == 1U);
        assert(core.state.gpr[3] == 42U);
        assert(core.state.gpr[4] == 0x44U);
        assert(recompiler.statistics().interpreter_fallbacks == 1U);
        assert(recompiler.statistics().native_instructions == 2U);
    }
    {
        EspressoCore core(0x1000U);
        const std::uint32_t import = core.hle.bind_import("missing", "symbol");
        core.state.cia = import;
        Recompiler recompiler;
        const RunResult result = recompiler.run(core, 10U);
        assert(result.reason == StopReason::unimplemented_hle_call);
        assert(result.steps == 0U);
        assert(result.hle_call == "missing::symbol");
    }
}

}

int main()
{
    recompiler_differential_arithmetic_test();
    recompiler_destination_aliases_second_source_test();
    recompiler_logical_differential_test();
    recompiler_condition_bit_differential_tests();
    recompiler_step_budget_test();
    recompiler_recent_history_window_test();
    recompiler_branch_test();
    recompiler_conditional_branch_tests();
    recompiler_special_register_move_tests();
    recompiler_interpreter_fallback_test();
    recompiler_memory_tests();
    recompiler_memory_fault_test();
    recompiler_self_modifying_code_test();
    recompiler_stop_reason_test();
}
