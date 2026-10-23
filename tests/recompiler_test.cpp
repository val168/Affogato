#include "cpu/espresso/recompiler.hpp"

#include <cassert>
#include <array>
#include <bit>
#include <cstdint>
#include <string>

using namespace affogato::cpu::espresso;

namespace
{

[[nodiscard]] constexpr std::uint32_t x_form(
    std::uint8_t rd, std::uint8_t ra, std::uint8_t rb, std::uint16_t xo,
    bool record = false)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(rd) << 21U) |
           (static_cast<std::uint32_t>(ra) << 16U) |
           (static_cast<std::uint32_t>(rb) << 11U) |
           (static_cast<std::uint32_t>(xo) << 1U) | static_cast<std::uint32_t>(record);
}

[[nodiscard]] constexpr std::uint32_t mcrf_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (19U << 26U) | (static_cast<std::uint32_t>(destination) << 23U) |
           (static_cast<std::uint32_t>(source) << 18U);
}

[[nodiscard]] constexpr std::uint32_t mtcrf_form(
    std::uint8_t source, std::uint8_t field_mask)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
           (static_cast<std::uint32_t>(field_mask) << 12U) | (144U << 1U);
}

[[nodiscard]] constexpr std::uint32_t mcrxr_form(std::uint8_t destination)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(destination) << 23U) |
           (512U << 1U);
}

[[nodiscard]] constexpr std::uint32_t cmpwi_form(
    std::uint8_t cr_field, std::uint8_t ra, std::int16_t immediate)
{
    return (11U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
           (static_cast<std::uint32_t>(ra) << 16U) |
           static_cast<std::uint16_t>(immediate);
}

[[nodiscard]] constexpr std::uint32_t cmplwi_form(
    std::uint8_t cr_field, std::uint8_t ra, std::uint16_t immediate)
{
    return (10U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
        (static_cast<std::uint32_t>(ra) << 16U) | immediate;
}

[[nodiscard]] constexpr std::uint32_t cmpw_form(
    std::uint8_t cr_field, std::uint8_t ra, std::uint8_t rb,
    bool unsigned_compare)
{
    const std::uint32_t xo = unsigned_compare ? 32U : 0U;
    return (31U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
           (static_cast<std::uint32_t>(ra) << 16U) |
           (static_cast<std::uint32_t>(rb) << 11U) | (xo << 1U);
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

[[nodiscard]] constexpr std::uint32_t rlwinm_form(
    std::uint8_t destination, std::uint8_t source, std::uint8_t shift,
    std::uint8_t begin, std::uint8_t end, bool record)
{
    return (21U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
           (static_cast<std::uint32_t>(destination) << 16U) |
           (static_cast<std::uint32_t>(shift) << 11U) |
           (static_cast<std::uint32_t>(begin) << 6U) |
           (static_cast<std::uint32_t>(end) << 1U) | (record ? 1U : 0U);
}

[[nodiscard]] constexpr std::uint32_t rlwimi_form(
    std::uint8_t destination, std::uint8_t source, std::uint8_t shift,
    std::uint8_t begin, std::uint8_t end, bool record)
{
    return (20U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
           (static_cast<std::uint32_t>(destination) << 16U) |
           (static_cast<std::uint32_t>(shift) << 11U) |
           (static_cast<std::uint32_t>(begin) << 6U) |
           (static_cast<std::uint32_t>(end) << 1U) | (record ? 1U : 0U);
}

[[nodiscard]] constexpr std::uint32_t ps_merge10_form(
    std::uint8_t destination, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(source_a) << 16U) |
           (static_cast<std::uint32_t>(source_b) << 11U) | (592U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_merge00_form(
    std::uint8_t destination, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(source_a) << 16U) |
           (static_cast<std::uint32_t>(source_b) << 11U) | (528U << 1U);
}

[[nodiscard]] constexpr std::uint32_t fctiwz_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (15U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfd_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (50U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(base) << 16U) |
           static_cast<std::uint16_t>(displacement);
}

[[nodiscard]] constexpr std::uint32_t lfs_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (48U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(base) << 16U) |
           static_cast<std::uint16_t>(displacement);
}

[[nodiscard]] constexpr std::uint32_t lbzu_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (35U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        static_cast<std::uint16_t>(displacement);
}

[[nodiscard]] constexpr std::uint32_t lwzu_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (33U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        static_cast<std::uint16_t>(displacement);
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

void recompiler_cmpwi_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 7> program{
        cmpwi_form(0U, 3U, 0),
        cmpwi_form(1U, 4U, -1),
        cmpwi_form(2U, 5U, 1),
        cmpwi_form(3U, 6U, -32768),
        cmpwi_form(4U, 7U, 32767),
        cmpwi_form(5U, 8U, -2),
        cmpwi_form(6U, 9U, 2),
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
        core->state.gpr[3] = 0U;
        core->state.gpr[4] = 0xFFFFFFFFU;
        core->state.gpr[5] = 1U;
        core->state.gpr[6] = 0x80000000U;
        core->state.gpr[7] = 0x7FFFFFFFU;
        core->state.gpr[8] = 0xFFFFFFFEU;
        core->state.gpr[9] = 3U;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x80000000U;
        core->state.lr = 0x12345678U;
        core->state.ctr = 0x87654321U;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.cr != 0xA5C39E71U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_cmplwi_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 8U> program{
        cmplwi_form(0U, 3U, 0U),
        cmplwi_form(1U, 4U, 0U),
        cmplwi_form(2U, 5U, 0x7FFFU),
        cmplwi_form(3U, 6U, 0x8000U),
        cmplwi_form(4U, 7U, 0xFFFFU),
        cmplwi_form(5U, 8U, 0xFFFFU),
        cmplwi_form(6U, 9U, 0xFFFFU),
        cmplwi_form(7U, 10U, 0x8000U),
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
        core->state.gpr[3] = 0U;
        core->state.gpr[4] = 1U;
        core->state.gpr[5] = 0x80000000U;
        core->state.gpr[6] = 0x7FFFU;
        core->state.gpr[7] = 0xFFFFU;
        core->state.gpr[8] = 0xFFFFFFFFU;
        core->state.gpr[9] = 0U;
        core->state.gpr[10] = 0x80000000U;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x80000000U;
        core->state.lr = 0x12345678U;
        core->state.ctr = 0x87654321U;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(!actual.instruction_history.empty());
    assert(actual.instruction_history.back().opcode_name == "cmplwi");
}

void recompiler_cmpw_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 8> program{
        cmpw_form(0U, 3U, 4U, false),
        cmpw_form(1U, 4U, 3U, false),
        cmpw_form(2U, 5U, 6U, false),
        cmpw_form(3U, 7U, 4U, false),
        cmpw_form(4U, 3U, 4U, true),
        cmpw_form(5U, 4U, 3U, true),
        cmpw_form(6U, 7U, 5U, true),
        cmpw_form(7U, 5U, 5U, true),
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
        core->state.gpr[3] = 0x80000000U;
        core->state.gpr[4] = 0x7FFFFFFFU;
        core->state.gpr[5] = 1U;
        core->state.gpr[6] = 1U;
        core->state.gpr[7] = 0xFFFFFFFFU;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x80000000U;
        core->state.lr = 0x12345678U;
        core->state.ctr = 0x87654321U;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.cr != 0xA5C39E71U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_fcmpu_differential_test()
{
    const auto encode_fcmpu = [](std::uint8_t cr_field, std::uint8_t fr_a,
                                 std::uint8_t fr_b) {
        return 0xFC000000U |
            (static_cast<std::uint32_t>(cr_field) << 23U) |
            (static_cast<std::uint32_t>(fr_a) << 16U) |
            (static_cast<std::uint32_t>(fr_b) << 11U);
    };
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint8_t cr_field;
        bool enable_invalid;
    };
    constexpr std::array cases{
        Case{0x3FF0000000000000ULL, 0x4000000000000000ULL, 0U, false},
        Case{0x4000000000000000ULL, 0x3FF0000000000000ULL, 1U, false},
        Case{0x3FF0000000000000ULL, 0x3FF0000000000000ULL, 2U, false},
        Case{0x0000000000000000ULL, 0x8000000000000000ULL, 3U, false},
        Case{0x7FF8000000000001ULL, 0x3FF0000000000000ULL, 4U, false},
        Case{0x3FF0000000000000ULL, 0x7FF0000000000001ULL, 5U, false},
        Case{0x7FF0000000000001ULL, 0x3FF0000000000000ULL, 6U, false},
        Case{0x3FF0000000000000ULL, 0xFFF0000000000001ULL, 7U, true},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, encode_fcmpu(test.cr_field, 1U, 31U));
            core->state.cia = 0x100U;
            core->state.fpr[1] = test.a;
            core->state.fpr[31] = test.b;
            core->state.fpr_ps1[1] = 0xDEADBEEF01234567ULL;
            core->state.fpr_ps1[31] = 0xBAD0BAD0BAD0BAD0ULL;
            core->state.cr = 0xA5C36E91U;
            core->state.xer = 0x80000000U;
            core->state.fpscr = 0x0000A025U |
                (test.enable_invalid ? fpscr::ve_mask : 0U);
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fcmpu");
        assert(history.has_fp_compare);
        assert(history.fp_compare_cr_field == test.cr_field);
        assert(history.fp_compare_a_raw == test.a);
        assert(history.fp_compare_b_raw == test.b);
    }
}

void recompiler_subfic_differential_test()
{
    struct Case
    {
        std::int16_t immediate;
        std::uint32_t source;
        std::uint8_t base;
        std::uint8_t destination;
        std::uint32_t initial_xer;
        std::uint32_t expected_result;
        bool expected_carry;
    };
    constexpr std::array cases{
        Case{5, 5U, 4U, 3U, 0xA5A5A5A5U, 0U, true},
        Case{5, 6U, 4U, 3U, 0x85A5A5A5U, 0xFFFFFFFFU, false},
        Case{5, 4U, 4U, 4U, 0x25A5A5A5U, 1U, true},
        Case{-1, 0U, 0U, 0U, 0xA5A5A5A5U, 0xFFFFFFFFU, true},
        Case{-1, 0xFFFFFFFFU, 7U, 7U, 0x85A5A5A5U, 0U, true},
        Case{-17, 0xFFFFFFF0U, 8U, 9U, 0x25A5A5A5U, 0xFFFFFFFFU, false},
        Case{-17, 0xFFFFFFEEU, 8U, 8U, 0xA5A5A5A5U, 1U, true},
        Case{0, 1U, 31U, 31U, 0x85A5A5A5U, 0xFFFFFFFFU, false},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = 0x20000000U |
            (static_cast<std::uint32_t>(test.destination) << 21U) |
            (static_cast<std::uint32_t>(test.base) << 16U) |
            static_cast<std::uint16_t>(test.immediate);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            for (std::uint32_t reg = 0U; reg < core->state.gpr.size(); ++reg)
            {
                core->state.gpr[reg] = 0x10203040U + reg * 0x01010101U;
            }
            core->state.gpr[test.base] = test.source;
            core->state.xer = test.initial_xer;
            core->state.cr = 0xA5C36E91U;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.destination] == test.expected_result);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == test.expected_carry);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert((compiled.state.xer & ~0x20000000U) ==
               (test.initial_xer & ~0x20000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "subfic");
        assert(history.has_immediate && history.immediate == test.immediate);
    }
}

void recompiler_subfc_differential_test()
{
    struct Case
    {
        std::uint32_t minuend;
        std::uint32_t subtrahend;
        std::uint8_t base;
        std::uint8_t source;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{5U, 5U, 4U, 3U, 2U, false, 0xA5A5A5A5U, 0x13579BDFU},
        Case{6U, 5U, 4U, 3U, 2U, false, 0x85A5A5A5U, 0xA5C36E91U},
        Case{5U, 6U, 4U, 3U, 2U, true, 0x25A5A5A5U, 0xA5C36E91U},
        Case{0U, 0xFFFFFFFFU, 4U, 3U, 3U, false, 0x80000001U, 0x12345678U},
        Case{0xFFFFFFFFU, 0U, 4U, 3U, 4U, true, 0x20000005U, 0xA5C36E91U},
        Case{0x80000000U, 0x80000000U, 7U, 7U, 7U, true, 0x80000000U, 0x87654321U},
        Case{0xFFFFFFFFU, 0x7FFFFFFFU, 4U, 3U, 5U, true, 0x60000000U, 0xFFFFFFFFU},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.base, test.source, 8U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.base] = test.subtrahend;
            core->state.gpr[test.source] = test.minuend;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[test.destination] ==
            static_cast<std::uint32_t>(test.minuend - test.subtrahend));
        assert(((compiled.state.xer & 0x20000000U) != 0U) ==
            (test.minuend >= test.subtrahend));
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        assert(actual.instruction_history.back().opcode_name ==
            (test.record ? "subfc." : "subfc"));
    }
}

void recompiler_subfe_differential_test()
{
    struct Case
    {
        std::uint32_t ra_value;
        std::uint32_t rb_value;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{5U, 5U, 4U, 3U, 2U, false, 0xA5A5A5A5U, 0x13579BDFU},
        Case{5U, 5U, 4U, 3U, 2U, true, 0x85A5A5A5U, 0xA5C36E91U},
        Case{0U, 0U, 4U, 3U, 3U, false, 0x25A5A5A5U, 0xA5C36E91U},
        Case{0U, 0U, 4U, 3U, 4U, true, 0xA5A5A5A5U, 0x87654321U},
        Case{0xFFFFFFFFU, 0U, 4U, 3U, 4U, false, 0x85A5A5A5U, 0x12345678U},
        Case{0xFFFFFFFFU, 0U, 4U, 3U, 3U, true, 0x25A5A5A5U, 0xA5C36E91U},
        Case{0x80000000U, 0x80000000U, 7U, 7U, 7U, true, 0x80000000U, 0xFFFFFFFFU},
        Case{0x80000000U, 0x7FFFFFFFU, 7U, 8U, 7U, true, 0x20000000U, 0x87654321U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 136U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.ra_value;
            core->state.gpr[test.rb] = test.rb_value;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);

        const std::uint32_t carry_in =
            (test.initial_xer & 0x20000000U) != 0U ? 1U : 0U;
        const std::uint64_t sum =
            static_cast<std::uint64_t>(~test.ra_value) + test.rb_value + carry_in;
        const std::uint32_t expected = static_cast<std::uint32_t>(sum);
        assert(compiled.state.gpr[test.destination] == expected);
        assert(((compiled.state.xer & 0x20000000U) != 0U) ==
            ((sum >> 32U) != 0U));
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        assert(actual.instruction_history.back().opcode_name ==
            (test.record ? "subfe." : "subfe"));
    }
}

void recompiler_addic_differential_test()
{
    struct Case
    {
        std::int16_t immediate;
        std::uint32_t source;
        std::uint8_t base;
        std::uint8_t destination;
        std::uint32_t initial_xer;
    };
    constexpr std::array cases{
        Case{5, 5U, 4U, 3U, 0xA5A5A5A5U},
        Case{1, 0xFFFFFFFFU, 4U, 4U, 0x85A5A5A5U},
        Case{-1, 1U, 0U, 0U, 0x25A5A5A5U},
        Case{-1, 0U, 7U, 7U, 0xA5A5A5A5U},
        Case{0, 0x80000000U, 31U, 31U, 0x85A5A5A5U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = 0x30000000U |
            (static_cast<std::uint32_t>(test.destination) << 21U) |
            (static_cast<std::uint32_t>(test.base) << 16U) |
            static_cast<std::uint16_t>(test.immediate);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            for (std::uint32_t reg = 0U; reg < core->state.gpr.size(); ++reg)
            {
                core->state.gpr[reg] = 0x10203040U + reg * 0x01010101U;
            }
            core->state.gpr[test.base] = test.source;
            core->state.xer = test.initial_xer;
            core->state.cr = 0xA5C36E91U;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert((compiled.state.xer & ~0x20000000U) ==
               (test.initial_xer & ~0x20000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "addic");
        assert(history.has_immediate && history.immediate == test.immediate);
    }

    struct RecordCase
    {
        std::uint32_t source;
        std::int16_t immediate;
        std::uint32_t xer;
        std::uint8_t expected_cr0;
    };
    constexpr std::array record_cases{
        RecordCase{0xFFFFFFFCU, 4, 0x80000000U, 3U},
        RecordCase{0x80000000U, 0, 0U, 8U},
        RecordCase{1U, 1, 0x80000000U, 5U},
    };
    for (const RecordCase& test : record_cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, 0x34000000U |
                (9U << 21U) | (9U << 16U) |
                static_cast<std::uint16_t>(test.immediate));
            core->state.cia = 0x100U;
            core->state.gpr[9] = test.source;
            core->state.xer = test.xer;
            core->state.cr = 0xA5C36E91U;
        }
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.cr == interpreted.state.cr);
        assert((compiled.state.cr >> 28U) == test.expected_cr0);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.back().opcode_name == "addic.");
        assert(oracle.steps == actual.steps);
    }
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

void recompiler_logical_record_differential_test()
{
    struct Case
    {
        std::uint32_t instruction;
        std::uint32_t source_a;
        std::uint32_t source_b;
        std::uint32_t xer;
        std::uint32_t expected_result;
        std::uint8_t expected_cr0;
        const char* name;
    };
    constexpr std::array cases{
        Case{x_form(3U, 5U, 4U, 444U, true), 0x80000000U, 0U,
             0x80000000U, 0x80000000U, 9U, "or."},
        Case{x_form(3U, 5U, 4U, 28U, true), 0xFFFFFFFFU, 0U,
             0U, 0U, 2U, "and."},
        Case{x_form(3U, 5U, 4U, 316U, true), 5U, 3U,
             0x80000000U, 6U, 5U, "xor."},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, test.instruction);
            core->state.cia = 0x100U;
            core->state.gpr[3] = test.source_a;
            core->state.gpr[4] = test.source_b;
            core->state.cr = 0xA5C36E91U;
            core->state.xer = test.xer;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[5] == test.expected_result);
        assert((compiled.state.cr >> 28U) == test.expected_cr0);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.back().opcode_name == test.name);
    }
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

void recompiler_mcrf_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> cr_values{
        0U, 0xFFFFFFFFU, 0xA5C39E71U, 0x13579BDFU};

    for (std::uint8_t destination = 0U; destination < 8U; ++destination)
    {
        for (std::uint8_t source = 0U; source < 8U; ++source)
        {
            for (const std::uint32_t initial_cr : cr_values)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint32_t instruction = mcrf_form(destination, source);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.cr = initial_cr;
                    core->state.xer = 0xE0000000U;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0x76543210U;
                    core->state.fpscr = 0x89ABCDEFU;
                    core->state.gpr[3] = 0x13579BDFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);

                const std::uint32_t destination_shift =
                    (7U - destination) * 4U;
                const std::uint32_t source_shift = (7U - source) * 4U;
                const std::uint32_t destination_mask = 0xFU << destination_shift;
                const std::uint32_t expected =
                    (initial_cr & ~destination_mask) |
                    (((initial_cr >> source_shift) & 0xFU) << destination_shift);
                assert(compiled.state.cr == expected);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name == "mcrf");
            }
        }
    }
}

void recompiler_mfcr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> cr_values{
        0U, 0xFFFFFFFFU, 0xA5C39E71U, 0x13579BDFU};

    for (std::uint8_t destination = 0U; destination < 32U; ++destination)
    {
        for (const std::uint32_t initial_cr : cr_values)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = x_form(destination, 0U, 0U, 19U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.cr = initial_cr;
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.gpr[destination] = 0x12345678U;
                core->state.xer = 0xE0000000U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0x76543210U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.cr == initial_cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(compiled.state.gpr[destination] == initial_cr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mfcr");
        }
    }
}

void recompiler_mtcrf_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 3U> cr_values{
        0U, 0xFFFFFFFFU, 0xA5C39E71U};
    constexpr std::array<std::uint32_t, 3U> source_values{
        0x13579BDFU, 0xECA86420U, 0x5AA59669U};

    for (std::uint32_t raw_mask = 0U; raw_mask < 256U; ++raw_mask)
    {
        const auto field_mask = static_cast<std::uint8_t>(raw_mask);
        const std::uint8_t source = static_cast<std::uint8_t>(raw_mask % 3U == 0U
            ? 0U
            : raw_mask % 3U == 1U ? 17U : 31U);
        for (std::size_t pattern = 0U; pattern < cr_values.size(); ++pattern)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = mtcrf_form(source, field_mask);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.cr = cr_values[pattern];
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.gpr[source] = source_values[pattern];
                core->state.xer = 0xE0000000U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0x76543210U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);

            std::uint32_t cr_mask = 0U;
            for (std::uint32_t field_index = 0U; field_index < 8U; ++field_index)
            {
                if ((field_mask & (1U << (7U - field_index))) != 0U)
                {
                    cr_mask |= 0xFU << (28U - 4U * field_index);
                }
            }
            const std::uint32_t expected =
                (cr_values[pattern] & ~cr_mask) |
                (source_values[pattern] & cr_mask);
            assert(compiled.state.cr == expected);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mtcrf");
        }
    }
}

void recompiler_mcrxr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 8U> xer_values{
        0x00000000U, 0x80000001U, 0x40000002U, 0x20000004U,
        0xE0000010U, 0xA5C39E71U, 0x5A3C618EU, 0xFFFFFFFFU};
    constexpr std::array<std::uint32_t, 3U> cr_values{
        0U, 0xFFFFFFFFU, 0xA5C39E71U};

    for (std::uint8_t destination = 0U; destination < 8U; ++destination)
    {
        for (std::size_t pattern = 0U; pattern < xer_values.size(); ++pattern)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = mcrxr_form(destination);
            const std::uint32_t initial_cr = cr_values[pattern % cr_values.size()];
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.cr = initial_cr;
                core->state.xer = xer_values[pattern];
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0x76543210U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);

            const std::uint32_t shift = (7U - destination) * 4U;
            const std::uint32_t field_mask = 0xFU << shift;
            const std::uint32_t expected_cr =
                (initial_cr & ~field_mask) |
                (((xer_values[pattern] >> 28U) & 0xFU) << shift);
            assert(compiled.state.cr == expected_cr);
            assert(compiled.state.xer == (xer_values[pattern] & 0x0FFFFFFFU));
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mcrxr");
        }
    }
}

void recompiler_mfxer_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> xer_values{
        0U, 0xE0000000U, 0xA5C39E71U, 0xFFFFFFFFU};

    for (std::uint8_t destination = 0U; destination < 32U; ++destination)
    {
        for (const std::uint32_t initial_xer : xer_values)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = x_form(destination, 1U, 0U, 339U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.gpr[destination] = 0x12345678U;
                core->state.cr = 0xA5C39E71U;
                core->state.xer = initial_xer;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0x76543210U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.gpr[destination] == initial_xer);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == initial_xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mfxer");
        }
    }
}

void recompiler_mtxer_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> source_values{
        0U, 0xE0000000U, 0xA5C39E71U, 0xFFFFFFFFU};

    for (std::uint8_t source = 0U; source < 32U; ++source)
    {
        for (const std::uint32_t source_value : source_values)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = x_form(source, 1U, 0U, 467U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.gpr[source] = source_value;
                core->state.cr = 0xA5C39E71U;
                core->state.xer = 0x12345678U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0x76543210U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.xer == source_value);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mtxer");
        }
    }
}

void recompiler_mfgqr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 8U> gqr_values{
        0x00000000U, 0xFFFFFFFFU, 0x01234567U, 0x89ABCDEFU,
        0xA5C39E71U, 0x80000001U, 0x7FFFFFFEU, 0x13579BDFU};

    for (std::uint8_t gqr = 0U; gqr < 8U; ++gqr)
    {
        for (std::uint8_t destination = 0U; destination < 32U; ++destination)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint16_t spr = static_cast<std::uint16_t>(912U + gqr);
            const std::uint32_t instruction = x_form(
                destination, static_cast<std::uint8_t>(spr & 0x1FU),
                static_cast<std::uint8_t>(spr >> 5U), 339U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.gpr.fill(0xDEADBEEFU);
                core->state.gpr[destination] = 0x12345678U;
                core->state.gqr = gqr_values;
                core->state.cr = 0xA5C39E71U;
                core->state.xer = 0x76543210U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0xFEDCBA98U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.gpr == interpreted.state.gpr);
            assert(compiled.state.gpr[destination] == gqr_values[gqr]);
            assert(compiled.state.gqr == interpreted.state.gqr);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "mfgqr");
        }
    }
}

void recompiler_mtgqr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> source_values{
        0U, 0xFFFFFFFFU, 0x89ABCDEFU, 0x13579BDFU};
    constexpr std::array<std::uint32_t, 8U> initial_gqr{
        0x01020304U, 0x11121314U, 0x21222324U, 0x31323334U,
        0x41424344U, 0x51525354U, 0x61626364U, 0x71727374U};

    for (std::uint8_t gqr = 0U; gqr < 8U; ++gqr)
    {
        for (std::uint8_t source = 0U; source < 32U; ++source)
        {
            for (const std::uint32_t source_value : source_values)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint16_t spr = static_cast<std::uint16_t>(912U + gqr);
                const std::uint32_t instruction = x_form(
                    source, static_cast<std::uint8_t>(spr & 0x1FU),
                    static_cast<std::uint8_t>(spr >> 5U), 467U);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[source] = source_value;
                    core->state.gqr = initial_gqr;
                    core->state.cr = 0xA5C39E71U;
                    core->state.xer = 0x76543210U;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0xFEDCBA98U;
                    core->state.fpscr = 0x89ABCDEFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.gqr == interpreted.state.gqr);
                assert(compiled.state.gqr[gqr] == source_value);
                for (std::uint8_t other = 0U; other < 8U; ++other)
                {
                    if (other != gqr)
                    {
                        assert(compiled.state.gqr[other] == initial_gqr[other]);
                    }
                }
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name == "mtgqr");
            }
        }
    }
}

void recompiler_fmr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint64_t, 32U> source_values = [] {
        std::array<std::uint64_t, 32U> values{};
        for (std::size_t i = 0U; i < values.size(); ++i)
        {
            values[i] = 0xFFF0000000000000ULL | (i * 0x102030405ULL);
        }
        values[0] = 0x8000000000000000ULL;
        values[1] = 0xFFF8123456789ABCULL;
        values[2] = 0xFFF0123456789ABCULL;
        return values;
    }();
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;

    for (std::uint8_t source = 0U; source < 32U; ++source)
    {
        for (std::uint8_t destination = 0U; destination < 32U; ++destination)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction =
                (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
                (static_cast<std::uint32_t>(source) << 11U) | (72U << 1U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.fpr.fill(0xA5A5A5A5A5A5A5A5ULL);
                core->state.fpr_ps1.fill(ps1_sentinel);
                core->state.fpr[source] = source_values[source];
                core->state.cr = 0xA5C39E71U;
                core->state.xer = 0x76543210U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0xFEDCBA98U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.fpr == interpreted.state.fpr);
            assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
            assert(compiled.state.fpr[destination] == source_values[source]);
            assert(compiled.state.fpr_ps1[destination] == ps1_sentinel);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "fmr");
        }
    }
}

void recompiler_fneg_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint64_t, 32U> source_values = [] {
        std::array<std::uint64_t, 32U> values{};
        for (std::size_t i = 0U; i < values.size(); ++i)
        {
            values[i] = 0xFFF0000000000000ULL | (i * 0x102030405ULL);
        }
        values[0] = 0x0000000000000000ULL;
        values[1] = 0x8000000000000000ULL;
        values[2] = 0xFFF8123456789ABCULL;
        values[3] = 0xFFF0123456789ABCULL;
        return values;
    }();
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;

    for (std::uint8_t source = 0U; source < 32U; ++source)
    {
        for (std::uint8_t destination = 0U; destination < 32U; ++destination)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction =
                (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
                (static_cast<std::uint32_t>(source) << 11U) | (40U << 1U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.fpr.fill(0xA5A5A5A5A5A5A5A5ULL);
                core->state.fpr_ps1.fill(ps1_sentinel);
                core->state.fpr[source] = source_values[source];
                core->state.cr = 0xA5C39E71U;
                core->state.xer = 0x76543210U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0xFEDCBA98U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.fpr == interpreted.state.fpr);
            assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
            assert(compiled.state.fpr[destination] ==
                (source_values[source] ^ 0x8000000000000000ULL));
            assert(compiled.state.fpr_ps1[destination] == ps1_sentinel);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "fneg");
        }
    }
}

void recompiler_fctiwz_differential_tests()
{
    struct Case
    {
        std::uint64_t source_raw;
        std::uint32_t initial_fpscr;
        std::uint8_t destination;
        std::uint8_t source;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t destination_sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array cases{
        Case{0x0000000000000000ULL, 0x0001F000U, 2U, 1U}, // +0
        Case{0x8000000000000000ULL, 0x0001F000U, 2U, 1U}, // -0 marker
        Case{0x4008000000000000ULL, 0x0001F000U, 2U, 1U}, // 3.0
        Case{0x400F333333333333ULL, 0x0001F000U, 2U, 1U}, // 3.9
        Case{0xC00F333333333333ULL, 0x0001F000U, 2U, 1U}, // -3.9
        Case{0x3FE0000000000000ULL, 0x0001F000U, 2U, 1U}, // +0.5
        Case{0xBFE0000000000000ULL, 0x0001F000U, 2U, 1U}, // -0.5 marker
        Case{0x41DFFFFFFFC00000ULL, 0x0001F000U, 2U, 1U}, // INT32_MAX
        Case{0xC1E0000000000000ULL, 0x0001F000U, 2U, 1U}, // INT32_MIN
        Case{0x41E0000000000000ULL, 0x0001F000U, 2U, 1U}, // +overflow
        Case{0x7FF0000000000000ULL, 0x0001F000U, 2U, 1U}, // +infinity
        Case{0x7FF8123456789ABCULL, 0x0001F000U, 2U, 1U}, // qNaN
        Case{0x7FF0123456789ABCULL, 0x0001F000U, 2U, 1U}, // sNaN
        Case{0x7FF0123456789ABCULL, 0x0001F200U, 2U, 1U}, // sNaN, VE
        Case{0x41E0000000000000ULL, 0x0001F200U, 2U, 1U}, // overflow, VE
        Case{0xC00F333333333333ULL, 0x0201F000U, 1U, 1U}, // alias, sticky XX
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = fctiwz_form(
            test.destination, test.source);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.fpr.fill(destination_sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[test.source] = test.source_raw;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x76543210U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.initial_fpscr;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fctiwz");
        assert(history.has_fp_integer_conversion);
        assert(history.fp_integer_conversion_source_raw == test.source_raw);
        assert(history.fp_integer_conversion_destination == test.destination);
    }
}

void recompiler_fabs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint64_t, 32U> source_values = [] {
        std::array<std::uint64_t, 32U> values{};
        for (std::size_t i = 0U; i < values.size(); ++i)
        {
            values[i] = 0xFFF0000000000000ULL | (i * 0x102030405ULL);
        }
        values[0] = 0x8000000000000000ULL;
        values[1] = 0x0000000000000000ULL;
        values[2] = 0xFFF8123456789ABCULL;
        values[3] = 0xFFF0123456789ABCULL;
        return values;
    }();
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;

    for (std::uint8_t source = 0U; source < 32U; ++source)
    {
        for (std::uint8_t destination = 0U; destination < 32U; ++destination)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction =
                (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
                (static_cast<std::uint32_t>(source) << 11U) | (264U << 1U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.fpr.fill(0xA5A5A5A5A5A5A5A5ULL);
                core->state.fpr_ps1.fill(ps1_sentinel);
                core->state.fpr[source] = source_values[source];
                core->state.cr = 0xA5C39E71U;
                core->state.xer = 0x76543210U;
                core->state.lr = 0xABCDEF00U;
                core->state.ctr = 0xFEDCBA98U;
                core->state.fpscr = 0x89ABCDEFU;
            }

            const RunResult oracle = interpreted.run(1U);
            Recompiler recompiler;
            const RunResult actual = recompiler.run(compiled, 1U);
            assert(actual.reason == oracle.reason);
            assert(actual.steps == oracle.steps && actual.steps == 1U);
            assert(actual.cia == oracle.cia);
            assert(compiled.state.fpr == interpreted.state.fpr);
            assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
            assert(compiled.state.fpr[destination] ==
                (source_values[source] & 0x7FFFFFFFFFFFFFFFULL));
            assert(compiled.state.fpr_ps1[destination] == ps1_sentinel);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "fabs");
        }
    }
}

void recompiler_ps_merge10_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    std::array<std::uint32_t, 32U> program{};
    program[0] = ps_merge10_form(1U, 1U, 1U);
    program[1] = ps_merge10_form(2U, 2U, 3U);
    program[2] = ps_merge10_form(4U, 5U, 4U);
    for (std::uint32_t index = 3U; index < program.size(); ++index)
    {
        const std::uint8_t source_a = static_cast<std::uint8_t>((index * 7U + 2U) % 32U);
        const std::uint8_t source_b = static_cast<std::uint8_t>((index * 13U + 9U) % 32U);
        std::uint8_t destination = static_cast<std::uint8_t>((index * 11U + 6U) % 32U);
        if (destination == 1U)
        {
            destination = 0U;
        }
        program[index] = ps_merge10_form(destination, source_a, source_b);
    }

    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < core->state.fpr.size(); ++reg)
        {
            core->state.fpr[reg] = 0x3FF0000000000000ULL +
                static_cast<std::uint64_t>(reg) * 0x0001000000000001ULL;
            core->state.fpr_ps1[reg] = 0xBFF0000000000000ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0101010101010101ULL);
        }
        core->state.fpr[1] = 0x7FF0123456789ABCULL;
        core->state.fpr_ps1[1] = 0xFFF8123456789ABCULL;
        core->state.fpr[5] = 0x8000000000000000ULL;
        core->state.fpr_ps1[5] = 0x0000000000000001ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const std::uint64_t expected_alias_ps0 = interpreted.state.fpr_ps1[1];
    const std::uint64_t expected_alias_ps1 = interpreted.state.fpr[1];
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.fpr[1] == expected_alias_ps0);
    assert(compiled.state.fpr_ps1[1] == expected_alias_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (const InstructionHistoryEntry& history : actual.instruction_history)
    {
        assert(history.opcode_name == "ps_merge10");
    }
}

void recompiler_ps_merge00_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    std::array<std::uint32_t, 32U> program{};
    program[0] = ps_merge00_form(1U, 1U, 1U);
    program[1] = ps_merge00_form(2U, 2U, 3U);
    program[2] = ps_merge00_form(4U, 5U, 4U);
    for (std::uint32_t index = 3U; index < program.size(); ++index)
    {
        const std::uint8_t source_a = static_cast<std::uint8_t>((index * 7U + 2U) % 32U);
        const std::uint8_t source_b = static_cast<std::uint8_t>((index * 13U + 9U) % 32U);
        std::uint8_t destination = static_cast<std::uint8_t>((index * 11U + 6U) % 32U);
        if (destination == 1U)
            destination = 0U;
        program[index] = ps_merge00_form(destination, source_a, source_b);
    }

    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < core->state.fpr.size(); ++reg)
        {
            core->state.fpr[reg] = 0x3FF0000000000000ULL +
                static_cast<std::uint64_t>(reg) * 0x0001000000000001ULL;
            core->state.fpr_ps1[reg] = 0xBFF0000000000000ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0101010101010101ULL);
        }
        core->state.fpr[1] = 0x7FF0123456789ABCULL;
        core->state.fpr_ps1[1] = 0xFFF8123456789ABCULL;
        core->state.fpr[5] = 0x8000000000000000ULL;
        core->state.fpr_ps1[5] = 0x0000000000000001ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const std::uint64_t expected_alias_ps0 = interpreted.state.fpr[1];
    const std::uint64_t expected_alias_ps1 = interpreted.state.fpr[1];
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.fpr[1] == expected_alias_ps0);
    assert(compiled.state.fpr_ps1[1] == expected_alias_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (const InstructionHistoryEntry& history : actual.instruction_history)
        assert(history.opcode_name == "ps_merge00");
}

void recompiler_stfs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t address = 0x300U;
    constexpr std::array<std::uint64_t, 8U> fp_values{
        0x0000000000000000ULL, 0x3FF0000000000000ULL,
        0xBFF0000000000000ULL, 0x3FB999999999999AULL,
        0x47EFFFFFE0000000ULL, 0x7FF8123456789ABCULL,
        0x7FF0123456789ABCULL, 0x7FF0000000000000ULL};

    for (std::uint8_t source = 0U; source < fp_values.size(); ++source)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = (52U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) | (6U << 16U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr[6] = address;
            core->state.fpr[source] = fp_values[source];
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x76543210U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.memory.read32_be(address) == interpreted.memory.read32_be(address));
        assert(compiled.memory.read32_be(address) == std::bit_cast<std::uint32_t>(
            static_cast<float>(std::bit_cast<double>(fp_values[source]))));
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const auto& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfs");
        assert(history.has_fp_source && history.fp_source_register == source);
        assert(history.fp_source_value == fp_values[source]);
        assert(history.has_stored_single_value);
        assert(history.stored_single_value == compiled.memory.read32_be(address));
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, (52U << 26U) | (1U << 21U) | (6U << 16U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.fpr[1] = 0x3FF0000000000000ULL;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stfsu_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t initial_base = 0x300U;
    constexpr std::int16_t displacement = 4;
    constexpr std::uint8_t base_register = 6U;
    constexpr std::uint8_t source_register = 1U;
    constexpr std::uint32_t instruction = (53U << 26U) |
        (static_cast<std::uint32_t>(source_register) << 21U) |
        (static_cast<std::uint32_t>(base_register) << 16U) |
        static_cast<std::uint16_t>(displacement);
    constexpr std::array<std::uint64_t, 5U> fp_values{
        0x0000000000000000ULL, 0x8000000000000000ULL,
        0x3FF0000000000000ULL, 0x7FF8123456789ABCULL,
        0x7FF0123456789ABCULL};

    for (const std::uint64_t fp_value : fp_values)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[base_register] = initial_base;
            core->state.fpr.fill(0xDEADBEEF01234567ULL);
            core->state.fpr[source_register] = fp_value;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const std::uint32_t expected_value = std::bit_cast<std::uint32_t>(
            static_cast<float>(std::bit_cast<double>(fp_value)));
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.memory.read32_be(initial_base + displacement) == expected_value);
        assert(compiled.memory.read32_be(initial_base + displacement) ==
            interpreted.memory.read32_be(initial_base + displacement));
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[base_register] == initial_base + displacement);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfsu");
        assert(history.has_fp_source && history.fp_source_register == source_register);
        assert(history.fp_source_value == fp_value);
        assert(history.has_stored_single_value &&
            history.stored_single_value == expected_value);
        assert(history.has_destination &&
            history.destination_register == base_register &&
            history.destination_value == initial_base + displacement);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, instruction);
    faulting.state.cia = start;
    faulting.state.gpr[base_register] = 0x2000U;
    faulting.state.fpr[source_register] = 0x3FF0000000000000ULL;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[base_register] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stfd_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t address = 0x300U;
    constexpr std::array<std::uint64_t, 8U> fp_values{
        0x0000000000000000ULL, 0x8000000000000000ULL,
        0x3FF0000000000000ULL, 0xBFF8000000000000ULL,
        0x0000000000000001ULL, 0x7FF8123456789ABCULL,
        0x7FF0123456789ABCULL, 0xFFFFFFFFFFFFFFFFULL};

    for (std::uint8_t source = 0U; source < fp_values.size(); ++source)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = (54U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) | (6U << 16U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr[6] = address;
            core->state.fpr[source] = fp_values[source];
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x76543210U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.memory.read64_be(address) == interpreted.memory.read64_be(address));
        assert(compiled.memory.read64_be(address) == fp_values[source]);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const auto& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfd");
        assert(history.has_fp_source && history.fp_source_register == source);
        assert(history.fp_source_value == fp_values[source]);
        assert(history.has_stored_double_value);
        assert(history.stored_double_value == fp_values[source]);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, (54U << 26U) | (1U << 21U) | (6U << 16U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.fpr[1] = 0x3FF0000000000000ULL;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 8 byte(s)") != std::string::npos);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stwu_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    struct Case
    {
        std::uint8_t source;
        std::uint8_t base;
        std::uint32_t base_value;
        std::int16_t displacement;
        std::uint32_t stored_value;
    };
    constexpr std::array<Case, 4U> cases{{
        {6U, 5U, 0x400U, -4, 0x89ABCDEFU},
        {5U, 5U, 0x400U, 4, 0x400U},
        {0U, 7U, 0xFFFFFFFCU, 8, 0x13579BDFU},
        {31U, 2U, 0x800U, 0, 0xFFFFFFFFU}}};

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = (37U << 26U) |
            (static_cast<std::uint32_t>(test.source) << 21U) |
            (static_cast<std::uint32_t>(test.base) << 16U) |
            static_cast<std::uint16_t>(test.displacement);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base] = test.base_value;
            core->state.gpr[test.source] = test.stored_value;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x76543210U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t expected_address = test.base_value +
            static_cast<std::uint32_t>(test.displacement);
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.memory.read32_be(expected_address) ==
            interpreted.memory.read32_be(expected_address));
        assert(compiled.memory.read32_be(expected_address) == test.stored_value);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base] == expected_address);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const auto& history = actual.instruction_history.back();
        assert(history.opcode_name == "stwu");
        assert(history.has_effective_address && history.effective_address == expected_address);
        assert(history.has_destination && history.destination_register == test.base);
        assert(history.destination_value == expected_address);
        assert(history.has_stored_integer_word_value);
        assert(history.stored_integer_word_value == test.stored_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, (37U << 26U) | (6U << 21U) | (5U << 16U));
    faulting.state.cia = start;
    faulting.state.gpr[5] = 0x2000U;
    faulting.state.gpr[6] = 0x12345678U;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(faulting.state.gpr[5] == 0x2000U);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_slw_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::array<std::uint8_t, 3U>, 4U> registers{{
        {{3U, 4U, 5U}}, {{3U, 3U, 5U}}, {{3U, 4U, 4U}}, {{3U, 3U, 3U}}}};
    constexpr std::array<std::uint32_t, 8U> source_values{
        0U, 1U, 0x80000000U, 0xFFFFFFFFU,
        0x12345678U, 0x40000001U, 0x7FFFFFFFU, 0xA5C39E71U};

    for (std::uint32_t count = 0U; count < 64U; ++count)
    {
        for (std::size_t pattern = 0U; pattern < registers.size(); ++pattern)
        {
            for (const bool record : {false, true})
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const auto [source, destination, shift_register] = registers[pattern];
                const std::uint32_t source_value = source_values[(count + pattern) %
                    source_values.size()];
                const std::uint32_t effective_source = source == shift_register
                    ? count : source_value;
                const std::uint32_t initial_cr = 0xA5C39E71U;
                const std::uint32_t initial_xer = (pattern & 1U) != 0U
                    ? 0x87654321U : 0x07654321U;
                const std::uint32_t instruction = x_form(
                    source, destination, shift_register, 24U, record);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[source] = source_value;
                    core->state.gpr[shift_register] = count;
                    core->state.cr = initial_cr;
                    core->state.xer = initial_xer;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0xFEDCBA98U;
                    core->state.fpscr = 0x89ABCDEFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.gpr[destination] == (count >= 32U
                    ? 0U : effective_source << count));
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name ==
                    (record ? "slw." : "slw"));
            }
        }
    }
}

void recompiler_srw_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::array<std::uint8_t, 3U>, 4U> registers{{
        {{3U, 4U, 5U}}, {{3U, 3U, 5U}}, {{3U, 4U, 4U}}, {{3U, 3U, 3U}}}};
    constexpr std::array<std::uint32_t, 8U> source_values{
        0U, 1U, 0x80000000U, 0xFFFFFFFFU,
        0x12345678U, 0x40000001U, 0x7FFFFFFFU, 0xA5C39E71U};

    for (std::uint32_t count = 0U; count < 64U; ++count)
    {
        for (std::size_t pattern = 0U; pattern < registers.size(); ++pattern)
        {
            for (const bool record : {false, true})
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const auto [source, destination, shift_register] = registers[pattern];
                const std::uint32_t source_value = source_values[(count + pattern) %
                    source_values.size()];
                const std::uint32_t effective_source = source == shift_register
                    ? count : source_value;
                const std::uint32_t initial_cr = 0xA5C39E71U;
                const std::uint32_t initial_xer = (pattern & 1U) != 0U
                    ? 0x87654321U : 0x07654321U;
                const std::uint32_t instruction = x_form(
                    source, destination, shift_register, 536U, record);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[source] = source_value;
                    core->state.gpr[shift_register] = count;
                    core->state.cr = initial_cr;
                    core->state.xer = initial_xer;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0xFEDCBA98U;
                    core->state.fpscr = 0x89ABCDEFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.gpr[destination] == (count >= 32U
                    ? 0U : effective_source >> count));
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name ==
                    (record ? "srw." : "srw"));
            }
        }
    }
}

void recompiler_srawi_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t carry_mask = 0x20000000U;
    constexpr std::array<std::array<std::uint8_t, 2U>, 3U> registers{{
        {{3U, 4U}}, {{3U, 3U}}, {{0U, 0U}}}};
    constexpr std::array<std::uint32_t, 6U> source_values{
        0U, 1U, 0x80000000U, 0x80000001U, 0xFFFFFFFFU, 0xA5C39E71U};

    for (std::uint8_t shift = 0U; shift < 32U; ++shift)
    {
        for (std::size_t pattern = 0U; pattern < registers.size(); ++pattern)
        {
            for (const bool record : {false, true})
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const auto [source, destination] = registers[pattern];
                const std::uint32_t source_value = source_values[(shift + pattern) %
                    source_values.size()];
                const std::uint32_t initial_xer = (shift & 1U) != 0U
                    ? 0xE7654321U : 0x87654321U;
                const std::uint32_t instruction = x_form(
                    source, destination, shift, 824U, record);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[source] = source_value;
                    core->state.cr = 0xA5C39E71U;
                    core->state.xer = initial_xer;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0xFEDCBA98U;
                    core->state.fpscr = 0x89ABCDEFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                const std::uint32_t low_mask = shift == 0U
                    ? 0U : (std::uint32_t{1U} << shift) - 1U;
                const bool expected_carry = (source_value & 0x80000000U) != 0U &&
                    (source_value & low_mask) != 0U;
                const std::int32_t signed_source = std::bit_cast<std::int32_t>(source_value);
                const std::uint32_t expected_result = static_cast<std::uint32_t>(
                    signed_source >> shift);
                const std::uint32_t expected_xer = (initial_xer & ~carry_mask) |
                    (expected_carry ? carry_mask : 0U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.gpr[destination] == expected_result);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.xer == expected_xer);
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name ==
                    (record ? "srawi." : "srawi"));
            }
        }
    }
}

void recompiler_rlwimi_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    std::array<std::uint32_t, 32U> program{};
    program[0] = rlwimi_form(12U, 30U, 6U, 24U, 25U, false);
    for (std::uint32_t index = 1U; index < program.size(); ++index)
    {
        const std::uint8_t source = static_cast<std::uint8_t>((index * 7U + 5U) % 32U);
        std::uint8_t destination = index % 4U == 1U
            ? source
            : static_cast<std::uint8_t>((index * 11U + 3U) % 32U);
        if (destination == 12U)
        {
            destination = 13U;
        }
        const std::uint8_t begin = static_cast<std::uint8_t>((index * 13U) % 32U);
        const std::uint8_t end = static_cast<std::uint8_t>((index * 17U + 31U) % 32U);
        program[index] = rlwimi_form(destination, source,
            static_cast<std::uint8_t>(index), begin, end, (index & 1U) != 0U);
    }

    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < core->state.gpr.size(); ++reg)
        {
            core->state.gpr[reg] = 0x9E3779B9U * (reg + 1U) ^ (reg << 27U);
        }
        core->state.gpr[30] = 3U;
        core->state.gpr[12] = 0x1234563FU;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[12] == 0x123456FFU);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const bool record = index != 0U && (index & 1U) != 0U;
        assert(actual.instruction_history[index].opcode_name ==
            (record ? "rlwimi." : "rlwimi"));
    }
}

void recompiler_cntlzw_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::array<std::uint8_t, 2U>, 3U> registers{{
        {{3U, 4U}}, {{3U, 3U}}, {{0U, 0U}}}};
    constexpr std::array<std::uint32_t, 36U> inputs = [] {
        std::array<std::uint32_t, 36U> values{};
        for (std::size_t bit = 0U; bit < 32U; ++bit)
        {
            values[bit + 1U] = std::uint32_t{1U} << bit;
        }
        values[33] = 0xFFFFFFFFU;
        values[34] = 0xA5C39E71U;
        values[35] = 0x00010000U;
        return values;
    }();

    for (const std::uint32_t input : inputs)
    {
        for (std::size_t pattern = 0U; pattern < registers.size(); ++pattern)
        {
            for (const bool record : {false, true})
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const auto [source, destination] = registers[pattern];
                const std::uint32_t initial_cr = 0xA5C39E71U;
                const std::uint32_t initial_xer = pattern == 0U
                    ? 0x87654321U : 0x07654321U;
                const std::uint32_t instruction = x_form(
                    source, destination, 0U, 26U, record);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[source] = input;
                    core->state.cr = initial_cr;
                    core->state.xer = initial_xer;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0xFEDCBA98U;
                    core->state.fpscr = 0x89ABCDEFU;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.gpr[destination] == static_cast<std::uint32_t>(
                    std::countl_zero(input)));
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.xer == initial_xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.back().opcode_name ==
                    (record ? "cntlzw." : "cntlzw"));
            }
        }
    }
}

void recompiler_rlwinm_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 6U> inputs{
        0U, 1U, 0x80000000U, 0xFFFFFFFFU, 0xA5C39E71U, 0x7FFFFFFFU};
    constexpr std::array<std::array<std::uint8_t, 3U>, 5U> fields{{
        {{0U, 0U, 31U}}, {{1U, 4U, 11U}}, {{31U, 28U, 3U}},
        {{17U, 0U, 31U}}, {{5U, 12U, 19U}}}};

    for (const std::uint32_t input : inputs)
    {
        for (const auto [shift, begin, end] : fields)
        {
            for (const bool record : {false, true})
            {
                for (const bool summary_overflow : {false, true})
                {
                    for (const bool alias : {false, true})
                    {
                        EspressoCore interpreted(0x1000U);
                        EspressoCore compiled(0x1000U);
                        const std::uint8_t destination = alias ? 3U : 4U;
                        const std::uint8_t source = 3U;
                        const std::uint32_t instruction = rlwinm_form(
                            destination, source, shift, begin, end, record);
                        for (EspressoCore* core : {&interpreted, &compiled})
                        {
                            core->memory.write32_be(start, instruction);
                            core->state.cia = start;
                            core->state.gpr[source] = input;
                            if (!alias)
                            {
                                core->state.gpr[destination] = 0xDEADBEEFU;
                            }
                            core->state.cr = 0x5A5AA5A5U;
                            core->state.xer = summary_overflow ? 0x80000000U : 0U;
                        }

                        const RunResult oracle = interpreted.run(1U);
                        Recompiler recompiler;
                        const RunResult actual = recompiler.run(compiled, 1U);
                        assert(actual.steps == oracle.steps && actual.steps == 1U);
                        assert(actual.cia == oracle.cia);
                        assert(compiled.state.gpr == interpreted.state.gpr);
                        assert(compiled.state.cr == interpreted.state.cr);
                        assert(compiled.state.xer == interpreted.state.xer);
                        assert(recompiler.statistics().native_instructions == 1U);
                        assert(recompiler.statistics().interpreter_fallbacks == 0U);
                        if (record)
                        {
                            const std::string history = format_instruction_history(actual);
                            assert(history.find("rlwinm.") != std::string::npos);
                        }
                    }
                }
            }
        }
    }
}

void recompiler_lfd_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array<std::uint64_t, 4U> patterns{
        0U, 0x8000000000000000ULL, 0x3FF0000000000000ULL,
        0xFFF8123456789ABCULL};
    for (std::size_t test_index = 0; test_index < patterns.size(); ++test_index)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint8_t destination = 5U;
        const std::uint8_t base = test_index == 1U ? 0U : 4U;
        const std::int16_t displacement = test_index == 1U ? 0x80 : 0x18;
        const std::uint32_t address = base == 0U
            ? static_cast<std::uint32_t>(displacement)
            : 0x218U;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfd_form(destination, base, displacement));
            core->memory.write64_be(address, patterns[test_index]);
            core->state.cia = start;
            core->state.gpr[0] = 0x400U;
            core->state.gpr[4] = 0x200U;
            core->state.fpr[destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.fpscr = 0x12345678U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.fpr[destination] == patterns[test_index]);
        assert(compiled.state.fpr_ps1[destination] == ps1_sentinel);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const std::string history = format_instruction_history(actual);
        assert(history.find("lfd") != std::string::npos);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfd_form(5U, 4U, 8));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.fpr[5] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[5] = ps1_sentinel;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 8 byte(s) at guest address 0x00001008") !=
           std::string::npos);
    assert(faulting.state.fpr[5] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[5] == ps1_sentinel);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
    assert(fault.instruction_history.back().opcode_name == "lfd");
}

void recompiler_lfs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t data = 0x300U;
    constexpr std::uint8_t destination = 6U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array<std::uint32_t, 9U> single_patterns{
        0x00000000U, 0x80000000U, 0x3FC00000U, 0xC2F68000U,
        0x00000001U, 0x7F800000U, 0xFF800000U, 0x7FC12345U,
        0x7F812345U};

    for (const std::uint32_t pattern : single_patterns)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfs_form(destination, 4U, 8));
            core->memory.write32_be(data, pattern);
            core->state.cia = start;
            core->state.gpr[4] = data - 8U;
            core->state.gpr[0] = 0xDEADBEEFU;
            core->state.fpr[destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.fpscr = 0x12345678U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.fpr[destination] == compiled.state.fpr_ps1[destination]);
        assert(compiled.state.gpr[0] == 0xDEADBEEFU);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.has_effective_address);
        assert(history.effective_address == data);
        assert(history.has_fp_destination);
        assert(history.fp_destination_register == destination);
        assert(history.fp_destination_value == compiled.state.fpr[destination]);
        assert(format_instruction_history(actual).find("lfs") != std::string::npos);
    }

    // D-form rA=0 means a literal zero base, independently of GPR0.
    EspressoCore zero_base_interpreted(0x1000U);
    EspressoCore zero_base_compiled(0x1000U);
    for (EspressoCore* core : {&zero_base_interpreted, &zero_base_compiled})
    {
        core->memory.write32_be(start, lfs_form(destination, 0U, 0x180));
        core->memory.write32_be(0x180U, 0x3F800000U);
        core->state.cia = start;
        core->state.gpr[0] = 0x400U;
    }
    static_cast<void>(zero_base_interpreted.run(1U));
    Recompiler zero_base_recompiler;
    const RunResult zero_base_actual = zero_base_recompiler.run(zero_base_compiled, 1U);
    assert(zero_base_actual.steps == 1U);
    assert(zero_base_compiled.state.fpr == zero_base_interpreted.state.fpr);
    assert(zero_base_compiled.state.fpr_ps1 == zero_base_interpreted.state.fpr_ps1);
    assert(zero_base_compiled.state.fpr[destination] == 0x3FF0000000000000ULL);
    assert(zero_base_compiled.state.fpr_ps1[destination] == 0x3FF0000000000000ULL);

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfs_form(destination, 4U, 8));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.fpr[destination] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[destination] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s) at guest address 0x00001008") !=
           std::string::npos);
    assert(faulting.state.fpr[destination] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[destination] == ps1_sentinel);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
    assert(fault.instruction_history.back().opcode_name == "lfs");
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
    constexpr std::size_t instruction_count = instruction_history_capacity + 8U;
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

void recompiler_isync_boundary_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 3U> program{
        0x38600007U, // addi r3,r0,7
        0x4C00012CU, // isync
        0x38830005U, // addi r4,r3,5
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(
                start + static_cast<std::uint32_t>(index * 4U), program[index]);
        }
        core->state.cia = start;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.gpr[3] == 7U && compiled.state.gpr[4] == 12U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(recompiler.statistics().compiled_blocks == 2U);
    assert(actual.instruction_history.size() == program.size());
    assert(actual.instruction_history[0].opcode_name == "addi");
    assert(actual.instruction_history[1].opcode_name == "isync");
    assert(actual.instruction_history[2].opcode_name == "addi");
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

void recompiler_fast_lookup_cache_test()
{
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, 0x48000021U); // b 0x120
    core.memory.write32_be(0x120U, 0x4BFFFFE0U); // b 0x100
    core.state.cia = 0x100U;

    Recompiler recompiler;
    const RunResult result = recompiler.run(core, 4U);

    assert(result.steps == 4U);
    assert(result.reason == StopReason::instruction_limit);
    assert(result.cia == 0x100U);
    assert(recompiler.statistics().compiled_blocks == 2U);
    assert(recompiler.statistics().fast_lookup_hits == 2U);
    assert(recompiler.statistics().cache_hits == 2U);
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
        machine->memory.write32_be(0x104U, 0x84650000U); // lwzu r3,0(r5)
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
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(recompiler.statistics().native_instructions == 3U);
}

void recompiler_lbzu_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::int16_t displacement;
        std::uint8_t memory_value;
        std::uint8_t destination;
        std::uint8_t base_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4, 0x00U, 3U, 6U},
        Case{0x300U, 8, 0x7FU, 4U, 7U},
        Case{0x300U, 12, 0x80U, 5U, 8U},
        Case{0x300U, 16, 0xFFU, 9U, 10U},
        Case{0xFFFFFFFCU, 8, 0xA5U, 11U, 12U}, // 32-bit EA wrap to 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lbzu_form(
            test.destination, test.base_register, test.displacement);
        const std::uint32_t address = test.base +
            static_cast<std::uint32_t>(test.displacement);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write8(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.destination] = 0x12345678U;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr[test.destination] == test.memory_value);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lbzu");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    EspressoCore faulting(0x1000U);
    constexpr std::uint8_t base_register = 6U;
    constexpr std::uint8_t destination = 3U;
    faulting.memory.write32_be(start, lbzu_form(destination, base_register, 4));
    faulting.state.cia = start;
    faulting.state.gpr[base_register] = 0x2000U;
    faulting.state.gpr[destination] = 0xDEADBEEFU;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 1 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[destination] == 0xDEADBEEFU);
    assert(faulting.state.gpr[base_register] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lwzu_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::int16_t displacement;
        std::uint32_t memory_value;
        std::uint8_t destination;
        std::uint8_t base_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4, 0x00000000U, 3U, 6U},
        Case{0x300U, 8, 0x7FFFFFFFU, 4U, 7U},
        Case{0x300U, 12, 0x80000000U, 5U, 8U},
        Case{0x300U, 16, 0xDEADBEEFU, 9U, 10U},
        Case{0xFFFFFFFCU, 8, 0x12345678U, 11U, 12U}, // wrap to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lwzu_form(
            test.destination, test.base_register, test.displacement);
        const std::uint32_t address = test.base +
            static_cast<std::uint32_t>(test.displacement);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write32_be(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.destination] = 0x12345678U;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.destination] == test.memory_value);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lwzu");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    EspressoCore faulting(0x1000U);
    constexpr std::uint8_t base_register = 6U;
    constexpr std::uint8_t destination = 3U;
    faulting.memory.write32_be(start, lwzu_form(destination, base_register, 4));
    faulting.state.cia = start;
    faulting.state.gpr[base_register] = 0x2000U;
    faulting.state.gpr[destination] = 0xDEADBEEFU;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[destination] == 0xDEADBEEFU);
    assert(faulting.state.gpr[base_register] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
}

void recompiler_dcbst_differential_tests()
{
    struct Case
    {
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t address;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0U, 5U, 0xDEADBEEFU, 0x40U, 0x40U},
        // rB==0 is a real GPR source; only rA==0 selects the literal base 0.
        Case{4U, 0U, 0x100U, 0x20U, 0x120U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            0U, test.base_register, test.index_register, 54U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write8(test.address, 0x5AU);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[0] = test.index_register == 0U
                ? test.index : 0xDEADBEEFU;
            if (test.base_register != 0U)
                core->state.gpr[test.base_register] = test.base;
            if (test.index_register != 0U)
                core->state.gpr[test.index_register] = test.index;
            core->state.cr = 0x12345678U;
            core->state.xer = 0xA00000A5U;
            core->state.lr = 0x10203040U;
            core->state.ctr = 0x50607080U;
            core->state.fpscr = 0xCAFEBABEU;
        }

        const auto gprs_before = interpreted.state.gpr;
        const auto fprs_before = interpreted.state.fpr;
        const auto ps1_before = interpreted.state.fpr_ps1;
        const std::uint32_t cr_before = interpreted.state.cr;
        const std::uint32_t xer_before = interpreted.state.xer;
        const std::uint32_t lr_before = interpreted.state.lr;
        const std::uint32_t ctr_before = interpreted.state.ctr;
        const std::uint32_t fpscr_before = interpreted.state.fpscr;

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 1U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr == gprs_before);
        assert(compiled.state.fpr == fprs_before &&
            compiled.state.fpr_ps1 == ps1_before);
        assert(compiled.state.cr == cr_before && compiled.state.xer == xer_before);
        assert(compiled.state.lr == lr_before && compiled.state.ctr == ctr_before);
        assert(compiled.state.fpscr == fpscr_before);
        assert(compiled.memory.read8(test.address) == 0x5AU);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "dcbst" && history.completed);
        assert(history.has_effective_address &&
            history.effective_address == test.address);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register == 0U)
        {
            assert(history.source_registers[0] == test.index_register);
            assert(history.source_values[0] == test.index);
        }
        else
        {
            assert(history.source_registers[0] == test.base_register);
            assert(history.source_values[0] == test.base);
            assert(history.source_registers[1] == test.index_register);
            assert(history.source_values[1] == test.index);
        }
        assert(!history.has_destination);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, x_form(0U, 6U, 5U, 54U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[5] = 0x20U;
    const auto gprs_before = faulting.state.gpr;
    Recompiler recompiler;
    const RunResult fault = recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 1 byte(s)") != std::string::npos);
    assert(faulting.state.gpr == gprs_before);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(fault.instruction_history.back().opcode_name == "dcbst");
    assert(fault.instruction_history.back().has_effective_address &&
        fault.instruction_history.back().effective_address == 0x2020U);
    assert(!fault.instruction_history.back().completed);
    assert(!fault.instruction_history.back().has_destination);
}

void recompiler_dcbf_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t address = 0x120U;
    constexpr std::uint32_t dcbf_r4_r0 = 0x7C0400ACU;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(start, dcbf_r4_r0);
        core->memory.write8(address, 0xA7U);
        core->state.cia = start;
        core->state.gpr.fill(0xC001D00DU);
        core->state.gpr[0] = 0x20U; // rB=0 is an ordinary GPR source.
        core->state.gpr[4] = 0x100U;
        core->state.cr = 0x91827364U;
        core->state.xer = 0xA0000042U;
        core->state.lr = 0x10203040U;
        core->state.ctr = 0x50607080U;
        core->state.fpscr = 0xCAFEBABEU;
    }
    const auto gprs_before = interpreted.state.gpr;
    const auto fprs_before = interpreted.state.fpr;
    const auto ps1_before = interpreted.state.fpr_ps1;
    const std::uint32_t cr_before = interpreted.state.cr;
    const std::uint32_t xer_before = interpreted.state.xer;
    const std::uint32_t lr_before = interpreted.state.lr;
    const std::uint32_t ctr_before = interpreted.state.ctr;
    const std::uint32_t fpscr_before = interpreted.state.fpscr;

    const RunResult oracle = interpreted.run(1U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 1U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 1U);
    assert(compiled.state.gpr == gprs_before);
    assert(compiled.state.fpr == fprs_before && compiled.state.fpr_ps1 == ps1_before);
    assert(compiled.state.cr == cr_before && compiled.state.xer == xer_before);
    assert(compiled.state.lr == lr_before && compiled.state.ctr == ctr_before);
    assert(compiled.state.fpscr == fpscr_before);
    assert(compiled.memory.read8(address) == 0xA7U);
    assert(recompiler.statistics().native_instructions == 1U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "dcbf" && history.completed);
    assert(history.has_effective_address && history.effective_address == address);
    assert(history.source_count == 2U && history.source_registers[0] == 4U);
    assert(history.source_values[0] == 0x100U);
    assert(history.source_registers[1] == 0U && history.source_values[1] == 0x20U);
    assert(!history.has_destination);

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, x_form(0U, 6U, 5U, 86U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[5] = 0x20U;
    const auto fault_gprs_before = faulting.state.gpr;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 1 byte(s)") != std::string::npos);
    assert(faulting.state.gpr == fault_gprs_before && faulting.state.cia == start);
    assert(!fault.instruction_history.empty());
    assert(fault.instruction_history.back().opcode_name == "dcbf");
    assert(fault.instruction_history.back().has_effective_address &&
        fault.instruction_history.back().effective_address == 0x2020U);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_memory_tests()
{
    EspressoCore watched_store(0x1000U);
    watched_store.memory.write32_be(0x100U, 0x90A40000U); // stw r5, 0(r4)
    watched_store.state.cia = 0x100U;
    watched_store.state.gpr[4] = 0x200U;
    watched_store.state.gpr[5] = 0xAABBCCDDU;
    watched_store.memory.set_write_watchpoint(0x200U, 4U);
    Recompiler watched_store_recompiler;
    const RunResult watched_store_result = watched_store_recompiler.run(
        watched_store, 1U);
    assert(watched_store_result.steps == 1U);
    const auto watched_store_events =
        watched_store.memory.write_watch_events_snapshot();
    assert(watched_store_events.size() == 1U);
    assert(watched_store_events[0].writer_cia == 0x100U);
    assert(watched_store_events[0].access_address == 0x200U);
    assert(watched_store_events[0].watched_value == 0xAABBCCDDU);

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
    assert(std::string(opcode_mnemonic(Opcode::floating_multiply_add_double)) == "fmadd");
    recompiler_differential_arithmetic_test();
    recompiler_cmpwi_differential_test();
    recompiler_cmplwi_differential_test();
    recompiler_cmpw_differential_test();
    recompiler_fcmpu_differential_test();
    recompiler_subfic_differential_test();
    recompiler_subfc_differential_test();
    recompiler_subfe_differential_test();
    recompiler_addic_differential_test();
    recompiler_destination_aliases_second_source_test();
    recompiler_logical_differential_test();
    recompiler_logical_record_differential_test();
    recompiler_condition_bit_differential_tests();
    recompiler_mcrf_differential_tests();
    recompiler_mfcr_differential_tests();
    recompiler_mtcrf_differential_tests();
    recompiler_mcrxr_differential_tests();
    recompiler_mfxer_differential_tests();
    recompiler_mtxer_differential_tests();
    recompiler_mfgqr_differential_tests();
    recompiler_mtgqr_differential_tests();
    recompiler_fmr_differential_tests();
    recompiler_fneg_differential_tests();
    recompiler_fctiwz_differential_tests();
    recompiler_fabs_differential_tests();
    recompiler_ps_merge10_differential_tests();
    recompiler_ps_merge00_differential_tests();
    recompiler_stfs_differential_tests();
    recompiler_stfsu_differential_tests();
    recompiler_stfd_differential_tests();
    recompiler_stwu_differential_tests();
    recompiler_slw_differential_tests();
    recompiler_srw_differential_tests();
    recompiler_srawi_differential_tests();
    recompiler_rlwimi_differential_tests();
    recompiler_cntlzw_differential_tests();
    recompiler_rlwinm_differential_tests();
    recompiler_lfd_differential_tests();
    recompiler_lfs_differential_tests();
    recompiler_step_budget_test();
    recompiler_recent_history_window_test();
    recompiler_isync_boundary_differential_test();
    recompiler_lbzu_differential_tests();
    recompiler_lwzu_differential_tests();
    recompiler_dcbst_differential_tests();
    recompiler_dcbf_differential_tests();
    recompiler_branch_test();
    recompiler_fast_lookup_cache_test();
    recompiler_conditional_branch_tests();
    recompiler_special_register_move_tests();
    recompiler_interpreter_fallback_test();
    recompiler_memory_tests();
    recompiler_memory_fault_test();
    recompiler_self_modifying_code_test();
    recompiler_stop_reason_test();
}
