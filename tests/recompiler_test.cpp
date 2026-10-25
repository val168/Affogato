#include "cpu/espresso/recompiler.hpp"

#include <cassert>
#include <array>
#include <bit>
#include <cstdint>
#include <limits>
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

[[nodiscard]] constexpr std::uint32_t andis_form(
    std::uint8_t source, std::uint8_t destination, std::uint16_t immediate)
{
    return (29U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
           (static_cast<std::uint32_t>(destination) << 16U) | immediate;
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

[[nodiscard]] constexpr std::uint32_t rlwnm_form(
    std::uint8_t destination, std::uint8_t source, std::uint8_t shift_register,
    std::uint8_t begin, std::uint8_t end, bool record = false)
{
    return (23U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
           (static_cast<std::uint32_t>(destination) << 16U) |
           (static_cast<std::uint32_t>(shift_register) << 11U) |
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

[[nodiscard]] constexpr std::uint32_t ps_mr_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (72U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_abs_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (264U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_nabs_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (136U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_res_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (24U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_rsqrte_form(
    std::uint8_t destination, std::uint8_t source)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (26U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_sel_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_c, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (23U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_muls1_form(
    std::uint8_t destination, std::uint8_t source_a, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (13U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_madds0_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (14U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_madds1_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (15U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_nmadd_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (30U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_nmsub_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (31U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_sum1_form(
    std::uint8_t destination, std::uint8_t source_a,
    std::uint8_t source_b, std::uint8_t source_c)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) |
        (static_cast<std::uint32_t>(source_c) << 6U) | (11U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_add_form(
    std::uint8_t destination, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(source_a) << 16U) |
           (static_cast<std::uint32_t>(source_b) << 11U) | (21U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_div_form(
    std::uint8_t destination, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) | (18U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_cmpu0_form(
    std::uint8_t cr_field, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U);
}

[[nodiscard]] constexpr std::uint32_t ps_cmpu1_form(
    std::uint8_t cr_field, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) | (64U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_cmpo0_form(
    std::uint8_t cr_field, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) | (32U << 1U);
}

[[nodiscard]] constexpr std::uint32_t ps_cmpo1_form(
    std::uint8_t cr_field, std::uint8_t source_a, std::uint8_t source_b)
{
    return (4U << 26U) | (static_cast<std::uint32_t>(cr_field) << 23U) |
        (static_cast<std::uint32_t>(source_a) << 16U) |
        (static_cast<std::uint32_t>(source_b) << 11U) | (96U << 1U);
}

[[nodiscard]] constexpr std::uint32_t fctiw_form(
    std::uint8_t destination, std::uint8_t source, std::uint32_t xo)
{
    return (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(source) << 11U) | (xo << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfd_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (50U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
           (static_cast<std::uint32_t>(base) << 16U) |
           static_cast<std::uint16_t>(displacement);
}

[[nodiscard]] constexpr std::uint32_t lfdu_form(
    std::uint8_t destination, std::uint8_t base, std::int16_t displacement)
{
    return (51U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
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

[[nodiscard]] constexpr std::uint32_t lwzux_form(
    std::uint8_t destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (55U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lbzux_form(
    std::uint8_t destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (119U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lhzux_form(
    std::uint8_t destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (311U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lhaux_form(
    std::uint8_t destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (375U << 1U);
}

[[nodiscard]] constexpr std::uint32_t stwux_form(
    std::uint8_t source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (183U << 1U);
}

[[nodiscard]] constexpr std::uint32_t stbux_form(
    std::uint8_t source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (247U << 1U);
}

[[nodiscard]] constexpr std::uint32_t sthux_form(
    std::uint8_t source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (439U << 1U);
}

[[nodiscard]] constexpr std::uint32_t stfsux_form(
    std::uint8_t fp_source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (695U << 1U);
}

[[nodiscard]] constexpr std::uint32_t stfdx_form(
    std::uint8_t fp_source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (727U << 1U);
}

[[nodiscard]] constexpr std::uint32_t stfdux_form(
    std::uint8_t fp_source, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_source) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (759U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfsx_form(
    std::uint8_t fp_destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (535U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfsux_form(
    std::uint8_t fp_destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (567U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfdx_form(
    std::uint8_t fp_destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (599U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lfdux_form(
    std::uint8_t fp_destination, std::uint8_t base, std::uint8_t index)
{
    return (31U << 26U) | (static_cast<std::uint32_t>(fp_destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        (static_cast<std::uint32_t>(index) << 11U) | (631U << 1U);
}

[[nodiscard]] constexpr std::uint32_t lmw_form(
    std::uint8_t first_destination, std::uint8_t base, std::int16_t displacement)
{
    return (46U << 26U) |
        (static_cast<std::uint32_t>(first_destination) << 21U) |
        (static_cast<std::uint32_t>(base) << 16U) |
        static_cast<std::uint16_t>(displacement);
}

[[nodiscard]] constexpr std::uint32_t stmw_form(
    std::uint8_t first_source, std::uint8_t base, std::int16_t displacement)
{
    return (47U << 26U) |
        (static_cast<std::uint32_t>(first_source) << 21U) |
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

void recompiler_execution_context_fp_history_reset_test()
{
    constexpr std::uint32_t fmul_f0_f1_f2 = (63U << 26U) | (1U << 16U) |
        (2U << 6U) | (25U << 1U);
    EspressoCore core(0x1000U);
    core.memory.write32_be(0x100U, fmul_f0_f1_f2);
    core.state.cia = 0x100U;
    core.state.fpr[0] = std::bit_cast<std::uint64_t>(-7.0);
    core.state.fpr[1] = std::bit_cast<std::uint64_t>(2.0);
    core.state.fpr[2] = std::bit_cast<std::uint64_t>(3.0);
    Recompiler recompiler;

    const RunResult normal = recompiler.run(core, 1U);
    assert(normal.steps == 1U);
    assert(core.state.fpr[0] == std::bit_cast<std::uint64_t>(6.0));
    assert(!normal.instruction_history.empty());
    assert(normal.instruction_history.back().has_fp_arithmetic_result);

    // Revisit the same cached native block with an invalid enabled operation.
    // Its reused trace record must not retain the prior result-present bit.
    core.state.cia = 0x100U;
    core.state.fpr[0] = 0x0123456789ABCDEFULL;
    core.state.fpr[1] = 0x7FF0000000000000ULL;
    core.state.fpr[2] = 0U;
    core.state.fpscr = fpscr::ve_mask;
    const RunResult suppressed = recompiler.run(core, 1U);
    assert(suppressed.steps == 1U);
    assert(core.state.fpr[0] == 0x0123456789ABCDEFULL);
    assert(!suppressed.instruction_history.empty());
    assert(!suppressed.instruction_history.back().has_fp_arithmetic_result);
    assert(recompiler.statistics().native_instructions == 2U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_mulli_differential_test()
{
    const auto encode_mulli = [](std::uint8_t destination, std::uint8_t base,
                                 std::int16_t immediate) {
        return (7U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            static_cast<std::uint16_t>(immediate);
    };
    constexpr std::uint32_t start = 0x100U;
    const std::array<std::uint32_t, 4U> program{
        encode_mulli(4U, 4U, -3), // destination aliases RA; -2 * -3 = 6.
        encode_mulli(6U, 5U, 3),  // low word of a signed product wraps modulo 2^32.
        encode_mulli(7U, 0U, -1), // RA=0 reads GPR0; no zero-base special case.
        encode_mulli(9U, 8U, 0),
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                                    program[index]);
        }
        core->state.cia = start;
        core->state.gpr[0] = 0x80000000U;
        core->state.gpr[4] = 0xFFFFFFFEU;
        core->state.gpr[5] = 0x7FFFFFFFU;
        core->state.gpr[8] = 0x12345678U;
        core->state.cr = 0xA1B2C3D4U;
        core->state.xer = 0xD2345678U;
    }
    const std::uint32_t cr_before = interpreted.state.cr;
    const std::uint32_t xer_before = interpreted.state.xer;
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[4] == 6U);
    assert(compiled.state.gpr[6] == 0x7FFFFFFDU);
    assert(compiled.state.gpr[7] == 0x80000000U);
    assert(compiled.state.gpr[9] == 0U);
    assert(compiled.state.cr == cr_before && compiled.state.xer == xer_before);
    assert(compiled.memory.write_generation() == interpreted.memory.write_generation());
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "mulli" && history.completed);
    assert(history.source_count == 1U && history.source_registers[0] == 8U);
    assert(history.source_values[0] == 0x12345678U);
    assert(history.has_immediate && history.immediate == 0);
    assert(history.has_destination && history.destination_register == 9U);
    assert(history.destination_value == 0U);
}

void recompiler_neg_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    const std::array<std::uint32_t, 3U> program{
        x_form(3U, 3U, 0U, 104U),       // destination aliases source; INT_MIN wraps.
        x_form(5U, 4U, 0U, 104U, true), // record -1, with SO already sticky.
        x_form(7U, 0U, 0U, 104U, true), // GPR0 is the real source; result is zero.
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                                    program[index]);
        }
        core->state.cia = start;
        core->state.gpr[0] = 0U;
        core->state.gpr[3] = 0x80000000U;
        core->state.gpr[4] = 1U;
        core->state.cr = 0xA1B2C3D4U;
        core->state.xer = 0xD2345678U;
    }
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[3] == 0x80000000U);
    assert(compiled.state.gpr[5] == 0xFFFFFFFFU);
    assert(compiled.state.gpr[7] == 0U);
    assert(compiled.state.cr == interpreted.state.cr);
    assert((compiled.state.cr >> 28U) == 3U); // EQ plus the existing XER.SO.
    assert(compiled.state.xer == interpreted.state.xer);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.back().opcode_name == "neg.");
}

void recompiler_mullw_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    const std::array<std::uint32_t, 3U> program{
        x_form(3U, 3U, 4U, 235U),       // r3 = INT_MIN * 3 modulo 2^32.
        x_form(5U, 6U, 7U, 235U, true),  // Negative wrapped result updates CR0.
        x_form(8U, 8U, 8U, 235U),       // All operands alias; (-1) * (-1) = 1.
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                                    program[index]);
        }
        core->state.cia = start;
        core->state.gpr[3] = 0x80000000U;
        core->state.gpr[4] = 3U;
        core->state.gpr[6] = 0x7FFFFFFFU;
        core->state.gpr[7] = 2U;
        core->state.gpr[8] = 0xFFFFFFFFU;
        core->state.cr = 0xA1B2C3D4U;
        core->state.xer = 0xD2345678U;
    }
    const std::uint32_t xer_before = interpreted.state.xer;
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[3] == 0x80000000U);
    assert(compiled.state.gpr[5] == 0xFFFFFFFEU);
    assert(compiled.state.gpr[8] == 1U);
    assert(compiled.state.cr == interpreted.state.cr);
    assert((compiled.state.cr >> 28U) == 9U); // LT plus the existing XER.SO.
    assert(compiled.state.xer == xer_before);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "mullw" && history.completed);
    assert(history.source_count == 2U);
    assert(history.source_registers[0] == 8U && history.source_values[0] == 0xFFFFFFFFU);
    assert(history.source_registers[1] == 8U && history.source_values[1] == 0xFFFFFFFFU);
    assert(history.has_destination && history.destination_register == 8U);
    assert(history.destination_value == 1U);
}

void recompiler_mullwo_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{7U, 9U, 4U, 5U, 3U, false, 0x600000A5U, 0x12345678U},
        Case{0x7FFFFFFFU, 2U, 4U, 5U, 4U, true, 0x200000A5U, 0x87654321U},
        Case{0x80000000U, 2U, 4U, 5U, 5U, false, 0xA00000A5U, 0xA5C36E91U},
        Case{0x40000000U, 4U, 4U, 5U, 4U, true, 0x600000A5U, 0xA5C36E91U},
        Case{0x7FFFFFFFU, 2U, 8U, 9U, 8U, true, 0x25A5A5A5U, 0x12345678U},
    };
    static_assert(x_form(3U, 4U, 5U, 747U) == 0x7C642DD6U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 747U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::int64_t product =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.a)) *
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.b));
        const std::uint32_t result = static_cast<std::uint32_t>(product);
        const bool overflow = product < std::numeric_limits<std::int32_t>::min() ||
            product > std::numeric_limits<std::int32_t>::max();
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == summary_overflow);
        assert((compiled.state.xer & 0x3FFFFFFFU) ==
            (test.initial_xer & 0x3FFFFFFFU));
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "mullwo." : "mullwo"));
        assert(history.source_count == (test.ra == test.rb ? 1U : 2U));
        assert(history.source_registers[0] == test.ra && history.source_values[0] == test.a);
        if (test.ra != test.rb)
        {
            assert(history.source_registers[1] == test.rb);
            assert(history.source_values[1] == test.b);
        }
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_mulhw_differential_tests()
{
    struct Registers
    {
        std::uint8_t destination;
        std::uint8_t base;
        std::uint8_t source;
    };
    constexpr std::array register_cases{
        Registers{3U, 4U, 5U},
        Registers{4U, 4U, 5U},
        Registers{5U, 4U, 5U},
        Registers{4U, 4U, 4U},
    };
    constexpr std::array<std::array<std::uint32_t, 2U>, 8U> operand_cases{{
        {0U, 0xFFFFFFFFU},
        {1U, 1U},
        {0xFFFFFFFFU, 2U},
        {0xFFFFFFFFU, 0xFFFFFFFFU},
        {0x80000000U, 1U},
        {0x80000000U, 0xFFFFFFFFU},
        {0x7FFFFFFFU, 0x7FFFFFFFU},
        {0x80000001U, 0x7FFFFFFEU},
    }};

    for (const Registers registers : register_cases)
    {
        for (const auto& [lhs, rhs] : operand_cases)
        {
            for (const bool record : {false, true})
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint32_t instruction = x_form(
                    registers.destination, registers.base, registers.source,
                    75U, record);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(0x100U, instruction);
                    core->state.cia = 0x100U;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.gpr[registers.base] = lhs;
                    core->state.gpr[registers.source] =
                        registers.source == registers.base ? lhs : rhs;
                    core->state.cr = 0xA5C39E71U;
                    core->state.xer = record ? 0x800000A5U : 0x200000A5U;
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
                const std::int64_t a = std::bit_cast<std::int32_t>(lhs);
                const std::int64_t b = std::bit_cast<std::int32_t>(
                    registers.source == registers.base ? lhs : rhs);
                const std::int64_t product = a * b;
                const std::uint32_t expected = static_cast<std::uint32_t>(
                    static_cast<std::uint64_t>(product) >> 32U);
                assert(compiled.state.gpr[registers.destination] == expected);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                const InstructionHistoryEntry& history =
                    actual.instruction_history.back();
                assert(history.opcode_name == (record ? "mulhw." : "mulhw"));
                assert(history.has_destination &&
                       history.destination_register == registers.destination);
                assert(history.destination_value == expected);
            }
        }
    }
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

void recompiler_fcmpo_differential_test()
{
    constexpr auto encode_fcmpo = [](std::uint8_t cr_field, std::uint8_t fr_a,
                                 std::uint8_t fr_b) {
        return 0xFC000040U |
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
        Case{0x7FF8000000000001ULL, 0x3FF0000000000000ULL, 7U, true},
        Case{0x7FF0000000000001ULL, 0x3FF0000000000000ULL, 0U, true},
    };
    static_assert(encode_fcmpo(0U, 1U, 31U) == 0xFC01F840U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, encode_fcmpo(test.cr_field, 1U, 31U));
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
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name == "fcmpo");
        assert(history.has_fp_compare && !history.fp_compare_uses_ps1);
        assert(history.fp_compare_cr_field == test.cr_field);
        assert(history.fp_compare_a_raw == test.a);
        assert(history.fp_compare_b_raw == test.b);
        assert(history.fp_compare_result == oracle.instruction_history.back().fp_compare_result);
    }
}

void recompiler_mffs_differential_test()
{
    constexpr std::array<std::uint32_t, 4U> fpscr_values{
        0U, 0xA5C3BEEFU, fpscr::vxsnan_mask | fpscr::vx_mask | fpscr::fx_mask,
        0xFFFFFFFFU,
    };
    constexpr std::array<std::uint8_t, 3U> destinations{0U, 5U, 31U};
    static_assert(((63U << 26U) | (5U << 21U) | (583U << 1U)) == 0xFCA0048EU);

    for (const std::uint32_t fpscr_value : fpscr_values)
    {
        for (const std::uint8_t destination : destinations)
        {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction = (63U << 26U) |
                (static_cast<std::uint32_t>(destination) << 21U) | (583U << 1U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(0x100U, instruction);
                core->state.cia = 0x100U;
                core->state.gpr.fill(0xA5A5A5A5U);
                core->state.fpr.fill(0x0123456789ABCDEFULL);
                core->state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
                core->state.fpr_ps1[destination] = 0xDEADBEEF01234567ULL;
                core->state.cr = 0x12345678U;
                core->state.xer = 0xA00000A5U;
                core->state.fpscr = fpscr_value;
                core->state.lr = 0x11223344U;
                core->state.ctr = 0x55667788U;
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
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpr[destination] ==
                (0xFFFFFFFF00000000ULL | fpscr_value));
            assert(compiled.state.fpr_ps1[destination] == 0xDEADBEEF01234567ULL);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.size() == 1U);
            const InstructionHistoryEntry& history = actual.instruction_history.back();
            assert(history.completed && history.opcode_name == "mffs");
            assert(history.has_fp_status_move);
            assert(history.fp_status_move_destination == destination);
            assert(history.fp_status_move_source == fpscr_value);
            assert(history.fp_status_move_result ==
                (0xFFFFFFFF00000000ULL | fpscr_value));
            assert(history.fp_status_move_destination_ps1 == 0xDEADBEEF01234567ULL);
        }
    }
}

void recompiler_mtfsb_differential_tests()
{
    constexpr std::array<std::uint32_t, 3U> fpscr_values{
        0U, 0xA5C3BEEFU,
        fpscr::fx_mask | fpscr::fex_mask | fpscr::vx_mask |
            fpscr::ve_mask | fpscr::vxsnan_mask | fpscr::ox_mask | fpscr::oe_mask,
    };
    static_assert(((63U << 26U) | (7U << 21U) | (70U << 1U)) == 0xFCE0008CU);

    for (const bool set_bit : {false, true})
    {
        for (std::uint8_t bit = 0U; bit < 32U; ++bit)
        {
            for (const std::uint32_t fpscr_value : fpscr_values)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint32_t instruction = (63U << 26U) |
                    (static_cast<std::uint32_t>(bit) << 21U) |
                    ((set_bit ? 38U : 70U) << 1U);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(0x100U, instruction);
                    core->state.cia = 0x100U;
                    core->state.gpr.fill(0xA5A5A5A5U);
                    core->state.fpr.fill(0x0123456789ABCDEFULL);
                    core->state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
                    core->state.cr = 0x12345678U;
                    core->state.xer = 0xA00000A5U;
                    core->state.fpscr = fpscr_value;
                    core->state.lr = 0x11223344U;
                    core->state.ctr = 0x55667788U;
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
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.size() == 1U);
                const InstructionHistoryEntry& history = actual.instruction_history.back();
                assert(history.completed && history.opcode_name ==
                    (set_bit ? "mtfsb1" : "mtfsb0"));
                assert(history.has_fpscr_bit_operation);
                assert(history.fpscr_bit_set == set_bit);
                assert(history.fpscr_bit_number == bit);
                assert(history.fpscr_before_bit_operation == fpscr_value);
                assert(history.fpscr_after_bit_operation == compiled.state.fpscr);
            }
        }
    }
}

void recompiler_mtfsfi_differential_tests()
{
    constexpr std::array<std::uint32_t, 3U> fpscr_values{
        0U, 0xA5C3BEEFU,
        fpscr::fx_mask | fpscr::fex_mask | fpscr::vx_mask |
            fpscr::ve_mask | fpscr::vxsnan_mask | fpscr::ox_mask | fpscr::oe_mask,
    };
    static_assert(((63U << 26U) | (6U << 23U) | (4U << 12U) | (134U << 1U)) ==
        0xFF00410CU);

    for (std::uint8_t field = 0U; field < 8U; ++field)
    {
        for (std::uint8_t immediate = 0U; immediate < 16U; ++immediate)
        {
            for (const std::uint32_t fpscr_value : fpscr_values)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint32_t instruction = (63U << 26U) |
                    (static_cast<std::uint32_t>(field) << 23U) |
                    (static_cast<std::uint32_t>(immediate) << 12U) | (134U << 1U);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(0x100U, instruction);
                    core->state.cia = 0x100U;
                    core->state.gpr.fill(0xA5A5A5A5U);
                    core->state.fpr.fill(0x0123456789ABCDEFULL);
                    core->state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
                    core->state.cr = 0x12345678U;
                    core->state.xer = 0xA00000A5U;
                    core->state.fpscr = fpscr_value;
                    core->state.lr = 0x11223344U;
                    core->state.ctr = 0x55667788U;
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
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);
                assert(actual.instruction_history.size() == 1U);
                const InstructionHistoryEntry& history = actual.instruction_history.back();
                assert(history.completed && history.opcode_name == "mtfsfi");
                assert(history.has_fpscr_field_write);
                assert(history.fpscr_field_number == field);
                assert(history.fpscr_field_immediate == immediate);
                assert(history.fpscr_before_field_write == fpscr_value);
                assert(history.fpscr_after_field_write == compiled.state.fpscr);
            }
        }
    }
}

void recompiler_mtfsf_differential_tests()
{
    constexpr std::uint8_t source_register = 3U;
    static_assert(((63U << 26U) | (0xFFU << 17U) | (3U << 11U) |
                   (711U << 1U)) == 0xFDFE1D8EU);

    for (std::uint16_t mask_value = 0U; mask_value < 256U; ++mask_value)
    {
        const std::uint8_t field_mask = static_cast<std::uint8_t>(mask_value);
        const std::uint32_t fpscr_value =
            (static_cast<std::uint32_t>(mask_value) * 0x01010101U) ^ 0xA5C3BEEFU;
        const std::uint64_t source_raw = 0xDEADBEEF00000000ULL |
            ((static_cast<std::uint32_t>(mask_value) * 0x01020304U) ^ 0xA5C3BEEFU);
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = (63U << 26U) |
            (static_cast<std::uint32_t>(field_mask) << 17U) |
            (static_cast<std::uint32_t>(source_register) << 11U) | (711U << 1U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.fpr.fill(0x0123456789ABCDEFULL);
            core->state.fpr[source_register] = source_raw;
            core->state.fpr_ps1.fill(0xFEDCBA9876543210ULL);
            core->state.cr = 0x12345678U;
            core->state.xer = 0xA00000A5U;
            core->state.fpscr = fpscr_value;
            core->state.lr = 0x11223344U;
            core->state.ctr = 0x55667788U;
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
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name == "mtfsf");
        assert(history.has_fpscr_masked_write);
        assert(history.fpscr_masked_write_source == source_register);
        assert(history.fpscr_masked_write_mask == field_mask);
        assert(history.fpscr_masked_write_source_raw == source_raw);
        assert(history.fpscr_before_masked_write == fpscr_value);
        assert(history.fpscr_after_masked_write == compiled.state.fpscr);
    }
}

void recompiler_ps_cmpu0_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_cmpu0_form(1U, 1U, 2U),
        ps_cmpu0_form(2U, 3U, 4U),
        ps_cmpu0_form(3U, 5U, 6U),
    };
    static_assert(ps_cmpu0_form(1U, 1U, 2U) == 0x10811000U);
    constexpr std::uint64_t positive_one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t positive_two = 0x4000000000000000ULL;
    constexpr std::uint64_t qnan = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t snan = 0x7FF0000000000001ULL;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.gpr.fill(0xC001D00DU);
        core->state.fpr.fill(positive_one);
        core->state.fpr_ps1.fill(positive_two);
        core->state.fpr[1] = positive_one;
        core->state.fpr_ps1[1] = positive_two;
        core->state.fpr[2] = positive_two;
        core->state.fpr_ps1[2] = positive_one;
        core->state.fpr[3] = qnan;
        core->state.fpr_ps1[3] = positive_one;
        core->state.fpr[4] = positive_two;
        core->state.fpr_ps1[4] = qnan;
        core->state.fpr[5] = snan;
        core->state.fpr_ps1[5] = positive_two;
        core->state.fpr[6] = positive_two;
        core->state.fpr_ps1[6] = positive_one;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x80000000U;
        core->state.lr = 0x12345678U;
        core->state.ctr = 0x87654321U;
        core->state.fpscr = 0x0000A025U | fpscr::ve_mask;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_fpr = compiled.state.fpr;
    const auto original_fpr_ps1 = compiled.state.fpr_ps1;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.fpr == original_fpr);
    assert(compiled.state.fpr_ps1 == original_fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert((compiled.state.fpscr & fpscr::vxsnan_mask) != 0U);
    assert((compiled.state.fpscr & fpscr::fex_mask) != 0U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.completed && history.opcode_name == "ps_cmpu0");
        assert(history.has_fp_compare);
        assert(history.fp_compare_cr_field == expected.fp_compare_cr_field);
        assert(history.fp_compare_a_register == expected.fp_compare_a_register);
        assert(history.fp_compare_b_register == expected.fp_compare_b_register);
        assert(history.fp_compare_a_raw == expected.fp_compare_a_raw);
        assert(history.fp_compare_b_raw == expected.fp_compare_b_raw);
        assert(history.fp_compare_result == expected.fp_compare_result);
        assert(!history.fp_compare_uses_ps1);
    }
    assert(actual.instruction_history[0].fp_compare_result == 0x8U);
    assert(actual.instruction_history[1].fp_compare_result == 0x1U);
    assert(actual.instruction_history[2].fp_compare_result == 0x1U);
}

void recompiler_ps_cmpu1_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_cmpu1_form(1U, 1U, 2U),
        ps_cmpu1_form(2U, 3U, 4U),
        ps_cmpu1_form(3U, 5U, 6U),
    };
    static_assert(ps_cmpu1_form(1U, 1U, 2U) == 0x10811080U);
    constexpr std::uint64_t positive_one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t positive_two = 0x4000000000000000ULL;
    constexpr std::uint64_t qnan = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t snan = 0xFFF0000000000001ULL;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.gpr.fill(0xC001D00DU);
        core->state.fpr.fill(positive_one);
        core->state.fpr_ps1.fill(positive_one);
        core->state.fpr[1] = 0x4022000000000000ULL; // PS0=9, opposite to PS1 compare
        core->state.fpr_ps1[1] = positive_one;
        core->state.fpr[2] = positive_one;
        core->state.fpr_ps1[2] = positive_two;
        core->state.fpr[3] = positive_one;
        core->state.fpr_ps1[3] = qnan;
        core->state.fpr[4] = positive_two;
        core->state.fpr_ps1[4] = positive_one;
        core->state.fpr[5] = positive_one;
        core->state.fpr_ps1[5] = positive_one;
        core->state.fpr[6] = positive_two;
        core->state.fpr_ps1[6] = snan;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x80000000U;
        core->state.lr = 0x12345678U;
        core->state.ctr = 0x87654321U;
        core->state.fpscr = 0x0000A025U | fpscr::ve_mask;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_fpr = compiled.state.fpr;
    const auto original_fpr_ps1 = compiled.state.fpr_ps1;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr &&
        compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr && compiled.state.fpr == original_fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1 &&
        compiled.state.fpr_ps1 == original_fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert((compiled.state.fpscr & fpscr::vxsnan_mask) != 0U);
    assert((compiled.state.fpscr & fpscr::fex_mask) != 0U);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.completed && history.opcode_name == "ps_cmpu1");
        assert(history.has_fp_compare && history.fp_compare_uses_ps1);
        assert(history.fp_compare_cr_field == expected.fp_compare_cr_field);
        assert(history.fp_compare_a_register == expected.fp_compare_a_register);
        assert(history.fp_compare_b_register == expected.fp_compare_b_register);
        assert(history.fp_compare_a_raw == expected.fp_compare_a_raw);
        assert(history.fp_compare_b_raw == expected.fp_compare_b_raw);
        assert(history.fp_compare_result == expected.fp_compare_result);
    }
    assert(actual.instruction_history[0].fp_compare_result == 0x8U);
    assert(actual.instruction_history[1].fp_compare_result == 0x1U);
    assert(actual.instruction_history[2].fp_compare_result == 0x1U);
}

void recompiler_ps_cmpo0_differential_test()
{
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        bool enable_invalid;
        std::uint8_t expected_result;
        bool expect_vxvc;
        bool expect_vxsnan;
    };
    constexpr std::array cases{
        Case{0x3FF0000000000000ULL, 0x4000000000000000ULL,
            false, 0x8U, false, false},
        Case{0x7FF8123456789ABCULL, 0x3FF0000000000000ULL,
            false, 0x1U, true, false},
        Case{0xFFF0000000000001ULL, 0x4000000000000000ULL,
            false, 0x1U, true, true},
        Case{0x3FF0000000000000ULL, 0x7FF0000000000001ULL,
            true, 0x1U, false, true},
    };
    static_assert(ps_cmpo0_form(1U, 1U, 2U) == 0x10811040U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        constexpr std::uint32_t instruction = ps_cmpo0_form(2U, 1U, 31U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0xC001D00DU);
            core->state.fpr.fill(0xA5A5A5A5A5A5A5A5ULL);
            core->state.fpr_ps1.fill(0x5A5A5A5A5A5A5A5AULL);
            core->state.fpr[1] = test.a;
            core->state.fpr[31] = test.b;
            core->state.fpr_ps1[1] = 0xDEADBEEF01234567ULL;
            core->state.fpr_ps1[31] = 0xBAD0BAD0BAD0BAD0ULL;
            core->state.cr = 0xA5C36E91U;
            core->state.xer = 0x80000000U;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x0000A025U |
                (test.enable_invalid ? fpscr::ve_mask : 0U);
        }

        const auto original_gprs = compiled.state.gpr;
        const auto original_fpr = compiled.state.fpr;
        const auto original_fpr_ps1 = compiled.state.fpr_ps1;
        const auto original_memory = compiled.memory.read32_be(0x100U);
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr &&
            compiled.state.gpr == original_gprs);
        assert(compiled.state.fpr == interpreted.state.fpr &&
            compiled.state.fpr == original_fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1 &&
            compiled.state.fpr_ps1 == original_fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(0x100U) == original_memory);
        assert(((compiled.state.fpscr & fpscr::vxvc_mask) != 0U) == test.expect_vxvc);
        assert(((compiled.state.fpscr & fpscr::vxsnan_mask) != 0U) == test.expect_vxsnan);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name == "ps_cmpo0");
        assert(history.has_fp_compare && !history.fp_compare_uses_ps1);
        assert(history.fp_compare_cr_field == 2U);
        assert(history.fp_compare_a_raw == test.a);
        assert(history.fp_compare_b_raw == test.b);
        assert(history.fp_compare_result == test.expected_result);
    }
}

void recompiler_ps_cmpo1_differential_test()
{
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        bool enable_invalid;
        std::uint8_t expected_result;
        bool expect_vxvc;
        bool expect_vxsnan;
    };
    constexpr std::array cases{
        Case{0x3FF0000000000000ULL, 0x4000000000000000ULL,
            false, 0x8U, false, false},
        Case{0x7FF8123456789ABCULL, 0x3FF0000000000000ULL,
            false, 0x1U, true, false},
        Case{0xFFF0000000000001ULL, 0x4000000000000000ULL,
            false, 0x1U, true, true},
        Case{0x3FF0000000000000ULL, 0x7FF0000000000001ULL,
            true, 0x1U, false, true},
    };
    static_assert(ps_cmpo1_form(1U, 1U, 2U) == 0x108110C0U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        constexpr std::uint32_t instruction = ps_cmpo1_form(2U, 1U, 31U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0xC001D00DU);
            core->state.fpr.fill(0xA5A5A5A5A5A5A5A5ULL);
            core->state.fpr_ps1.fill(0x5A5A5A5A5A5A5A5AULL);
            core->state.fpr[1] = 0x4022000000000000ULL; // PS0 compares greater
            core->state.fpr_ps1[1] = test.a;
            core->state.fpr[31] = 0x3FF0000000000000ULL;
            core->state.fpr_ps1[31] = test.b;
            core->state.cr = 0xA5C36E91U;
            core->state.xer = 0x80000000U;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x0000A025U |
                (test.enable_invalid ? fpscr::ve_mask : 0U);
        }

        const auto original_gprs = compiled.state.gpr;
        const auto original_fpr = compiled.state.fpr;
        const auto original_fpr_ps1 = compiled.state.fpr_ps1;
        const auto original_memory = compiled.memory.read32_be(0x100U);
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr &&
            compiled.state.gpr == original_gprs);
        assert(compiled.state.fpr == interpreted.state.fpr &&
            compiled.state.fpr == original_fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1 &&
            compiled.state.fpr_ps1 == original_fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(0x100U) == original_memory);
        assert(((compiled.state.fpscr & fpscr::vxvc_mask) != 0U) == test.expect_vxvc);
        assert(((compiled.state.fpscr & fpscr::vxsnan_mask) != 0U) == test.expect_vxsnan);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name == "ps_cmpo1");
        assert(history.has_fp_compare && history.fp_compare_uses_ps1);
        assert(history.fp_compare_cr_field == 2U);
        assert(history.fp_compare_a_raw == test.a);
        assert(history.fp_compare_b_raw == test.b);
        assert(history.fp_compare_result == test.expected_result);
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

void recompiler_subfeo_differential_test()
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
        Case{0xFFFFFFFFU, 0x7FFFFFFFU, 4U, 3U, 2U, false,
            0x25A5A5A5U, 0xA5C36E91U},
        Case{0x7FFFFFFFU, 0x80000000U, 4U, 3U, 4U, true,
            0xA5A5A5A5U, 0x87654321U},
        Case{0x80000000U, 0x7FFFFFFFU, 7U, 8U, 7U, true,
            0x80000000U, 0xFFFFFFFFU},
        Case{0x80000000U, 0x80000000U, 7U, 7U, 7U, true,
            0x20000000U, 0x87654321U},
    };
    static_assert(x_form(3U, 4U, 5U, 648U) == 0x7C642D10U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 648U, test.record);
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
        const std::uint64_t extended_sum =
            static_cast<std::uint64_t>(~test.ra_value) +
            test.rb_value + carry_in;
        const std::uint32_t expected =
            static_cast<std::uint32_t>(extended_sum);
        const std::int64_t signed_result =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.rb_value)) -
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.ra_value)) -
            static_cast<std::int64_t>(carry_in == 0U);
        const bool overflow =
            signed_result < std::numeric_limits<std::int32_t>::min() ||
            signed_result > std::numeric_limits<std::int32_t>::max();
        const bool carry_out = (extended_sum >> 32U) != 0U;
        const bool old_so = (test.initial_xer & 0x80000000U) != 0U;
        assert(compiled.state.gpr[test.destination] == expected);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == carry_out);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == (old_so || overflow));
        assert((compiled.state.xer & ~0xE0000000U) ==
            (test.initial_xer & ~0xE0000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == (test.record ? "subfeo." : "subfeo"));
        assert(history.has_carry_input && history.carry_input == (carry_in != 0U));
        assert(history.has_carry_result && history.carry_result == carry_out);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == (old_so || overflow));
    }
}

void recompiler_divwuo_differential_test()
{
    struct Case
    {
        std::uint32_t dividend;
        std::uint32_t divisor;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{100U, 5U, 4U, 5U, 3U, false, 0xE5A5A5A5U, 0x13579BDFU},
        Case{0xFFFFFFFFU, 1U, 4U, 5U, 4U, true, 0x60000000U, 0xA5C36E91U},
        Case{0U, 0U, 4U, 5U, 5U, true, 0x20000000U, 0x87654321U},
        Case{0x80000000U, 0U, 7U, 8U, 7U, false, 0x40000000U, 0xFFFFFFFFU},
        Case{0x100U, 0x100U, 9U, 9U, 9U, true, 0xA0000000U, 0xA5A5A5A5U},
    };
    static_assert(x_form(3U, 4U, 5U, 971U) == 0x7C642F96U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 971U, test.record);
        assert(decode(instruction).opcode == Opcode::divide_word_unsigned_overflow);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.dividend;
            core->state.gpr[test.rb] = test.divisor;
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

        const bool overflow = test.divisor == 0U;
        const std::uint32_t expected = overflow ? 0U : test.dividend / test.divisor;
        const bool old_so = (test.initial_xer & 0x80000000U) != 0U;
        assert(compiled.state.gpr[test.destination] == expected);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == (old_so || overflow));
        assert((compiled.state.xer & ~0xC0000000U) ==
            (test.initial_xer & ~0xC0000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == (test.record ? "divwuo." : "divwuo"));
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == (old_so || overflow));
    }
}

void recompiler_divwo_differential_test()
{
    struct Case
    {
        std::uint32_t dividend;
        std::uint32_t divisor;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0xFFFFFFF9U, 2U, 4U, 5U, 3U, false, 0xE5A5A5A5U, 0x13579BDFU},
        Case{0x80000000U, 0xFFFFFFFFU, 4U, 5U, 4U, true, 0x20000000U, 0xA5C36E91U},
        Case{0xFFFFFFF9U, 0U, 4U, 5U, 5U, true, 0x40000000U, 0x87654321U},
        Case{7U, 0U, 7U, 8U, 7U, false, 0xA0000000U, 0xFFFFFFFFU},
        Case{0xFFFFFFFEU, 0xFFFFFFFEU, 9U, 9U, 9U, true,
            0x60000000U, 0xA5A5A5A5U},
    };
    static_assert(x_form(3U, 4U, 5U, 1003U) == 0x7C642FD6U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 1003U, test.record);
        assert(decode(instruction).opcode == Opcode::divide_word_signed_overflow);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.dividend;
            core->state.gpr[test.rb] = test.divisor;
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

        const std::int32_t dividend = std::bit_cast<std::int32_t>(test.dividend);
        const std::int32_t divisor = std::bit_cast<std::int32_t>(test.divisor);
        const bool overflow = divisor == 0 ||
            (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1);
        std::uint32_t expected = 0U;
        if (divisor == 0)
        {
            expected = dividend < 0 ? 0xFFFFFFFFU : 0U;
        }
        else if (dividend == std::numeric_limits<std::int32_t>::min() && divisor == -1)
        {
            expected = 0xFFFFFFFFU;
        }
        else
        {
            expected = static_cast<std::uint32_t>(dividend / divisor);
        }
        const bool old_so = (test.initial_xer & 0x80000000U) != 0U;
        assert(compiled.state.gpr[test.destination] == expected);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == (old_so || overflow));
        assert((compiled.state.xer & ~0xC0000000U) ==
            (test.initial_xer & ~0xC0000000U));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == (test.record ? "divwo." : "divwo"));
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == (old_so || overflow));
    }
}

void recompiler_adde_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
        std::uint32_t expected_result;
        bool expected_carry;
    };
    constexpr std::array cases{
        Case{5U, 7U, 2U, 3U, 4U, false, 0x80000000U, 0x13579BDFU, 12U, false},
        Case{0xFFFFFFFFU, 0U, 4U, 3U, 4U, true, 0xA0000000U, 0x87654321U,
            0U, true},
        Case{0x7FFFFFFFU, 0U, 5U, 6U, 7U, true, 0x20000000U, 0xA5C36E91U,
            0x80000000U, false},
        Case{0xFFFFFFFFU, 0U, 4U, 3U, 4U, false, 0x80000000U, 0xA5C36E91U,
            0xFFFFFFFFU, false},
        Case{0x7FFFFFFFU, 0x7FFFFFFFU, 7U, 7U, 7U, false, 0xA0000000U,
            0x12345678U, 0xFFFFFFFFU, false},
    };
    static_assert(x_form(3U, 4U, 5U, 138U) == 0x7C642914U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 138U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t carry_in =
            (test.initial_xer & 0x20000000U) != 0U ? 1U : 0U;
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
        assert(compiled.state.gpr[test.destination] == test.expected_result);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == test.expected_carry);
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        assert(!test.record ||
            (compiled.state.cr >> 28U) ==
                ((test.expected_result >> 31U) != 0U ?
                    ((test.initial_xer >> 31U) | 8U) :
                    test.expected_result == 0U ? ((test.initial_xer >> 31U) | 2U) :
                    ((test.initial_xer >> 31U) | 4U)));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.record ? "adde." : "adde"));
        assert(history.source_count == (test.ra == test.rb ? 1U : 2U));
        assert(history.source_registers[0] == test.ra);
        assert(history.source_values[0] == test.a);
        if (test.ra != test.rb)
        {
            assert(history.source_registers[1] == test.rb);
            assert(history.source_values[1] == test.b);
        }
        assert(history.has_carry_input && history.carry_input == (carry_in != 0U));
        assert(history.has_carry_result && history.carry_result == test.expected_carry);
    }
}

void recompiler_addeo_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0x7FFFFFFFU, 0U, 4U, 5U, 6U, false, 0x12345678U, 0xA5C36E91U},
        Case{0xFFFFFFFFU, 0U, 4U, 5U, 4U, true, 0x80000000U, 0x87654321U},
        Case{0x7FFFFFFFU, 0U, 6U, 7U, 7U, true, 0x20000000U, 0xA5C36E91U},
        Case{0x80000000U, 0xFFFFFFFFU, 8U, 9U, 8U, false,
            0xA0000000U, 0x13579BDFU},
        Case{0xFFFFFFFFU, 0U, 10U, 11U, 10U, false,
            0x60000000U, 0x2468ACE0U},
        Case{0x7FFFFFFFU, 0x7FFFFFFFU, 12U, 12U, 12U, true,
            0x40000000U, 0xFFFFFFFFU},
        Case{0U, 0U, 13U, 14U, 15U, false, 0xE0000000U, 0x11223344U},
    };
    static_assert(x_form(3U, 4U, 5U, 650U) == 0x7C642D14U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 650U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t carry_in =
            (test.initial_xer & 0x20000000U) != 0U ? 1U : 0U;
        const std::uint64_t sum = static_cast<std::uint64_t>(test.a) +
            test.b + carry_in;
        const std::uint32_t expected_result = static_cast<std::uint32_t>(sum);
        const std::int64_t signed_sum =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.a)) +
            std::bit_cast<std::int32_t>(test.b) + carry_in;
        const bool expected_overflow = signed_sum <
                std::numeric_limits<std::int32_t>::min() ||
            signed_sum > std::numeric_limits<std::int32_t>::max();
        const bool expected_carry = (sum >> 32U) != 0U;
        const bool expected_so =
            (test.initial_xer & 0x80000000U) != 0U || expected_overflow;

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
        assert(compiled.state.gpr[test.destination] == expected_result);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == expected_carry);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == expected_overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == expected_so);
        assert((compiled.state.xer & 0x1FFFFFFFU) ==
            (test.initial_xer & 0x1FFFFFFFU & ~0x20000000U));
        assert(!test.record ||
            (compiled.state.cr >> 28U) ==
                ((expected_result >> 31U) != 0U ?
                    ((expected_so ? 1U : 0U) | 8U) :
                    expected_result == 0U ? ((expected_so ? 1U : 0U) | 2U) :
                    ((expected_so ? 1U : 0U) | 4U)));
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.record ? "addeo." : "addeo"));
        assert(history.has_carry_input && history.carry_input == (carry_in != 0U));
        assert(history.has_carry_result && history.carry_result == expected_carry);
        assert(history.has_overflow_result &&
            history.overflow_result == expected_overflow);
        assert(history.summary_overflow_result == expected_so);
    }
}

void recompiler_addo_subfo_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool subtract;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0x7FFFFFFFU, 1U, 4U, 5U, 3U, false, false,
            0xA5A5A5A5U, 0x12345678U},
        Case{0x80000000U, 0xFFFFFFFFU, 4U, 5U, 4U, false, true,
            0x25A5A5A5U, 0x87654321U},
        Case{5U, 7U, 4U, 5U, 5U, false, false, 0xE5A5A5A5U, 0xA5C36E91U},
        Case{0x7FFFFFFFU, 1U, 6U, 7U, 7U, false, true,
            0x65A5A5A5U, 0xA5C36E91U},
        Case{0x80000000U, 0x80000000U, 8U, 8U, 8U, false, true,
            0xA5A5A5A5U, 0xFFFFFFFFU},
        Case{0x80000000U, 1U, 4U, 5U, 3U, true, true,
            0xE5A5A5A5U, 0x12345678U},
        Case{0x7FFFFFFFU, 0xFFFFFFFFU, 4U, 5U, 5U, true, false,
            0x65A5A5A5U, 0x87654321U},
        Case{0x80000000U, 0x7FFFFFFFU, 8U, 9U, 8U, true, true,
            0x25A5A5A5U, 0xA5C36E91U},
    };
    static_assert(x_form(3U, 4U, 5U, 778U) == 0x7C642E14U);
    static_assert(x_form(3U, 4U, 5U, 552U) == 0x7C642C50U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb,
            test.subtract ? 552U : 778U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const std::uint32_t result = test.subtract
            ? test.b - test.a : test.a + test.b;
        const bool overflow = test.subtract
            ? (((test.b ^ test.a) & (test.b ^ result)) & 0x80000000U) != 0U
            : ((~(test.a ^ test.b) & (test.a ^ result)) & 0x80000000U) != 0U;
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;

        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[test.destination] == result);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == summary_overflow);
        assert((compiled.state.xer & 0x3FFFFFFFU) ==
            (test.initial_xer & 0x3FFFFFFFU));
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.subtract
                ? (test.record ? "subfo." : "subfo")
                : (test.record ? "addo." : "addo")));
        assert(history.source_count == (test.ra == test.rb ? 1U : 2U));
        assert(history.source_registers[0] == test.ra);
        assert(history.source_values[0] == test.a);
        if (test.ra != test.rb)
        {
            assert(history.source_registers[1] == test.rb);
            assert(history.source_values[1] == test.b);
        }
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_nego_differential_test()
{
    struct Case
    {
        std::uint32_t source_value;
        std::uint8_t source;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0U, 4U, 3U, false, 0x600000A5U, 0x12345678U},
        Case{1U, 4U, 3U, true, 0xA00000A5U, 0x87654321U},
        Case{0x7FFFFFFFU, 4U, 4U, false, 0x200000A5U, 0xA5C36E91U},
        Case{0x80000000U, 4U, 4U, false, 0x200000A5U, 0xA5C36E91U},
        Case{0x80000000U, 4U, 6U, true, 0x600000A5U, 0xA5C36E91U},
    };
    constexpr auto encode_nego = [](std::uint8_t destination, std::uint8_t source,
                                    bool record) {
        return (31U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 16U) | (616U << 1U) |
            static_cast<std::uint32_t>(record);
    };
    static_assert(encode_nego(3U, 4U, false) == 0x7C6404D0U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = encode_nego(
            test.destination, test.source, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.source_value;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const std::uint32_t result = 0U - test.source_value;
        const bool overflow = test.source_value == 0x80000000U;
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[test.destination] == result);
        assert(((compiled.state.xer & 0x40000000U) != 0U) == overflow);
        assert(((compiled.state.xer & 0x80000000U) != 0U) == summary_overflow);
        assert((compiled.state.xer & 0x3FFFFFFFU) ==
            (test.initial_xer & 0x3FFFFFFFU));
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "nego." : "nego"));
        assert(history.source_count == 1U && history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.source_value);
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_addco_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0xFFFFFFFFU, 1U, 4U, 5U, 3U, false, 0x600000A5U, 0x12345678U},
        Case{0x7FFFFFFFU, 1U, 4U, 5U, 4U, true, 0x200000A5U, 0x87654321U},
        Case{0x80000000U, 0x80000000U, 4U, 5U, 5U, false,
            0xA00000A5U, 0xA5C36E91U},
        Case{1U, 2U, 4U, 5U, 4U, true, 0xE00000A5U, 0xA5C36E91U},
        Case{0x40000000U, 0x40000000U, 8U, 8U, 8U, true,
            0x25A5A5A5U, 0x12345678U},
    };
    static_assert(x_form(3U, 4U, 5U, 522U) == 0x7C642C14U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 522U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint64_t sum = static_cast<std::uint64_t>(test.a) + test.b;
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const bool carry = (sum >> 32U) != 0U;
        const bool overflow = ((~(test.a ^ test.b) & (test.a ^ result)) &
            0x80000000U) != 0U;
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer &
            ~(0x20000000U | 0x40000000U | 0x80000000U);
        if (carry)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "addco." : "addco"));
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_result && history.carry_result == carry);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_subfco_differential_test()
{
    struct Case
    {
        std::uint32_t a;
        std::uint32_t b;
        std::uint8_t ra;
        std::uint8_t rb;
        std::uint8_t destination;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{2U, 1U, 4U, 5U, 3U, false, 0x600000A5U, 0x12345678U},
        Case{1U, 0x80000000U, 4U, 5U, 4U, true, 0x200000A5U, 0x87654321U},
        Case{0xFFFFFFFFU, 0x7FFFFFFFU, 4U, 5U, 5U, false,
            0xA00000A5U, 0xA5C36E91U},
        Case{2U, 1U, 4U, 5U, 4U, true, 0xE00000A5U, 0xA5C36E91U},
        Case{0x80000000U, 0x80000000U, 8U, 8U, 8U, true,
            0x25A5A5A5U, 0x12345678U},
    };
    static_assert(x_form(3U, 4U, 5U, 520U) == 0x7C642C10U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.ra, test.rb, 520U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.ra] = test.a;
            core->state.gpr[test.rb] = test.b;
            core->state.xer = test.initial_xer;
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t result = test.b - test.a;
        const bool carry = test.b >= test.a;
        const bool overflow = (((test.b ^ test.a) & (test.b ^ result)) &
            0x80000000U) != 0U;
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer &
            ~(0x20000000U | 0x40000000U | 0x80000000U);
        if (carry)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "subfco." : "subfco"));
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_result && history.carry_result == carry);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_addzeo_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0x7FFFFFFFU, 4U, 5U, true, false, 0x400000A5U, 0x13579BDFU},
        Case{0xFFFFFFFFU, 6U, 6U, true, true, 0x8000005AU, 0x87654321U},
        Case{0x80000000U, 7U, 8U, true, true, 0x60000033U, 0xA5C36E91U},
        Case{0U, 9U, 10U, false, false, 0xA00000C3U, 0x2468ACE0U},
        Case{0x7FFFFFFFU, 12U, 12U, false, true, 0xE000005AU, 0x12345678U},
    };
    static_assert(x_form(3U, 4U, 0U, 714U) == 0x7C640594U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 714U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t result = test.operand +
            static_cast<std::uint32_t>(test.carry_in);
        const bool carry_out = test.carry_in && test.operand == 0xFFFFFFFFU;
        const bool overflow = test.carry_in && test.operand == 0x7FFFFFFFU;
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer &
            ~(0xE0000000U);
        if (carry_out)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        else
        {
            const std::uint32_t relation = result == 0U ? 2U
                : (result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (relation | (summary_overflow ? 1U : 0U)));
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "addzeo." : "addzeo"));
        assert(history.source_count == 1U && history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == carry_out);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_addmeo_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0U, 4U, 4U, false, false, 0x800000A5U, 0x13579BDFU},
        Case{0U, 5U, 6U, true, true, 0x4000005AU, 0x87654321U},
        Case{1U, 7U, 7U, false, true, 0xA0000033U, 0xA5C36E91U},
        Case{0x80000000U, 8U, 9U, false, true, 0x200000C3U, 0x12345678U},
        Case{0x7FFFFFFFU, 10U, 11U, false, false, 0x600000A5U, 0x2468ACE0U},
        Case{0xFFFFFFFFU, 12U, 12U, true, true, 0x2000005AU, 0xFFFFFFFFU},
        Case{0x80000000U, 13U, 14U, true, false, 0x80000033U, 0x11223344U},
    };
    static_assert(x_form(3U, 4U, 0U, 746U) == 0x7C6405D4U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 746U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint64_t sum = static_cast<std::uint64_t>(test.operand) +
            0xFFFFFFFFULL + static_cast<std::uint32_t>(test.carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t signed_sum =
            static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.operand)) - 1 +
            static_cast<std::int64_t>(test.carry_in);
        const bool carry_out = (sum >> 32U) != 0U;
        const bool overflow = signed_sum < std::numeric_limits<std::int32_t>::min() ||
            signed_sum > std::numeric_limits<std::int32_t>::max();
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer & ~0xE0000000U;
        if (carry_out)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        else
        {
            const std::uint32_t relation = result == 0U ? 2U
                : (result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (relation | (summary_overflow ? 1U : 0U)));
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "addmeo." : "addmeo"));
        assert(history.source_count == 1U && history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == carry_out);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_addme_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
        std::uint32_t expected_result;
        bool expected_carry;
    };
    constexpr std::array cases{
        Case{0U, 4U, 4U, false, false, 0x80000000U, 0x13579BDFU,
            0xFFFFFFFFU, false},
        Case{0U, 4U, 4U, true, true, 0xA0000000U, 0x87654321U,
            0U, true},
        Case{1U, 4U, 5U, false, true, 0U, 0xA5C36E91U,
            0U, true},
        Case{0xFFFFFFFFU, 3U, 4U, false, false, 0x80000000U, 0xA5C36E91U,
            0xFFFFFFFEU, true},
        Case{0x80000000U, 0U, 0U, false, true, 0x80000000U, 0x12345678U,
            0x7FFFFFFFU, true},
        Case{0xFFFFFFFFU, 7U, 7U, true, true, 0x20000000U, 0xA5C36E91U,
            0xFFFFFFFFU, true},
    };
    static_assert(x_form(3U, 4U, 0U, 234U) == 0x7C6401D4U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 234U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
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
        assert(compiled.state.gpr[test.destination] == test.expected_result);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == test.expected_carry);
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        if (test.record)
        {
            const std::uint32_t expected_cr0 = test.expected_result == 0U ? 2U
                : (test.expected_result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (expected_cr0 | (test.initial_xer >> 31U)));
        }
        else
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.record ? "addme." : "addme"));
        assert(history.source_count == 1U);
        assert(history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == test.expected_carry);
    }
}

void recompiler_subfmeo_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0U, 4U, 5U, false, false, 0x800000A5U, 0x13579BDFU},
        Case{0U, 6U, 6U, true, true, 0x4000005AU, 0x87654321U},
        Case{0x7FFFFFFFU, 7U, 8U, false, true, 0x20000033U, 0xA5C36E91U},
        Case{0x7FFFFFFFU, 9U, 9U, true, false, 0xA00000C3U, 0x12345678U},
        Case{0x80000000U, 10U, 11U, false, true, 0x600000A5U, 0x2468ACE0U},
        Case{0x80000000U, 12U, 12U, true, true, 0x2000005AU, 0xFFFFFFFFU},
        Case{0xFFFFFFFFU, 13U, 14U, false, false, 0x80000033U, 0x11223344U},
    };
    static_assert(x_form(3U, 4U, 0U, 744U) == 0x7C6405D0U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 744U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t complement = ~test.operand;
        const std::uint64_t sum = static_cast<std::uint64_t>(complement) +
            0xFFFFFFFFULL + static_cast<std::uint32_t>(test.carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t signed_sum =
            -static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.operand)) - 2 +
            static_cast<std::int64_t>(test.carry_in);
        const bool carry_out = (sum >> 32U) != 0U;
        const bool overflow = signed_sum < std::numeric_limits<std::int32_t>::min() ||
            signed_sum > std::numeric_limits<std::int32_t>::max();
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer & ~0xE0000000U;
        if (carry_out)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        else
        {
            const std::uint32_t relation = result == 0U ? 2U
                : (result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (relation | (summary_overflow ? 1U : 0U)));
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "subfmeo." : "subfmeo"));
        assert(history.source_count == 1U && history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == carry_out);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_subfme_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
        std::uint32_t expected_result;
        bool expected_carry;
    };
    constexpr std::array cases{
        Case{0xFFFFFFFFU, 4U, 4U, false, false, 0x80000000U, 0x13579BDFU,
            0xFFFFFFFFU, false},
        Case{0xFFFFFFFFU, 4U, 4U, true, true, 0xA0000000U, 0x87654321U,
            0U, true},
        Case{0U, 5U, 6U, false, true, 0U, 0xA5C36E91U,
            0xFFFFFFFEU, true},
        Case{0U, 5U, 6U, true, false, 0x20000000U, 0xA5C36E91U,
            0xFFFFFFFFU, true},
        Case{0x80000000U, 0U, 0U, false, true, 0x80000000U, 0x12345678U,
            0x7FFFFFFEU, true},
        Case{0x7FFFFFFFU, 7U, 7U, false, true, 0U, 0xA5C36E91U,
            0x7FFFFFFFU, true},
    };
    static_assert(x_form(3U, 4U, 0U, 232U) == 0x7C6401D0U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 232U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
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
        assert(compiled.state.gpr[test.destination] == test.expected_result);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == test.expected_carry);
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        if (test.record)
        {
            const std::uint32_t expected_cr0 = test.expected_result == 0U ? 2U
                : (test.expected_result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (expected_cr0 | (test.initial_xer >> 31U)));
        }
        else
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.record ? "subfme." : "subfme"));
        assert(history.source_count == 1U);
        assert(history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == test.expected_carry);
    }
}

void recompiler_sraw_differential_test()
{
    struct Case
    {
        std::uint8_t source_register;
        std::uint8_t destination;
        std::uint8_t shift_register;
        std::uint32_t source;
        std::uint32_t shift;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
        std::uint32_t expected_result;
        bool expected_carry;
    };
    constexpr std::array cases{
        Case{4U, 3U, 5U, 0x80000001U, 1U, false, 0xA00000A5U,
            0x12345678U, 0xC0000000U, true},
        Case{4U, 3U, 5U, 0x80000000U, 1U, false, 0x400000A5U,
            0xA5C36E91U, 0xC0000000U, false},
        Case{4U, 3U, 5U, 0x7FFFFFFFU, 1U, false, 0x800000A5U,
            0xA5C36E91U, 0x3FFFFFFFU, false},
        Case{4U, 3U, 5U, 0x80000001U, 0U, false, 0xA00000A5U,
            0x87654321U, 0x80000001U, false},
        Case{4U, 3U, 5U, 0x80000000U, 32U, true, 0x800000A5U,
            0xA5C36E91U, 0xFFFFFFFFU, true},
        Case{4U, 3U, 5U, 0x7FFFFFFFU, 32U, false, 0x400000A5U,
            0x12345678U, 0U, false},
        Case{4U, 3U, 5U, 0x80000001U, 63U, false, 0x200000A5U,
            0x87654321U, 0xFFFFFFFFU, true},
        Case{4U, 3U, 5U, 0x80000001U, 64U, false, 0xA00000A5U,
            0xA5C36E91U, 0x80000001U, false},
        Case{3U, 3U, 3U, 0x80000001U, 0x80000001U, true, 0x80000000U,
            0x13579BDFU, 0xC0000000U, true},
    };
    static_assert(x_form(4U, 3U, 5U, 792U) == 0x7C832E30U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(test.source_register,
            test.destination, test.shift_register, 792U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source_register] = test.source;
            core->state.gpr[test.shift_register] = test.shift;
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
        assert(compiled.state.gpr[test.destination] == test.expected_result);
        assert(((compiled.state.xer & 0x20000000U) != 0U) == test.expected_carry);
        assert((compiled.state.xer & ~0x20000000U) ==
            (test.initial_xer & ~0x20000000U));
        if (test.record)
        {
            const std::uint32_t expected_cr0 = test.expected_result == 0U ? 2U
                : (test.expected_result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (expected_cr0 | (test.initial_xer >> 31U)));
        }
        else
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.completed && history.opcode_name ==
            (test.record ? "sraw." : "sraw"));
        assert(history.source_count ==
            (test.source_register == test.shift_register ? 1U : 2U));
        assert(history.source_registers[0] == test.source_register);
        assert(history.source_values[0] == test.source);
        if (test.source_register != test.shift_register)
        {
            assert(history.source_registers[1] == test.shift_register);
            assert(history.source_values[1] == test.shift);
        }
        assert(history.has_carry_result && history.carry_result == test.expected_carry);
    }
}

void recompiler_subfzeo_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0U, 4U, 5U, false, false, 0x800000A5U, 0x13579BDFU},
        Case{0U, 6U, 6U, true, true, 0x4000005AU, 0x87654321U},
        Case{0x80000000U, 7U, 8U, false, true, 0x20000033U, 0xA5C36E91U},
        Case{0x80000000U, 9U, 9U, true, false, 0xA00000C3U, 0x12345678U},
        Case{0x7FFFFFFFU, 10U, 11U, false, true, 0x600000A5U, 0x2468ACE0U},
        Case{0xFFFFFFFFU, 12U, 12U, true, true, 0x2000005AU, 0xFFFFFFFFU},
    };
    static_assert(x_form(3U, 4U, 0U, 712U) == 0x7C640590U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 712U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
            core->state.cr = test.initial_cr;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const std::uint32_t complement = ~test.operand;
        const std::uint64_t sum = static_cast<std::uint64_t>(complement) +
            static_cast<std::uint32_t>(test.carry_in);
        const std::uint32_t result = static_cast<std::uint32_t>(sum);
        const std::int64_t signed_sum =
            -static_cast<std::int64_t>(std::bit_cast<std::int32_t>(test.operand)) - 1 +
            static_cast<std::int64_t>(test.carry_in);
        const bool carry_out = (sum >> 32U) != 0U;
        const bool overflow = signed_sum < std::numeric_limits<std::int32_t>::min() ||
            signed_sum > std::numeric_limits<std::int32_t>::max();
        const bool summary_overflow =
            ((test.initial_xer & 0x80000000U) != 0U) || overflow;
        std::uint32_t expected_xer = test.initial_xer & ~0xE0000000U;
        if (carry_out)
        {
            expected_xer |= 0x20000000U;
        }
        if (overflow)
        {
            expected_xer |= 0x40000000U;
        }
        if (summary_overflow)
        {
            expected_xer |= 0x80000000U;
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
        assert(compiled.state.gpr[test.destination] == result);
        assert(compiled.state.xer == expected_xer);
        if (!test.record)
        {
            assert(compiled.state.cr == test.initial_cr);
        }
        else
        {
            const std::uint32_t relation = result == 0U ? 2U
                : (result & 0x80000000U) != 0U ? 8U : 4U;
            assert((compiled.state.cr >> 28U) ==
                (relation | (summary_overflow ? 1U : 0U)));
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.completed && history.opcode_name ==
            (test.record ? "subfzeo." : "subfzeo"));
        assert(history.source_count == 1U && history.source_registers[0] == test.source);
        assert(history.source_values[0] == test.operand);
        assert(history.has_destination && history.destination_register == test.destination);
        assert(history.destination_value == result);
        assert(history.has_carry_input && history.carry_input == test.carry_in);
        assert(history.has_carry_result && history.carry_result == carry_out);
        assert(history.has_overflow_result && history.overflow_result == overflow);
        assert(history.summary_overflow_result == summary_overflow);
    }
}

void recompiler_subfze_differential_test()
{
    struct Case
    {
        std::uint32_t operand;
        std::uint8_t source;
        std::uint8_t destination;
        bool carry_in;
        bool record;
        std::uint32_t initial_xer;
        std::uint32_t initial_cr;
    };
    constexpr std::array cases{
        Case{0U, 4U, 3U, false, false, 0xA5A5A5A5U, 0x13579BDFU},
        Case{0U, 4U, 3U, true, false, 0x25A5A5A5U, 0xA5C36E91U},
        Case{1U, 4U, 4U, false, false, 0x85A5A5A5U, 0xA5C36E91U},
        Case{1U, 4U, 4U, true, true, 0x25A5A5A5U, 0x87654321U},
        Case{0xFFFFFFFFU, 4U, 5U, false, true, 0x80000000U, 0x12345678U},
        Case{0xFFFFFFFFU, 4U, 5U, true, true, 0xA5A5A5A5U, 0xA5C36E91U},
        Case{0x80000000U, 0U, 0U, true, true, 0x80000000U, 0xFFFFFFFFU},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = x_form(
            test.destination, test.source, 0U, 200U, test.record);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0x10203040U);
            core->state.gpr[test.source] = test.operand;
            core->state.xer = (test.initial_xer & ~0x20000000U) |
                (test.carry_in ? 0x20000000U : 0U);
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
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(!actual.instruction_history.empty());
        assert(actual.instruction_history.back().opcode_name ==
            (test.record ? "subfze." : "subfze"));
        assert(actual.instruction_history.back().has_carry_input);
        assert(actual.instruction_history.back().carry_input == test.carry_in);
        assert(actual.instruction_history.back().has_carry_result);
        assert(actual.instruction_history.back().carry_result ==
            ((compiled.state.xer & 0x20000000U) != 0U));
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
    const std::array<std::uint32_t, 9> program{
        0x38601234U,                 // addi r3,r0,0x1234
        0x6064ABCDU,                 // ori r4,r3,0xABCD
        0x688500FFU,                 // xori r5,r4,0x00FF
        x_form(5, 6, 3, 28),         // and r6,r5,r3
        x_form(6, 7, 4, 316),        // xor r7,r6,r4
        x_form(7, 8, 6, 124),        // nor r8,r7,r6
        x_form(7, 7, 6, 476),        // nand r7,r7,r6 (destination aliases RS)
        x_form(7, 7, 6, 284),        // eqv r7,r7,r6 (destination aliases RS)
        x_form(7, 7, 6, 412),        // orc r7,r7,r6 (destination aliases RS)
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
    assert(compiled.state.gpr[8] == 0xFFFF4402U);
    assert(compiled.state.gpr[7] == 0xFFFFFFFFU);
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
        Case{x_form(3U, 5U, 4U, 284U, true), 0xAAU, 0x0FU,
             0U, 0xFFFFFF5AU, 8U, "eqv."},
        Case{x_form(3U, 5U, 4U, 412U, true), 0x00FF00FFU, 0x0F0F0F0FU,
             0U, 0xF0FFF0FFU, 8U, "orc."},
        Case{x_form(3U, 5U, 4U, 476U, true), 0xFFFFFFFFU, 0U,
             0U, 0xFFFFFFFFU, 8U, "nand."},
        Case{x_form(3U, 5U, 4U, 124U, true), 0U, 0U,
             0x80000000U, 0xFFFFFFFFU, 9U, "nor."},
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

void recompiler_mcrfs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 4U> fpscr_values{
        0U, 0xFFFFFFFFU, 0xA5C39E71U,
        fpscr::fx_mask | fpscr::ox_mask | fpscr::ux_mask | fpscr::zx_mask |
            fpscr::xx_mask | fpscr::vxsnan_mask | fpscr::vxisi_mask |
            fpscr::vxidi_mask | fpscr::vxzdz_mask | fpscr::vximz_mask |
            fpscr::vxvc_mask | fpscr::vxsoft_mask | fpscr::vxsqrt_mask |
            fpscr::vxcvi_mask | fpscr::ve_mask | fpscr::ze_mask};

    for (std::uint8_t destination = 0U; destination < 8U; ++destination)
    {
        for (std::uint8_t source = 0U; source < 8U; ++source)
        {
            for (const std::uint32_t initial_fpscr : fpscr_values)
            {
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                const std::uint32_t instruction = 0xFC000080U |
                    (static_cast<std::uint32_t>(destination) << 23U) |
                    (static_cast<std::uint32_t>(source) << 18U);
                for (EspressoCore* core : {&interpreted, &compiled})
                {
                    core->memory.write32_be(start, instruction);
                    core->state.cia = start;
                    core->state.cr = 0xA5C39E71U ^
                        (static_cast<std::uint32_t>(destination) << 12U);
                    core->state.fpscr = initial_fpscr;
                    core->state.gpr.fill(0xDEADBEEFU);
                    core->state.xer = 0xE00000A5U;
                    core->state.lr = 0xABCDEF00U;
                    core->state.ctr = 0x76543210U;
                    core->state.fpr[7] = 0x0123456789ABCDEFULL;
                    core->state.fpr_ps1[7] = 0xFEDCBA9876543210ULL;
                }

                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                assert(actual.reason == oracle.reason);
                assert(actual.steps == oracle.steps && actual.steps == 1U);
                assert(actual.cia == oracle.cia);
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.fpscr == interpreted.state.fpscr);
                assert(compiled.state.gpr == interpreted.state.gpr);
                assert(compiled.state.fpr == interpreted.state.fpr);
                assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
                assert(compiled.state.xer == interpreted.state.xer);
                assert(compiled.state.lr == interpreted.state.lr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(recompiler.statistics().native_instructions == 1U);
                assert(recompiler.statistics().interpreter_fallbacks == 0U);

                const InstructionHistoryEntry& expected = oracle.instruction_history.back();
                const InstructionHistoryEntry& received = actual.instruction_history.back();
                assert(received.opcode_name == "mcrfs");
                assert(received.has_fpscr_cr_move == expected.has_fpscr_cr_move);
                assert(received.fpscr_cr_source == expected.fpscr_cr_source);
                assert(received.fpscr_cr_destination == expected.fpscr_cr_destination);
                assert(received.fpscr_cr_value == expected.fpscr_cr_value);
                assert(received.fpscr_before_cr_move == expected.fpscr_before_cr_move);
                assert(received.fpscr_after_cr_move == expected.fpscr_after_cr_move);
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
        const std::uint32_t instruction = fctiw_form(
            test.destination, test.source, 15U);
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fctiwz");
        assert(history.has_fp_integer_conversion);
        assert(history.fp_integer_conversion_source_raw == test.source_raw);
        assert(history.fp_integer_conversion_destination == test.destination);
    }
}

void recompiler_fctiw_differential_tests()
{
    struct Case
    {
        std::uint64_t source_raw;
        std::uint32_t fpscr;
        std::uint8_t destination;
        std::uint8_t source;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t destination_sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::uint64_t qnan = 0x7FF8123456789ABCULL;
    constexpr std::uint64_t snan = 0x7FF0000000004321ULL;
    constexpr auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::array cases{
        Case{raw(1.5), 0x00015000U, 2U, 1U}, // nearest-even -> 2, FR
        Case{raw(1.5), 0x00015001U, 2U, 1U}, // toward zero -> 1
        Case{raw(1.5), 0x00015002U, 2U, 1U}, // toward +infinity -> 2, FR
        Case{raw(1.5), 0x00015003U, 2U, 1U}, // toward -infinity -> 1
        Case{raw(-1.5), 0x00015000U, 2U, 1U}, // nearest-even -> -2, FR
        Case{raw(-1.5), 0x00015001U, 2U, 1U}, // toward zero -> -1
        Case{raw(-1.5), 0x00015002U, 2U, 1U}, // toward +infinity -> -1
        Case{raw(-1.5), 0x00015003U, 2U, 1U}, // toward -infinity -> -2, FR
        Case{raw(2.5), 0x00015000U, 2U, 1U}, // ties to even -> 2
        Case{raw(3.5), 0x00015000U, 2U, 1U}, // ties to even -> 4, FR
        Case{raw(1.25), 0x00015002U, 2U, 1U}, // ceil -> 2, FR
        Case{raw(-1.25), 0x00015003U, 2U, 1U}, // floor -> -2, FR
        Case{raw(-0.25), 0x00015002U, 2U, 1U}, // negative-to-zero marker
        Case{raw(2147483647.4), 0x00015000U, 2U, 1U}, // rounds to max
        Case{raw(2147483647.4), 0x00015002U, 2U, 1U}, // rounds out of range
        Case{qnan, 0x00015000U, 2U, 1U}, // qNaN
        Case{snan, 0x00015200U, 2U, 1U}, // sNaN, VE
        Case{raw(1.5), 0x00015000U, 1U, 1U}, // destination/source alias
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = fctiw_form(
            test.destination, test.source, 14U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write32_be(start + 8U, 0xDEADBEEFU);
            core->state.cia = start;
            core->state.fpr.fill(destination_sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[test.source] = test.source_raw;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x76543210U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.fpscr;
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
        assert(compiled.memory.read32_be(start + 8U) == 0xDEADBEEFU);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fctiw");
        assert(history.has_fp_integer_conversion);
        assert(history.fp_integer_conversion_source_raw == test.source_raw);
        assert(history.fp_integer_conversion_destination == test.destination);
    }
}

void recompiler_fmsub_differential_tests()
{
    const auto encode = [](std::uint8_t destination, std::uint8_t source_a,
                           std::uint8_t source_b, std::uint8_t source_c) {
        return (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (28U << 1U);
    };
    constexpr auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t fpr_sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint64_t c;
        std::uint32_t fpscr;
        std::uint8_t destination;
        std::uint8_t source_a;
        std::uint8_t source_b;
        std::uint8_t source_c;
    };
    constexpr std::array cases{
        Case{raw(2.0), raw(5.0), raw(3.0), 0x00000000U, 4U, 1U, 2U, 3U},
        Case{0x3FF0000002000000ULL, raw(1.0), 0x3FEFFFFFFC000000ULL,
            0x00000000U, 4U, 1U, 2U, 3U}, // fused result -2^-54
        Case{raw(2.0), raw(5.0), raw(3.0), 0x00000000U, 2U, 1U, 2U, 3U},
        Case{0x7FF8123456789ABCULL, raw(1.0), raw(2.0), 0U, 4U, 1U, 2U, 3U},
        Case{raw(2.0), raw(1.0), 0x7FF8123456789ABCULL, 0U, 4U, 1U, 2U, 3U},
        Case{raw(2.0), 0x7FF8123456789ABCULL, raw(1.0), 0U, 4U, 1U, 2U, 3U},
        Case{0x7FF0000000004321ULL, raw(1.0), raw(2.0), 0U, 4U, 1U, 2U, 3U},
        Case{raw(2.0), raw(1.0), 0x7FF0000000004321ULL, fpscr::ve_mask,
            4U, 1U, 2U, 3U},
        Case{0x7FF0000000000000ULL, raw(1.0), 0U, 0U, 4U, 1U, 2U, 3U},
        Case{0x7FF0000000000000ULL, 0x7FF0000000000000ULL, raw(1.0), 0U,
            4U, 1U, 2U, 3U},
        Case{0x7FF0000000000000ULL, 0x7FF0000000000000ULL, raw(1.0),
            fpscr::ve_mask, 4U, 1U, 2U, 3U},
        Case{0x7FF0000000000000ULL, 0xFFF0000000000000ULL, raw(1.0), 0U,
            4U, 1U, 2U, 3U},
    };
    static_assert(encode(30U, 0U, 30U, 1U) == 0xFFC0F078U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = encode(test.destination, test.source_a,
            test.source_b, test.source_c);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write32_be(start + 8U, 0xA55AA55AU);
            core->state.cia = start;
            core->state.fpr.fill(fpr_sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[test.source_a] = test.a;
            core->state.fpr[test.source_b] = test.b;
            core->state.fpr[test.source_c] = test.c;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.fpscr;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(start + 8U) == 0xA55AA55AU);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fmsub");
        assert(history.has_fp_arithmetic && history.fp_arithmetic_source_count == 3U);
        assert(history.fp_arithmetic_a_raw == test.a);
        assert(history.fp_arithmetic_b_raw == test.b);
        assert(history.fp_arithmetic_c_raw == test.c);
    }

    EspressoCore fused_core(0x1000U);
    fused_core.memory.write32_be(start, encode(4U, 1U, 2U, 3U));
    fused_core.state.cia = start;
    fused_core.state.fpr[1] = 0x3FF0000002000000ULL;
    fused_core.state.fpr[2] = raw(1.0);
    fused_core.state.fpr[3] = 0x3FEFFFFFFC000000ULL;
    Recompiler recompiler;
    const RunResult fused = recompiler.run(fused_core, 1U);
    assert(fused_core.state.fpr[4] == 0xBC90000000000000ULL);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_fnmadd_differential_tests()
{
    const auto encode = [](std::uint8_t destination, std::uint8_t source_a,
                           std::uint8_t source_b, std::uint8_t source_c) {
        return (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source_a) << 16U) |
            (static_cast<std::uint32_t>(source_b) << 11U) |
            (static_cast<std::uint32_t>(source_c) << 6U) | (31U << 1U);
    };
    constexpr auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t fpr_sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint64_t c;
        std::uint32_t fpscr;
        std::uint8_t destination;
    };
    constexpr std::array cases{
        Case{0x3FF0000002000000ULL, 0xBFF0000000000000ULL,
            0x3FEFFFFFFC000000ULL, 0U, 4U}, // fused +2^-54
        Case{raw(2.0), raw(4.0), raw(3.0), 0U, 4U}, // -(2*3+4) = -10
        Case{0xFFF8123456789ABCULL, raw(1.0), raw(2.0), 0U, 4U}, // NaN is not negated
        Case{0x7FF0000000004321ULL, raw(1.0), raw(2.0), fpscr::ve_mask, 4U},
        Case{0x7FF0000000000000ULL, 0xFFF0000000000000ULL, raw(1.0), 0U, 4U},
        Case{0x7FF0000000000000ULL, 0xFFF0000000000000ULL, raw(1.0),
            fpscr::ve_mask, 4U},
        Case{0x7FF0000000000000ULL, 0x7FF0000000000000ULL, raw(1.0), 0U, 4U},
    };
    static_assert(encode(1U, 2U, 3U, 4U) == 0xFC22193EU);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, encode(test.destination, 1U, 2U, 3U));
            core->memory.write32_be(start + 8U, 0xA55AA55AU);
            core->state.cia = start;
            core->state.fpr.fill(fpr_sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[1] = test.a;
            core->state.fpr[2] = test.b;
            core->state.fpr[3] = test.c;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.fpscr;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(start + 8U) == 0xA55AA55AU);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fnmadd");
        assert(history.has_fp_arithmetic && history.fp_arithmetic_source_count == 3U);
    }

    EspressoCore alias_interpreted(0x1000U);
    EspressoCore alias_compiled(0x1000U);
    for (EspressoCore* core : {&alias_interpreted, &alias_compiled})
    {
        core->memory.write32_be(start, encode(2U, 1U, 2U, 3U));
        core->state.cia = start;
        core->state.fpr[1] = raw(2.0);
        core->state.fpr[2] = raw(4.0);
        core->state.fpr[3] = raw(3.0);
        core->state.fpr_ps1[2] = ps1_sentinel;
    }
    const RunResult alias_oracle = alias_interpreted.run(1U);
    Recompiler alias_recompiler;
    const RunResult alias_actual = alias_recompiler.run(alias_compiled, 1U);
    assert(alias_actual.reason == alias_oracle.reason);
    assert(alias_compiled.state.fpr == alias_interpreted.state.fpr);
    assert(alias_compiled.state.fpscr == alias_interpreted.state.fpscr);
    assert(alias_compiled.state.fpr[2] == 0xC024000000000000ULL);
    assert(alias_compiled.state.fpr_ps1[2] == ps1_sentinel);
    assert(alias_recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_fnmadds_differential_tests()
{
    const auto encode = [](std::uint8_t destination) {
        return (59U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
            (1U << 16U) | (2U << 11U) | (3U << 6U) | (31U << 1U);
    };
    constexpr auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t fpr_sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    struct Case
    {
        std::uint64_t a;
        std::uint64_t b;
        std::uint64_t c;
        std::uint32_t fpscr;
        std::uint8_t destination;
        std::uint64_t expected;
    };
    constexpr std::array cases{
        Case{raw(2.0), raw(4.0), raw(3.0), 0U, 4U, raw(-10.0)},
        Case{0x3FF0000002000000ULL, 0xBFF0000000000000ULL, raw(1.0),
            0U, 4U, raw(-0x1p-27)},
        Case{0xFFF8123456789ABCULL, raw(1.0), raw(2.0),
            0U, 4U, 0U}, // NaN source sign is preserved by the interpreter.
        Case{0x7FF0000000004321ULL, raw(1.0), raw(2.0),
            fpscr::ve_mask, 4U, fpr_sentinel},
        Case{0x7FF0000000000000ULL, raw(1.0), 0U,
            0U, 4U, 0x7FF8000000000000ULL}, // VXIMZ
        Case{0x7FF0000000000000ULL, 0xFFF0000000000000ULL, raw(1.0),
            0U, 4U, 0x7FF8000000000000ULL}, // VXISI
        Case{0x7FF0000000000000ULL, 0x7FF0000000000000ULL, raw(1.0),
            0U, 4U, 0xFFF0000000000000ULL}, // valid +infinity sum, then negate
        Case{raw(2.0), raw(4.0), raw(3.0), 0U, 2U, raw(-10.0)}, // d == B
    };
    static_assert(encode(1U) == 0xEC2110FEU);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, encode(test.destination));
            core->memory.write32_be(start + 8U, 0xA55AA55AU);
            core->state.cia = start;
            core->state.fpr.fill(fpr_sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[1] = test.a;
            core->state.fpr[2] = test.b;
            core->state.fpr[3] = test.c;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.fpscr;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(start + 8U) == 0xA55AA55AU);
        assert(compiled.state.fpr[test.destination] == interpreted.state.fpr[test.destination]);
        if (test.expected != 0U)
        {
            assert(compiled.state.fpr[test.destination] == test.expected);
        }
        else
        {
            const std::uint64_t nan_result = compiled.state.fpr[test.destination];
            assert((nan_result & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL);
            assert((nan_result & 0x000FFFFFFFFFFFFFULL) != 0U);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fnmadds");
        assert(history.has_fp_arithmetic);
        assert(history.has_fp_arithmetic_single_bits ==
            (test.expected != fpr_sentinel));
        assert(history.fp_arithmetic_source_count == 3U);
    }
}

void recompiler_fres_differential_tests()
{
    constexpr auto encode = [](std::uint8_t destination, std::uint8_t source) {
        return (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(source) << 11U) | (24U << 1U);
    };
    constexpr auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t sentinel = 0x123456789ABCDEF0ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    struct Case
    {
        std::uint64_t source;
        std::uint32_t fpscr;
        std::uint8_t destination;
        std::uint64_t expected;
    };
    constexpr std::array cases{
        Case{raw(2.0), 0U, 4U, raw(0.5)},
        Case{raw(3.0), 0U, 4U, 0U}, // estimate rounds to binary32
        Case{raw(-4.0), 0U, 4U, raw(-0.25)},
        Case{0U, 0U, 4U, 0x7FF0000000000000ULL},
        Case{0x8000000000000000ULL, 0U, 4U, 0xFFF0000000000000ULL},
        Case{0U, fpscr::ze_mask, 4U, sentinel},
        Case{0x7FF0000000000000ULL, 0U, 4U, 0U},
        Case{0xFFF0000000000000ULL, 0U, 4U, 0x8000000000000000ULL},
        Case{0xFFF8123456789ABCULL, 0U, 4U, 0U},
        Case{0xFFF0123456789ABCULL, 0U, 4U, 0U},
        Case{0x7FF0123456789ABCULL, fpscr::ve_mask, 4U, sentinel},
        Case{raw(2.0), 0U, 2U, raw(0.5)}, // destination aliases source
    };
    static_assert(encode(1U, 2U) == 0xFC201030U);

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, encode(test.destination, 2U));
            core->memory.write32_be(start + 8U, 0xA55AA55AU);
            core->state.cia = start;
            core->state.fpr.fill(sentinel);
            core->state.fpr_ps1.fill(ps1_sentinel);
            core->state.fpr[2] = test.source;
            core->state.fpr[4] = sentinel;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = test.fpscr;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.memory.read32_be(start + 8U) == 0xA55AA55AU);
        if (test.expected != 0U)
        {
            assert(compiled.state.fpr[test.destination] == test.expected);
        }
        else if (test.source == 0xFFF8123456789ABCULL ||
            test.source == 0xFFF0123456789ABCULL)
        {
            const std::uint64_t result = compiled.state.fpr[test.destination];
            assert((result & 0x7FF8000000000000ULL) == 0x7FF8000000000000ULL);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "fres");
        assert(history.has_fp_source && history.fp_source_register == 2U);
        assert(history.has_fp_destination && history.fp_destination_register == test.destination);
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

    for (const bool negative_absolute : {false, true})
    {
        for (std::uint8_t source = 0U; source < 32U; ++source)
        {
            for (std::uint8_t destination = 0U; destination < 32U; ++destination)
            {
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            const std::uint32_t instruction =
                (63U << 26U) | (static_cast<std::uint32_t>(destination) << 21U) |
                (static_cast<std::uint32_t>(source) << 11U) |
                ((negative_absolute ? 136U : 264U) << 1U);
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
                (negative_absolute
                    ? source_values[source] | 0x8000000000000000ULL
                    : source_values[source] & 0x7FFFFFFFFFFFFFFFULL));
            assert(compiled.state.fpr_ps1[destination] == ps1_sentinel);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name ==
                (negative_absolute ? "fnabs" : "fabs"));
            }
        }
    }
}

void recompiler_ps_add_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t fpr_sentinel = 0xA5A5A5A5A5A5A5A5ULL;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::uint32_t gpr_sentinel = 0xC001D00DU;
    const auto raw = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    const std::array<std::array<std::uint64_t, 4U>, 4U> lane_inputs{{
        {raw(1.0000000596046448), raw(-4.25), raw(0.000000059604644775390625), raw(1.5)},
        {0x0000000000000000ULL, 0x8000000000000000ULL,
            0x8000000000000000ULL, 0x8000000000000000ULL},
        {0x7FF0000000000000ULL, 0xFFF0000000000000ULL,
            0x3FF0000000000000ULL, 0xFFF0000000000000ULL},
        {0x7FF8123456789ABCULL, 0x3FF0000000000000ULL,
            0x4000000000000000ULL, 0xFFF0123456789ABCULL},
    }};
    constexpr std::array<std::array<std::uint8_t, 3U>, 4U> register_sets{{
        {2U, 3U, 4U}, // distinct destination
        {3U, 3U, 4U}, // destination aliases A
        {4U, 3U, 4U}, // destination aliases B
        {5U, 5U, 5U}, // all operands alias
    }};

    for (std::size_t input_index = 0; input_index < lane_inputs.size(); ++input_index)
    {
        for (const auto& registers : register_sets)
        {
            const std::uint8_t destination = registers[0];
            const std::uint8_t source_a = registers[1];
            const std::uint8_t source_b = registers[2];
            const auto& input = lane_inputs[input_index];
            const std::uint64_t a_ps0 = input[0];
            const std::uint64_t a_ps1 = input[1];
            const std::uint64_t b_ps0 = source_a == source_b ? a_ps0 : input[2];
            const std::uint64_t b_ps1 = source_a == source_b ? a_ps1 : input[3];
            const std::uint32_t instruction = ps_add_form(destination, source_a, source_b);
            EspressoCore interpreted(0x1000U);
            EspressoCore compiled(0x1000U);
            for (EspressoCore* core : {&interpreted, &compiled})
            {
                core->memory.write32_be(start, instruction);
                core->state.cia = start;
                core->state.gpr.fill(gpr_sentinel);
                core->state.fpr.fill(fpr_sentinel);
                core->state.fpr_ps1.fill(ps1_sentinel);
                core->state.fpr[source_a] = a_ps0;
                core->state.fpr_ps1[source_a] = a_ps1;
                core->state.fpr[source_b] = b_ps0;
                core->state.fpr_ps1[source_b] = b_ps1;
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
            assert(compiled.state.fpr == interpreted.state.fpr);
            assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
            assert(compiled.state.cr == interpreted.state.cr);
            assert(compiled.state.xer == interpreted.state.xer);
            assert(compiled.state.lr == interpreted.state.lr);
            assert(compiled.state.ctr == interpreted.state.ctr);
            assert(compiled.state.fpscr == interpreted.state.fpscr);
            assert(compiled.state.gpr[0] == gpr_sentinel);
            assert(compiled.state.fpr_ps1[destination] ==
                interpreted.state.fpr_ps1[destination]);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(!actual.instruction_history.empty());
            const InstructionHistoryEntry& history = actual.instruction_history.back();
            assert(history.completed && history.opcode_name == "ps_add");
            assert(history.has_paired_fp_state);
            assert(history.paired_fp_source_a_ps0 == a_ps0);
            assert(history.paired_fp_source_a_ps1 == a_ps1);
            assert(history.paired_fp_source_b_ps0 == b_ps0);
            assert(history.paired_fp_source_b_ps1 == b_ps1);
            assert(history.paired_fp_destination_ps0 == compiled.state.fpr[destination]);
            assert(history.paired_fp_destination_ps1 ==
                compiled.state.fpr_ps1[destination]);
        }
    }
}

void recompiler_ps_div_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_div_form(2U, 1U, 3U),
        ps_div_form(4U, 4U, 5U), // destination aliases numerator
        ps_div_form(6U, 7U, 6U), // destination aliases denominator
        ps_div_form(8U, 9U, 10U),
    };
    static_assert(ps_div_form(2U, 1U, 3U) == 0x10411824U);
    constexpr std::uint64_t positive_zero = 0x0000000000000000ULL;
    constexpr std::uint64_t negative_zero = 0x8000000000000000ULL;
    constexpr std::uint64_t positive_one = 0x3FF0000000000000ULL;
    constexpr std::uint64_t negative_one = 0xBFF0000000000000ULL;
    constexpr std::uint64_t qnan = 0x7FF8123456789ABCULL;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.gpr.fill(0xC001D00DU);
        core->state.fpr.fill(positive_one);
        core->state.fpr_ps1.fill(negative_one);
        core->state.fpr[1] = 0x4018000000000000ULL; // 6.0
        core->state.fpr_ps1[1] = 0xC018000000000000ULL; // -6.0
        core->state.fpr[3] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[3] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[4] = positive_one;
        core->state.fpr_ps1[4] = negative_one;
        core->state.fpr[5] = positive_zero;
        core->state.fpr_ps1[5] = positive_zero;
        core->state.fpr[6] = positive_zero;
        core->state.fpr_ps1[6] = positive_zero;
        core->state.fpr[7] = positive_zero;
        core->state.fpr_ps1[7] = negative_zero;
        core->state.fpr[9] = qnan;
        core->state.fpr_ps1[9] = 0xFFF8123456789ABCULL;
        core->state.fpr[10] = 0x4000000000000000ULL;
        core->state.fpr_ps1[10] = 0x4000000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_cr = compiled.state.cr;
    const auto original_xer = compiled.state.xer;
    const auto original_lr = compiled.state.lr;
    const auto original_ctr = compiled.state.ctr;
    const auto original_fpscr = compiled.state.fpscr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr && compiled.state.cr == original_cr);
    assert(compiled.state.xer == interpreted.state.xer && compiled.state.xer == original_xer);
    assert(compiled.state.lr == interpreted.state.lr && compiled.state.lr == original_lr);
    assert(compiled.state.ctr == interpreted.state.ctr && compiled.state.ctr == original_ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr &&
        compiled.state.fpscr == original_fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[2] == 0x4008000000000000ULL); // 3.0
    assert(compiled.state.fpr_ps1[2] == 0x4000000000000000ULL); // 2.0
    assert(compiled.state.fpr[4] == 0x7FF0000000000000ULL);
    assert(compiled.state.fpr_ps1[4] == 0xFFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.completed && history.opcode_name == "ps_div");
        assert(history.has_paired_fp_state);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_reservation_instruction_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t address = 0x208U;

    // lwarx snapshots its effective-address inputs before writing RT, including
    // when RT aliases either indexed-address register.
    for (const auto [destination, base, index] : {
             std::array<std::uint8_t, 3U>{5U, 4U, 6U},
             std::array<std::uint8_t, 3U>{4U, 4U, 6U},
             std::array<std::uint8_t, 3U>{6U, 4U, 6U},
             std::array<std::uint8_t, 3U>{5U, 0U, 0U}})
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t effective_address = base == 0U ? 0x208U : address;
        const std::uint32_t word = x_form(destination, base, index, 20U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, word);
            core->memory.write32_be(effective_address, 0xA1B2C3D4U);
            core->state.cia = start;
            core->state.gpr[destination] = 0xDEADBEEFU;
            core->state.gpr[0] = 0x208U;
            core->state.gpr[4] = 0x200U;
            core->state.gpr[6] = 8U;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x81234567U;
            core->state.reservation_address = 0x44U;
            core->state.reservation_valid = false;
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
        assert(compiled.state.reservation_valid == interpreted.state.reservation_valid);
        assert(compiled.state.reservation_address == interpreted.state.reservation_address);
        assert(compiled.state.gpr[destination] == 0xA1B2C3D4U);
        assert(compiled.state.reservation_valid);
        assert(compiled.state.reservation_address == effective_address);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.back().opcode_name == "lwarx");
        assert(actual.instruction_history.back().has_effective_address);
        assert(actual.instruction_history.back().effective_address == effective_address);
    }

    const auto run_stwcx = [&](bool reservation_valid,
                               std::uint32_t reservation_address,
                               bool summary_overflow,
                               bool misaligned) {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, x_form(5U, 4U, 6U, 150U, true));
            core->memory.write32_be(address, 0x11223344U);
            core->state.cia = start;
            core->state.gpr[4] = 0x200U;
            core->state.gpr[6] = misaligned ? 9U : 8U;
            core->state.gpr[5] = 0xAABBCCDDU;
            core->state.cr = 0x5A5AA5A5U;
            core->state.xer = summary_overflow ? 0x80000000U : 0x01234567U;
            core->state.reservation_address = reservation_address;
            core->state.reservation_valid = reservation_valid;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.reservation_valid == interpreted.state.reservation_valid);
        assert(compiled.state.reservation_address == interpreted.state.reservation_address);
        assert(compiled.memory.read32_be(address) == interpreted.memory.read32_be(address));
        if (misaligned)
        {
            assert(actual.reason == StopReason::memory_fault);
            assert(actual.detail.find("write 4 byte(s) at guest address 0x00000209") !=
                   std::string::npos);
            assert(compiled.state.reservation_valid);
            assert(compiled.state.cr == 0x5A5AA5A5U);
            assert(!actual.instruction_history.empty());
            assert(!actual.instruction_history.back().completed);
        }
        else
        {
            assert(actual.steps == 1U);
            assert(!compiled.state.reservation_valid);
            const bool success = reservation_valid && reservation_address == address;
            assert(compiled.memory.read32_be(address) ==
                (success ? 0xAABBCCDDU : 0x11223344U));
            const std::uint32_t expected_cr0 =
                (success ? 0x2U : 0U) | (summary_overflow ? 1U : 0U);
            assert((compiled.state.cr >> 28U) == expected_cr0);
            assert(recompiler.statistics().native_instructions == 1U);
            assert(recompiler.statistics().interpreter_fallbacks == 0U);
            assert(actual.instruction_history.back().opcode_name == "stwcx.");
            assert(actual.instruction_history.back().effective_address == address);
        }
    };

    run_stwcx(true, address, false, false);
    run_stwcx(true, address, true, false);
    run_stwcx(false, address, true, false);
    run_stwcx(true, address + 4U, false, false);
    run_stwcx(true, address, false, true);

    // A failed lwarx read leaves both RT and any previous reservation intact.
    EspressoCore interpreted_fault(0x1000U);
    EspressoCore compiled_fault(0x1000U);
    for (EspressoCore* core : {&interpreted_fault, &compiled_fault})
    {
        core->memory.write32_be(start, x_form(5U, 4U, 6U, 20U));
        core->state.cia = start;
        core->state.gpr[4] = 0x1000U;
        core->state.gpr[6] = 0U;
        core->state.gpr[5] = 0xDEADBEEFU;
        core->state.reservation_address = 0x88U;
        core->state.reservation_valid = true;
    }
    const RunResult fault_oracle = interpreted_fault.run(1U);
    Recompiler fault_recompiler;
    const RunResult fault_actual = fault_recompiler.run(compiled_fault, 1U);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("read 4 byte(s)") != std::string::npos);
    assert(compiled_fault.state.gpr == interpreted_fault.state.gpr);
    assert(compiled_fault.state.gpr[5] == 0xDEADBEEFU);
    assert(compiled_fault.state.reservation_valid);
    assert(compiled_fault.state.reservation_address == 0x88U);
    assert(!fault_actual.instruction_history.empty());
    assert(!fault_actual.instruction_history.back().completed);

    EspressoCore misaligned_interpreted(0x1000U);
    EspressoCore misaligned_compiled(0x1000U);
    for (EspressoCore* core : {&misaligned_interpreted, &misaligned_compiled})
    {
        core->memory.write32_be(start, x_form(5U, 4U, 6U, 20U));
        core->state.cia = start;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[6] = 1U;
        core->state.gpr[5] = 0xDEADBEEFU;
        core->state.reservation_address = 0x88U;
        core->state.reservation_valid = true;
    }
    const RunResult misaligned_oracle = misaligned_interpreted.run(1U);
    Recompiler misaligned_recompiler;
    const RunResult misaligned_actual = misaligned_recompiler.run(misaligned_compiled, 1U);
    assert(misaligned_actual.reason == misaligned_oracle.reason);
    assert(misaligned_actual.detail.find("read 4 byte(s) at guest address 0x00000201") !=
           std::string::npos);
    assert(misaligned_compiled.state.gpr == misaligned_interpreted.state.gpr);
    assert(misaligned_compiled.state.gpr[5] == 0xDEADBEEFU);
    assert(misaligned_compiled.state.reservation_valid);
    assert(misaligned_compiled.state.reservation_address == 0x88U);
    assert(!misaligned_actual.instruction_history.back().completed);
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

void recompiler_ps_mr_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_mr_form(1U, 5U),
        ps_mr_form(6U, 6U), // full source/destination alias
        ps_mr_form(31U, 0U),
        ps_mr_form(0U, 31U),
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < 32U; ++reg)
        {
            core->state.fpr[reg] = 0x3FF0000000000000ULL + reg * 0x10101ULL;
            core->state.fpr_ps1[reg] = 0xBFF0000000000000ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0101010101010101ULL);
        }
        core->state.fpr[5] = 0x7FF0123456789ABCULL;
        core->state.fpr_ps1[5] = 0xFFF8123456789ABCULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_mr");
        assert(history.has_paired_fp_state && history.paired_fp_unary);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_abs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_abs_form(1U, 5U),
        ps_abs_form(6U, 6U), // full source/destination alias
        ps_abs_form(31U, 0U),
        ps_abs_form(0U, 31U),
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < 32U; ++reg)
        {
            core->state.fpr[reg] = 0xBFF0000000000000ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0101010101010101ULL);
            core->state.fpr_ps1[reg] = 0xFFF0000000000001ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0001000100010001ULL);
        }
        core->state.fpr[5] = 0xFFF8123456789ABCULL;
        core->state.fpr_ps1[5] = 0xFFF0123456789ABCULL;
        core->state.fpr[6] = 0x8000000000000000ULL;
        core->state.fpr_ps1[6] = 0xFFF8000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_abs");
        assert(history.has_paired_fp_state && history.paired_fp_unary);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_nabs_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_nabs_form(1U, 5U),
        ps_nabs_form(6U, 6U), // full source/destination alias
        ps_nabs_form(31U, 0U),
        ps_nabs_form(0U, 31U),
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        for (std::uint32_t reg = 0U; reg < 32U; ++reg)
        {
            core->state.fpr[reg] = 0x3FF0000000000000ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0101010101010101ULL);
            core->state.fpr_ps1[reg] = 0x7FF0000000000001ULL ^
                (static_cast<std::uint64_t>(reg) * 0x0001000100010001ULL);
        }
        core->state.fpr[5] = 0x7FF8123456789ABCULL;
        core->state.fpr_ps1[5] = 0x7FF0123456789ABCULL;
        core->state.fpr[6] = 0x0000000000000000ULL;
        core->state.fpr_ps1[6] = 0xFFF8000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_nabs");
        assert(history.has_paired_fp_state && history.paired_fp_unary);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_sum1_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_sum1_form(1U, 5U, 6U, 7U),
        ps_sum1_form(6U, 6U, 6U, 6U), // destination aliases every source
        ps_sum1_form(5U, 0U, 0U, 5U), // destination aliases C
    };
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0xBFE0000000000000ULL);
        core->state.fpr_ps1.fill(0x3FE0000000000000ULL);
        core->state.fpr[5] = 0x3FF8000000000000ULL; // 1.5
        core->state.fpr_ps1[5] = 0x4000000000000000ULL;
        core->state.fpr[6] = 0xBFF0000000000000ULL; // -1
        core->state.fpr_ps1[6] = 0x4002000000000000ULL; // 2.25
        core->state.fpr[7] = 0xC012000000000000ULL; // -4.5
        core->state.fpr_ps1[7] = 0x400A000000000000ULL; // 3.25
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_sum1");
        assert(history.has_paired_fp_state && history.paired_fp_sum1 &&
            !history.paired_fp_sum0);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_source_c_ps0 == expected.paired_fp_source_c_ps0);
        assert(history.paired_fp_source_c_ps1 == expected.paired_fp_source_c_ps1);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_res_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_res_form(4U, 1U),
        ps_res_form(6U, 6U), // source/destination alias
        ps_res_form(8U, 7U),
        ps_res_form(10U, 9U),
    };
    static_assert(ps_res_form(4U, 1U) == 0x10800830U);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0 -> 0.5
        core->state.fpr_ps1[1] = 0xC010000000000000ULL; // -4.0 -> -0.25
        core->state.fpr[6] = 0x0000000000000000ULL; // +0 -> +infinity
        core->state.fpr_ps1[6] = 0x8000000000000000ULL; // -0 -> -infinity
        core->state.fpr[7] = 0x7FF0000000000000ULL; // +infinity -> +0
        core->state.fpr_ps1[7] = 0xFFF0000000000000ULL; // -infinity -> -0
        core->state.fpr[9] = 0x7FF8123456789ABCULL; // qNaN
        core->state.fpr_ps1[9] = 0x7FF0123456789ABCULL; // sNaN
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[4] == 0x3FE0000000000000ULL);
    assert(compiled.state.fpr_ps1[4] == 0xBFD0000000000000ULL);
    assert(compiled.state.fpr[6] == 0x7FF0000000000000ULL);
    assert(compiled.state.fpr_ps1[6] == 0xFFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_res");
        assert(history.has_paired_fp_state && history.paired_fp_unary);
        assert(history.paired_fp_source_b_register ==
            expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_rsqrte_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_rsqrte_form(4U, 1U),
        ps_rsqrte_form(6U, 6U), // source/destination alias
        ps_rsqrte_form(8U, 7U),
        ps_rsqrte_form(10U, 9U),
        ps_rsqrte_form(12U, 11U),
    };
    static_assert(ps_rsqrte_form(4U, 1U) == 0x10800834U);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4010000000000000ULL; // 4.0 -> 0.5
        core->state.fpr_ps1[1] = 0x4030000000000000ULL; // 16.0 -> 0.25
        core->state.fpr[6] = 0x0000000000000000ULL; // +0 -> +infinity
        core->state.fpr_ps1[6] = 0x8000000000000000ULL; // -0 -> NaN
        core->state.fpr[7] = 0x7FF0000000000000ULL; // +infinity -> +0
        core->state.fpr_ps1[7] = 0xFFF0000000000000ULL; // -infinity -> NaN
        core->state.fpr[9] = 0xC010000000000000ULL; // negative -> NaN
        core->state.fpr_ps1[9] = 0x7FF8123456789ABCULL; // qNaN -> NaN
        core->state.fpr[11] = 0x7FF8123456789ABCULL; // qNaN
        core->state.fpr_ps1[11] = 0x7FF0123456789ABCULL; // sNaN
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[4] == 0x3FE0000000000000ULL);
    assert(compiled.state.fpr_ps1[4] == 0x3FD0000000000000ULL);
    assert(compiled.state.fpr[6] == 0x7FF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_rsqrte");
        assert(history.has_paired_fp_state && history.paired_fp_unary);
        assert(history.paired_fp_source_b_register ==
            expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_select_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_sel_form(1U, 2U, 3U, 4U),
        ps_sel_form(6U, 6U, 8U, 7U), // destination aliases A
        ps_sel_form(9U, 10U, 12U, 11U),
        ps_sel_form(13U, 14U, 16U, 15U),
        ps_sel_form(17U, 18U, 20U, 19U),
    };
    static_assert(ps_sel_form(1U, 2U, 3U, 4U) == 0x102220EEU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[2] = 0x3FF0000000000000ULL; // +1 selects C
        core->state.fpr_ps1[2] = 0xBFF0000000000000ULL; // -1 selects B
        core->state.fpr[3] = 0x4022000000000000ULL;
        core->state.fpr_ps1[3] = 0x4024000000000000ULL;
        core->state.fpr[4] = 0x4008000000000000ULL;
        core->state.fpr_ps1[4] = 0x4010000000000000ULL;
        core->state.fpr[6] = 0x7FF8123456789ABCULL; // NaN selects B
        core->state.fpr_ps1[6] = 0x8000000000000000ULL; // -0 selects C
        core->state.fpr[7] = 0x4016000000000000ULL;
        core->state.fpr_ps1[7] = 0x401C000000000000ULL;
        core->state.fpr[8] = 0x4024000000000000ULL;
        core->state.fpr_ps1[8] = 0x4026000000000000ULL;
        core->state.fpr[10] = 0x0000000000000000ULL; // +0 selects C
        core->state.fpr_ps1[10] = 0x8000000000000000ULL; // -0 selects C
        core->state.fpr[11] = 0x4030000000000000ULL;
        core->state.fpr_ps1[11] = 0x4031000000000000ULL;
        core->state.fpr[12] = 0x4032000000000000ULL;
        core->state.fpr_ps1[12] = 0x4033000000000000ULL;
        core->state.fpr[14] = 0xC000000000000000ULL; // -2 selects B
        core->state.fpr_ps1[14] = 0x7FF0000000000000ULL; // +inf selects C
        core->state.fpr[15] = 0x4034000000000000ULL;
        core->state.fpr_ps1[15] = 0x4035000000000000ULL;
        core->state.fpr[16] = 0x4036000000000000ULL;
        core->state.fpr_ps1[16] = 0x4037000000000000ULL;
        core->state.fpr[18] = 0x7FF8123456789ABCULL; // qNaN selects B
        core->state.fpr_ps1[18] = 0x7FF0123456789ABCULL; // sNaN selects B
        core->state.fpr[19] = 0x4038000000000000ULL;
        core->state.fpr_ps1[19] = 0x4039000000000000ULL;
        core->state.fpr[20] = 0x403A000000000000ULL;
        core->state.fpr_ps1[20] = 0x403B000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[1] == 0x4022000000000000ULL);
    assert(compiled.state.fpr_ps1[1] == 0x4010000000000000ULL);
    assert(compiled.state.fpr[6] == 0x4016000000000000ULL);
    assert(compiled.state.fpr_ps1[6] == 0x4026000000000000ULL);
    assert(compiled.state.fpr[9] == 0x4032000000000000ULL);
    assert(compiled.state.fpr_ps1[9] == 0x4033000000000000ULL);
    assert(compiled.state.fpr[13] == 0x4034000000000000ULL);
    assert(compiled.state.fpr_ps1[13] == 0x4037000000000000ULL);
    assert(compiled.state.fpr[17] == 0x4038000000000000ULL);
    assert(compiled.state.fpr_ps1[17] == 0x4039000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_sel" && history.has_paired_fp_select);
        assert(history.paired_fp_select_destination == expected.paired_fp_select_destination);
        assert(history.paired_fp_select_a == expected.paired_fp_select_a);
        assert(history.paired_fp_select_b == expected.paired_fp_select_b);
        assert(history.paired_fp_select_c == expected.paired_fp_select_c);
        assert(history.paired_fp_select_a_ps0 == expected.paired_fp_select_a_ps0);
        assert(history.paired_fp_select_a_ps1 == expected.paired_fp_select_a_ps1);
        assert(history.paired_fp_select_b_ps0 == expected.paired_fp_select_b_ps0);
        assert(history.paired_fp_select_b_ps1 == expected.paired_fp_select_b_ps1);
        assert(history.paired_fp_select_c_ps0 == expected.paired_fp_select_c_ps0);
        assert(history.paired_fp_select_c_ps1 == expected.paired_fp_select_c_ps1);
        assert(history.paired_fp_select_result_ps0 == expected.paired_fp_select_result_ps0);
        assert(history.paired_fp_select_result_ps1 == expected.paired_fp_select_result_ps1);
    }
}

void recompiler_ps_muls1_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_muls1_form(4U, 1U, 3U),
        ps_muls1_form(3U, 1U, 3U), // destination aliases scalar source C
        ps_muls1_form(8U, 5U, 7U), // scalar operand's low bits are rounded
    };
    static_assert(ps_muls1_form(4U, 1U, 3U) == 0x108100DAU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x4000000000000000ULL);
        core->state.fpr_ps1.fill(0xC008000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[1] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[3] = 0x401C000000000000ULL; // 7.0
        core->state.fpr_ps1[3] = 0x4010000000000000ULL; // scalar 4.0
        core->state.fpr[5] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[5] = 0xBFF0000000000000ULL; // -1.0
        core->state.fpr[7] = 0x0000000000000000ULL;
        core->state.fpr_ps1[7] = 0x3FF0000004000000ULL; // 1 + 2^-26
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[4] == 0x4020000000000000ULL); // 8.0
    assert(compiled.state.fpr_ps1[4] == 0xC028000000000000ULL); // -12.0
    assert(compiled.state.fpr[3] == 0x4020000000000000ULL);
    assert(compiled.state.fpr_ps1[3] == 0xC028000000000000ULL);
    assert(compiled.state.fpr[8] == 0x3FF0000000000000ULL);
    assert(compiled.state.fpr_ps1[8] == 0xBFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_muls1");
        assert(history.has_paired_fp_state && history.paired_fp_scalar_multiply);
        assert(history.paired_fp_scalar_source_ps1);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_madds0_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_madds0_form(5U, 1U, 2U, 3U),
        ps_madds0_form(3U, 1U, 2U, 3U), // destination aliases scalar source C
        ps_madds0_form(4U, 6U, 7U, 8U), // scalar operand's low bits are rounded
    };
    static_assert(ps_madds0_form(5U, 1U, 2U, 3U) == 0x10A110DCU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[1] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[2] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[2] = 0x4014000000000000ULL; // 5.0
        core->state.fpr[3] = 0x4010000000000000ULL; // scalar 4.0
        core->state.fpr_ps1[3] = 0x4022000000000000ULL; // 9.0
        core->state.fpr[6] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[6] = 0x3FF0000000000000ULL;
        core->state.fpr[7] = 0x0000000000000000ULL;
        core->state.fpr_ps1[7] = 0x0000000000000000ULL;
        core->state.fpr[8] = 0x3FF0000004000000ULL; // 1 + 2^-26
        core->state.fpr_ps1[8] = 0x3FF0000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[5] == 0x4022000000000000ULL); // 9.0
    assert(compiled.state.fpr_ps1[5] == 0xC01C000000000000ULL); // -7.0
    assert(compiled.state.fpr[3] == 0x4022000000000000ULL);
    assert(compiled.state.fpr_ps1[3] == 0xC01C000000000000ULL); // -7.0
    assert(compiled.state.fpr[4] == 0x3FF0000000000000ULL);
    assert(compiled.state.fpr_ps1[4] == 0x3FF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_madds0");
        assert(history.has_paired_fp_state && history.paired_fp_multiply_add_scalar0);
        assert(!history.paired_fp_multiply_add && !history.paired_fp_multiply_subtract);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_source_c_register == expected.paired_fp_source_c_register);
        assert(history.paired_fp_source_c_ps0 == expected.paired_fp_source_c_ps0);
        assert(history.paired_fp_source_c_ps1 == expected.paired_fp_source_c_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_madds1_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_madds1_form(5U, 1U, 2U, 3U),
        ps_madds1_form(8U, 6U, 7U, 8U), // destination aliases scalar source C
        ps_madds1_form(4U, 9U, 10U, 11U), // scalar operand's low bits are rounded
    };
    static_assert(ps_madds1_form(5U, 1U, 2U, 3U) == 0x10A110DEU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[1] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[2] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[2] = 0x4014000000000000ULL; // 5.0
        core->state.fpr[3] = 0x4010000000000000ULL; // 4.0
        core->state.fpr_ps1[3] = 0x4022000000000000ULL; // scalar 9.0
        core->state.fpr[6] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[6] = 0x4008000000000000ULL; // 3.0
        core->state.fpr[7] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[7] = 0x4000000000000000ULL; // 2.0
        core->state.fpr[8] = 0x4010000000000000ULL; // 4.0
        core->state.fpr_ps1[8] = 0x4014000000000000ULL; // scalar 5.0
        core->state.fpr[9] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[9] = 0xBFF0000000000000ULL; // -1.0
        core->state.fpr[10] = 0x0000000000000000ULL;
        core->state.fpr_ps1[10] = 0x0000000000000000ULL;
        core->state.fpr[11] = 0x3FF0000000000000ULL;
        core->state.fpr_ps1[11] = 0x3FF0000004000000ULL; // 1 + 2^-26
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[5] == 0x4033000000000000ULL); // 19.0
    assert(compiled.state.fpr_ps1[5] == 0xC036000000000000ULL); // -22.0
    assert(compiled.state.fpr[8] == 0x4026000000000000ULL); // 11.0
    assert(compiled.state.fpr_ps1[8] == 0x4031000000000000ULL); // 17.0
    assert(compiled.state.fpr[4] == 0x3FF0000000000000ULL);
    assert(compiled.state.fpr_ps1[4] == 0xBFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_madds1");
        assert(history.has_paired_fp_state && history.paired_fp_multiply_add_scalar1);
        assert(!history.paired_fp_multiply_add && !history.paired_fp_multiply_subtract);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_source_c_register == expected.paired_fp_source_c_register);
        assert(history.paired_fp_source_c_ps0 == expected.paired_fp_source_c_ps0);
        assert(history.paired_fp_source_c_ps1 == expected.paired_fp_source_c_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_nmadd_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_nmadd_form(2U, 1U, 3U, 4U),
        ps_nmadd_form(6U, 5U, 6U, 7U), // destination aliases B
        ps_nmadd_form(8U, 9U, 10U, 11U), // 25-bit scalar operand rounding
    };
    static_assert(ps_nmadd_form(2U, 1U, 3U, 4U) == 0x1041193CU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[1] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[3] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[3] = 0x4000000000000000ULL; // 2.0
        core->state.fpr[4] = 0x4010000000000000ULL; // 4.0
        core->state.fpr_ps1[4] = 0x4014000000000000ULL; // 5.0
        core->state.fpr[5] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[5] = 0x4008000000000000ULL; // 3.0
        core->state.fpr[6] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[6] = 0x4000000000000000ULL; // 2.0
        core->state.fpr[7] = 0x4008000000000000ULL; // 3.0
        core->state.fpr_ps1[7] = 0x4010000000000000ULL; // 4.0
        core->state.fpr[9] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[9] = 0x3FF0000000000000ULL;
        core->state.fpr[10] = 0x0000000000000000ULL;
        core->state.fpr_ps1[10] = 0x0000000000000000ULL;
        core->state.fpr[11] = 0x3FF0000004000000ULL; // 1 + 2^-26
        core->state.fpr_ps1[11] = 0x3FF0000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[2] == 0xC022000000000000ULL); // -9.0
    assert(compiled.state.fpr_ps1[2] == 0x402A000000000000ULL); // 13.0
    assert(compiled.state.fpr[6] == 0xC01C000000000000ULL); // -7.0
    assert(compiled.state.fpr_ps1[6] == 0xC02C000000000000ULL); // -14.0
    assert(compiled.state.fpr[8] == 0xBFF0000000000000ULL);
    assert(compiled.state.fpr_ps1[8] == 0xBFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_nmadd");
        assert(history.has_paired_fp_state && history.paired_fp_negative_multiply_add);
        assert(!history.paired_fp_negative_multiply_subtract);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_source_c_register == expected.paired_fp_source_c_register);
        assert(history.paired_fp_source_c_ps0 == expected.paired_fp_source_c_ps0);
        assert(history.paired_fp_source_c_ps1 == expected.paired_fp_source_c_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
}

void recompiler_ps_nmsub_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array program{
        ps_nmsub_form(2U, 1U, 3U, 4U),
        ps_nmsub_form(6U, 5U, 6U, 7U), // destination aliases B
        ps_nmsub_form(8U, 9U, 10U, 11U), // 25-bit scalar operand rounding
    };
    static_assert(ps_nmsub_form(2U, 1U, 3U, 4U) == 0x1041193EU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        for (std::size_t index = 0U; index < program.size(); ++index)
        {
            core->memory.write32_be(start + static_cast<std::uint32_t>(index * 4U),
                program[index]);
        }
        core->state.cia = start;
        core->state.fpr.fill(0x3FF0000000000000ULL);
        core->state.fpr_ps1.fill(0xBFF0000000000000ULL);
        core->state.fpr[1] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[1] = 0xC008000000000000ULL; // -3.0
        core->state.fpr[3] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[3] = 0x4000000000000000ULL; // 2.0
        core->state.fpr[4] = 0x4010000000000000ULL; // 4.0
        core->state.fpr_ps1[4] = 0x4014000000000000ULL; // 5.0
        core->state.fpr[5] = 0x4000000000000000ULL; // 2.0
        core->state.fpr_ps1[5] = 0x4008000000000000ULL; // 3.0
        core->state.fpr[6] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[6] = 0x4000000000000000ULL; // 2.0
        core->state.fpr[7] = 0x4008000000000000ULL; // 3.0
        core->state.fpr_ps1[7] = 0x4010000000000000ULL; // 4.0
        core->state.fpr[9] = 0x3FF0000000000000ULL; // 1.0
        core->state.fpr_ps1[9] = 0x3FF0000000000000ULL;
        core->state.fpr[10] = 0x0000000000000000ULL;
        core->state.fpr_ps1[10] = 0x0000000000000000ULL;
        core->state.fpr[11] = 0x3FF0000004000000ULL; // 1 + 2^-26
        core->state.fpr_ps1[11] = 0x3FF0000000000000ULL;
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const auto original_gprs = compiled.state.gpr;
    const auto original_memory = compiled.memory.read32_be(start);
    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason);
    assert(actual.steps == oracle.steps && actual.steps == program.size());
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr == original_gprs);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(start) == original_memory);
    assert(compiled.state.fpr[2] == 0xC01C000000000000ULL); // -7.0
    assert(compiled.state.fpr_ps1[2] == 0x4031000000000000ULL); // 17.0
    assert(compiled.state.fpr[6] == 0xC014000000000000ULL); // -5.0
    assert(compiled.state.fpr_ps1[6] == 0xC024000000000000ULL); // -10.0
    assert(compiled.state.fpr[8] == 0xBFF0000000000000ULL);
    assert(compiled.state.fpr_ps1[8] == 0xBFF0000000000000ULL);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == program.size());
    for (std::size_t index = 0U; index < program.size(); ++index)
    {
        const InstructionHistoryEntry& expected = oracle.instruction_history[index];
        const InstructionHistoryEntry& history = actual.instruction_history[index];
        assert(history.opcode_name == "ps_nmsub");
        assert(history.has_paired_fp_state && history.paired_fp_negative_multiply_subtract);
        assert(!history.paired_fp_negative_multiply_add);
        assert(history.paired_fp_source_a_register == expected.paired_fp_source_a_register);
        assert(history.paired_fp_source_a_ps0 == expected.paired_fp_source_a_ps0);
        assert(history.paired_fp_source_a_ps1 == expected.paired_fp_source_a_ps1);
        assert(history.paired_fp_source_b_register == expected.paired_fp_source_b_register);
        assert(history.paired_fp_source_b_ps0 == expected.paired_fp_source_b_ps0);
        assert(history.paired_fp_source_b_ps1 == expected.paired_fp_source_b_ps1);
        assert(history.paired_fp_source_c_register == expected.paired_fp_source_c_register);
        assert(history.paired_fp_source_c_ps0 == expected.paired_fp_source_c_ps0);
        assert(history.paired_fp_source_c_ps1 == expected.paired_fp_source_c_ps1);
        assert(history.paired_fp_destination_register ==
            expected.paired_fp_destination_register);
        assert(history.paired_fp_destination_ps0 == expected.paired_fp_destination_ps0);
        assert(history.paired_fp_destination_ps1 == expected.paired_fp_destination_ps1);
    }
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

void recompiler_stfdx_differential_tests()
{
    struct Case
    {
        std::uint8_t fp_source;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint64_t fp_value;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{1U, 6U, 7U, 0x300U, 8U, 0x3FF199999999999AULL},
        Case{2U, 6U, 4U, 0x300U, 16U, 0x8000000000000000ULL},
        Case{3U, 6U, 6U, 0x100U, 0x100U, 0x7FF0123456789ABCULL},
        Case{4U, 0U, 5U, 0xDEADBEEFU, 0x300U, 0xFFFFFFFFFFFFFFFFULL},
        Case{5U, 12U, 13U, 0xFFFFFFFCU, 8U, 0x0000000000000001ULL},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address =
            (test.base_register == 0U ? 0U : test.base) + test.index;
        const std::uint32_t instruction = stfdx_form(
            test.fp_source, test.base_register, test.index_register);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xDEADBEEF01234567ULL);
            core->state.fpr[test.fp_source] = test.fp_value;
            core->state.fpr_ps1.fill(0x123456789ABCDEF0ULL);
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
        assert(compiled.memory.read64_be(address) == test.fp_value);
        assert(interpreted.memory.read64_be(address) == test.fp_value);
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
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfdx");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register != 0U)
        {
            assert(history.source_registers[0] == test.base_register);
            assert(history.source_values[0] == test.base);
        }
        const std::uint8_t index_source = test.base_register == 0U ? 0U : 1U;
        assert(history.source_registers[index_source] == test.index_register);
        assert(history.source_values[index_source] == test.index);
        assert(history.has_fp_source && history.fp_source_register == test.fp_source);
        assert(history.fp_source_value == test.fp_value);
        assert(history.has_stored_double_value &&
            history.stored_double_value == test.fp_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, stfdx_form(1U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.fpr[1] = 0x3FF0000000000000ULL;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 8 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.gpr[7] == 0U);
    assert(faulting.state.fpr[1] == 0x3FF0000000000000ULL);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stfdux_differential_tests()
{
    struct Case
    {
        std::uint8_t fp_source;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint64_t fp_value;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{1U, 6U, 7U, 0x300U, 8U, 0x3FF199999999999AULL},
        Case{2U, 6U, 4U, 0x300U, 16U, 0x8000000000000000ULL},
        Case{3U, 6U, 6U, 0x100U, 0x100U, 0x7FF0123456789ABCULL},
        Case{4U, 12U, 13U, 0xFFFFFFFCU, 8U, 0xFFFFFFFFFFFFFFFFULL},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address = test.base + test.index;
        const std::uint32_t instruction = stfdux_form(
            test.fp_source, test.base_register, test.index_register);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xDEADBEEF01234567ULL);
            core->state.fpr[test.fp_source] = test.fp_value;
            core->state.fpr_ps1.fill(0x123456789ABCDEF0ULL);
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
        assert(compiled.memory.read64_be(address) == test.fp_value);
        assert(interpreted.memory.read64_be(address) == test.fp_value);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfdux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_fp_source && history.fp_source_register == test.fp_source);
        assert(history.fp_source_value == test.fp_value);
        assert(history.has_stored_double_value &&
            history.stored_double_value == test.fp_value);
        assert(history.has_destination &&
            history.destination_register == test.base_register &&
            history.destination_value == address);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, stfdux_form(1U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.fpr[1] = 0x3FF0000000000000ULL;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 8 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.fpr[1] == 0x3FF0000000000000ULL);
    assert(faulting.state.cia == start);
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

void recompiler_stbu_differential_tests()
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
    constexpr std::array<Case, 3U> cases{{
        {6U, 5U, 0x400U, -1, 0x123456ABU},
        {5U, 5U, 0x400U, 4, 0x00000400U}, // Store old RA, then update RA.
        {31U, 2U, 0x800U, 0, 0xFFFFFFFFU}}};

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = (39U << 26U) |
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
        assert(compiled.memory.read8(expected_address) ==
            interpreted.memory.read8(expected_address));
        assert(compiled.memory.read8(expected_address) ==
            static_cast<std::uint8_t>(test.stored_value));
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base] == expected_address);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stbu");
        assert(history.has_effective_address && history.effective_address == expected_address);
        assert(history.has_destination && history.destination_register == test.base);
        assert(history.destination_value == expected_address);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, (39U << 26U) | (6U << 21U) | (5U << 16U));
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

void recompiler_rlwnm_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 5U> inputs{
        0U, 1U, 0x80000000U, 0xA5C39E71U, 0xFFFFFFFFU};
    constexpr std::array<std::uint32_t, 6U> shifts{
        0U, 1U, 31U, 32U, 33U, 0xFFFFFFFFU};
    constexpr std::array<std::array<std::uint8_t, 2U>, 4U> masks{{
        {{0U, 31U}}, {{0U, 0U}}, {{8U, 15U}}, {{28U, 3U}}}};

    for (const std::uint32_t input : inputs)
    {
        for (const std::uint32_t shift_value : shifts)
        {
            for (const auto [begin, end] : masks)
            {
                for (const bool record : {false, true})
                {
                    for (const std::uint8_t alias_case : {0U, 1U, 2U, 3U})
                    {
                        constexpr std::uint8_t source_register = 3U;
                        constexpr std::uint8_t shift_register = 5U;
                        const std::uint8_t destination = alias_case == 1U
                            ? source_register
                            : alias_case == 2U
                                ? shift_register
                                : alias_case == 3U ? source_register : 7U;
                        const std::uint8_t actual_shift_register = alias_case == 3U
                            ? source_register
                            : shift_register;
                        const std::uint32_t instruction = rlwnm_form(
                            destination, source_register, actual_shift_register,
                            begin, end, record);

                        EspressoCore interpreted(0x1000U);
                        EspressoCore compiled(0x1000U);
                        for (EspressoCore* machine : {&interpreted, &compiled})
                        {
                            machine->memory.write32_be(start, instruction);
                            machine->state.cia = start;
                            machine->state.gpr.fill(0x6A5B4C3DU);
                            machine->state.gpr[source_register] = input;
                            machine->state.gpr[actual_shift_register] =
                                alias_case == 3U ? input : shift_value;
                            machine->state.gpr[destination] =
                                alias_case == 1U || alias_case == 3U
                                    ? input
                                    : alias_case == 2U ? shift_value : 0xDEADBEEFU;
                            machine->state.cr = 0xA5C39E71U;
                            machine->state.xer = 0x80000000U;
                            machine->state.lr = 0x12345678U;
                            machine->state.ctr = 0x87654321U;
                            machine->state.fpscr = 0x13579BDFU;
                        }

                        const RunResult oracle = interpreted.run(1U);
                        Recompiler recompiler;
                        const RunResult actual = recompiler.run(compiled, 1U);
                        assert(actual.steps == oracle.steps && actual.steps == 1U);
                        assert(actual.reason == oracle.reason);
                        assert(actual.cia == oracle.cia);
                        assert(compiled.state.gpr == interpreted.state.gpr);
                        assert(compiled.state.cr == interpreted.state.cr);
                        assert(compiled.state.xer == interpreted.state.xer);
                        assert(compiled.state.lr == interpreted.state.lr);
                        assert(compiled.state.ctr == interpreted.state.ctr);
                        assert(compiled.state.fpscr == interpreted.state.fpscr);
                        assert(recompiler.statistics().native_instructions == 1U);
                        assert(recompiler.statistics().interpreter_fallbacks == 0U);
                        assert(!actual.instruction_history.empty());
                        assert(actual.instruction_history.back().opcode_name ==
                            (record ? "rlwnm." : "rlwnm"));
                    }
                }
            }
        }
    }
}

void recompiler_andis_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 5U> inputs{
        0U, 0xFFFFFFFFU, 0xA1B2C3D4U, 0x8000FFFFU, 0x7FFFFFFFU};
    constexpr std::array<std::uint16_t, 5U> immediates{
        0U, 1U, 0x00FFU, 0x8000U, 0xFFFFU};

    for (const std::uint32_t input : inputs)
    {
        for (const std::uint16_t immediate : immediates)
        {
            for (const bool alias : {false, true})
            {
                for (const bool summary_overflow : {false, true})
                {
                    constexpr std::uint8_t source = 4U;
                    const std::uint8_t destination = alias ? source : 7U;
                    const std::uint32_t instruction =
                        andis_form(source, destination, immediate);
                    EspressoCore interpreted(0x1000U);
                    EspressoCore compiled(0x1000U);
                    for (EspressoCore* machine : {&interpreted, &compiled})
                    {
                        machine->memory.write32_be(start, instruction);
                        machine->state.cia = start;
                        machine->state.gpr.fill(0x6A5B4C3DU);
                        machine->state.gpr[source] = input;
                        if (!alias)
                        {
                            machine->state.gpr[destination] = 0xDEADBEEFU;
                        }
                        machine->state.cr = 0xA5C36E91U;
                        machine->state.xer = summary_overflow ? 0x80000000U : 0U;
                        machine->state.lr = 0x12345678U;
                        machine->state.ctr = 0x87654321U;
                        machine->state.fpscr = 0x13579BDFU;
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
                    assert(recompiler.statistics().native_instructions == 1U);
                    assert(recompiler.statistics().interpreter_fallbacks == 0U);
                    assert(!actual.instruction_history.empty());
                    const auto& history = actual.instruction_history.back();
                    assert(history.opcode_name == "andis.");
                    assert(history.has_immediate && history.immediate == immediate);
                    assert(history.immediate_hex);
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

void recompiler_lfdx_differential_tests()
{
    struct Case
    {
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint64_t memory_bits;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array cases{
        Case{6U, 4U, 5U, 0x300U, 8U, 0x3FF0000000000000ULL},
        Case{7U, 6U, 6U, 0x200U, 0x200U, 0x8000000000000000ULL}, // RA aliases RB
        Case{8U, 0U, 0U, 0xDEADBEEFU, 0x320U, 0x7FF8123456789ABCULL}, // RA=0, RB=GPR0
        Case{9U, 12U, 13U, 0xFFFFFF00U, 0x500U, 0x0000000000000001ULL}, // wraps
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address =
            (test.base_register == 0U ? 0U : test.base) + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfdx_form(
                test.destination, test.base_register, test.index_register));
            core->memory.write64_be(address, test.memory_bits);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xFEDCBA9876543210ULL);
            core->state.fpr[test.destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[test.destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x12345678U;
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
        assert(compiled.state.fpr[test.destination] == test.memory_bits);
        assert(compiled.state.fpr_ps1[test.destination] == ps1_sentinel);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lfdx");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register != 0U)
        {
            assert(history.source_registers[0] == test.base_register);
            assert(history.source_values[0] == test.base);
        }
        const std::uint8_t index_source = test.base_register == 0U ? 0U : 1U;
        assert(history.source_registers[index_source] == test.index_register);
        assert(history.source_values[index_source] == test.index);
        assert(history.has_fp_destination);
        assert(history.fp_destination_register == test.destination);
        assert(history.fp_destination_value == test.memory_bits);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfdx_form(6U, 4U, 5U));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.gpr[5] = 8U;
    faulting.state.fpr[6] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[6] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 8 byte(s) at guest address 0x00001008") !=
        std::string::npos);
    assert(faulting.state.fpr[6] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[6] == ps1_sentinel);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lfdux_differential_tests()
{
    struct Case
    {
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint64_t memory_bits;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array cases{
        Case{6U, 4U, 5U, 0x300U, 8U, 0x3FF0000000000000ULL},
        Case{7U, 6U, 6U, 0x200U, 0x200U, 0x8000000000000000ULL}, // RA aliases RB
        Case{8U, 12U, 13U, 0xFFFFFF00U, 0x500U, 0x7FF8123456789ABCULL}, // wraps
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfdux_form(
                test.destination, test.base_register, test.index_register));
            core->memory.write64_be(address, test.memory_bits);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xFEDCBA9876543210ULL);
            core->state.fpr[test.destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[test.destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x12345678U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.fpr[test.destination] == test.memory_bits);
        assert(compiled.state.fpr_ps1[test.destination] == ps1_sentinel);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lfdux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_fp_destination);
        assert(history.fp_destination_register == test.destination);
        assert(history.fp_destination_value == test.memory_bits);
        assert(history.has_destination &&
            history.destination_register == test.base_register &&
            history.destination_value == address);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfdux_form(6U, 4U, 5U));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.gpr[5] = 8U;
    faulting.state.fpr[6] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[6] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 8 byte(s) at guest address 0x00001008") !=
        std::string::npos);
    assert(faulting.state.gpr[4] == 0x1000U);
    assert(faulting.state.fpr[6] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[6] == ps1_sentinel);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lfdu_differential_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::uint64_t loaded_value = 0xFFF8123456789ABCULL;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(start, lfdu_form(5U, 4U, 0x18));
        core->memory.write64_be(0x218U, loaded_value);
        core->state.cia = start;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[0] = 0xDEADBEEFU;
        core->state.fpr[5] = 0xA5A5A5A5A5A5A5A5ULL;
        core->state.fpr_ps1[5] = ps1_sentinel;
        core->state.cr = 0x76543210U;
        core->state.xer = 0xE0000000U;
        core->state.fpscr = 0x12345678U;
    }

    const RunResult oracle = interpreted.run(1U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 1U);
    assert(actual.steps == oracle.steps && actual.steps == 1U);
    assert(actual.reason == oracle.reason);
    assert(actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.gpr[4] == 0x218U);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.fpr[5] == loaded_value);
    assert(compiled.state.fpr_ps1[5] == ps1_sentinel);
    assert(recompiler.statistics().native_instructions == 1U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const std::string history = format_instruction_history(actual);
    assert(history.find("lfdu") != std::string::npos);
    assert(history.find("r4=0x00000218") != std::string::npos);

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfdu_form(5U, 4U, 8));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.fpr[5] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[5] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 8 byte(s) at guest address 0x00001008") !=
           std::string::npos);
    assert(faulting.state.gpr[4] == 0x1000U);
    assert(faulting.state.fpr[5] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[5] == ps1_sentinel);
    assert(!fault.instruction_history.empty());
    assert(!fault.instruction_history.back().completed);
    assert(fault.instruction_history.back().opcode_name == "lfdu");
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

void recompiler_lfsx_differential_tests()
{
    struct Case
    {
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t memory_bits;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array cases{
        Case{6U, 4U, 5U, 0x300U, 8U, 0x3FC00000U},
        Case{7U, 6U, 6U, 0x200U, 0x200U, 0x80000000U}, // RA aliases RB
        Case{8U, 0U, 0U, 0xDEADBEEFU, 0x320U, 0x7FC12345U}, // RA=0, RB=GPR0
        Case{9U, 12U, 13U, 0xFFFFFF00U, 0x500U, 0xFF812345U}, // wraps to 0x400
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address =
            (test.base_register == 0U ? 0U : test.base) + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfsx_form(
                test.destination, test.base_register, test.index_register));
            core->memory.write32_be(address, test.memory_bits);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xFEDCBA9876543210ULL);
            core->state.fpr[test.destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[test.destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x12345678U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const double single_value = std::bit_cast<float>(test.memory_bits);
        const std::uint64_t expected = std::bit_cast<std::uint64_t>(single_value);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        if (test.base_register != 0U)
        {
            assert(compiled.state.gpr[test.base_register] == test.base);
        }
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.fpr[test.destination] == expected);
        assert(compiled.state.fpr_ps1[test.destination] == expected);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lfsx");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register != 0U)
        {
            assert(history.source_registers[0] == test.base_register);
            assert(history.source_values[0] == test.base);
        }
        const std::uint8_t index_source = test.base_register == 0U ? 0U : 1U;
        assert(history.source_registers[index_source] == test.index_register);
        assert(history.source_values[index_source] == test.index);
        assert(history.has_fp_destination);
        assert(history.fp_destination_register == test.destination);
        assert(history.fp_destination_value == expected);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfsx_form(6U, 4U, 5U));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.gpr[5] = 8U;
    faulting.state.fpr[6] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[6] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s) at guest address 0x00001008") !=
        std::string::npos);
    assert(faulting.state.fpr[6] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[6] == ps1_sentinel);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lfsux_differential_tests()
{
    struct Case
    {
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t memory_bits;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint64_t ps1_sentinel = 0xDEADBEEF01234567ULL;
    constexpr std::array cases{
        Case{6U, 4U, 5U, 0x300U, 8U, 0x3FC00000U},
        Case{7U, 6U, 6U, 0x200U, 0x200U, 0x80000000U}, // RA aliases RB
        Case{8U, 12U, 13U, 0xFFFFFF00U, 0x500U, 0x7FC12345U}, // wraps to 0x400
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, lfsux_form(
                test.destination, test.base_register, test.index_register));
            core->memory.write32_be(address, test.memory_bits);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xFEDCBA9876543210ULL);
            core->state.fpr[test.destination] = 0xA5A5A5A5A5A5A5A5ULL;
            core->state.fpr_ps1[test.destination] = ps1_sentinel;
            core->state.cr = 0x76543210U;
            core->state.xer = 0xE0000000U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x12345678U;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const double single_value = std::bit_cast<float>(test.memory_bits);
        const std::uint64_t expected = std::bit_cast<std::uint64_t>(single_value);
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.fpr[test.destination] == expected);
        assert(compiled.state.fpr_ps1[test.destination] == expected);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lfsux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_fp_destination);
        assert(history.fp_destination_register == test.destination);
        assert(history.fp_destination_value == expected);
        assert(history.has_destination &&
            history.destination_register == test.base_register &&
            history.destination_value == address);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lfsux_form(6U, 4U, 5U));
    faulting.state.cia = start;
    faulting.state.gpr[4] = 0x1000U;
    faulting.state.gpr[5] = 8U;
    faulting.state.fpr[6] = 0xA5A5A5A5A5A5A5A5ULL;
    faulting.state.fpr_ps1[6] = ps1_sentinel;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s) at guest address 0x00001008") !=
        std::string::npos);
    assert(faulting.state.gpr[4] == 0x1000U);
    assert(faulting.state.fpr[6] == 0xA5A5A5A5A5A5A5A5ULL);
    assert(faulting.state.fpr_ps1[6] == ps1_sentinel);
    assert(!fault.instruction_history.back().completed);
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

void recompiler_eieio_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t eieio = 0x7C0006ACU;
    const DecodedInstruction decoded = decode(eieio);
    assert(decoded.opcode == Opcode::enforce_in_order_io);

    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(start, eieio);
        core->memory.write32_be(start + 4U, 0x3860002AU);
        core->memory.write32_be(0x300U, 0xA55AA55AU);
        core->state.cia = start;
        core->state.gpr.fill(0xA5A5A5A5U);
        core->state.fpr.fill(0x123456789ABCDEF0ULL);
        core->state.fpr_ps1.fill(0xDEADBEEF01234567ULL);
        core->state.cr = 0xA5C39E71U;
        core->state.xer = 0x87654321U;
        core->state.lr = 0xABCDEF00U;
        core->state.ctr = 0xFEDCBA98U;
        core->state.fpscr = 0x89ABCDEFU;
    }

    const RunResult oracle = interpreted.run(2U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 2U);
    assert(actual.reason == oracle.reason);
    assert(actual.steps == 2U && actual.steps == oracle.steps);
    assert(actual.cia == oracle.cia && actual.cia == start + 8U);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.gpr[3] == 42U);
    assert(compiled.memory.read32_be(0x300U) == 0xA55AA55AU);
    assert(recompiler.statistics().native_instructions == 2U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(actual.instruction_history.size() == 2U);
    assert(actual.instruction_history[0].opcode_name == "eieio");
    assert(actual.instruction_history[1].opcode_name == "addi");
}

void recompiler_memory_synchronize_differential_test()
{
    struct FenceCase
    {
        std::uint32_t instruction;
        const char* mnemonic;
        Opcode opcode;
    };
    constexpr std::array cases{
        FenceCase{0x7C0004ACU, "sync", Opcode::memory_synchronize},
        FenceCase{0x7C2004ACU, "lwsync", Opcode::memory_synchronize_lightweight},
    };
    constexpr std::uint32_t start = 0x100U;

    for (const FenceCase& test : cases)
    {
        assert(decode(test.instruction).opcode == test.opcode);
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, 0x90830000U); // stw r4,0(r3)
            core->memory.write32_be(start + 4U, test.instruction);
            core->memory.write32_be(start + 8U, 0x80A30000U); // lwz r5,0(r3)
            core->state.cia = start;
            core->state.gpr[3] = 0x300U;
            core->state.gpr[4] = 0xCAFEBABEU;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(3U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 3U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 3U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[5] == 0xCAFEBABEU);
        assert(compiled.memory.read32_be(0x300U) == 0xCAFEBABEU);
        assert(recompiler.statistics().native_instructions == 3U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls > 0U);
        assert(actual.instruction_history.size() == 3U);
        assert(actual.instruction_history[1].opcode_name == test.mnemonic);
    }
}

void recompiler_move_from_time_base_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array<std::uint32_t, 3U> program{
        0x7C6C42E6U, // mftb r3
        0x7C8D42E6U, // mftbu r4
        0x7CAC42E6U, // mftb r5
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
        core->state.gpr.fill(0xA5A5A5A5U);
        core->state.gpr[3] = 0xDEADBEEFU;
        core->state.gpr[4] = 0xCAFEBABEU;
        core->state.gpr[5] = 0x13579BDFU;
        core->state.cr = 0x12345678U;
        core->state.xer = 0xE00000A5U;
        core->state.lr = 0x10203040U;
        core->state.ctr = 0x50607080U;
        core->state.fpscr = 0xCAFEBABEU;
    }

    const RunResult oracle = interpreted.run(program.size());
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, program.size());
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == program.size() && actual.cia == oracle.cia);
    assert(compiled.state.gpr[3] != 0xDEADBEEFU);
    assert(compiled.state.gpr[4] == 0U);
    assert(compiled.state.gpr[5] >= compiled.state.gpr[3]);
    assert(interpreted.state.gpr[3] != 0xDEADBEEFU);
    assert(interpreted.state.gpr[4] == 0U);
    assert(interpreted.state.gpr[5] >= interpreted.state.gpr[3]);
    for (std::size_t reg = 0U; reg < compiled.state.gpr.size(); ++reg)
    {
        if (reg != 3U && reg != 4U && reg != 5U)
        {
            assert(compiled.state.gpr[reg] == interpreted.state.gpr[reg]);
        }
    }
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(recompiler.statistics().memory_slow_path_calls == 0U);
    assert(actual.instruction_history.size() == program.size());
    assert(actual.instruction_history[0].opcode_name == "mftb");
    assert(actual.instruction_history[0].destination_value == compiled.state.gpr[3]);
    assert(actual.instruction_history[1].opcode_name == "mftbu");
    assert(actual.instruction_history[1].destination_value == compiled.state.gpr[4]);
    assert(actual.instruction_history[2].opcode_name == "mftb");
    assert(actual.instruction_history[2].destination_value == compiled.state.gpr[5]);
}

void recompiler_dcbt_differential_test()
{
    struct Case
    {
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t expected_address;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t dcbt_r4_r0 = x_form(0U, 4U, 0U, 278U);
    constexpr std::uint32_t dcbt_r0_r0 = x_form(0U, 0U, 0U, 278U);
    static_assert(dcbt_r4_r0 == 0x7C04022CU);
    constexpr std::array cases{
        // The target is unmapped: dcbt is a non-faulting hint, not a load.
        Case{4U, 0U, 0x2000U, 0x20U, 0x2020U},
        // rA=0 is literal zero while rB=0 reads the actual GPR0.
        Case{0U, 0U, 0xDEADBEEFU, 0x2020U, 0x2020U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, test.base_register == 0U
                ? dcbt_r0_r0 : dcbt_r4_r0);
            core->memory.write32_be(start + 4U, 0x38C0002AU); // addi r6,r0,42
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[0] = test.index_register == 0U
                ? test.index : 0xFACEB00CU;
            if (test.base_register != 0U)
                core->state.gpr[test.base_register] = test.base;
            if (test.index_register != 0U)
                core->state.gpr[test.index_register] = test.index;
            core->state.cr = 0x91827364U;
            core->state.xer = 0xA0000042U;
            core->state.lr = 0x10203040U;
            core->state.ctr = 0x50607080U;
            core->state.fpscr = 0xCAFEBABEU;
        }

        const RunResult oracle = interpreted.run(2U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 2U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 2U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[6] == 42U);
        assert(recompiler.statistics().native_instructions == 2U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        assert(actual.instruction_history.size() == 2U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.opcode_name == "dcbt" && history.completed);
        assert(history.has_effective_address &&
            history.effective_address == test.expected_address);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register == 0U)
        {
            assert(history.source_registers[0] == 0U);
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
}

void recompiler_dcbtst_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t dcbtst_r4_r0 = x_form(0U, 4U, 0U, 246U);
    static_assert(dcbtst_r4_r0 == 0x7C0401ECU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(start, dcbtst_r4_r0);
        core->memory.write32_be(start + 4U, 0x38C0002AU); // addi r6,r0,42
        core->state.cia = start;
        core->state.gpr.fill(0xA5A5A5A5U);
        core->state.gpr[0] = 0x20U;
        core->state.gpr[4] = 0x6000U;
        core->state.cr = 0x91827364U;
        core->state.xer = 0xA0000042U;
        core->state.lr = 0x10203040U;
        core->state.ctr = 0x50607080U;
        core->state.fpscr = 0xCAFEBABEU;
    }

    const RunResult oracle = interpreted.run(2U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 2U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 2U && actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.gpr[6] == 42U);
    assert(recompiler.statistics().native_instructions == 2U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(recompiler.statistics().memory_slow_path_calls == 0U);
    assert(actual.instruction_history.size() == 2U);
    const InstructionHistoryEntry& history = actual.instruction_history.front();
    assert(history.opcode_name == "dcbtst" && history.completed);
    assert(history.has_effective_address && history.effective_address == 0x6020U);
    assert(history.source_count == 2U);
    assert(history.source_registers[0] == 4U && history.source_values[0] == 0x6000U);
    assert(history.source_registers[1] == 0U && history.source_values[1] == 0x20U);
    assert(!history.has_destination);
}

void recompiler_icbi_differential_test()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t icbi_r4_r0 = x_form(0U, 4U, 0U, 982U);
    static_assert(icbi_r4_r0 == 0x7C0407ACU);
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.write32_be(start, icbi_r4_r0);
        core->memory.write32_be(start + 4U, 0x38C0002AU); // addi r6,r0,42
        core->state.cia = start;
        core->state.gpr.fill(0xA5A5A5A5U);
        core->state.gpr[0] = 0x20U;
        core->state.gpr[4] = 0x6000U;
        core->state.cr = 0x91827364U;
        core->state.xer = 0xA0000042U;
        core->state.lr = 0x10203040U;
        core->state.ctr = 0x50607080U;
        core->state.fpscr = 0xCAFEBABEU;
    }

    const RunResult oracle = interpreted.run(2U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 2U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 2U && actual.cia == oracle.cia);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr);
    assert(compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.lr == interpreted.state.lr);
    assert(compiled.state.ctr == interpreted.state.ctr);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.gpr[6] == 42U);
    assert(recompiler.statistics().native_instructions == 2U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    assert(recompiler.statistics().memory_slow_path_calls == 0U);
    assert(actual.instruction_history.size() == 2U);
    const InstructionHistoryEntry& history = actual.instruction_history.front();
    assert(history.opcode_name == "icbi" && history.completed);
    assert(history.has_effective_address && history.effective_address == 0x6020U);
    assert(history.source_count == 2U);
    assert(history.source_registers[0] == 4U && history.source_values[0] == 0x6000U);
    assert(history.source_registers[1] == 0U && history.source_values[1] == 0x20U);
    assert(!history.has_destination);

    // Guest writes publish code-page generations, so the next dispatch after
    // icbi must not reuse a previously translated version of the target.
    EspressoCore self_modified(0x1000U);
    self_modified.memory.write32_be(0x100U, 0x90850000U); // stw r4,0(r5)
    self_modified.memory.write32_be(0x104U, x_form(0U, 5U, 0U, 982U)); // icbi r5,r0
    self_modified.memory.write32_be(0x108U, 0x480000F8U); // b 0x200
    self_modified.memory.write32_be(0x200U, 0x38C00001U); // addi r6,r0,1
    self_modified.state.cia = 0x200U;
    Recompiler coherent_code_recompiler;
    assert(coherent_code_recompiler.run(self_modified, 1U).steps == 1U);
    assert(self_modified.state.gpr[6] == 1U);

    self_modified.state.gpr[0] = 0U;
    self_modified.state.gpr[4] = 0x38C00002U; // replacement: addi r6,r0,2
    self_modified.state.gpr[5] = 0x200U;
    self_modified.state.cia = 0x100U;
    const RunResult writer = coherent_code_recompiler.run(self_modified, 3U);
    assert(writer.steps == 3U && self_modified.state.cia == 0x200U);
    assert(self_modified.memory.read32_be(0x200U) == 0x38C00002U);
    assert(coherent_code_recompiler.run(self_modified, 1U).steps == 1U);
    assert(self_modified.state.gpr[6] == 2U);
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

    // If bl .+4 ends a compiled block, it must still publish the fallthrough
    // CIA before the dispatcher handles the following unsupported word.
    EspressoCore ending_link(0x1000U);
    EspressoCore ending_link_interpreted(0x1000U);
    for (EspressoCore* candidate : {&ending_link, &ending_link_interpreted})
    {
        candidate->memory.write32_be(0x100U, 0x48000005U); // bl .+4
        candidate->memory.write32_be(0x104U, 0xFFFFFFFFU); // unsupported
        candidate->state.cia = 0x100U;
    }
    const RunResult ending_link_oracle = ending_link_interpreted.run(2U);
    Recompiler ending_link_recompiler;
    const RunResult ending_link_result = ending_link_recompiler.run(ending_link, 2U);
    assert(ending_link_result.reason == ending_link_oracle.reason);
    assert(ending_link_result.reason == StopReason::unsupported_instruction);
    assert(ending_link_result.steps == ending_link_oracle.steps);
    assert(ending_link_result.cia == 0x104U);
    assert(ending_link.state.cia == 0x104U);
    assert(ending_link.state.lr == 0x104U);

    // A link branch to its own fallthrough is commonly used to capture the
    // guest PC. Keep it in the same native block as the following mflr.
    EspressoCore pc_relative(0x1000U);
    EspressoCore pc_relative_interpreted(0x1000U);
    for (EspressoCore* candidate : {&pc_relative, &pc_relative_interpreted})
    {
        candidate->memory.write32_be(0x100U, 0x48000005U); // bl 0x104
        candidate->memory.write32_be(0x104U, 0x7C6802A6U); // mflr r3
        candidate->memory.write32_be(0x108U, 0x4BFFFFF8U); // b 0x100
        candidate->state.cia = 0x100U;
    }
    const RunResult pc_oracle = pc_relative_interpreted.run(6U);
    Recompiler pc_recompiler;
    const RunResult pc_result = pc_recompiler.run(pc_relative, 6U);
    assert(pc_result.reason == pc_oracle.reason);
    assert(pc_result.steps == 6U && pc_result.steps == pc_oracle.steps);
    assert(pc_relative.state.cia == pc_relative_interpreted.state.cia);
    assert(pc_relative.state.lr == 0x104U);
    assert(pc_relative.state.gpr[3] == 0x104U);
    assert(pc_relative.state.gpr == pc_relative_interpreted.state.gpr);
    assert(pc_recompiler.statistics().native_instructions == 6U);
    assert(pc_recompiler.statistics().compiled_blocks == 1U);
    assert(pc_recompiler.statistics().dispatcher_entries == 2U);
    assert(pc_recompiler.statistics().interpreter_fallbacks == 0U);

    // Conditional fallthrough branches may also stay in the block; their
    // architectural CTR effects still occur even though either path reaches
    // the same next instruction.
    EspressoCore ctr_fallthrough(0x1000U);
    EspressoCore ctr_fallthrough_interpreted(0x1000U);
    for (EspressoCore* candidate : {&ctr_fallthrough,
                                    &ctr_fallthrough_interpreted})
    {
        candidate->memory.write32_be(0x100U, 0x42000004U); // bdnz .+4
        candidate->memory.write32_be(0x104U, 0x38600007U); // addi r3,r0,7
        candidate->memory.write32_be(0x108U, 0x4BFFFFF8U); // b 0x100
        candidate->state.cia = 0x100U;
        candidate->state.ctr = 1U;
    }
    const RunResult ctr_oracle = ctr_fallthrough_interpreted.run(6U);
    Recompiler ctr_recompiler;
    const RunResult ctr_result = ctr_recompiler.run(ctr_fallthrough, 6U);
    assert(ctr_result.reason == ctr_oracle.reason);
    assert(ctr_result.steps == 6U && ctr_result.steps == ctr_oracle.steps);
    assert(ctr_fallthrough.state.cia == ctr_fallthrough_interpreted.state.cia);
    assert(ctr_fallthrough.state.ctr == 0xFFFFFFFFU);
    assert(ctr_fallthrough.state.ctr == ctr_fallthrough_interpreted.state.ctr);
    assert(ctr_fallthrough.state.gpr == ctr_fallthrough_interpreted.state.gpr);
    assert(ctr_recompiler.statistics().native_instructions == 6U);
    assert(ctr_recompiler.statistics().compiled_blocks == 1U);
    assert(ctr_recompiler.statistics().dispatcher_entries == 2U);
    assert(ctr_recompiler.statistics().interpreter_fallbacks == 0U);

    // A CR-tested linked branch to its fallthrough has identical control flow
    // for either CR value, but LK must be written in both cases.
    for (const std::uint32_t cr : {0U, 0x20000000U})
    {
        EspressoCore linked_fallthrough(0x1000U);
        EspressoCore linked_fallthrough_interpreted(0x1000U);
        for (EspressoCore* candidate : {&linked_fallthrough,
                                        &linked_fallthrough_interpreted})
        {
            candidate->memory.write32_be(0x100U, 0x41820005U); // beq+ .+4
            candidate->memory.write32_be(0x104U, 0x38630001U); // addi r3,r3,1
            candidate->memory.write32_be(0x108U, 0x4BFFFFF8U); // b 0x100
            candidate->state.cia = 0x100U;
            candidate->state.cr = cr;
            candidate->state.ctr = 0x12345678U;
        }
        const RunResult linked_oracle = linked_fallthrough_interpreted.run(6U);
        Recompiler linked_recompiler;
        const RunResult linked_result = linked_recompiler.run(linked_fallthrough, 6U);
        assert(linked_result.reason == linked_oracle.reason);
        assert(linked_result.steps == 6U && linked_result.steps == linked_oracle.steps);
        assert(linked_fallthrough.state.cia == linked_fallthrough_interpreted.state.cia);
        assert(linked_fallthrough.state.lr == 0x104U);
        assert(linked_fallthrough.state.ctr == 0x12345678U);
        assert(linked_fallthrough.state.gpr == linked_fallthrough_interpreted.state.gpr);
        assert(linked_fallthrough.state.cr == linked_fallthrough_interpreted.state.cr);
        assert(linked_recompiler.statistics().compiled_blocks == 1U);
        assert(linked_recompiler.statistics().interpreter_fallbacks == 0U);
    }
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

    for (std::uint8_t bi = 0; bi < 32U; ++bi)
    {
        const std::uint32_t cr_mask = 1U << (31U - bi);
        for (const bool condition_sense : {false, true})
        {
            const std::uint32_t bo = 0x04U | (condition_sense ? 0x08U : 0U);
            const std::uint32_t word = 0x40000000U | (bo << 21U) |
                (static_cast<std::uint32_t>(bi) << 16U) | 0x8U;
            for (const bool bit_set : {false, true})
            {
                const std::uint32_t cr = bit_set ? cr_mask : 0U;
                EspressoCore interpreted(0x1000U);
                EspressoCore compiled(0x1000U);
                for (EspressoCore* candidate : {&interpreted, &compiled})
                {
                    candidate->memory.write32_be(0x100U, word);
                    candidate->state.cia = 0x100U;
                    candidate->state.cr = cr;
                    candidate->state.ctr = 0x89ABCDEFU;
                }
                const RunResult oracle = interpreted.run(1U);
                Recompiler recompiler;
                const RunResult actual = recompiler.run(compiled, 1U);
                const bool taken = bit_set == condition_sense;
                const std::uint32_t expected_cia = taken ? 0x108U : 0x104U;
                assert(oracle.cia == expected_cia && actual.cia == expected_cia);
                assert(compiled.state.cr == interpreted.state.cr);
                assert(compiled.state.ctr == interpreted.state.ctr);
                assert(compiled.state.lr == interpreted.state.lr);
            }
        }
    }
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
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
    assert(recompiler.statistics().memory_slow_path_calls == 1U);
}

void recompiler_direct_flat_halfword_load_tests()
{
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint8_t destination = 5U;
    constexpr std::uint8_t base_register = 6U;
    constexpr std::uint32_t instruction = (40U << 26U) |
        (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(base_register) << 16U) | 2U;
    constexpr std::array<std::uint16_t, 4> values{
        0x0000U, 0x7FFFU, 0x8000U, 0xFFFFU};

    for (const std::uint16_t value : values)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write16_be(0x302U, value);
            core->state.cia = start;
            core->state.gpr[base_register] = 0x300U;
            core->state.gpr[destination] = 0xDEADBEEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 1U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[destination] == value);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
    }

    // When code is in a mapped region, a one-byte flat backing store still
    // permits an in-bounds byte load but must send a halfword load to the
    // checked helper, which reports the missing second byte.
    constexpr std::uint32_t mapped_code = 0x1000U;
    constexpr std::uint8_t mapped_base = 6U;
    EspressoCore one_byte_memory(1U);
    one_byte_memory.memory.map_region(mapped_code, 0x1000U);
    const std::uint32_t lbz = (34U << 26U) |
        (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(mapped_base) << 16U);
    one_byte_memory.memory.write32_be(mapped_code, lbz);
    one_byte_memory.memory.write8(0U, 0xA5U);
    one_byte_memory.state.cia = mapped_code;
    one_byte_memory.state.gpr[mapped_base] = 0U;
    Recompiler byte_recompiler;
    const RunResult byte_result = byte_recompiler.run(one_byte_memory, 1U);
    assert(byte_result.reason == StopReason::instruction_limit);
    assert(one_byte_memory.state.gpr[destination] == 0xA5U);
    assert(byte_recompiler.statistics().native_instructions == 1U);
    assert(byte_recompiler.statistics().memory_slow_path_calls == 0U);

    EspressoCore short_halfword_memory(1U);
    short_halfword_memory.memory.map_region(mapped_code, 0x1000U);
    const std::uint32_t lhz = (40U << 26U) |
        (static_cast<std::uint32_t>(destination) << 21U) |
        (static_cast<std::uint32_t>(mapped_base) << 16U);
    short_halfword_memory.memory.write32_be(mapped_code, lhz);
    short_halfword_memory.memory.write8(0U, 0xA5U);
    short_halfword_memory.state.cia = mapped_code;
    short_halfword_memory.state.gpr[mapped_base] = 0U;
    short_halfword_memory.state.gpr[destination] = 0xDEADBEEFU;
    Recompiler halfword_recompiler;
    const RunResult halfword_result = halfword_recompiler.run(
        short_halfword_memory, 1U);
    assert(halfword_result.reason == StopReason::memory_fault);
    assert(halfword_result.detail.find("read 2 byte(s)") != std::string::npos);
    assert(short_halfword_memory.state.gpr[destination] == 0xDEADBEEFU);
    assert(halfword_recompiler.statistics().memory_slow_path_calls == 1U);
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lwzu");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    // Guest aliases are outside the flat backing range and must resolve
    // through GuestMemory rather than the direct flat-RAM pointer path.
    EspressoCore aliased(0x5000U);
    constexpr std::uint8_t alias_base_register = 6U;
    constexpr std::uint8_t alias_destination = 3U;
    constexpr std::uint32_t alias_address = 0x10000U;
    constexpr std::uint32_t alias_target = 0x2000U;
    constexpr std::uint32_t alias_size = 0x1000U;
    constexpr std::uint32_t alias_value = 0xA1B2C3D4U;
    aliased.memory.map_alias(alias_address, alias_target, alias_size);
    aliased.memory.write32_be(start, lwzu_form(alias_destination,
        alias_base_register, 4));
    aliased.memory.write32_be(alias_target + 4U, alias_value);
    aliased.state.cia = start;
    aliased.state.gpr[alias_base_register] = alias_address;
    aliased.state.gpr[alias_destination] = 0xDEADBEEFU;
    Recompiler alias_recompiler;
    const RunResult alias_result = alias_recompiler.run(aliased, 1U);
    assert(alias_result.reason == StopReason::instruction_limit);
    assert(aliased.state.gpr[alias_destination] == alias_value);
    assert(aliased.state.gpr[alias_base_register] == alias_address + 4U);
    assert(alias_recompiler.statistics().native_instructions == 1U);
    assert(alias_recompiler.statistics().interpreter_fallbacks == 0U);
    assert(alias_recompiler.statistics().memory_slow_path_calls == 1U);

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
    assert(recompiler.statistics().memory_slow_path_calls == 1U);
}

void recompiler_lwzux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t memory_value;
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0x00000000U, 3U, 6U, 7U},
        Case{0x300U, 8U, 0x7FFFFFFFU, 4U, 7U, 4U}, // RT aliases RB
        Case{0x100U, 0x100U, 0xA1B2C3D4U, 5U, 6U, 6U}, // RA aliases RB
        Case{0xFFFFFFFCU, 8U, 0x12345678U, 11U, 12U, 13U}, // wrap to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lwzux_form(
            test.destination, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write32_be(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            if (test.destination != test.index_register)
            {
                core->state.gpr[test.destination] = 0x87654321U;
            }
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lwzux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.map_region(0x2000U, 2U);
    faulting.memory.write16_be(0x2000U, 0x1234U);
    faulting.memory.write32_be(start, lwzux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.gpr[3] = 0xDEADBEEFU;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 4 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xDEADBEEFU);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lbzux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint8_t memory_value;
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0x00U, 3U, 6U, 7U},
        Case{0x300U, 8U, 0x80U, 4U, 7U, 4U}, // RT aliases RB
        Case{0x100U, 0x100U, 0xFFU, 5U, 6U, 6U}, // RA aliases RB
        Case{0xFFFFFFFCU, 8U, 0xA5U, 11U, 12U, 13U}, // wrap to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lbzux_form(
            test.destination, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write8(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            if (test.destination != test.index_register)
            {
                core->state.gpr[test.destination] = 0x87654321U;
            }
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lbzux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, lbzux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.gpr[3] = 0xDEADBEEFU;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 1 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xDEADBEEFU);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lhzux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint16_t memory_value;
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0x0000U, 3U, 6U, 7U},
        Case{0x300U, 8U, 0x8000U, 4U, 7U, 4U}, // RT aliases RB
        Case{0x100U, 0x100U, 0xFFFFU, 5U, 6U, 6U}, // RA aliases RB
        Case{0xFFFFFFFCU, 8U, 0xA5C3U, 11U, 12U, 13U}, // wrap to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lhzux_form(
            test.destination, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write16_be(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            if (test.destination != test.index_register)
            {
                core->state.gpr[test.destination] = 0x87654321U;
            }
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
        assert(recompiler.statistics().memory_slow_path_calls == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lhzux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.memory_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.map_region(0x2000U, 1U);
    faulting.memory.write8(0x2000U, 0x12U);
    faulting.memory.write32_be(start, lhzux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.gpr[3] = 0xDEADBEEFU;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 2 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xDEADBEEFU);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lhaux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint16_t memory_value;
        std::uint32_t expected_value;
        std::uint8_t destination;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0x0000U, 0x00000000U, 3U, 6U, 7U},
        Case{0x300U, 8U, 0x7FFFU, 0x00007FFFU, 4U, 7U, 4U}, // RT aliases RB
        Case{0x100U, 0x100U, 0x8000U, 0xFFFF8000U, 5U, 6U, 6U}, // RA aliases RB
        Case{0xFFFFFFFCU, 8U, 0xFFFFU, 0xFFFFFFFFU, 11U, 12U, 13U}, // wraps to 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lhaux_form(
            test.destination, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write16_be(address, test.memory_value);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            if (test.destination != test.index_register)
            {
                core->state.gpr[test.destination] = 0x87654321U;
            }
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
        assert(compiled.state.gpr[test.destination] == test.expected_value);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "lhaux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_destination &&
            history.destination_register == test.destination &&
            history.destination_value == test.expected_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.map_region(0x2000U, 1U);
    faulting.memory.write8(0x2000U, 0x80U);
    faulting.memory.write32_be(start, lhaux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.gpr[3] = 0xDEADBEEFU;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("read 2 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xDEADBEEFU);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stwux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t stored_value;
        std::uint8_t source_register;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0xA1B2C3D4U, 3U, 6U, 7U},
        Case{0x300U, 8U, 8U, 4U, 6U, 4U}, // RS aliases RB
        Case{0x300U, 8U, 0x300U, 6U, 6U, 7U}, // RS aliases RA
        Case{0x300U, 0x300U, 0x300U, 6U, 6U, 6U}, // all registers alias
        Case{0xFFFFFFFCU, 8U, 0x55667788U, 11U, 12U, 13U}, // wraps to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = stwux_form(
            test.source_register, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.gpr[test.source_register] = test.stored_value;
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
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.memory.read32_be(address) == test.stored_value);
        assert(interpreted.memory.read32_be(address) == test.stored_value);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stwux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 3U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.source_registers[2] == test.source_register);
        assert(history.source_values[2] == test.stored_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, stwux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[3] = 0xA1B2C3D4U;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xA1B2C3D4U);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stbux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t stored_value;
        std::uint8_t source_register;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0x12345678U, 3U, 6U, 7U},
        Case{0x300U, 8U, 8U, 4U, 6U, 4U}, // RS aliases RB
        Case{0x300U, 8U, 0x300U, 6U, 6U, 7U}, // RS aliases RA
        Case{0x300U, 0x300U, 0x300U, 6U, 6U, 6U}, // all registers alias
        Case{0xFFFFFFFCU, 8U, 0xAABBCCDDU, 11U, 12U, 13U}, // wraps to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = stbux_form(
            test.source_register, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.gpr[test.source_register] = test.stored_value;
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
        assert(compiled.state.gpr[test.base_register] == address);
        const std::uint8_t expected_byte = static_cast<std::uint8_t>(test.stored_value);
        assert(compiled.memory.read8(address) == expected_byte);
        assert(interpreted.memory.read8(address) == expected_byte);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stbux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 3U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.source_registers[2] == test.source_register);
        assert(history.source_values[2] == test.stored_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, stbux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[3] = 0xA1B2C3D4U;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 1 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xA1B2C3D4U);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_sthux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint32_t stored_value;
        std::uint8_t source_register;
        std::uint8_t base_register;
        std::uint8_t index_register;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 0xA1B2C3D4U, 3U, 6U, 7U},
        Case{0x300U, 8U, 8U, 4U, 6U, 4U}, // RS aliases RB
        Case{0x300U, 8U, 0x300U, 6U, 6U, 7U}, // RS aliases RA
        Case{0x300U, 0x300U, 0x300U, 6U, 6U, 6U}, // all registers alias
        Case{0xFFFFFFFCU, 8U, 0xAABBCCDDU, 11U, 12U, 13U}, // wraps to address 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = sthux_form(
            test.source_register, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.gpr[test.source_register] = test.stored_value;
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
        assert(compiled.state.gpr[test.base_register] == address);
        const std::uint16_t expected_halfword =
            static_cast<std::uint16_t>(test.stored_value);
        assert(compiled.memory.read16_be(address) == expected_halfword);
        assert(interpreted.memory.read16_be(address) == expected_halfword);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "sthux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 3U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.source_registers[2] == test.source_register);
        assert(history.source_values[2] == test.stored_value);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, sthux_form(3U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[3] = 0xA1B2C3D4U;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 2 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[3] == 0xA1B2C3D4U);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_stfsux_differential_tests()
{
    struct Case
    {
        std::uint32_t base;
        std::uint32_t index;
        std::uint8_t fp_source;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint64_t fp_value;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{0x300U, 4U, 1U, 6U, 7U, 0x3FF199999999999AULL}, // 1.1 rounds to f32
        Case{0x300U, 8U, 2U, 6U, 4U, 0x8000000000000000ULL}, // negative zero
        Case{0x100U, 0x100U, 3U, 6U, 6U, 0x7FF8123456789ABCULL}, // RA aliases RB
        Case{0xFFFFFFFCU, 8U, 4U, 12U, 13U, 0x3FF0000000000000ULL}, // wraps to 4
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = stfsux_form(
            test.fp_source, test.base_register, test.index_register);
        const std::uint32_t address = test.base + test.index;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xA5A5A5A5U);
            core->state.gpr[test.base_register] = test.base;
            core->state.gpr[test.index_register] = test.index;
            core->state.fpr.fill(0xDEADBEEF01234567ULL);
            core->state.fpr[test.fp_source] = test.fp_value;
            core->state.fpr_ps1.fill(0x123456789ABCDEF0ULL);
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        const std::uint32_t expected_bits = std::bit_cast<std::uint32_t>(
            static_cast<float>(std::bit_cast<double>(test.fp_value)));
        assert(actual.reason == oracle.reason);
        assert(actual.steps == oracle.steps && actual.steps == 1U);
        assert(actual.cia == oracle.cia);
        assert(compiled.memory.read32_be(address) == expected_bits);
        assert(interpreted.memory.read32_be(address) == expected_bits);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.gpr[test.base_register] == address);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(recompiler.statistics().memory_slow_path_calls == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stfsux");
        assert(history.has_effective_address && history.effective_address == address);
        assert(history.source_count == 2U);
        assert(history.source_registers[0] == test.base_register);
        assert(history.source_values[0] == test.base);
        assert(history.source_registers[1] == test.index_register);
        assert(history.source_values[1] == test.index);
        assert(history.has_fp_source && history.fp_source_register == test.fp_source);
        assert(history.fp_source_value == test.fp_value);
        assert(history.has_stored_single_value &&
            history.stored_single_value == expected_bits);
        assert(history.has_destination &&
            history.destination_register == test.base_register &&
            history.destination_value == address);
    }

    EspressoCore faulting(0x1000U);
    faulting.memory.write32_be(start, stfsux_form(1U, 6U, 7U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[7] = 0U;
    faulting.state.fpr[1] = 0x3FF0000000000000ULL;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(faulting.state.gpr[6] == 0x2000U);
    assert(faulting.state.fpr[1] == 0x3FF0000000000000ULL);
    assert(faulting.state.cia == start);
    assert(!fault.instruction_history.back().completed);
}

void recompiler_lmw_differential_tests()
{
    struct Case
    {
        std::uint8_t first_destination;
        std::uint8_t base_register;
        std::uint32_t base_value;
        std::int16_t displacement;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{29U, 6U, 0x300U, -4},
        Case{4U, 4U, 0x300U, 8}, // rA is overwritten; EA must use its old value.
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = lmw_form(
            test.first_destination, test.base_register, test.displacement);
        const std::uint32_t address = test.base_value +
            static_cast<std::uint32_t>(test.displacement);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base_value;
            for (std::uint32_t reg = test.first_destination; reg < 32U; ++reg)
            {
                const std::uint32_t value = 0x10203040U + reg * 0x01010101U;
                core->memory.write32_be(address + (reg - test.first_destination) * 4U,
                    value);
            }
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }

        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 1U && actual.cia == oracle.cia);
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
        assert(history.opcode_name == "lmw" && history.completed);
        assert(history.has_register_range &&
            history.range_first_register == test.first_destination &&
            history.range_last_register == 31U);
        assert(history.has_effective_address && history.effective_address == address);
    }

    EspressoCore interpreted_fault(0x1000U);
    EspressoCore compiled_fault(0x1000U);
    for (EspressoCore* core : {&interpreted_fault, &compiled_fault})
    {
        core->memory.write32_be(start, lmw_form(29U, 6U, 0));
        core->memory.write32_be(0xFF8U, 0x11223344U);
        core->memory.write32_be(0xFFCU, 0x55667788U);
        core->state.cia = start;
        core->state.gpr[6] = 0xFF8U;
        core->state.gpr[29] = 0xAAAAAAAAU;
        core->state.gpr[30] = 0xBBBBBBBBU;
        core->state.gpr[31] = 0xCCCCCCCCU;
    }
    const RunResult oracle_fault = interpreted_fault.run(1U);
    Recompiler fault_recompiler;
    const RunResult actual_fault = fault_recompiler.run(compiled_fault, 1U);
    assert(actual_fault.reason == oracle_fault.reason);
    assert(actual_fault.reason == StopReason::memory_fault);
    assert(actual_fault.detail.find("read 4 byte(s)") != std::string::npos);
    assert(actual_fault.detail.find("0x00001000") != std::string::npos);
    assert(compiled_fault.state.gpr == interpreted_fault.state.gpr);
    assert(compiled_fault.state.gpr[29] == 0x11223344U);
    assert(compiled_fault.state.gpr[30] == 0x55667788U);
    assert(compiled_fault.state.gpr[31] == 0xCCCCCCCCU);
    assert(compiled_fault.state.cia == start);
    assert(!actual_fault.instruction_history.empty());
    assert(actual_fault.instruction_history.back().opcode_name == "lmw");
    assert(actual_fault.instruction_history.back().has_register_range);
    assert(!actual_fault.instruction_history.back().completed);
}

void recompiler_stmw_differential_tests()
{
    struct Case
    {
        std::uint8_t first_source;
        std::uint8_t base_register;
        std::uint32_t base_value;
        std::int16_t displacement;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::array cases{
        Case{29U, 6U, 0x300U, -4},
        Case{4U, 4U, 0x300U, 8}, // EA snapshots the old rA before its range store.
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = stmw_form(
            test.first_source, test.base_register, test.displacement);
        const std::uint32_t address = test.base_value +
            static_cast<std::uint32_t>(test.displacement);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->state.cia = start;
            core->state.gpr.fill(0xDEADBEEFU);
            core->state.gpr[test.base_register] = test.base_value;
            core->state.cr = 0xA5C39E71U;
            core->state.xer = 0x87654321U;
            core->state.lr = 0xABCDEF00U;
            core->state.ctr = 0xFEDCBA98U;
            core->state.fpscr = 0x89ABCDEFU;
        }
        const auto gprs_before = interpreted.state.gpr;
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 1U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == gprs_before);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        for (std::uint32_t reg = test.first_source; reg < 32U; ++reg)
        {
            const std::uint32_t stored = gprs_before[reg];
            const std::uint32_t stored_address =
                address + (reg - test.first_source) * 4U;
            assert(compiled.memory.read32_be(stored_address) == stored);
        }
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        assert(history.opcode_name == "stmw" && history.completed);
        assert(history.has_register_range &&
            history.range_first_register == test.first_source &&
            history.range_last_register == 31U);
        assert(history.has_effective_address && history.effective_address == address);
    }

    EspressoCore interpreted_fault(0x1000U);
    EspressoCore compiled_fault(0x1000U);
    for (EspressoCore* core : {&interpreted_fault, &compiled_fault})
    {
        core->memory.write32_be(start, stmw_form(29U, 6U, 0));
        core->state.cia = start;
        core->state.gpr[6] = 0xFF8U;
        core->state.gpr[29] = 0x11223344U;
        core->state.gpr[30] = 0x55667788U;
        core->state.gpr[31] = 0x99AABBCCU;
    }
    compiled_fault.memory.set_write_watchpoint(0xFF8U, 8U);
    const RunResult oracle_fault = interpreted_fault.run(1U);
    Recompiler fault_recompiler;
    const RunResult actual_fault = fault_recompiler.run(compiled_fault, 1U);
    assert(actual_fault.reason == oracle_fault.reason);
    assert(actual_fault.reason == StopReason::memory_fault);
    assert(actual_fault.detail.find("write 4 byte(s)") != std::string::npos);
    assert(actual_fault.detail.find("0x00001000") != std::string::npos);
    assert(compiled_fault.memory.read32_be(0xFF8U) == 0x11223344U);
    assert(compiled_fault.memory.read32_be(0xFFCU) == 0x55667788U);
    assert(compiled_fault.memory.read32_be(0xFF8U) ==
        interpreted_fault.memory.read32_be(0xFF8U));
    assert(compiled_fault.memory.read32_be(0xFFCU) ==
        interpreted_fault.memory.read32_be(0xFFCU));
    const auto writes = compiled_fault.memory.write_watch_events_snapshot();
    assert(writes.size() == 2U); // The fault path did not replay prior stores.
    assert(writes[0].writer_cia == start && writes[1].writer_cia == start);
    assert(!actual_fault.instruction_history.empty());
    assert(actual_fault.instruction_history.back().opcode_name == "stmw");
    assert(actual_fault.instruction_history.back().has_register_range);
    assert(!actual_fault.instruction_history.back().completed);

    EspressoCore interpreted_code(0x1000U);
    EspressoCore compiled_code(0x1000U);
    constexpr std::uint32_t replacement_addi = 0x38600009U;
    for (EspressoCore* core : {&interpreted_code, &compiled_code})
    {
        core->memory.write32_be(start, stmw_form(31U, 6U, 4));
        core->memory.write32_be(start + 4U, 0x38600001U);
        core->state.cia = start;
        core->state.gpr[6] = start;
        core->state.gpr[31] = replacement_addi;
    }
    const RunResult code_oracle = interpreted_code.run(2U);
    Recompiler code_recompiler;
    const RunResult code_actual = code_recompiler.run(compiled_code, 2U);
    assert(code_actual.reason == code_oracle.reason);
    assert(code_actual.steps == 2U && code_actual.steps == code_oracle.steps);
    assert(compiled_code.state.gpr == interpreted_code.state.gpr);
    assert(compiled_code.state.gpr[3] == 9U);
    assert(compiled_code.memory.read32_be(start + 4U) == replacement_addi);
    assert(code_recompiler.statistics().native_instructions == 2U);
    assert(code_recompiler.statistics().interpreter_fallbacks == 0U);
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

void recompiler_dcbz_differential_tests()
{
    struct Case
    {
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint32_t base;
        std::uint32_t index;
    };
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t dcbz_r3_r4 = x_form(0U, 3U, 4U, 1014U);
    constexpr std::uint32_t dcbz_r0_r0 = x_form(0U, 0U, 0U, 1014U);
    static_assert(dcbz_r3_r4 == 0x7C0327ECU);
    constexpr std::array cases{
        Case{3U, 4U, 0x130U, 0x2FU},
        // rA=0 uses literal zero; rB=0 still reads the actual GPR0 value.
        Case{0U, 0U, 0xDEADBEEFU, 0x15FU},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x1000U);
        EspressoCore compiled(0x1000U);
        const std::uint32_t instruction = test.base_register == 0U
            ? dcbz_r0_r0 : dcbz_r3_r4;
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(start, instruction);
            core->memory.write32_be(start + 4U, 0x38E0002AU); // addi r7,r0,42
            for (std::uint32_t offset = 0U; offset < 64U; ++offset)
            {
                core->memory.write8(0x140U + offset,
                    offset < 32U ? 0xA5U : 0x5AU);
            }
            core->state.cia = start;
            core->state.gpr.fill(0xC001D00DU);
            core->state.gpr[0] = test.index_register == 0U
                ? test.index : 0xFACEB00CU;
            if (test.base_register != 0U)
                core->state.gpr[test.base_register] = test.base;
            if (test.index_register != 0U)
                core->state.gpr[test.index_register] = test.index;
            core->state.cr = 0x91827364U;
            core->state.xer = 0xA0000042U;
            core->state.lr = 0x10203040U;
            core->state.ctr = 0x50607080U;
            core->state.fpscr = 0xCAFEBABEU;
        }

        const RunResult oracle = interpreted.run(2U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 2U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        assert(actual.steps == 2U && actual.cia == oracle.cia);
        assert(compiled.state.gpr == interpreted.state.gpr);
        assert(compiled.state.fpr == interpreted.state.fpr);
        assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
        assert(compiled.state.cr == interpreted.state.cr);
        assert(compiled.state.xer == interpreted.state.xer);
        assert(compiled.state.lr == interpreted.state.lr);
        assert(compiled.state.ctr == interpreted.state.ctr);
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(compiled.state.gpr[7] == 42U);
        assert(recompiler.statistics().native_instructions == 2U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 2U);
        const InstructionHistoryEntry& history = actual.instruction_history.front();
        assert(history.opcode_name == "dcbz" && history.completed);
        assert(history.has_effective_address && history.effective_address == 0x140U);
        assert(history.source_count == (test.base_register == 0U ? 1U : 2U));
        if (test.base_register == 0U)
        {
            assert(history.source_registers[0] == 0U);
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
        for (std::uint32_t offset = 0U; offset < 32U; ++offset)
            assert(compiled.memory.read8(0x140U + offset) == 0U);
        for (std::uint32_t offset = 32U; offset < 64U; ++offset)
            assert(compiled.memory.read8(0x140U + offset) == 0x5AU);
    }

    // The whole-line write must validate before changing any bytes.
    EspressoCore faulting(0x1000U);
    faulting.memory.map_region(0x2000U, 31U);
    for (std::uint32_t offset = 0U; offset < 31U; ++offset)
        faulting.memory.write8(0x2000U + offset, 0xA5U);
    faulting.memory.write32_be(start, x_form(0U, 6U, 5U, 1014U));
    faulting.state.cia = start;
    faulting.state.gpr[6] = 0x2000U;
    faulting.state.gpr[5] = 0x1FU;
    const auto gprs_before = faulting.state.gpr;
    Recompiler fault_recompiler;
    const RunResult fault = fault_recompiler.run(faulting, 1U);
    assert(fault.reason == StopReason::memory_fault);
    assert(fault.detail.find("write 32 byte(s)") != std::string::npos);
    assert(faulting.state.gpr == gprs_before && faulting.state.cia == start);
    for (std::uint32_t offset = 0U; offset < 31U; ++offset)
        assert(faulting.memory.read8(0x2000U + offset) == 0xA5U);
    assert(!fault.instruction_history.empty());
    assert(fault.instruction_history.back().opcode_name == "dcbz");
    assert(fault.instruction_history.back().has_effective_address &&
        fault.instruction_history.back().effective_address == 0x2000U);
    assert(!fault.instruction_history.back().completed);
    assert(!fault.instruction_history.back().has_destination);
    assert(fault_recompiler.statistics().native_instructions == 0U);
    assert(fault_recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_psq_st_differential_tests()
{
    const auto encode_psq_st = [](
        std::uint8_t source, std::uint8_t base, std::int16_t displacement,
        bool single_element, std::uint8_t gqr) {
        return (60U << 26U) |
            (static_cast<std::uint32_t>(source) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(single_element) << 15U) |
            (static_cast<std::uint32_t>(gqr & 0x7U) << 12U) |
            (static_cast<std::uint16_t>(displacement) & 0x0FFFU);
    };
    const auto double_bits = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;

    const auto compare_cores = [](EspressoCore& interpreted, EspressoCore& compiled,
                                  std::uint32_t instruction) {
        interpreted.memory.write32_be(start, instruction);
        compiled.memory.write32_be(start, instruction);
        interpreted.state.cia = start;
        compiled.state.cia = start;
        const RunResult oracle = interpreted.run(1U);
        Recompiler recompiler;
        const RunResult actual = recompiler.run(compiled, 1U);
        assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
        return std::pair{actual, recompiler.statistics()};
    };

    // Float32 pair, exact endian representation, and architectural state
    // preservation (including both source lanes).
    EspressoCore float_interpreted(0x1000U);
    EspressoCore float_compiled(0x1000U);
    for (EspressoCore* core : {&float_interpreted, &float_compiled})
    {
        core->state.gpr.fill(0xC001D00DU);
        core->state.gpr[0] = 0x20U;
        core->state.gpr[4] = 0x200U;
        core->state.fpr[13] = double_bits(1.5);
        core->state.fpr_ps1[13] = double_bits(-2.25);
        core->state.cr = 0x12345678U;
        core->state.xer = 0xA0000042U;
        core->state.lr = 0x10203040U;
        core->state.ctr = 0x50607080U;
        core->state.fpscr = 0xCAFEBABEU;
    }
    const auto float_gprs = float_interpreted.state.gpr;
    const auto [float_result, float_stats] = compare_cores(
        float_interpreted, float_compiled,
        encode_psq_st(13U, 4U, 16, false, 0U));
    assert(float_result.steps == 1U);
    assert(float_compiled.memory.read32_be(0x210U) == std::bit_cast<std::uint32_t>(1.5F));
    assert(float_compiled.memory.read32_be(0x214U) == std::bit_cast<std::uint32_t>(-2.25F));
    assert(float_compiled.state.gpr == float_gprs);
    assert(float_compiled.state.fpr == float_interpreted.state.fpr);
    assert(float_compiled.state.fpr_ps1 == float_interpreted.state.fpr_ps1);
    assert(float_compiled.state.cr == float_interpreted.state.cr);
    assert(float_compiled.state.xer == float_interpreted.state.xer);
    assert(float_compiled.state.lr == float_interpreted.state.lr);
    assert(float_compiled.state.ctr == float_interpreted.state.ctr);
    assert(float_compiled.state.fpscr == float_interpreted.state.fpscr);
    assert(float_stats.native_instructions == 1U && float_stats.interpreter_fallbacks == 0U);
    const auto& float_history = float_result.instruction_history.back();
    assert(float_history.opcode_name == "psq_st" && float_history.completed);
    assert(float_history.has_effective_address && float_history.effective_address == 0x210U);
    assert(float_history.has_paired_fp_state && float_history.paired_fp_quantized_store);
    assert(float_history.paired_fp_source_a_ps0 == double_bits(1.5));
    assert(float_history.paired_fp_source_a_ps1 == double_bits(-2.25));
    assert(float_history.paired_fp_store_value0 == std::bit_cast<std::uint32_t>(1.5F));
    assert(float_history.paired_fp_store_value1 == std::bit_cast<std::uint32_t>(-2.25F));

    // Integer quantization applies the signed GQR scale and clamps before
    // truncation. W=1 must not touch the next byte or read PS1 for quantization.
    EspressoCore integer_interpreted(0x1000U);
    EspressoCore integer_compiled(0x1000U);
    for (EspressoCore* core : {&integer_interpreted, &integer_compiled})
    {
        core->state.gpr[4] = 0x300U;
        core->state.gqr[2] = (1U << 8U) | 4U; // U8, scale +1.
        core->state.fpr[3] = double_bits(10.75);
        core->state.fpr_ps1[3] = double_bits(200.0);
        core->memory.write8(0x302U, 0xCCU);
    }
    const auto [integer_result, integer_stats] = compare_cores(
        integer_interpreted, integer_compiled,
        encode_psq_st(3U, 4U, 2, true, 2U));
    assert(integer_result.steps == 1U);
    assert(integer_compiled.memory.read8(0x302U) == 21U);
    assert(integer_compiled.memory.read8(0x303U) == 0U);
    assert(integer_stats.native_instructions == 1U && integer_stats.interpreter_fallbacks == 0U);

    // Signed 16-bit values use two's complement in guest big-endian order.
    EspressoCore signed_interpreted(0x1000U);
    EspressoCore signed_compiled(0x1000U);
    for (EspressoCore* core : {&signed_interpreted, &signed_compiled})
    {
        core->state.gpr[5] = 0x400U;
        core->state.gqr[1] = 7U | (0x3FU << 8U); // S16, scale -1.
        core->state.fpr[6] = double_bits(100000.0);
        core->state.fpr_ps1[6] = double_bits(-3.5);
    }
    const auto [signed_result, signed_stats] = compare_cores(
        signed_interpreted, signed_compiled,
        encode_psq_st(6U, 5U, 0, false, 1U));
    assert(signed_result.steps == 1U);
    assert(signed_compiled.memory.read16_be(0x400U) == 0x7FFFU);
    assert(signed_compiled.memory.read16_be(0x402U) == 0xFFFFU);
    assert(signed_stats.native_instructions == 1U && signed_stats.interpreter_fallbacks == 0U);

    // Reserved GQR types and integer NaN conversion remain interpreter
    // unsupported cases; they must not write memory or count as native.
    EspressoCore reserved(0x1000U);
    reserved.state.gqr[0] = 2U;
    reserved.state.gpr[4] = 0x500U;
    reserved.memory.write8(0x500U, 0xA5U);
    reserved.memory.write32_be(start, encode_psq_st(1U, 4U, 0, true, 0U));
    reserved.state.cia = start;
    Recompiler reserved_recompiler;
    const RunResult reserved_result = reserved_recompiler.run(reserved, 1U);
    assert(reserved_result.reason == StopReason::unsupported_instruction);
    assert(reserved_result.steps == 0U && reserved.memory.read8(0x500U) == 0xA5U);
    assert(reserved_result.instruction_history.back().opcode_name == "psq_st");
    assert(!reserved_result.instruction_history.back().completed);
    assert(reserved_recompiler.statistics().native_instructions == 0U);

    EspressoCore nan_store(0x1000U);
    nan_store.state.gqr[0] = 4U;
    nan_store.state.gpr[4] = 0x510U;
    nan_store.state.fpr[1] = double_bits(std::numeric_limits<double>::quiet_NaN());
    nan_store.memory.write8(0x510U, 0x5AU);
    nan_store.memory.write32_be(start, encode_psq_st(1U, 4U, 0, true, 0U));
    nan_store.state.cia = start;
    Recompiler nan_recompiler;
    const RunResult nan_result = nan_recompiler.run(nan_store, 1U);
    assert(nan_result.reason == StopReason::unsupported_instruction);
    assert(nan_store.memory.read8(0x510U) == 0x5AU);

    // A pair crossing an unmapped boundary faults transactionally just like
    // the interpreter's complete-range validation.
    EspressoCore fault_interpreted(0x1000U);
    EspressoCore fault_compiled(0x1000U);
    for (EspressoCore* core : {&fault_interpreted, &fault_compiled})
    {
        core->memory.map_region(0x2000U, 4U);
        core->memory.write32_be(0x2000U, 0xAABBCCDDU);
        core->state.gpr[7] = 0x2000U;
        core->state.fpr[8] = double_bits(9.0);
        core->state.fpr_ps1[8] = double_bits(10.0);
    }
    const auto [fault_result, fault_stats] = compare_cores(
        fault_interpreted, fault_compiled,
        encode_psq_st(8U, 7U, 0, false, 0U));
    assert(fault_result.reason == StopReason::memory_fault);
    assert(fault_result.detail.find("write 8 byte(s)") != std::string::npos);
    assert(fault_compiled.memory.read32_be(0x2000U) == 0xAABBCCDDU);
    assert(fault_compiled.state.cia == start);
    assert(!fault_result.instruction_history.back().completed);
    assert(fault_result.instruction_history.back().opcode_name == "psq_st");
    assert(fault_stats.native_instructions == 0U);
}

void recompiler_psq_l_differential_tests()
{
    const auto encode_psq_l = [](
        std::uint8_t destination, std::uint8_t base, std::int16_t displacement,
        bool single_element, std::uint8_t gqr) {
        return (56U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(single_element) << 15U) |
            (static_cast<std::uint32_t>(gqr & 0x7U) << 12U) |
            (static_cast<std::uint16_t>(displacement) & 0x0FFFU);
    };
    const auto double_bits = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;

    // Float32 lanes are widened to doubles; W=1 reads one lane and supplies
    // the architectural constant 1.0 in PS1.
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->state.gpr[0] = 0xDEADBEEFU;
        core->state.gqr[0] = 0U;
        core->state.fpr[7] = 0xAAAAAAAAAAAAAAAAULL;
        core->state.fpr_ps1[7] = 0xBBBBBBBBBBBBBBBBULL;
        core->memory.write32_be(0x20U, std::bit_cast<std::uint32_t>(-3.25F));
        core->memory.write32_be(0x24U, std::bit_cast<std::uint32_t>(8.5F));
    }
    interpreted.memory.write32_be(start, encode_psq_l(7U, 0U, 0x20, false, 0U));
    compiled.memory.write32_be(start, encode_psq_l(7U, 0U, 0x20, false, 0U));
    interpreted.state.cia = start;
    compiled.state.cia = start;
    const RunResult oracle = interpreted.run(1U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 1U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 1U);
    assert(compiled.state.fpr[7] == double_bits(-3.25));
    assert(compiled.state.fpr_ps1[7] == double_bits(8.5));
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr && compiled.state.xer == interpreted.state.xer);
    assert(recompiler.statistics().native_instructions == 1U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "psq_l" && history.completed);
    assert(history.has_effective_address && history.effective_address == 0x20U);
    assert(history.has_paired_fp_state && history.paired_fp_destination_register == 7U);
    assert(history.paired_fp_destination_ps0 == double_bits(-3.25));
    assert(history.paired_fp_destination_ps1 == double_bits(8.5));

    EspressoCore scalar_interpreted(0x1000U);
    EspressoCore scalar_compiled(0x1000U);
    for (EspressoCore* core : {&scalar_interpreted, &scalar_compiled})
    {
        core->state.gpr[4] = 0x300U;
        core->state.gqr[2] = (6U << 16U) | (1U << 24U); // S8, load scale +1.
        core->state.fpr[3] = 0x123456789ABCDEF0ULL;
        core->state.fpr_ps1[3] = 0x0FEDCBA987654321ULL;
        core->memory.write8(0x302U, 0xFEU);
        core->memory.write8(0x303U, 0x7FU);
    }
    scalar_interpreted.memory.write32_be(start, encode_psq_l(3U, 4U, 2, false, 2U));
    scalar_compiled.memory.write32_be(start, encode_psq_l(3U, 4U, 2, false, 2U));
    scalar_interpreted.state.cia = start;
    scalar_compiled.state.cia = start;
    const RunResult scalar_oracle = scalar_interpreted.run(1U);
    Recompiler scalar_recompiler;
    const RunResult scalar_actual = scalar_recompiler.run(scalar_compiled, 1U);
    assert(scalar_actual.reason == scalar_oracle.reason && scalar_actual.steps == 1U);
    assert(scalar_compiled.state.fpr[3] == double_bits(-1.0));
    assert(scalar_compiled.state.fpr_ps1[3] == double_bits(63.5));
    assert(scalar_recompiler.statistics().native_instructions == 1U);

    EspressoCore w_interpreted(0x1000U);
    EspressoCore w_compiled(0x1000U);
    for (EspressoCore* core : {&w_interpreted, &w_compiled})
    {
        core->state.gpr[5] = 0x180U;
        core->state.gqr[1] = (5U << 16U) | (0x3FU << 24U); // U16, scale -1.
        core->memory.write16_be(0x184U, 0x1234U);
    }
    w_interpreted.memory.write32_be(start, encode_psq_l(8U, 5U, 4, true, 1U));
    w_compiled.memory.write32_be(start, encode_psq_l(8U, 5U, 4, true, 1U));
    w_interpreted.state.cia = start;
    w_compiled.state.cia = start;
    const RunResult w_oracle = w_interpreted.run(1U);
    Recompiler w_recompiler;
    const RunResult w_actual = w_recompiler.run(w_compiled, 1U);
    assert(w_actual.reason == w_oracle.reason && w_actual.steps == 1U);
    assert(w_compiled.state.fpr[8] == double_bits(9320.0F));
    assert(w_compiled.state.fpr_ps1[8] == double_bits(1.0));
    assert(w_recompiler.statistics().native_instructions == 1U);

    // If lane 1 faults, PS0 from lane 0 must not be committed.
    EspressoCore fault_interpreted(0x1000U);
    EspressoCore fault_compiled(0x1000U);
    for (EspressoCore* core : {&fault_interpreted, &fault_compiled})
    {
        core->memory.map_region(0x2000U, 4U);
        core->memory.write32_be(0x2000U, std::bit_cast<std::uint32_t>(5.0F));
        core->state.gqr[0] = 0U;
        core->state.fpr[2] = 0x1111222233334444ULL;
        core->state.fpr_ps1[2] = 0x5555666677778888ULL;
        core->state.gpr[6] = 0x2000U;
    }
    fault_interpreted.memory.write32_be(start, encode_psq_l(2U, 6U, 0, false, 0U));
    fault_compiled.memory.write32_be(start, encode_psq_l(2U, 6U, 0, false, 0U));
    fault_interpreted.state.cia = start;
    fault_compiled.state.cia = start;
    const auto fault_oracle = fault_interpreted.run(1U);
    Recompiler fault_recompiler;
    const auto fault_actual = fault_recompiler.run(fault_compiled, 1U);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("read 4 byte(s) at guest address 0x00002004") !=
        std::string::npos);
    assert(fault_compiled.state.fpr[2] == 0x1111222233334444ULL);
    assert(fault_compiled.state.fpr_ps1[2] == 0x5555666677778888ULL);
    assert(fault_compiled.state.cia == start);
    assert(!fault_actual.instruction_history.back().completed);
    assert(fault_recompiler.statistics().native_instructions == 0U);

    EspressoCore reserved(0x1000U);
    reserved.state.gqr[0] = 1U << 16U;
    reserved.state.gpr[4] = 0x20U;
    reserved.state.fpr[1] = 0x1122334455667788ULL;
    reserved.state.fpr_ps1[1] = 0x8877665544332211ULL;
    reserved.memory.write32_be(start, encode_psq_l(1U, 4U, 0, true, 0U));
    reserved.state.cia = start;
    Recompiler reserved_recompiler;
    const RunResult reserved_result = reserved_recompiler.run(reserved, 1U);
    assert(reserved_result.reason == StopReason::unsupported_instruction);
    assert(reserved.state.fpr[1] == 0x1122334455667788ULL);
    assert(reserved.state.fpr_ps1[1] == 0x8877665544332211ULL);
    assert(!reserved_result.instruction_history.back().completed);
    assert(reserved_recompiler.statistics().native_instructions == 0U);
}

void recompiler_psq_lx_differential_tests()
{
    const auto encode_psq_lx = [](
        std::uint8_t destination, std::uint8_t base, std::uint8_t index,
        bool single_element, std::uint8_t gqr) {
        return (4U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) |
            (static_cast<std::uint32_t>(single_element) << 10U) |
            (static_cast<std::uint32_t>(gqr & 7U) << 7U) | (6U << 1U);
    };
    const auto double_bits = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;

    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->memory.map_region(0x2000U, 8U);
        core->state.gpr[0] = 0x2000U;
        core->state.gqr[2] = 0U;
        core->state.fpr[4] = 0x1111222233334444ULL;
        core->state.fpr_ps1[4] = 0x5555666677778888ULL;
        core->state.cia = start;
        core->memory.write32_be(0x2000U, std::bit_cast<std::uint32_t>(1.5F));
        core->memory.write32_be(0x2004U, std::bit_cast<std::uint32_t>(-2.25F));
        core->memory.write32_be(start, encode_psq_lx(4U, 0U, 0U, false, 2U));
    }
    const RunResult oracle = interpreted.run(1U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 1U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 1U && compiled.state.cia == interpreted.state.cia);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.cr == interpreted.state.cr && compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.memory.read32_be(0x2000U) == interpreted.memory.read32_be(0x2000U));
    assert(recompiler.statistics().native_instructions == 1U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "psq_lx" && history.completed);
    assert(history.has_effective_address && history.effective_address == 0x2000U);
    assert(history.source_count == 1U && history.source_registers[0] == 0U);
    assert(history.source_values[0] == 0x2000U);
    assert(history.paired_fp_destination_ps0 == double_bits(1.5));
    assert(history.paired_fp_destination_ps1 == double_bits(-2.25));

    // W=1 with an indexed GQR U16 load reads one element and writes PS1=1.
    EspressoCore scalar_interpreted(0x1000U);
    EspressoCore scalar_compiled(0x1000U);
    for (EspressoCore* core : {&scalar_interpreted, &scalar_compiled})
    {
        core->memory.map_region(0x3000U, 2U);
        core->state.gpr[5] = 0x2FF0U;
        core->state.gpr[6] = 0x10U;
        core->state.gqr[1] = (5U << 16U) | (1U << 24U); // U16, scale +1.
        core->state.cia = start;
        core->memory.write16_be(0x3000U, 0x1234U);
        core->memory.write32_be(start, encode_psq_lx(8U, 5U, 6U, true, 1U));
    }
    const RunResult scalar_oracle = scalar_interpreted.run(1U);
    Recompiler scalar_recompiler;
    const RunResult scalar_actual = scalar_recompiler.run(scalar_compiled, 1U);
    assert(scalar_actual.reason == scalar_oracle.reason && scalar_actual.steps == 1U);
    assert(scalar_compiled.state.fpr == scalar_interpreted.state.fpr);
    assert(scalar_compiled.state.fpr_ps1 == scalar_interpreted.state.fpr_ps1);
    assert(scalar_compiled.state.gpr == scalar_interpreted.state.gpr);
    assert(scalar_compiled.state.fpr_ps1[8] == double_bits(1.0));
    assert(scalar_recompiler.statistics().native_instructions == 1U);
    assert(scalar_recompiler.statistics().interpreter_fallbacks == 0U);

    // A fault on lane 1 leaves both destination lanes unchanged in either engine.
    EspressoCore fault_interpreted(0x1000U);
    EspressoCore fault_compiled(0x1000U);
    for (EspressoCore* core : {&fault_interpreted, &fault_compiled})
    {
        core->memory.map_region(0x4000U, 4U);
        core->state.gpr[2] = 0x3FF0U;
        core->state.gpr[3] = 0x10U;
        core->state.fpr[6] = 0x1111222233334444ULL;
        core->state.fpr_ps1[6] = 0x5555666677778888ULL;
        core->state.cia = start;
        core->memory.write32_be(0x4000U, std::bit_cast<std::uint32_t>(5.0F));
        core->memory.write32_be(start, encode_psq_lx(6U, 2U, 3U, false, 0U));
    }
    const RunResult fault_oracle = fault_interpreted.run(1U);
    Recompiler fault_recompiler;
    const RunResult fault_actual = fault_recompiler.run(fault_compiled, 1U);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("read 4 byte(s) at guest address 0x00004004") !=
        std::string::npos);
    assert(fault_compiled.state.fpr == fault_interpreted.state.fpr);
    assert(fault_compiled.state.fpr_ps1 == fault_interpreted.state.fpr_ps1);
    assert(fault_compiled.state.cia == start);
    assert(!fault_actual.instruction_history.back().completed);
    assert(fault_actual.instruction_history.back().opcode_name == "psq_lx");
    assert(fault_actual.instruction_history.back().has_effective_address &&
        fault_actual.instruction_history.back().effective_address == 0x4000U);
    assert(fault_recompiler.statistics().native_instructions == 0U);
}

void recompiler_psq_lux_differential_tests()
{
    const auto encode_psq_lux = [](
        std::uint8_t destination, std::uint8_t base, std::uint8_t index,
        bool single_element, std::uint8_t gqr) {
        return (4U << 26U) |
            (static_cast<std::uint32_t>(destination) << 21U) |
            (static_cast<std::uint32_t>(base) << 16U) |
            (static_cast<std::uint32_t>(index) << 11U) |
            (static_cast<std::uint32_t>(single_element) << 10U) |
            (static_cast<std::uint32_t>(gqr & 7U) << 7U) | (38U << 1U);
    };
    const auto double_bits = [](double value) { return std::bit_cast<std::uint64_t>(value); };
    constexpr std::uint32_t start = 0x100U;

    EspressoCore interpreted(0x3000U);
    EspressoCore compiled(0x3000U);
    for (EspressoCore* core : {&interpreted, &compiled})
    {
        core->state.gpr[5] = 0x800U;
        core->state.gqr[0] = 0U;
        core->state.fpr[9] = 0x1111222233334444ULL;
        core->state.fpr_ps1[9] = 0x5555666677778888ULL;
        core->state.cia = start;
        core->memory.write32_be(start, encode_psq_lux(9U, 5U, 5U, false, 0U));
        core->memory.write32_be(0x1000U, std::bit_cast<std::uint32_t>(2.5F));
        core->memory.write32_be(0x1004U, std::bit_cast<std::uint32_t>(-1.25F));
    }
    const RunResult oracle = interpreted.run(1U);
    Recompiler recompiler;
    const RunResult actual = recompiler.run(compiled, 1U);
    assert(actual.reason == oracle.reason && actual.steps == oracle.steps);
    assert(actual.steps == 1U);
    assert(compiled.state.gpr[5] == 0x1000U);
    assert(compiled.state.gpr == interpreted.state.gpr);
    assert(compiled.state.fpr == interpreted.state.fpr);
    assert(compiled.state.fpr_ps1 == interpreted.state.fpr_ps1);
    assert(compiled.state.cr == interpreted.state.cr && compiled.state.xer == interpreted.state.xer);
    assert(compiled.state.fpscr == interpreted.state.fpscr);
    assert(compiled.state.fpr[9] == double_bits(2.5));
    assert(compiled.state.fpr_ps1[9] == double_bits(-1.25));
    assert(recompiler.statistics().native_instructions == 1U);
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    const InstructionHistoryEntry& history = actual.instruction_history.back();
    assert(history.opcode_name == "psq_lux" && history.completed);
    assert(history.has_effective_address && history.effective_address == 0x1000U);
    assert(history.has_destination && history.destination_register == 5U &&
        history.destination_value == 0x1000U);
    assert(history.source_count == 1U);
    assert(history.source_registers[0] == 5U && history.source_values[0] == 0x800U);

    EspressoCore fault_interpreted(0x3000U);
    EspressoCore fault_compiled(0x3000U);
    for (EspressoCore* core : {&fault_interpreted, &fault_compiled})
    {
        core->memory.map_region(0x4000U, 4U);
        core->state.gpr[7] = 0x3FF0U;
        core->state.gpr[8] = 0x10U;
        core->state.fpr[3] = 0x1111222233334444ULL;
        core->state.fpr_ps1[3] = 0x5555666677778888ULL;
        core->state.cia = start;
        core->memory.write32_be(start, encode_psq_lux(3U, 7U, 8U, false, 0U));
        core->memory.write32_be(0x4000U, std::bit_cast<std::uint32_t>(3.0F));
    }
    const RunResult fault_oracle = fault_interpreted.run(1U);
    Recompiler fault_recompiler;
    const RunResult fault_actual = fault_recompiler.run(fault_compiled, 1U);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("read 4 byte(s) at guest address 0x00004004") !=
        std::string::npos);
    assert(fault_compiled.state.gpr == fault_interpreted.state.gpr);
    assert(fault_compiled.state.fpr == fault_interpreted.state.fpr);
    assert(fault_compiled.state.fpr_ps1 == fault_interpreted.state.fpr_ps1);
    assert(fault_compiled.state.gpr[7] == 0x3FF0U);
    assert(!fault_actual.instruction_history.back().completed);
    assert(fault_actual.instruction_history.back().opcode_name == "psq_lux");
    assert(fault_recompiler.statistics().native_instructions == 0U);
}

void recompiler_memory_tests()
{
    EspressoCore cached_watched_store(0x1000U);
    cached_watched_store.memory.write32_be(0x100U, 0x3CA01234U); // lis r5, 0x1234
    cached_watched_store.memory.write32_be(0x104U, 0x60A55678U); // ori r5,r5,0x5678
    cached_watched_store.memory.write32_be(0x108U, 0x90A40000U); // stw r5,0(r4)
    cached_watched_store.state.cia = 0x100U;
    cached_watched_store.state.gpr[4] = 0x200U;
    Recompiler cached_watched_store_recompiler;
    const RunResult initial_compile = cached_watched_store_recompiler.run(
        cached_watched_store, 1U);
    assert(initial_compile.steps == 1U);
    assert(cached_watched_store.state.gpr[5] == 0x12340000U);
    assert(cached_watched_store.memory.write_watch_events_snapshot().empty());

    // The block was compiled before diagnostic watching was enabled. Its cache
    // entry must be recompiled so the store publishes host-cached GPRs first.
    cached_watched_store.memory.set_write_watch_gpr_source(
        &cached_watched_store.state.gpr);
    cached_watched_store.memory.set_write_watchpoint(0x200U, 4U);
    cached_watched_store.state.cia = 0x100U;
    const RunResult watched_cached_result = cached_watched_store_recompiler.run(
        cached_watched_store, 3U);
    assert(watched_cached_result.steps == 3U);
    const auto cached_watch_events =
        cached_watched_store.memory.write_watch_events_snapshot();
    assert(cached_watch_events.size() == 1U);
    assert(cached_watch_events[0].writer_cia == 0x108U);
    assert(cached_watch_events[0].watched_value == 0x12345678U);
    assert(cached_watch_events[0].has_gpr_snapshot);
    assert(cached_watch_events[0].gpr_snapshot[4] == 0x200U);
    assert(cached_watch_events[0].gpr_snapshot[5] == 0x12345678U);

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
    assert(watched_store_recompiler.statistics().memory_slow_path_calls == 1U);

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
    assert(compiled.memory.write_generation() == interpreted.memory.write_generation());
    assert(compiled.memory.page_write_generation(0x200U) ==
        interpreted.memory.page_write_generation(0x200U));
    assert(recompiler.statistics().native_instructions == program.size());
    assert(recompiler.statistics().interpreter_fallbacks == 0U);
    // LHA and the three integer stores continue through the checked helper.
    assert(recompiler.statistics().memory_slow_path_calls == 4U);
    const std::string history = format_instruction_history(actual);
    assert(history.find("lwz") != std::string::npos);
    assert(history.find("0x00000200") != std::string::npos);

    // Indexed memory operations use GPR[RB] even when RB is r0, while RA=0 is
    // the architectural zero base. Check both BE data and sign extension.
    EspressoCore indexed_interpreted(0x1000U);
    EspressoCore indexed_compiled(0x1000U);
    const std::array<std::uint32_t, 7> indexed_program{
        x_form(3, 4, 0, 23),  // lwzx r3,r4,r0 (r0 is an ordinary index)
        x_form(3, 4, 0, 151), // stwx r3,r4,r0
        x_form(6, 4, 0, 87),  // lbzx r6,r4,r0
        x_form(7, 4, 5, 343),// lhax r7,r4,r5
        x_form(8, 4, 0, 534), // lwbrx r8,r4,r0
        x_form(9, 4, 0, 790), // lhbrx r9,r4,r0
        x_form(4, 4, 0, 534), // lwbrx r4,r4,r0 (RA/destination alias)
    };
    static_assert(x_form(8, 4, 0, 534) == 0x7D04042CU);
    static_assert(x_form(9, 4, 0, 790) == 0x7D24062CU);
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
    assert(indexed_compiled.state.gpr[8] == 0x98BADCFEU);
    assert(indexed_compiled.state.gpr[9] == 0x0000DCFEU);
    assert(indexed_compiled.state.gpr[4] == 0x98BADCFEU);
    assert(indexed_recompiler.statistics().native_instructions == indexed_program.size());
    assert(indexed_recompiler.statistics().interpreter_fallbacks == 0U);

    // stwbrx reverses the source word before the checked big-endian store,
    // while retaining the original register value in native history.
    EspressoCore reverse_store_interpreted(0x1000U);
    EspressoCore reverse_store_compiled(0x1000U);
    constexpr std::uint32_t stwbrx = x_form(6, 4, 5, 662);
    static_assert(stwbrx == 0x7CC42D2CU);
    for (EspressoCore* core : {&reverse_store_interpreted,
                               &reverse_store_compiled})
    {
        core->memory.write32_be(0x100U, stwbrx); // stwbrx r6,r4,r5
        core->state.cia = 0x100U;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[5] = 0x20U;
        core->state.gpr[6] = 0x11223344U;
    }
    const RunResult reverse_store_oracle = reverse_store_interpreted.run(1U);
    Recompiler reverse_store_recompiler;
    const RunResult reverse_store_actual = reverse_store_recompiler.run(
        reverse_store_compiled, 1U);
    assert(reverse_store_actual.steps == reverse_store_oracle.steps);
    assert(reverse_store_compiled.memory.read32_be(0x220U) == 0x44332211U);
    assert(reverse_store_compiled.state.gpr == reverse_store_interpreted.state.gpr);
    assert(reverse_store_compiled.memory.read32_be(0x220U) ==
           reverse_store_interpreted.memory.read32_be(0x220U));
    assert(reverse_store_actual.instruction_history.front().opcode_name == "stwbrx");
    assert(reverse_store_actual.instruction_history.front().source_count == 3U);
    assert(reverse_store_actual.instruction_history.front().source_values[2] ==
           0x11223344U);
    assert(reverse_store_recompiler.statistics().native_instructions == 1U);
    assert(reverse_store_recompiler.statistics().interpreter_fallbacks == 0U);

    // The source may alias rA: address and data both use its pre-instruction value.
    EspressoCore reverse_store_alias(0x1000U);
    reverse_store_alias.memory.write32_be(0x100U, x_form(4, 4, 5, 662));
    reverse_store_alias.state.cia = 0x100U;
    reverse_store_alias.state.gpr[4] = 0x240U;
    reverse_store_alias.state.gpr[5] = 4U;
    Recompiler reverse_store_alias_recompiler;
    const RunResult reverse_store_alias_result =
        reverse_store_alias_recompiler.run(reverse_store_alias, 1U);
    assert(reverse_store_alias_result.steps == 1U);
    assert(reverse_store_alias.memory.read32_be(0x244U) == 0x40020000U);
    assert(reverse_store_alias.state.gpr[4] == 0x240U);

    // A partial mapping cannot be modified by the failed four-byte store.
    EspressoCore reverse_store_fault(0x1000U);
    reverse_store_fault.memory.map_region(0x2000U, 2U);
    reverse_store_fault.memory.write16_be(0x2000U, 0xAABBU);
    reverse_store_fault.memory.write32_be(0x100U, x_form(6, 4, 5, 662));
    reverse_store_fault.state.cia = 0x100U;
    reverse_store_fault.state.gpr[4] = 0x2000U;
    reverse_store_fault.state.gpr[5] = 0U;
    reverse_store_fault.state.gpr[6] = 0x11223344U;
    Recompiler reverse_store_fault_recompiler;
    const RunResult reverse_store_fault_result =
        reverse_store_fault_recompiler.run(reverse_store_fault, 1U);
    assert(reverse_store_fault_result.reason == StopReason::memory_fault);
    assert(reverse_store_fault_result.detail.find("write 4 byte(s)") !=
           std::string::npos);
    assert(reverse_store_fault.memory.read16_be(0x2000U) == 0xAABBU);
    assert(reverse_store_fault.state.gpr[6] == 0x11223344U);
    assert(!reverse_store_fault_result.instruction_history.front().completed);
    assert(reverse_store_fault_recompiler.statistics().native_instructions == 0U);
    assert(reverse_store_fault_recompiler.statistics().interpreter_fallbacks == 0U);

    // sthbrx reverses only the low halfword and preserves the complete source
    // GPR value in history, matching the interpreter's checked store behavior.
    EspressoCore reverse_half_store_interpreted(0x1000U);
    EspressoCore reverse_half_store_compiled(0x1000U);
    constexpr std::uint32_t sthbrx = x_form(6, 4, 5, 918);
    static_assert(sthbrx == 0x7CC42F2CU);
    for (EspressoCore* core : {&reverse_half_store_interpreted,
                               &reverse_half_store_compiled})
    {
        core->memory.write32_be(0x100U, sthbrx); // sthbrx r6,r4,r5
        core->state.cia = 0x100U;
        core->state.gpr[4] = 0x200U;
        core->state.gpr[5] = 0x20U;
        core->state.gpr[6] = 0x12345678U;
    }
    const RunResult reverse_half_store_oracle =
        reverse_half_store_interpreted.run(1U);
    Recompiler reverse_half_store_recompiler;
    const RunResult reverse_half_store_actual =
        reverse_half_store_recompiler.run(reverse_half_store_compiled, 1U);
    assert(reverse_half_store_actual.steps == reverse_half_store_oracle.steps);
    assert(reverse_half_store_compiled.memory.read16_be(0x220U) == 0x7856U);
    assert(reverse_half_store_compiled.state.gpr ==
           reverse_half_store_interpreted.state.gpr);
    assert(reverse_half_store_compiled.memory.read16_be(0x220U) ==
           reverse_half_store_interpreted.memory.read16_be(0x220U));
    assert(reverse_half_store_actual.instruction_history.front().opcode_name == "sthbrx");
    assert(reverse_half_store_actual.instruction_history.front().source_count == 3U);
    assert(reverse_half_store_actual.instruction_history.front().source_values[2] ==
           0x12345678U);
    assert(reverse_half_store_actual.instruction_history.front().has_stored_halfword_value);
    assert(reverse_half_store_actual.instruction_history.front().stored_halfword_value ==
           0x7856U);
    assert(reverse_half_store_recompiler.statistics().native_instructions == 1U);
    assert(reverse_half_store_recompiler.statistics().interpreter_fallbacks == 0U);

    EspressoCore reverse_half_store_fault(0x1000U);
    reverse_half_store_fault.memory.map_region(0x2000U, 1U);
    reverse_half_store_fault.memory.write8(0x2000U, 0xA5U);
    reverse_half_store_fault.memory.write32_be(0x100U, sthbrx);
    reverse_half_store_fault.state.cia = 0x100U;
    reverse_half_store_fault.state.gpr[4] = 0x2000U;
    reverse_half_store_fault.state.gpr[5] = 0U;
    reverse_half_store_fault.state.gpr[6] = 0x12345678U;
    Recompiler reverse_half_store_fault_recompiler;
    const RunResult reverse_half_store_fault_result =
        reverse_half_store_fault_recompiler.run(reverse_half_store_fault, 1U);
    assert(reverse_half_store_fault_result.reason == StopReason::memory_fault);
    assert(reverse_half_store_fault_result.detail.find("write 2 byte(s)") !=
           std::string::npos);
    assert(reverse_half_store_fault.memory.read8(0x2000U) == 0xA5U);
    assert(reverse_half_store_fault.state.gpr[6] == 0x12345678U);
    assert(!reverse_half_store_fault_result.instruction_history.front().completed);
    assert(reverse_half_store_fault_recompiler.statistics().native_instructions == 0U);
    assert(reverse_half_store_fault_recompiler.statistics().interpreter_fallbacks == 0U);

    // A checked store crossing a page boundary advances both page generations
    // and the global write generation exactly once.
    EspressoCore boundary_store(0x2000U);
    boundary_store.memory.write32_be(0x100U, 0x90A40000U); // stw r5,0(r4)
    boundary_store.state.cia = 0x100U;
    boundary_store.state.gpr[4] = 0xFFEU;
    boundary_store.state.gpr[5] = 0x12345678U;
    const std::uint64_t write_generation_before =
        boundary_store.memory.write_generation();
    const std::uint64_t first_page_before =
        boundary_store.memory.page_write_generation(0xFFEU);
    const std::uint64_t second_page_before =
        boundary_store.memory.page_write_generation(0x1000U);
    Recompiler boundary_recompiler;
    const RunResult boundary_result = boundary_recompiler.run(boundary_store, 1U);
    assert(boundary_result.steps == 1U);
    assert(boundary_store.memory.read32_be(0xFFEU) == 0x12345678U);
    assert(boundary_store.memory.write_generation() == write_generation_before + 1U);
    assert(boundary_store.memory.page_write_generation(0xFFEU) == first_page_before + 1U);
    assert(boundary_store.memory.page_write_generation(0x1000U) == second_page_before + 1U);
    assert(boundary_recompiler.statistics().memory_slow_path_calls == 1U);

    // A high guest alias must use the checked store path and still update the
    // flat backing bytes and generation counters coherently.
    EspressoCore aliased_store(0x3000U);
    aliased_store.memory.map_alias(0x4000U, 0x1000U, 0x1000U);
    aliased_store.memory.write32_be(0x100U, 0x90A40000U);
    aliased_store.state.cia = 0x100U;
    aliased_store.state.gpr[4] = 0x4000U;
    aliased_store.state.gpr[5] = 0x89ABCDEFU;
    Recompiler alias_store_recompiler;
    const RunResult alias_store_result = alias_store_recompiler.run(aliased_store, 1U);
    assert(alias_store_result.steps == 1U);
    assert(aliased_store.memory.read32_be(0x1000U) == 0x89ABCDEFU);
    assert(alias_store_recompiler.statistics().memory_slow_path_calls == 1U);

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

void recompiler_lswx_differential_tests()
{
    struct Case
    {
        std::uint8_t first_register;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint8_t byte_count;
        std::uint32_t base_value;
        std::uint32_t index_value;
    };
    constexpr std::array cases{
        // Zero-byte lswx is a no-op and must not probe its effective address.
        Case{6U, 4U, 5U, 0U, 0x2000U, 0x100U},
        Case{6U, 4U, 5U, 1U, 0x200U, 0x30U},
        Case{31U, 4U, 5U, 5U, 0x200U, 0x30U},
        Case{31U, 0U, 0U, 6U, 0xDEADBEEFU, 0x400U},
        Case{29U, 4U, 5U, 127U, 0x200U, 0x30U},
        Case{4U, 4U, 5U, 5U, 0x200U, 0x30U},
        Case{5U, 4U, 5U, 5U, 0x200U, 0x30U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x2000U);
        EspressoCore compiled(0x2000U);
        const std::uint32_t instruction = x_form(
            test.first_register, test.base_register, test.index_register, 533U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0xA5A50000U);
            core->state.gpr[test.base_register] = test.base_value;
            core->state.gpr[test.index_register] = test.index_value;
            core->state.xer = 0x81234500U | test.byte_count;
            core->state.cr = 0x89ABCDEFU;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0xA5A55A5AU;
            const std::uint32_t base = test.base_register == 0U
                ? 0U : core->state.gpr[test.base_register];
            const std::uint32_t address = base + core->state.gpr[test.index_register];
            for (std::uint32_t byte = 0U; byte < test.byte_count; ++byte)
            {
                const std::uint8_t value = static_cast<std::uint8_t>(byte * 37U + 11U);
                core->memory.write8(address + byte, value);
            }
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
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        if (test.byte_count == 0U)
            assert(recompiler.statistics().memory_slow_path_calls == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        const InstructionHistoryEntry& expected = oracle.instruction_history.back();
        assert(history.completed && history.opcode_name == "lswx");
        assert(history.has_string_transfer && history.string_transfer_indexed);
        assert(history.string_transfer_first_register == test.first_register);
        assert(history.string_transfer_base_register == test.base_register);
        assert(history.string_transfer_index_register == test.index_register);
        assert(history.string_transfer_byte_count == test.byte_count);
        assert(history.string_transfer_register_count == expected.string_transfer_register_count);
        assert(history.effective_address == expected.effective_address);
        assert(history.string_transfer_base_value == expected.string_transfer_base_value);
        assert(history.string_transfer_index_value == expected.string_transfer_index_value);
        assert(history.string_transfer_bytes == expected.string_transfer_bytes);
        assert(history.string_transfer_registers == expected.string_transfer_registers);
        assert(history.string_transfer_register_values == expected.string_transfer_register_values);
    }

    // A failed multi-byte read must not partially write any destination GPR.
    EspressoCore interpreted_fault(0x1000U);
    EspressoCore compiled_fault(0x1000U);
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t fault_address = 0xFFDU;
    const std::uint32_t fault_instruction = x_form(31U, 4U, 5U, 533U);
    for (EspressoCore* core : {&interpreted_fault, &compiled_fault})
    {
        core->memory.write32_be(start, fault_instruction);
        core->memory.write8(fault_address, 0x11U);
        core->memory.write8(fault_address + 1U, 0x22U);
        core->memory.write8(fault_address + 2U, 0x33U);
        core->state.cia = start;
        core->state.gpr.fill(0xDEADBEEFU);
        core->state.gpr[4] = fault_address;
        core->state.gpr[5] = 0U;
        core->state.xer = 5U;
    }
    const auto original_gprs = compiled_fault.state.gpr;
    const RunResult fault_oracle = interpreted_fault.run(1U);
    Recompiler fault_recompiler;
    const RunResult fault_actual = fault_recompiler.run(compiled_fault, 1U);
    assert(fault_actual.reason == StopReason::memory_fault);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("read 5 byte(s) at guest address 0x00000FFD") !=
        std::string::npos);
    assert(compiled_fault.state.gpr == original_gprs);
    assert(compiled_fault.state.gpr == interpreted_fault.state.gpr);
    assert(fault_actual.instruction_history.size() == 1U);
    const InstructionHistoryEntry& fault_history =
        fault_actual.instruction_history.back();
    assert(!fault_history.completed && fault_history.opcode_name == "lswx");
    assert(fault_history.has_string_transfer && fault_history.string_transfer_indexed);
    assert(fault_history.effective_address == fault_address);
    assert(fault_history.string_transfer_byte_count == 5U);
    assert(fault_recompiler.statistics().native_instructions == 0U);
    assert(fault_recompiler.statistics().interpreter_fallbacks == 0U);
}

void recompiler_stswx_differential_tests()
{
    struct Case
    {
        std::uint8_t first_register;
        std::uint8_t base_register;
        std::uint8_t index_register;
        std::uint8_t byte_count;
        std::uint32_t base_value;
        std::uint32_t index_value;
    };
    constexpr std::array cases{
        Case{6U, 4U, 5U, 0U, 0x2000U, 0x100U},
        Case{6U, 4U, 5U, 1U, 0x200U, 0x30U},
        Case{6U, 4U, 5U, 6U, 0x200U, 0x30U},
        Case{31U, 0U, 0U, 6U, 0xDEADBEEFU, 0x400U},
        Case{29U, 4U, 5U, 127U, 0x200U, 0x30U},
        Case{4U, 4U, 5U, 5U, 0x200U, 0x30U},
        Case{5U, 4U, 5U, 5U, 0x200U, 0x30U},
    };

    for (const Case& test : cases)
    {
        EspressoCore interpreted(0x4000U);
        EspressoCore compiled(0x4000U);
        const std::uint32_t instruction = x_form(
            test.first_register, test.base_register, test.index_register, 661U);
        for (EspressoCore* core : {&interpreted, &compiled})
        {
            core->memory.write32_be(0x100U, instruction);
            core->state.cia = 0x100U;
            core->state.gpr.fill(0xC0DE0000U);
            core->state.gpr[test.base_register] = test.base_value;
            core->state.gpr[test.index_register] = test.index_value;
            core->state.xer = 0x81234500U | test.byte_count;
            core->state.cr = 0x89ABCDEFU;
            core->state.lr = 0x12345678U;
            core->state.ctr = 0x87654321U;
            core->state.fpscr = 0xA5A55A5AU;
            const std::uint32_t base = test.base_register == 0U
                ? 0U : core->state.gpr[test.base_register];
            const std::uint32_t address = base + core->state.gpr[test.index_register];
            for (std::uint32_t byte = 0U; byte < test.byte_count; ++byte)
                core->memory.write8(address + byte, 0xEEU);
        }
        const auto original_memory = interpreted.memory.read8(0x200U);
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
        assert(compiled.state.fpscr == interpreted.state.fpscr);
        assert(recompiler.statistics().native_instructions == 1U);
        assert(recompiler.statistics().interpreter_fallbacks == 0U);
        assert(actual.instruction_history.size() == 1U);
        const InstructionHistoryEntry& history = actual.instruction_history.back();
        const InstructionHistoryEntry& expected = oracle.instruction_history.back();
        assert(history.completed && history.opcode_name == "stswx");
        assert(history.has_string_transfer && history.string_transfer_store);
        assert(history.string_transfer_indexed);
        assert(history.string_transfer_first_register == test.first_register);
        assert(history.string_transfer_base_register == test.base_register);
        assert(history.string_transfer_index_register == test.index_register);
        assert(history.string_transfer_byte_count == test.byte_count);
        assert(history.string_transfer_register_count == expected.string_transfer_register_count);
        assert(history.effective_address == expected.effective_address);
        assert(history.string_transfer_base_value == expected.string_transfer_base_value);
        assert(history.string_transfer_index_value == expected.string_transfer_index_value);
        assert(history.string_transfer_bytes == expected.string_transfer_bytes);
        assert(history.string_transfer_registers == expected.string_transfer_registers);
        assert(history.string_transfer_register_values == expected.string_transfer_register_values);
        for (std::uint32_t byte = 0U; byte < test.byte_count; ++byte)
        {
            const std::uint32_t address = history.effective_address + byte;
            assert(compiled.memory.read8(address) == interpreted.memory.read8(address));
        }
        if (test.byte_count == 0U)
        {
            assert(recompiler.statistics().memory_slow_path_calls == 0U);
            assert(compiled.memory.read8(0x200U) == original_memory);
        }
    }

    // A fault in the final byte must not leave a prefix store committed.
    EspressoCore interpreted_fault(0x1000U);
    EspressoCore compiled_fault(0x1000U);
    constexpr std::uint32_t start = 0x100U;
    constexpr std::uint32_t fault_address = 0xFFDU;
    const std::uint32_t fault_instruction = x_form(31U, 4U, 5U, 661U);
    for (EspressoCore* core : {&interpreted_fault, &compiled_fault})
    {
        core->memory.write32_be(start, fault_instruction);
        core->memory.write8(fault_address, 0xAAU);
        core->memory.write8(fault_address + 1U, 0xBBU);
        core->memory.write8(fault_address + 2U, 0xCCU);
        core->state.cia = start;
        core->state.gpr.fill(0xDEADBEEFU);
        core->state.gpr[4] = fault_address;
        core->state.gpr[5] = 0U;
        core->state.gpr[31] = 0x11223344U;
        core->state.gpr[0] = 0x55667788U;
        core->state.xer = 5U;
    }
    const auto original_gprs = compiled_fault.state.gpr;
    const RunResult fault_oracle = interpreted_fault.run(1U);
    Recompiler fault_recompiler;
    const RunResult fault_actual = fault_recompiler.run(compiled_fault, 1U);
    assert(fault_actual.reason == StopReason::memory_fault);
    assert(fault_actual.reason == fault_oracle.reason);
    assert(fault_actual.detail.find("write 5 byte(s) at guest address 0x00000FFD") !=
        std::string::npos);
    assert(compiled_fault.state.gpr == original_gprs);
    assert(compiled_fault.state.gpr == interpreted_fault.state.gpr);
    assert(compiled_fault.memory.read8(fault_address) == 0xAAU);
    assert(compiled_fault.memory.read8(fault_address + 1U) == 0xBBU);
    assert(compiled_fault.memory.read8(fault_address + 2U) == 0xCCU);
    assert(fault_actual.instruction_history.size() == 1U);
    const InstructionHistoryEntry& fault_history = fault_actual.instruction_history.back();
    assert(!fault_history.completed && fault_history.opcode_name == "stswx");
    assert(fault_history.has_string_transfer && fault_history.string_transfer_store);
    assert(fault_history.string_transfer_indexed);
    assert(fault_history.effective_address == fault_address);
    assert(fault_history.string_transfer_byte_count == 5U);
    assert(fault_recompiler.statistics().native_instructions == 0U);
    assert(fault_recompiler.statistics().interpreter_fallbacks == 0U);
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
    recompiler_execution_context_fp_history_reset_test();
    recompiler_mulli_differential_test();
    recompiler_neg_differential_test();
    recompiler_mullw_differential_test();
    recompiler_mulhw_differential_tests();
    recompiler_cmpwi_differential_test();
    recompiler_cmplwi_differential_test();
    recompiler_cmpw_differential_test();
    recompiler_fcmpu_differential_test();
    recompiler_fcmpo_differential_test();
    recompiler_mffs_differential_test();
    recompiler_mtfsb_differential_tests();
    recompiler_mtfsfi_differential_tests();
    recompiler_mtfsf_differential_tests();
    recompiler_ps_cmpu0_differential_test();
    recompiler_ps_cmpu1_differential_test();
    recompiler_ps_cmpo0_differential_test();
    recompiler_ps_cmpo1_differential_test();
    recompiler_subfic_differential_test();
    recompiler_subfc_differential_test();
    recompiler_subfe_differential_test();
    recompiler_subfeo_differential_test();
    recompiler_divwuo_differential_test();
    recompiler_divwo_differential_test();
    recompiler_adde_differential_test();
    recompiler_addeo_differential_test();
    recompiler_addo_subfo_differential_test();
    recompiler_nego_differential_test();
    recompiler_addco_differential_test();
    recompiler_subfco_differential_test();
    recompiler_addzeo_differential_test();
    recompiler_addmeo_differential_test();
    recompiler_addme_differential_test();
    recompiler_mullwo_differential_test();
    recompiler_subfme_differential_test();
    recompiler_subfmeo_differential_test();
    recompiler_sraw_differential_test();
    recompiler_subfze_differential_test();
    recompiler_subfzeo_differential_test();
    recompiler_addic_differential_test();
    recompiler_destination_aliases_second_source_test();
    recompiler_logical_differential_test();
    recompiler_logical_record_differential_test();
    recompiler_condition_bit_differential_tests();
    recompiler_mcrf_differential_tests();
    recompiler_mcrfs_differential_tests();
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
    recompiler_fctiw_differential_tests();
    recompiler_fmsub_differential_tests();
    recompiler_fnmadd_differential_tests();
    recompiler_fnmadds_differential_tests();
    recompiler_fres_differential_tests();
    recompiler_fabs_differential_tests();
    recompiler_ps_add_differential_tests();
    recompiler_ps_div_differential_tests();
    recompiler_reservation_instruction_differential_tests();
    recompiler_ps_merge10_differential_tests();
    recompiler_ps_merge00_differential_tests();
    recompiler_ps_mr_differential_tests();
    recompiler_ps_abs_differential_tests();
    recompiler_ps_nabs_differential_tests();
    recompiler_ps_sum1_differential_tests();
    recompiler_ps_res_differential_tests();
    recompiler_ps_rsqrte_differential_tests();
    recompiler_ps_select_differential_tests();
    recompiler_ps_muls1_differential_tests();
    recompiler_ps_madds0_differential_tests();
    recompiler_ps_madds1_differential_tests();
    recompiler_ps_nmadd_differential_tests();
    recompiler_ps_nmsub_differential_tests();
    recompiler_stfs_differential_tests();
    recompiler_stfsu_differential_tests();
    recompiler_stfd_differential_tests();
    recompiler_stfdx_differential_tests();
    recompiler_stfdux_differential_tests();
    recompiler_stwu_differential_tests();
    recompiler_stbu_differential_tests();
    recompiler_slw_differential_tests();
    recompiler_srw_differential_tests();
    recompiler_srawi_differential_tests();
    recompiler_rlwimi_differential_tests();
    recompiler_cntlzw_differential_tests();
    recompiler_rlwinm_differential_tests();
    recompiler_rlwnm_differential_tests();
    recompiler_andis_differential_tests();
    recompiler_lfd_differential_tests();
    recompiler_lfdx_differential_tests();
    recompiler_lfdux_differential_tests();
    recompiler_lfdu_differential_tests();
    recompiler_lfs_differential_tests();
    recompiler_lfsx_differential_tests();
    recompiler_lfsux_differential_tests();
    recompiler_step_budget_test();
    recompiler_recent_history_window_test();
    recompiler_isync_boundary_differential_test();
    recompiler_eieio_differential_test();
    recompiler_memory_synchronize_differential_test();
    recompiler_move_from_time_base_differential_test();
    recompiler_dcbt_differential_test();
    recompiler_dcbtst_differential_test();
    recompiler_icbi_differential_test();
    recompiler_lbzu_differential_tests();
    recompiler_direct_flat_halfword_load_tests();
    recompiler_lwzu_differential_tests();
    recompiler_lwzux_differential_tests();
    recompiler_lbzux_differential_tests();
    recompiler_lhzux_differential_tests();
    recompiler_lhaux_differential_tests();
    recompiler_stwux_differential_tests();
    recompiler_stbux_differential_tests();
    recompiler_sthux_differential_tests();
    recompiler_stfsux_differential_tests();
    recompiler_lmw_differential_tests();
    recompiler_stmw_differential_tests();
    recompiler_dcbst_differential_tests();
    recompiler_dcbf_differential_tests();
    recompiler_dcbz_differential_tests();
    recompiler_psq_st_differential_tests();
    recompiler_psq_l_differential_tests();
    recompiler_psq_lx_differential_tests();
    recompiler_psq_lux_differential_tests();
    recompiler_branch_test();
    recompiler_fast_lookup_cache_test();
    recompiler_conditional_branch_tests();
    recompiler_special_register_move_tests();
    recompiler_interpreter_fallback_test();
    recompiler_memory_tests();
    recompiler_lswx_differential_tests();
    recompiler_stswx_differential_tests();
    recompiler_memory_fault_test();
    recompiler_self_modifying_code_test();
    recompiler_stop_reason_test();
}
