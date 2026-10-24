#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/recompiler.hpp"

#include <chrono>
#include <cstdint>
#include <iomanip>
#include <iostream>

namespace
{

constexpr std::uint32_t guest_code = 0x100U;
constexpr std::size_t benchmark_steps = 1'000'000U;
constexpr std::size_t pc_relative_benchmark_steps = 10'000'000U;
constexpr std::uint32_t loop_instructions = 64U;

void install_loop(affogato::cpu::espresso::EspressoCore& core)
{
    for (std::uint32_t i = 0; i < loop_instructions - 1U; ++i)
    {
        // addi r3,r3,1
        core.memory.write32_be(guest_code + i * 4U, 0x38630001U);
    }
    // b guest_code (relative displacement -252) at instruction 63.
    const std::uint32_t branch_address = guest_code + (loop_instructions - 1U) * 4U;
    core.memory.write32_be(branch_address,
        (18U << 26U) | (static_cast<std::uint32_t>(-252) & 0x03FFFFFCU));
    core.state.cia = guest_code;
}

void install_tiny_loop(affogato::cpu::espresso::EspressoCore& core)
{
    // addi r3,r3,1; b back to the addi. This deliberately makes every
    // compiled block only two guest instructions to expose dispatch overhead.
    core.memory.write32_be(guest_code, 0x38630001U);
    core.memory.write32_be(guest_code + 4U,
        (18U << 26U) | (static_cast<std::uint32_t>(-4) & 0x03FFFFFCU));
    core.state.cia = guest_code;
}

void install_memory_loop(affogato::cpu::espresso::EspressoCore& core)
{
    // lwz r4,0(r5); addi r4,r4,1; stw r4,0(r5); b back to lwz.
    core.memory.write32_be(guest_code, 0x80850000U);
    core.memory.write32_be(guest_code + 4U, 0x38840001U);
    core.memory.write32_be(guest_code + 8U, 0x90850000U);
    core.memory.write32_be(guest_code + 12U,
        (18U << 26U) | (static_cast<std::uint32_t>(-12) & 0x03FFFFFCU));
    core.state.gpr[5] = 0x800U;
    core.state.cia = guest_code;
}

void install_pc_relative_loop(affogato::cpu::espresso::EspressoCore& core)
{
    // bl .+4; mflr r3; b guest_code. This models a common PPC PC-capture
    // sequence and exposes the cost of dispatching at the link branch.
    core.memory.write32_be(guest_code, 0x48000005U);
    core.memory.write32_be(guest_code + 4U, 0x7C6802A6U);
    core.memory.write32_be(guest_code + 8U, 0x4BFFFFF8U);
    core.state.cia = guest_code;
}

}

int main()
{
    using namespace affogato::cpu::espresso;
    EspressoCore interpreted(0x1000U);
    EspressoCore compiled(0x1000U);
    install_loop(interpreted);
    install_loop(compiled);
    Recompiler recompiler;

    static_cast<void>(recompiler.run(compiled, loop_instructions)); // compile/warm block cache
    compiled.state.cia = guest_code;
    compiled.state.gpr[3] = 0U;
    const std::uint64_t native_before = recompiler.statistics().native_instructions;
    const std::uint64_t fallbacks_before = recompiler.statistics().interpreter_fallbacks;
    const RecompilerStatistics stats_before = recompiler.statistics();

    const auto interpreter_start = std::chrono::steady_clock::now();
    const RunResult interpreter_result = interpreted.run(benchmark_steps);
    const auto interpreter_end = std::chrono::steady_clock::now();

    const auto recompiler_start = std::chrono::steady_clock::now();
    const RunResult recompiler_result = recompiler.run(compiled, benchmark_steps);
    const auto recompiler_end = std::chrono::steady_clock::now();

    const auto interpreter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        interpreter_end - interpreter_start).count();
    const auto recompiler_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        recompiler_end - recompiler_start).count();
    const RecompilerStatistics& stats_after = recompiler.statistics();
    std::cout << "Guest steps: " << benchmark_steps
              << "\nInterpreter: " << interpreter_ms << " ms"
              << "\nRecompiler: " << recompiler_ms << " ms"
              << "\nSpeedup: " << std::fixed << std::setprecision(2)
              << static_cast<double>(interpreter_ms) /
                    static_cast<double>(recompiler_ms)
              << "x vs interpreter"
              << "\nCompiled blocks: "
              << stats_after.compiled_blocks - stats_before.compiled_blocks
              << "\nDispatcher entries: "
              << stats_after.dispatcher_entries - stats_before.dispatcher_entries
              << "\nBlock cache hits: "
              << stats_after.cache_hits - stats_before.cache_hits
              << "\nLast-block cache hits: "
              << stats_after.last_block_cache_hits - stats_before.last_block_cache_hits
              << "\nBlock lookup misses: "
              << stats_after.block_lookup_misses - stats_before.block_lookup_misses
              << "\nDirect-link hits: "
              << stats_after.direct_link_hits - stats_before.direct_link_hits
              << "\nNative guest instructions: "
              << recompiler.statistics().native_instructions - native_before
              << "\nMemory slow-path calls: "
              << stats_after.memory_slow_path_calls - stats_before.memory_slow_path_calls
              << "\nInterpreter fallbacks: "
              << recompiler.statistics().interpreter_fallbacks - fallbacks_before
              << "\nGPR3 interpreter/JIT: " << interpreted.state.gpr[3]
              << '/' << compiled.state.gpr[3] << '\n';

    EspressoCore tiny_interpreted(0x1000U);
    EspressoCore tiny_compiled(0x1000U);
    install_tiny_loop(tiny_interpreted);
    install_tiny_loop(tiny_compiled);
    Recompiler tiny_recompiler;
    static_cast<void>(tiny_recompiler.run(tiny_compiled, 2U));
    tiny_compiled.state.cia = guest_code;
    tiny_compiled.state.gpr[3] = 0U;
    const RecompilerStatistics tiny_stats_before = tiny_recompiler.statistics();
    const auto tiny_interpreter_start = std::chrono::steady_clock::now();
    const RunResult tiny_interpreter_result = tiny_interpreted.run(benchmark_steps);
    const auto tiny_interpreter_end = std::chrono::steady_clock::now();
    const auto tiny_recompiler_start = std::chrono::steady_clock::now();
    const RunResult tiny_recompiler_result = tiny_recompiler.run(tiny_compiled, benchmark_steps);
    const auto tiny_recompiler_end = std::chrono::steady_clock::now();
    const auto tiny_interpreter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tiny_interpreter_end - tiny_interpreter_start).count();
    const auto tiny_recompiler_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        tiny_recompiler_end - tiny_recompiler_start).count();
    const RecompilerStatistics& tiny_stats_after = tiny_recompiler.statistics();
    std::cout << "\nDispatch-heavy workload (2 guest instructions/block):"
              << "\nInterpreter: " << tiny_interpreter_ms << " ms"
              << "\nRecompiler: " << tiny_recompiler_ms << " ms"
              << "\nSpeedup: " << std::fixed << std::setprecision(2)
              << static_cast<double>(tiny_interpreter_ms) /
                    static_cast<double>(tiny_recompiler_ms)
              << "x vs interpreter"
              << "\nDispatcher entries: "
              << tiny_stats_after.dispatcher_entries -
                    tiny_stats_before.dispatcher_entries
              << "\nLast-block cache hits: "
              << tiny_stats_after.last_block_cache_hits -
                    tiny_stats_before.last_block_cache_hits
              << "\nNative guest instructions: "
              << tiny_stats_after.native_instructions -
                    tiny_stats_before.native_instructions
              << "\nInterpreter fallbacks: "
              << tiny_stats_after.interpreter_fallbacks -
                    tiny_stats_before.interpreter_fallbacks
              << "\nGPR3 interpreter/JIT: " << tiny_interpreted.state.gpr[3]
              << '/' << tiny_compiled.state.gpr[3] << '\n';

    EspressoCore pc_interpreted(0x1000U);
    EspressoCore pc_compiled(0x1000U);
    install_pc_relative_loop(pc_interpreted);
    install_pc_relative_loop(pc_compiled);
    Recompiler pc_recompiler;
    static_cast<void>(pc_recompiler.run(pc_compiled, 3U));
    pc_compiled.state.cia = guest_code;
    pc_compiled.state.gpr[3] = 0U;
    pc_compiled.state.lr = 0U;
    const RecompilerStatistics pc_stats_before = pc_recompiler.statistics();
    const auto pc_interpreter_start = std::chrono::steady_clock::now();
    const RunResult pc_interpreter_result =
        pc_interpreted.run(pc_relative_benchmark_steps);
    const auto pc_interpreter_end = std::chrono::steady_clock::now();
    const auto pc_recompiler_start = std::chrono::steady_clock::now();
    const RunResult pc_recompiler_result =
        pc_recompiler.run(pc_compiled, pc_relative_benchmark_steps);
    const auto pc_recompiler_end = std::chrono::steady_clock::now();
    const auto pc_interpreter_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        pc_interpreter_end - pc_interpreter_start).count();
    const auto pc_recompiler_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        pc_recompiler_end - pc_recompiler_start).count();
    const RecompilerStatistics& pc_stats_after = pc_recompiler.statistics();
    std::cout << "\nPC-relative link-register workload (3 guest instructions/block):"
              << "\nGuest steps: " << pc_relative_benchmark_steps
              << "\nInterpreter: " << pc_interpreter_ms << " ms"
              << "\nRecompiler: " << pc_recompiler_ms << " ms"
              << "\nSpeedup: " << std::fixed << std::setprecision(2)
              << static_cast<double>(pc_interpreter_ms) /
                    static_cast<double>(pc_recompiler_ms)
              << "x vs interpreter"
              << "\nDispatcher entries: "
              << pc_stats_after.dispatcher_entries - pc_stats_before.dispatcher_entries
              << "\nCompiled blocks: "
              << pc_stats_after.compiled_blocks - pc_stats_before.compiled_blocks
              << "\nNative guest instructions: "
              << pc_stats_after.native_instructions - pc_stats_before.native_instructions
              << "\nInterpreter fallbacks: "
              << pc_stats_after.interpreter_fallbacks -
                    pc_stats_before.interpreter_fallbacks
              << "\nGPR3 interpreter/JIT: " << pc_interpreted.state.gpr[3]
              << '/' << pc_compiled.state.gpr[3] << '\n';

    EspressoCore memory_interpreted(0x1000U);
    EspressoCore memory_compiled(0x1000U);
    install_memory_loop(memory_interpreted);
    install_memory_loop(memory_compiled);
    Recompiler memory_recompiler;
    static_cast<void>(memory_recompiler.run(memory_compiled, 4U));
    memory_compiled.state.cia = guest_code;
    memory_compiled.state.gpr[4] = 0U;
    memory_compiled.memory.write32_be(0x800U, 0U);
    const auto memory_interpreter_start = std::chrono::steady_clock::now();
    const RunResult memory_interpreter_result = memory_interpreted.run(benchmark_steps);
    const auto memory_interpreter_end = std::chrono::steady_clock::now();
    const auto memory_recompiler_start = std::chrono::steady_clock::now();
    const RunResult memory_recompiler_result =
        memory_recompiler.run(memory_compiled, benchmark_steps);
    const auto memory_recompiler_end = std::chrono::steady_clock::now();
    const auto memory_interpreter_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            memory_interpreter_end - memory_interpreter_start).count();
    const auto memory_recompiler_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            memory_recompiler_end - memory_recompiler_start).count();
    std::cout << "\nMemory-heavy workload (lwz/addi/stw/branch loop):"
              << "\nInterpreter: " << memory_interpreter_ms << " ms"
              << "\nRecompiler: " << memory_recompiler_ms << " ms"
              << "\nSpeedup: " << std::fixed << std::setprecision(2)
              << static_cast<double>(memory_interpreter_ms) /
                    static_cast<double>(memory_recompiler_ms)
              << "x vs interpreter"
              << "\nGuest memory result interpreter/JIT: "
              << memory_interpreted.memory.read32_be(0x800U) << '/'
              << memory_compiled.memory.read32_be(0x800U)
              << "\nMemory slow-path calls: "
              << memory_recompiler.statistics().memory_slow_path_calls
              << "\nInterpreter fallbacks: "
              << memory_recompiler.statistics().interpreter_fallbacks << '\n';

    return interpreter_result.steps == benchmark_steps &&
        recompiler_result.steps == benchmark_steps &&
        interpreted.state.gpr[3] == compiled.state.gpr[3] &&
        tiny_interpreter_result.steps == benchmark_steps &&
        tiny_recompiler_result.steps == benchmark_steps &&
        tiny_interpreted.state.gpr[3] == tiny_compiled.state.gpr[3] &&
        pc_interpreter_result.steps == pc_relative_benchmark_steps &&
        pc_recompiler_result.steps == pc_relative_benchmark_steps &&
        pc_interpreted.state.gpr[3] == pc_compiled.state.gpr[3] &&
        memory_interpreter_result.steps == benchmark_steps &&
        memory_recompiler_result.steps == benchmark_steps &&
        memory_interpreted.memory.read32_be(0x800U) ==
            memory_compiled.memory.read32_be(0x800U) ? 0 : 1;
}
