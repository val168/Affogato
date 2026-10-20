#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/recompiler.hpp"

#include <chrono>
#include <cstdint>
#include <iostream>

namespace
{

constexpr std::uint32_t guest_code = 0x100U;
constexpr std::size_t benchmark_steps = 1'000'000U;
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
    std::cout << "Guest steps: " << benchmark_steps
              << "\nInterpreter: " << interpreter_ms << " ms"
              << "\nRecompiler: " << recompiler_ms << " ms"
              << "\nNative guest instructions: "
              << recompiler.statistics().native_instructions - native_before
              << "\nInterpreter fallbacks: "
              << recompiler.statistics().interpreter_fallbacks - fallbacks_before
              << "\nGPR3 interpreter/JIT: " << interpreted.state.gpr[3]
              << '/' << compiled.state.gpr[3] << '\n';

    return interpreter_result.steps == benchmark_steps &&
        recompiler_result.steps == benchmark_steps &&
        interpreted.state.gpr[3] == compiled.state.gpr[3] ? 0 : 1;
}
