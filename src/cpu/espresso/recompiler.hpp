#pragma once

#include "cpu/espresso/interpreter.hpp"
#include "cpu/espresso/decoder.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <unordered_map>

namespace affogato::cpu::espresso
{

struct RecompilerStatistics
{
    std::uint64_t compilation_time_ns{};
    std::uint64_t native_execution_time_ns{};
    std::uint64_t compiled_blocks{};
    std::uint64_t cache_hits{};
    std::uint64_t last_block_cache_hits{};
    std::uint64_t block_lookup_misses{};
    std::uint64_t dispatcher_entries{};
    std::uint64_t direct_link_hits{};
    std::uint64_t memory_slow_path_calls{};
    std::uint64_t native_instructions{};
    std::uint64_t interpreter_fallbacks{};
    std::unordered_map<std::uint16_t, std::uint64_t> fallbacks_by_opcode;
};

// An opt-in, instance-local x86-64 JIT. Unsupported JIT operations are always
// delegated to EspressoCore::step(); the interpreter remains the oracle.
class Recompiler
{
public:
    Recompiler();
    ~Recompiler();
    Recompiler(const Recompiler&) = delete;
    Recompiler& operator=(const Recompiler&) = delete;
    Recompiler(Recompiler&&) noexcept;
    Recompiler& operator=(Recompiler&&) noexcept;

    [[nodiscard]] RunResult run(EspressoCore& core, std::size_t max_steps);
    void clear_cache() noexcept;
    [[nodiscard]] const RecompilerStatistics& statistics() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}
