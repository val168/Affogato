#pragma once

#include "gpu/latte_shader_program.hpp"
#include "gpu/latte_vertex_fetch.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace affogato::gpu
{

struct LatteVertexShaderSubsetResult
{
    std::size_t fetched_attribute_count{};
    LatteShaderExecutionResult shader_execution;
};

// Execute only the supported CPU shader subset after resolving one explicitly
// selected vertex through Affogato's HLE-generated fetch-shader subset. The
// caller supplies reusable register scratch, which receives the fetched inputs.
[[nodiscard]] inline LatteVertexShaderSubsetResult
execute_latte_vertex_shader_subset(
    std::span<const std::uint32_t> guest_big_endian_shader_words,
    const cpu::espresso::Gx2FetchShaderBinding& fetch_shader,
    std::span<const cpu::espresso::Gx2VertexBufferSnapshot> vertex_buffers,
    std::span<const std::uint32_t> semantic_table,
    std::uint32_t vertex_shader_gpr_base,
    std::uint32_t vertex_index,
    std::span<LatteShaderRegister> shader_registers)
{
    const std::vector<LatteVertexAttribute> attributes =
        decode_latte_vertex_fetch_shader(fetch_shader);
    const std::size_t fetched_attribute_count =
        apply_latte_vertex_shader_inputs(
            attributes,
            vertex_buffers,
            semantic_table,
            vertex_shader_gpr_base,
            vertex_index,
            shader_registers);

    return {
        fetched_attribute_count,
        execute_latte_shader_subset(
            guest_big_endian_shader_words,
            shader_registers)};
}

} // namespace affogato::gpu
