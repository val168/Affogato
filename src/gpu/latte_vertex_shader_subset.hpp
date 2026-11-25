#pragma once

#include "gpu/gx2_register_state.hpp"
#include "gpu/latte_shader_program.hpp"
#include "gpu/latte_vertex_fetch.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace affogato::gpu
{

struct LatteVertexShaderSubsetResult
{
    std::size_t fetched_attribute_count{};
    LatteShaderExecutionResult shader_execution;
};

[[nodiscard]] inline LatteShaderConstantFile
capture_latte_shader_constant_file(
    std::span<const Gx2RegisterSnapshotEntry> registers,
    Gx2ShaderStage stage)
{
    constexpr std::uint32_t pixel_constant_register_base = 0x00030000U;
    constexpr std::uint32_t vertex_constant_register_base = 0x00031000U;
    constexpr std::uint32_t constant_file_byte_size =
        256U * 4U * sizeof(std::uint32_t);

    std::uint32_t register_base{};
    switch (stage)
    {
    case Gx2ShaderStage::pixel:
        register_base = pixel_constant_register_base;
        break;
    case Gx2ShaderStage::vertex:
        register_base = vertex_constant_register_base;
        break;
    default:
        throw std::runtime_error(
            "Latte CPU subset constant files support vertex or pixel stages");
    }

    const std::uint32_t register_limit = register_base + constant_file_byte_size;
    LatteShaderConstantFile constant_file{};
    for (const Gx2RegisterSnapshotEntry& register_entry : registers)
    {
        if (register_entry.address < register_base ||
            register_entry.address >= register_limit)
        {
            continue;
        }

        const std::uint32_t byte_offset =
            register_entry.address - register_base;
        if ((byte_offset % sizeof(std::uint32_t)) != 0U)
        {
            throw std::runtime_error(
                "Latte shader constant register address is not dword-aligned");
        }
        if (register_entry.known_mask != 0xFFFFFFFFU)
        {
            continue;
        }

        const std::uint32_t word_index =
            byte_offset / sizeof(std::uint32_t);
        const std::size_t vector_index = word_index / 4U;
        const std::size_t component_index = word_index % 4U;
        constant_file.entries[vector_index][component_index] =
            register_entry.value;
    }

    return constant_file;
}

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
    std::span<LatteShaderRegister> shader_registers,
    const LatteShaderConstantFile* constant_file = nullptr)
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
        execute_latte_vertex_shader_program_subset(
            guest_big_endian_shader_words,
            shader_registers,
            constant_file)};
}

[[nodiscard]] inline LatteVertexShaderSubsetResult
execute_latte_vertex_shader_subset(
    const Gx2DrawSnapshotView& draw_snapshot,
    std::uint32_t vertex_shader_gpr_base,
    std::uint32_t vertex_index,
    std::span<LatteShaderRegister> shader_registers)
{
    constexpr std::size_t vertex_shader_stage_index =
        static_cast<std::size_t>(cpu::espresso::Gx2ProgramStage::vertex);

    if (draw_snapshot.command.type != cpu::espresso::Gx2CommandType::draw ||
        !draw_snapshot.command.draw_call)
    {
        throw std::runtime_error(
            "Latte vertex shader subset requires a captured GX2 draw command");
    }
    if (!draw_snapshot.fetch_shader)
    {
        throw std::runtime_error(
            "Latte vertex shader subset requires a draw-time fetch shader");
    }

    const std::optional<cpu::espresso::Gx2ProgramShaderBinding>&
        vertex_shader =
            draw_snapshot.program_shaders[vertex_shader_stage_index];
    if (!vertex_shader.has_value() ||
        vertex_shader->stage != cpu::espresso::Gx2ProgramStage::vertex)
    {
        throw std::runtime_error(
            "Latte vertex shader subset requires a draw-time vertex shader");
    }

    const std::optional<std::array<std::uint32_t,
        latte_vertex_semantic_entry_count>> semantic_table =
        capture_latte_vertex_semantic_table(draw_snapshot.registers);
    if (!semantic_table.has_value())
    {
        throw std::runtime_error(
            "Latte vertex shader subset requires a complete draw-time semantic table");
    }

    const LatteShaderConstantFile constant_file =
        capture_latte_shader_constant_file(
            draw_snapshot.registers,
            Gx2ShaderStage::vertex);

    return execute_latte_vertex_shader_subset(
        vertex_shader->program_words,
        *draw_snapshot.fetch_shader,
        draw_snapshot.command.draw_vertex_data,
        *semantic_table,
        vertex_shader_gpr_base,
        vertex_index,
        shader_registers,
        &constant_file);
}

} // namespace affogato::gpu
