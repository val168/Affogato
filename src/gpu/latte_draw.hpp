#pragma once

#include "cpu/espresso/gx2_hle.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace affogato::gpu
{

// Resolve the captured element stream into the vertex indices observed by a
// non-instanced draw. Index-buffer bytes are decoded according to GX2's
// explicitly endian-qualified index types.
[[nodiscard]] inline std::vector<std::uint32_t>
decode_latte_draw_vertex_indices(
    const cpu::espresso::Gx2Command& command)
{
    constexpr std::uint32_t gx2_index_type_u16_little_endian = 0U;
    constexpr std::uint32_t gx2_index_type_u32_little_endian = 1U;
    constexpr std::uint32_t gx2_index_type_u16_big_endian = 4U;
    constexpr std::uint32_t gx2_index_type_u32_big_endian = 9U;

    if (command.type != cpu::espresso::Gx2CommandType::draw ||
        !command.draw_call)
    {
        throw std::runtime_error(
            "Latte draw index decoding requires a captured draw command");
    }

    const cpu::espresso::Gx2DrawCall& draw = *command.draw_call;
    std::vector<std::uint32_t> vertex_indices;
    vertex_indices.reserve(draw.element_count);

    if (!draw.indexed)
    {
        const std::uint64_t end_vertex =
            static_cast<std::uint64_t>(draw.element_offset) +
            draw.element_count;
        if (end_vertex >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::uint32_t>::max()) + 1U)
        {
            throw std::runtime_error(
                "Latte non-indexed draw vertex range exceeds 32-bit indices");
        }

        for (std::uint32_t vertex_offset = 0U;
             vertex_offset < draw.element_count;
             ++vertex_offset)
        {
            vertex_indices.push_back(draw.element_offset + vertex_offset);
        }

        return vertex_indices;
    }

    std::size_t index_size_bytes = 0U;
    bool little_endian = false;
    switch (draw.index_type)
    {
    case gx2_index_type_u16_little_endian:
        index_size_bytes = sizeof(std::uint16_t);
        little_endian = true;
        break;
    case gx2_index_type_u32_little_endian:
        index_size_bytes = sizeof(std::uint32_t);
        little_endian = true;
        break;
    case gx2_index_type_u16_big_endian:
        index_size_bytes = sizeof(std::uint16_t);
        break;
    case gx2_index_type_u32_big_endian:
        index_size_bytes = sizeof(std::uint32_t);
        break;
    default:
        throw std::runtime_error(
            "Latte draw uses an unsupported GX2 index type");
    }

    if (draw.element_count >
        std::numeric_limits<std::size_t>::max() / index_size_bytes)
    {
        throw std::runtime_error(
            "Latte indexed draw byte count exceeds host size limits");
    }

    const std::size_t expected_byte_count =
        static_cast<std::size_t>(draw.element_count) * index_size_bytes;
    if (!command.draw_index_data ||
        !command.draw_index_data->is_valid ||
        command.draw_index_data->index_type != draw.index_type ||
        command.draw_index_data->byte_offset != draw.element_offset ||
        command.draw_index_data->bytes.size() != expected_byte_count)
    {
        throw std::runtime_error(
            "Latte indexed draw has an invalid or mismatched index snapshot");
    }

    const auto& index_bytes = command.draw_index_data->bytes;
    const std::uint32_t primitive_restart_index = index_size_bytes ==
            sizeof(std::uint16_t)
        ? std::numeric_limits<std::uint16_t>::max()
        : std::numeric_limits<std::uint32_t>::max();
    for (std::size_t index_number = 0U;
         index_number < draw.element_count;
         ++index_number)
    {
        const std::size_t byte_offset = index_number * index_size_bytes;
        std::uint32_t index_value = 0U;
        for (std::size_t byte_number = 0U;
             byte_number < index_size_bytes;
             ++byte_number)
        {
            const std::size_t shift = little_endian
                ? byte_number * 8U
                : (index_size_bytes - byte_number - 1U) * 8U;
            index_value |= static_cast<std::uint32_t>(
                index_bytes[byte_offset + byte_number]) << shift;
        }

        if (index_value == primitive_restart_index)
        {
            throw std::runtime_error(
                "Latte draw index snapshot contains a primitive-restart index");
        }

        const std::uint64_t vertex_index =
            static_cast<std::uint64_t>(index_value) + draw.base_vertex;
        if (vertex_index > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(
                "Latte indexed draw index plus base vertex exceeds 32-bit range");
        }

        vertex_indices.push_back(static_cast<std::uint32_t>(vertex_index));
    }

    return vertex_indices;
}

} // namespace affogato::gpu
