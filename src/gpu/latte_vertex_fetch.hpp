#pragma once

#include "cpu/espresso/gx2_hle.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace affogato::gpu
{

struct LatteVertexAttribute
{
    std::uint32_t location{};
    std::uint32_t buffer_index{};
    std::uint32_t byte_offset{};
    std::uint32_t byte_size{};
    std::uint32_t component_count{};
    std::uint32_t data_format{};
    std::uint32_t number_format{};
    std::uint32_t endian_swap{};
    std::array<std::uint32_t, 4U> selectors{};

    bool operator==(const LatteVertexAttribute&) const = default;
};

// Decode the VTX words emitted by Affogato's GX2InitFetchShaderEx HLE.
// Arbitrary guest Latte fetch programs are intentionally rejected.
[[nodiscard]] inline std::vector<LatteVertexAttribute>
decode_latte_vertex_fetch_shader(
    const cpu::espresso::Gx2FetchShaderBinding& fetch_shader)
{
    constexpr std::uint32_t words_per_vtx_instruction = 4U;
    constexpr std::uint32_t bytes_per_component = sizeof(float);
    if (fetch_shader.type != 0U || fetch_shader.divisor_count != 0U ||
        fetch_shader.attribute_count > std::numeric_limits<std::uint32_t>::max() /
            (words_per_vtx_instruction * sizeof(std::uint32_t)) ||
        fetch_shader.attribute_count > std::numeric_limits<std::size_t>::max() /
            words_per_vtx_instruction ||
        fetch_shader.program_words.size() !=
            static_cast<std::size_t>(fetch_shader.attribute_count) *
                words_per_vtx_instruction ||
        fetch_shader.program_size !=
            fetch_shader.attribute_count * words_per_vtx_instruction *
                sizeof(std::uint32_t))
    {
        throw std::runtime_error(
            "Latte vertex fetch decoder requires an HLE-generated, non-instanced fetch program");
    }

    std::vector<LatteVertexAttribute> attributes;
    attributes.reserve(fetch_shader.attribute_count);
    for (std::uint32_t index = 0U; index < fetch_shader.attribute_count; ++index)
    {
        const std::size_t word_index =
            static_cast<std::size_t>(index) * words_per_vtx_instruction;
        const std::uint32_t word0 = fetch_shader.program_words[word_index];
        const std::uint32_t word1 = fetch_shader.program_words[word_index + 1U];
        const std::uint32_t word2 = fetch_shader.program_words[word_index + 2U];
        const std::uint32_t word3 = fetch_shader.program_words[word_index + 3U];

        if ((word0 & 0xFFU) != 1U || word3 != 0U)
        {
            throw std::runtime_error(
                "Latte vertex fetch decoder encountered an unsupported instruction layout");
        }

        LatteVertexAttribute attribute{};
        attribute.location = word1 & 0x7FU;
        attribute.buffer_index = (word0 >> 8U) & 0xFU;
        attribute.byte_offset = word2 & 0xFFFFU;
        attribute.endian_swap = (word2 >> 16U) & 0x7U;
        attribute.data_format = (word1 >> 22U) & 0x3FU;
        attribute.number_format = (word1 >> 28U) & 0x3U;
        attribute.selectors = {
            (word1 >> 9U) & 0x7U,
            (word1 >> 12U) & 0x7U,
            (word1 >> 15U) & 0x7U,
            (word1 >> 18U) & 0x7U};

        // GX2AttribFormat FLOAT_32 through FLOAT_32_32_32_32 use these
        // Latte data-format IDs after GX2InitFetchShaderEx encodes format&0x3F.
        switch (attribute.data_format)
        {
        case 0x06U:
            attribute.component_count = 1U;
            break;
        case 0x0DU:
            attribute.component_count = 2U;
            break;
        case 0x11U:
            attribute.component_count = 3U;
            break;
        case 0x13U:
            attribute.component_count = 4U;
            break;
        default:
            throw std::runtime_error(
                "Latte vertex fetch decoder does not support data format " +
                std::to_string(attribute.data_format));
        }

        if (attribute.number_format != 2U || attribute.endian_swap != 2U)
        {
            throw std::runtime_error(
                "Latte vertex fetch decoder currently supports 32-bit float attributes with 8-in-32 endian swap");
        }
        attribute.byte_size = attribute.component_count * bytes_per_component;
        const std::uint32_t encoded_fetch_size = ((word0 >> 26U) & 0x3FU) + 1U;
        if (encoded_fetch_size != attribute.byte_size)
        {
            throw std::runtime_error(
                "Latte vertex fetch byte count does not match its decoded format");
        }

        attributes.push_back(attribute);
    }

    return attributes;
}

}
