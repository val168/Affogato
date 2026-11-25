#pragma once

#include "cpu/espresso/gx2_hle.hpp"
#include "gpu/gx2_register_state.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace affogato::gpu
{

struct LatteVertexAttribute
{
    std::uint32_t semantic_id{};
    std::uint32_t buffer_index{};
    std::uint32_t byte_offset{};
    std::uint32_t byte_size{};
    std::uint32_t component_count{};
    std::uint32_t component_size_bytes{};
    std::uint32_t data_format{};
    std::uint32_t number_format{};
    std::uint32_t endian_swap{};
    std::array<std::uint32_t, 4U> selectors{};

    bool operator==(const LatteVertexAttribute&) const = default;
};

struct LatteVertexShaderInput
{
    std::uint32_t destination_gpr{};
    std::array<std::optional<std::uint32_t>, 4U> components{};

    bool operator==(const LatteVertexShaderInput&) const = default;
};

inline constexpr std::size_t latte_vertex_semantic_entry_count = 32U;
inline constexpr std::uint32_t latte_vertex_semantic_register_base =
    0x00028380U;

// Read the semantic table from one draw's register snapshot. Requiring all
// entries to be explicitly captured avoids guessing reset values.
[[nodiscard]] inline std::optional<std::array<std::uint32_t,
    latte_vertex_semantic_entry_count>>
capture_latte_vertex_semantic_table(
    std::span<const Gx2RegisterSnapshotEntry> register_snapshot)
{
    constexpr std::uint32_t register_stride = sizeof(std::uint32_t);
    constexpr std::uint32_t cleared_semantic_entry = 0xFFFFFFFFU;

    std::array<std::uint32_t, latte_vertex_semantic_entry_count> semantic_table{};
    semantic_table.fill(cleared_semantic_entry);
    std::array<bool, latte_vertex_semantic_entry_count> has_entry{};

    for (const Gx2RegisterSnapshotEntry& entry : register_snapshot)
    {
        if (entry.address < latte_vertex_semantic_register_base)
        {
            continue;
        }

        const std::uint32_t address_offset =
            entry.address - latte_vertex_semantic_register_base;
        if (address_offset >=
                latte_vertex_semantic_entry_count * register_stride ||
            address_offset % register_stride != 0U)
        {
            continue;
        }

        const std::size_t table_index = address_offset / register_stride;
        if (entry.type != cpu::espresso::Gx2CommandType::set_context_register ||
            has_entry[table_index])
        {
            return std::nullopt;
        }

        semantic_table[table_index] = entry.value;
        has_entry[table_index] = true;
    }

    if (!std::all_of(has_entry.begin(), has_entry.end(), [](bool present) {
            return present;
        }))
    {
        return std::nullopt;
    }

    return semantic_table;
}

// Resolve a semantic fetched by the vertex fetch shader to its destination VS
// GPR. The caller supplies VS_GPR_BASE because its value is shader state.
[[nodiscard]] inline std::optional<std::uint32_t>
resolve_latte_vertex_input_gpr(
    std::uint8_t semantic_id,
    std::span<const std::uint32_t> semantic_table,
    std::uint32_t vertex_shader_gpr_base)
{
    constexpr std::uint32_t cleared_semantic_entry = 0xFFFFFFFFU;
    constexpr std::uint32_t semantic_id_mask = 0xFFU;
    constexpr std::uint32_t system_input_gpr_count = 1U;

    if (semantic_table.size() > latte_vertex_semantic_entry_count)
    {
        throw std::runtime_error(
            "Latte vertex semantic table exceeds its 32-entry hardware limit");
    }

    std::optional<std::uint32_t> destination_gpr;
    for (std::size_t table_index = 0U;
         table_index < semantic_table.size();
         ++table_index)
    {
        const std::uint32_t entry = semantic_table[table_index];
        if (entry == cleared_semantic_entry ||
            (entry & semantic_id_mask) != semantic_id)
        {
            continue;
        }

        if (destination_gpr.has_value())
        {
            throw std::runtime_error(
                "Latte vertex semantic maps to multiple shader input GPRs");
        }

        const std::uint64_t resolved_gpr =
            static_cast<std::uint64_t>(vertex_shader_gpr_base) +
            system_input_gpr_count + table_index;
        if (resolved_gpr > std::numeric_limits<std::uint32_t>::max())
        {
            throw std::runtime_error(
                "Latte vertex semantic destination GPR overflows 32 bits");
        }

        destination_gpr = static_cast<std::uint32_t>(resolved_gpr);
    }

    return destination_gpr;
}

[[nodiscard]] inline float decode_ieee754_binary16(std::uint16_t value) noexcept
{
    const std::uint32_t sign =
        static_cast<std::uint32_t>(value & 0x8000U) << 16U;
    std::uint32_t exponent = (value >> 10U) & 0x1FU;
    std::uint32_t significand = value & 0x03FFU;
    std::uint32_t float_bits = sign;

    if (exponent == 0U)
    {
        if (significand != 0U)
        {
            std::int32_t adjusted_exponent = -14;
            while ((significand & 0x0400U) == 0U)
            {
                significand <<= 1U;
                --adjusted_exponent;
            }
            significand &= 0x03FFU;
            const auto float_exponent = static_cast<std::uint32_t>(
                adjusted_exponent + 127);
            float_bits |= float_exponent << 23U;
            float_bits |= significand << 13U;
        }
    }
    else if (exponent == 0x1FU)
    {
        float_bits |= 0x7F800000U | (significand << 13U);
    }
    else
    {
        exponent += 112U;
        float_bits |= exponent << 23U;
        float_bits |= significand << 13U;
    }

    return std::bit_cast<float>(float_bits);
}

// Decode the VTX words emitted by Affogato's GX2InitFetchShaderEx HLE.
// Arbitrary guest Latte fetch programs are intentionally rejected.
[[nodiscard]] inline std::vector<LatteVertexAttribute>
decode_latte_vertex_fetch_shader(
    const cpu::espresso::Gx2FetchShaderBinding& fetch_shader)
{
    constexpr std::uint32_t words_per_vtx_instruction = 4U;
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
        // Word 0 selects semantic mode, so the overlapping word-1 field is
        // the semantic ID rather than a direct destination GPR.
        attribute.semantic_id = word1 & 0x7FU;
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

        // These are the Latte VTX data-format IDs written by
        // GX2InitFetchShaderEx for the host diagnostic position formats
        // supported below.
        switch (attribute.data_format)
        {
        case 0x00U: // GX2_ATTRIB_FORMAT_UNORM_8
            attribute.component_count = 1U;
            attribute.component_size_bytes = sizeof(std::uint8_t);
            break;
        case 0x04U: // GX2_ATTRIB_FORMAT_UNORM_8_8
            attribute.component_count = 2U;
            attribute.component_size_bytes = sizeof(std::uint8_t);
            break;
        case 0x0AU: // GX2_ATTRIB_FORMAT_UNORM_8_8_8_8
            attribute.component_count = 4U;
            attribute.component_size_bytes = sizeof(std::uint8_t);
            break;
        case 0x06U:
            attribute.component_count = 1U;
            attribute.component_size_bytes = sizeof(float);
            break;
        case 0x08U: // GX2_ATTRIB_TYPE_16_16_FLOAT
            attribute.component_count = 2U;
            attribute.component_size_bytes = sizeof(std::uint16_t);
            break;
        case 0x0DU:
            attribute.component_count = 2U;
            attribute.component_size_bytes = sizeof(float);
            break;
        case 0x11U:
            attribute.component_count = 3U;
            attribute.component_size_bytes = sizeof(float);
            break;
        case 0x13U:
            attribute.component_count = 4U;
            attribute.component_size_bytes = sizeof(float);
            break;
        default:
            throw std::runtime_error(
                "Latte vertex fetch decoder does not support data format " +
                std::to_string(attribute.data_format));
        }

        const bool is_unorm8 = attribute.data_format == 0x00U ||
            attribute.data_format == 0x04U ||
            attribute.data_format == 0x0AU;
        const bool is_float16x2 = attribute.data_format == 0x08U;
        if ((is_unorm8 &&
             (attribute.number_format != 0U || attribute.endian_swap != 0U ||
              ((word1 >> 30U) & 0x1U) != 0U)) ||
            (is_float16x2 &&
             (attribute.number_format != 0U || attribute.endian_swap != 1U)) ||
            (!is_unorm8 && !is_float16x2 &&
             (attribute.number_format != 2U || attribute.endian_swap != 2U)))
        {
            throw std::runtime_error(
                "Latte vertex fetch decoder received an unsupported numeric format or endian mode");
        }
        attribute.byte_size =
            attribute.component_count * attribute.component_size_bytes;
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

[[nodiscard]] inline float decode_latte_vertex_component(
    std::span<const std::uint8_t> vertex_bytes,
    const LatteVertexAttribute& attribute,
    std::size_t component_index)
{
    if (component_index >= attribute.component_count ||
        attribute.component_size_bytes == 0U)
    {
        throw std::runtime_error(
            "Latte vertex component index is outside the decoded attribute");
    }

    const std::size_t attribute_offset = attribute.byte_offset;
    if (component_index >
        (std::numeric_limits<std::size_t>::max() - attribute_offset) /
            attribute.component_size_bytes)
    {
        throw std::runtime_error(
            "Latte vertex component offset overflows host address space");
    }

    const std::size_t component_offset = attribute_offset +
        component_index * attribute.component_size_bytes;
    if (component_offset > vertex_bytes.size() ||
        attribute.component_size_bytes > vertex_bytes.size() - component_offset)
    {
        throw std::runtime_error(
            "Latte vertex component exceeds the captured vertex bytes");
    }

    if (attribute.data_format == 0x00U || attribute.data_format == 0x04U ||
        attribute.data_format == 0x0AU)
    {
        constexpr float unorm8_max = 255.0F;
        return static_cast<float>(vertex_bytes[component_offset]) / unorm8_max;
    }

    if (attribute.data_format == 0x08U)
    {
        const std::uint16_t value = static_cast<std::uint16_t>(
            (static_cast<std::uint16_t>(vertex_bytes[component_offset]) << 8U) |
            static_cast<std::uint16_t>(vertex_bytes[component_offset + 1U]));
        return decode_ieee754_binary16(value);
    }

    if (attribute.component_size_bytes == sizeof(float))
    {
        const std::uint32_t value =
            (static_cast<std::uint32_t>(vertex_bytes[component_offset]) << 24U) |
            (static_cast<std::uint32_t>(vertex_bytes[component_offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(vertex_bytes[component_offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(vertex_bytes[component_offset + 3U]);
        return std::bit_cast<float>(value);
    }

    throw std::runtime_error(
        "Latte vertex component uses an unsupported data format");
}

[[nodiscard]] inline std::array<std::optional<std::uint32_t>, 4U>
decode_latte_vertex_attribute_vector(
    std::span<const std::uint8_t> vertex_bytes,
    const LatteVertexAttribute& attribute)
{
    constexpr std::uint32_t selector_zero = 4U;
    constexpr std::uint32_t selector_one = 5U;
    constexpr std::uint32_t reserved_selector = 6U;
    constexpr std::uint32_t selector_mask = 7U;
    constexpr std::uint32_t float_one_bits = 0x3F800000U;

    std::array<std::uint32_t, 4U> fetched_components{};
    if (attribute.component_count > fetched_components.size())
    {
        throw std::runtime_error(
            "Latte vertex attribute exceeds the four-component vector width");
    }

    for (std::size_t component_index = 0U;
         component_index < attribute.component_count;
         ++component_index)
    {
        fetched_components[component_index] = std::bit_cast<std::uint32_t>(
            decode_latte_vertex_component(
                vertex_bytes,
                attribute,
                component_index));
    }

    std::array<std::optional<std::uint32_t>, 4U> result{};
    for (std::size_t destination_component = 0U;
         destination_component < attribute.selectors.size();
         ++destination_component)
    {
        const std::uint32_t selector =
            attribute.selectors[destination_component];
        if (selector < attribute.component_count)
        {
            result[destination_component] = fetched_components[selector];
            continue;
        }

        if (selector == selector_zero)
        {
            result[destination_component] = 0U;
            continue;
        }

        if (selector == selector_one)
        {
            result[destination_component] = float_one_bits;
            continue;
        }

        if (selector == selector_mask)
        {
            continue;
        }

        if (selector == reserved_selector)
        {
            throw std::runtime_error(
                "Latte vertex attribute uses a reserved destination selector");
        }

        throw std::runtime_error(
            "Latte vertex attribute selects a component absent from its format");
    }

    return result;
}

[[nodiscard]] inline std::optional<LatteVertexShaderInput>
decode_latte_vertex_shader_input(
    const LatteVertexAttribute& attribute,
    std::span<const cpu::espresso::Gx2VertexBufferSnapshot> vertex_buffers,
    std::span<const std::uint32_t> semantic_table,
    std::uint32_t vertex_shader_gpr_base,
    std::uint32_t vertex_index)
{
    if (attribute.semantic_id > std::numeric_limits<std::uint8_t>::max())
    {
        throw std::runtime_error(
            "Latte vertex attribute semantic ID exceeds the supported 8-bit table range");
    }

    const std::optional<std::uint32_t> destination_gpr =
        resolve_latte_vertex_input_gpr(
            static_cast<std::uint8_t>(attribute.semantic_id),
            semantic_table,
            vertex_shader_gpr_base);
    if (!destination_gpr.has_value())
    {
        return std::nullopt;
    }

    const auto vertex_buffer = std::ranges::find_if(
        vertex_buffers,
        [&attribute](
            const cpu::espresso::Gx2VertexBufferSnapshot& snapshot)
        {
            return snapshot.binding.index == attribute.buffer_index;
        });
    if (vertex_buffer == vertex_buffers.end())
    {
        throw std::runtime_error(
            "Latte vertex attribute has no draw-time snapshot for its buffer slot");
    }

    const cpu::espresso::Gx2VertexBufferBinding& binding =
        vertex_buffer->binding;
    if (!vertex_buffer->is_valid || binding.stride == 0U ||
        attribute.byte_offset > binding.stride ||
        attribute.byte_size > binding.stride - attribute.byte_offset)
    {
        throw std::runtime_error(
            "Latte vertex attribute has an invalid draw-time buffer snapshot or stride");
    }

    const std::uint64_t vertex_byte_offset =
        static_cast<std::uint64_t>(vertex_index) * binding.stride;
    if (vertex_byte_offset < vertex_buffer->byte_offset ||
        vertex_byte_offset > binding.size ||
        attribute.byte_offset > binding.size - vertex_byte_offset ||
        attribute.byte_size > binding.size - vertex_byte_offset -
            attribute.byte_offset)
    {
        throw std::runtime_error(
            "Latte vertex index or attribute exceeds the guest vertex-buffer binding");
    }

    const std::uint64_t captured_vertex_offset =
        vertex_byte_offset - vertex_buffer->byte_offset;
    if (captured_vertex_offset > vertex_buffer->bytes.size() ||
        binding.stride > vertex_buffer->bytes.size() - captured_vertex_offset)
    {
        throw std::runtime_error(
            "Latte vertex index exceeds its immutable draw-time snapshot");
    }

    const std::span<const std::uint8_t> captured_bytes(vertex_buffer->bytes);
    const auto components = decode_latte_vertex_attribute_vector(
        captured_bytes.subspan(
            static_cast<std::size_t>(captured_vertex_offset),
            binding.stride),
        attribute);
    return LatteVertexShaderInput{*destination_gpr, components};
}

}
