#pragma once

#include "cpu/espresso/gx2_hle.hpp"
#include "gpu/latte_shader_interpreter.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace affogato::gpu
{

enum class LattePixelInputDefaultValue : std::uint8_t
{
    zero,
    zero_rgb_one_alpha,
    one_rgb_zero_alpha,
    one,
};

struct LattePixelShaderInputControl
{
    std::uint8_t semantic{};
    LattePixelInputDefaultValue default_value{};
    std::uint8_t cylindrical_wrap_mask{};
    bool flat_shade{};
    bool centroid{};
    bool linear{};
    bool point_sprite_texture{};
    bool sample{};

    bool operator==(const LattePixelShaderInputControl&) const = default;
};

struct LatteVertexShaderOutputId
{
    std::array<std::uint8_t, 4U> semantics{};

    bool operator==(const LatteVertexShaderOutputId&) const = default;
};

struct LatteVertexShaderOutputInterface
{
    bool outputs_per_component{};
    std::uint8_t parameter_export_count{};
    std::vector<LatteVertexShaderOutputId> output_ids;

    bool operator==(const LatteVertexShaderOutputInterface&) const = default;
};

[[nodiscard]] inline LatteVertexShaderOutputInterface
decode_latte_vertex_shader_output_interface(
    const cpu::espresso::Gx2ProgramShaderBinding& shader)
{
    constexpr std::size_t minimum_vertex_shader_descriptor_words = 4U;
    constexpr std::size_t output_configuration_register_index = 2U;
    constexpr std::size_t output_id_count_register_index = 3U;
    constexpr std::size_t first_output_id_register_index = 4U;
    constexpr std::size_t maximum_vertex_shader_output_ids = 10U;
    constexpr std::uint32_t output_count_mask = 0x3EU;
    constexpr std::uint32_t output_count_shift = 1U;
    constexpr std::uint32_t maximum_parameter_exports = 32U;
    constexpr std::uint32_t output_per_component_mask = 1U;

    if (shader.stage != cpu::espresso::Gx2ProgramStage::vertex)
    {
        throw std::invalid_argument(
            "Latte vertex output decoding requires a vertex shader descriptor");
    }
    if (shader.descriptor_words.size() < minimum_vertex_shader_descriptor_words)
    {
        throw std::invalid_argument(
            "Latte vertex shader descriptor is missing output registers");
    }

    const std::uint32_t output_id_count =
        shader.descriptor_words[output_id_count_register_index];
    if (output_id_count > maximum_vertex_shader_output_ids ||
        shader.descriptor_words.size() <
            first_output_id_register_index + output_id_count)
    {
        throw std::invalid_argument(
            "Latte vertex shader output ID range exceeds its descriptor");
    }

    LatteVertexShaderOutputInterface output_interface{};
    output_interface.outputs_per_component =
        (shader.descriptor_words[output_configuration_register_index] &
         output_per_component_mask) != 0U;
    output_interface.parameter_export_count = static_cast<std::uint8_t>(
        ((shader.descriptor_words[output_configuration_register_index] &
          output_count_mask) >> output_count_shift) + 1U);
    if (output_interface.parameter_export_count > maximum_parameter_exports)
    {
        throw std::invalid_argument(
            "Latte vertex shader parameter export count exceeds hardware capacity");
    }
    const std::size_t required_output_id_count =
        (output_interface.parameter_export_count + 3U) / 4U;
    if (output_id_count < required_output_id_count)
    {
        throw std::invalid_argument(
            "Latte vertex shader is missing parameter output semantic IDs");
    }

    output_interface.output_ids.reserve(output_id_count);
    for (std::uint32_t output_index = 0U;
         output_index < output_id_count;
         ++output_index)
    {
        const std::uint32_t packed_semantics = shader.descriptor_words[
            first_output_id_register_index + output_index];
        output_interface.output_ids.push_back({{
            static_cast<std::uint8_t>(packed_semantics & 0xFFU),
            static_cast<std::uint8_t>((packed_semantics >> 8U) & 0xFFU),
            static_cast<std::uint8_t>((packed_semantics >> 16U) & 0xFFU),
            static_cast<std::uint8_t>((packed_semantics >> 24U) & 0xFFU)}});
    }

    return output_interface;
}

enum class LattePixelInputSource : std::uint8_t
{
    vertex_parameter,
    default_value,
};

struct LatteLinkedPixelShaderInput
{
    std::size_t pixel_input_index{};
    std::uint8_t semantic{};
    LattePixelInputSource source{};
    std::optional<std::uint32_t> vertex_export_index;
    std::array<std::uint32_t, 4U> default_component_bits{};
    bool flat_shade{};
    bool centroid{};
    bool linear{};
    bool point_sprite_texture{};
    bool sample{};

    bool operator==(const LatteLinkedPixelShaderInput&) const = default;
};

[[nodiscard]] inline std::array<std::uint32_t, 4U>
get_latte_pixel_input_default_bits(LattePixelInputDefaultValue default_value)
{
    constexpr std::uint32_t one_float_bits = 0x3F800000U;

    switch (default_value)
    {
    case LattePixelInputDefaultValue::zero:
        return {0U, 0U, 0U, 0U};
    case LattePixelInputDefaultValue::zero_rgb_one_alpha:
        return {0U, 0U, 0U, one_float_bits};
    case LattePixelInputDefaultValue::one_rgb_zero_alpha:
        return {one_float_bits, one_float_bits, one_float_bits, 0U};
    case LattePixelInputDefaultValue::one:
        return {one_float_bits, one_float_bits, one_float_bits, one_float_bits};
    }

    throw std::invalid_argument("Latte pixel input default value is out of range");
}

[[nodiscard]] inline std::vector<LatteLinkedPixelShaderInput>
link_latte_pixel_shader_inputs(
    const LatteVertexShaderOutputInterface& vertex_output_interface,
    std::span<const LatteShaderExportValue> vertex_exports,
    std::span<const LattePixelShaderInputControl> pixel_input_controls)
{
    if (vertex_output_interface.outputs_per_component)
    {
        throw std::invalid_argument(
            "Latte per-component vertex output linking is unsupported");
    }

    const std::size_t required_output_id_count =
        (vertex_output_interface.parameter_export_count + 3U) / 4U;
    if (vertex_output_interface.parameter_export_count == 0U ||
        vertex_output_interface.parameter_export_count > 32U ||
        vertex_output_interface.output_ids.size() < required_output_id_count)
    {
        throw std::invalid_argument(
            "Latte vertex output interface is incomplete or invalid");
    }

    for (std::size_t export_list_index = 0U;
         export_list_index < vertex_exports.size();
         ++export_list_index)
    {
        const LatteShaderExportValue& vertex_export =
            vertex_exports[export_list_index];
        if (vertex_export.type != LatteExportType::parameter)
        {
            continue;
        }

        const std::uint32_t export_index = vertex_export.array_index;
        if (export_index >= vertex_output_interface.parameter_export_count)
        {
            throw std::invalid_argument(
                "Latte vertex parameter export index exceeds VS_EXPORT_COUNT");
        }
        for (std::size_t previous_index = 0U;
             previous_index < export_list_index;
             ++previous_index)
        {
            const LatteShaderExportValue& previous_export =
                vertex_exports[previous_index];
            if (previous_export.type == LatteExportType::parameter &&
                previous_export.array_index == export_index)
            {
                throw std::invalid_argument(
                    "Latte vertex parameter export index is duplicated");
            }
        }
    }

    std::vector<LatteLinkedPixelShaderInput> linked_inputs;
    linked_inputs.reserve(pixel_input_controls.size());
    for (std::size_t pixel_input_index = 0U;
         pixel_input_index < pixel_input_controls.size();
         ++pixel_input_index)
    {
        const LattePixelShaderInputControl& pixel_input =
            pixel_input_controls[pixel_input_index];
        std::optional<std::uint32_t> matching_export_index;
        for (const LatteShaderExportValue& vertex_export : vertex_exports)
        {
            if (vertex_export.type != LatteExportType::parameter)
            {
                continue;
            }

            const std::uint32_t export_index = vertex_export.array_index;
            const std::size_t semantic_register_index = export_index / 4U;
            const std::size_t semantic_lane_index = export_index % 4U;
            const std::uint8_t output_semantic = vertex_output_interface
                .output_ids[semantic_register_index]
                .semantics[semantic_lane_index];
            if (output_semantic != pixel_input.semantic)
            {
                continue;
            }
            if (matching_export_index.has_value())
            {
                throw std::invalid_argument(
                    "Latte pixel input semantic matches multiple vertex exports");
            }
            matching_export_index = export_index;
        }

        LatteLinkedPixelShaderInput linked_input{};
        linked_input.pixel_input_index = pixel_input_index;
        linked_input.semantic = pixel_input.semantic;
        linked_input.vertex_export_index = matching_export_index;
        linked_input.source = matching_export_index.has_value()
            ? LattePixelInputSource::vertex_parameter
            : LattePixelInputSource::default_value;
        linked_input.default_component_bits =
            get_latte_pixel_input_default_bits(pixel_input.default_value);
        linked_input.flat_shade = pixel_input.flat_shade;
        linked_input.centroid = pixel_input.centroid;
        linked_input.linear = pixel_input.linear;
        linked_input.point_sprite_texture = pixel_input.point_sprite_texture;
        linked_input.sample = pixel_input.sample;
        linked_inputs.push_back(linked_input);
    }

    return linked_inputs;
}

[[nodiscard]] inline std::vector<LattePixelShaderInputControl>
decode_latte_pixel_shader_input_controls(
    const cpu::espresso::Gx2ProgramShaderBinding& shader)
{
    constexpr std::size_t pixel_shader_register_count = 41U;
    constexpr std::size_t input_count_register_index = 4U;
    constexpr std::size_t first_input_register_index = 5U;
    constexpr std::size_t maximum_pixel_shader_inputs = 32U;

    if (shader.stage != cpu::espresso::Gx2ProgramStage::pixel)
    {
        throw std::invalid_argument(
            "Latte pixel input decoding requires a pixel shader descriptor");
    }
    if (shader.descriptor_words.size() < pixel_shader_register_count)
    {
        throw std::invalid_argument(
            "Latte pixel shader descriptor is missing register words");
    }

    const std::uint32_t input_count =
        shader.descriptor_words[input_count_register_index];
    if (input_count > maximum_pixel_shader_inputs)
    {
        throw std::invalid_argument(
            "Latte pixel shader input count exceeds SPI input control capacity");
    }

    std::vector<LattePixelShaderInputControl> input_controls;
    input_controls.reserve(input_count);
    for (std::uint32_t input_index = 0U;
         input_index < input_count;
         ++input_index)
    {
        const std::uint32_t control = shader.descriptor_words[
            first_input_register_index + input_index];
        input_controls.push_back({
            static_cast<std::uint8_t>(control & 0xFFU),
            static_cast<LattePixelInputDefaultValue>((control >> 8U) & 0x3U),
            static_cast<std::uint8_t>((control >> 13U) & 0xFU),
            (control & (1U << 10U)) != 0U,
            (control & (1U << 11U)) != 0U,
            (control & (1U << 12U)) != 0U,
            (control & (1U << 17U)) != 0U,
            (control & (1U << 18U)) != 0U});
    }

    return input_controls;
}

}
