#pragma once

#include "cpu/espresso/gx2_hle.hpp"

#include <cstddef>
#include <cstdint>
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
