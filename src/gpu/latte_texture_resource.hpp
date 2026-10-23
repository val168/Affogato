#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace affogato::gpu
{

using LatteTextureResourceWords = std::array<std::uint32_t, 7U>;
using LatteSamplerWords = std::array<std::uint32_t, 3U>;

struct LatteTextureResourceDescriptor
{
    LatteTextureResourceWords raw_words{};
    std::uint32_t dimension{};
    std::uint32_t tile_mode{};
    bool depth_tile{};
    std::uint32_t pitch_pixels{};
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t depth_or_layers{};
    std::uint32_t format{};
    std::uint32_t image_address{};
    std::uint32_t mipmap_address{};
    std::uint32_t numeric_format{};
    std::uint32_t endian_swap{};
    bool force_degamma{};
    std::array<std::uint8_t, 4U> component_selectors{};
    std::uint32_t base_level{};
    std::uint32_t last_level{};
    std::uint32_t base_array{};
    std::uint32_t last_array{};

    bool operator==(const LatteTextureResourceDescriptor&) const = default;
};

// Decode only valid texture resources. This does not translate hardware
// formats, detile memory, or validate the guest address against GuestMemory;
// those belong to resource upload where the memory map is available.
[[nodiscard]] inline std::optional<LatteTextureResourceDescriptor>
decode_latte_texture_resource(const LatteTextureResourceWords& words)
{
    constexpr std::uint32_t maximum_encoded_address = 0x00FFFFFFU;
    constexpr std::uint32_t maximum_component_selector = 5U;

    const std::uint32_t type = words[6] >> 30U;
    const std::uint32_t base_address = words[2];
    const std::uint32_t mipmap_address = words[3];
    if (type != 2U || base_address > maximum_encoded_address ||
        mipmap_address > maximum_encoded_address)
    {
        return std::nullopt;
    }

    const std::uint32_t dimension = words[0] & 7U;
    const std::uint32_t encoded_depth = (words[1] >> 13U) & 0x1FFFU;
    const std::uint32_t format = (words[1] >> 26U) & 0x3FU;
    const std::uint32_t numeric_format = (words[4] >> 8U) & 3U;
    if (format == 0U || numeric_format == 3U)
    {
        return std::nullopt;
    }
    const std::uint32_t depth_or_layers = dimension == 3U
        ? (encoded_depth + 1U) * 6U
        : encoded_depth + 1U;
    const std::uint32_t component_map = words[4];
    const std::array<std::uint8_t, 4U> selectors{
        static_cast<std::uint8_t>((component_map >> 16U) & 7U),
        static_cast<std::uint8_t>((component_map >> 19U) & 7U),
        static_cast<std::uint8_t>((component_map >> 22U) & 7U),
        static_cast<std::uint8_t>((component_map >> 25U) & 7U)};
    for (const std::uint8_t selector : selectors)
    {
        if (selector > maximum_component_selector)
        {
            return std::nullopt;
        }
    }

    LatteTextureResourceDescriptor result{};
    result.raw_words = words;
    result.dimension = dimension;
    result.tile_mode = (words[0] >> 3U) & 0xFU;
    result.depth_tile = ((words[0] >> 7U) & 1U) != 0U;
    result.pitch_pixels = (((words[0] >> 8U) & 0x7FFU) + 1U) * 8U;
    result.width = ((words[0] >> 19U) & 0x1FFFU) + 1U;
    result.height = (words[1] & 0x1FFFU) + 1U;
    result.depth_or_layers = depth_or_layers;
    result.format = format;
    result.image_address = base_address << 8U;
    result.mipmap_address = mipmap_address << 8U;
    result.numeric_format = numeric_format;
    result.force_degamma = ((words[4] >> 11U) & 1U) != 0U;
    result.endian_swap = (words[4] >> 12U) & 3U;
    result.component_selectors = selectors;
    result.base_level = words[4] >> 28U;
    result.last_level = words[5] & 0xFU;
    result.base_array = (words[5] >> 4U) & 0x1FFFU;
    result.last_array = (words[5] >> 17U) & 0x1FFFU;
    return result;
}

}
