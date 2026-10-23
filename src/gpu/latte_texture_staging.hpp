#pragma once

#include "gpu/latte_texture_resource.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace affogato::gpu
{

struct Rgba8TextureStaging
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;

    bool operator==(const Rgba8TextureStaging&) const = default;
};

// Converts explicitly supported linear GX2 formats to tightly packed RGBA8
// bytes suitable for a Vulkan staging buffer. GuestMemory range
// validation and snapshotting are the caller's responsibility; guest_bytes
// starts at descriptor.image_address.
[[nodiscard]] inline std::optional<Rgba8TextureStaging>
stage_linear_rgba8_texture(
    const LatteTextureResourceDescriptor& descriptor,
    std::span<const std::uint8_t> guest_bytes)
{
    constexpr std::uint32_t gx2_2d_dimension = 2U;
    constexpr std::uint32_t gx2_linear_aligned_tile_mode = 1U;
    constexpr std::uint32_t gx2_unorm_r8_format = 0x01U;
    constexpr std::uint32_t gx2_unorm_rg8_format = 0x07U;
    constexpr std::uint32_t gx2_unorm_rgba8_format = 0x1AU;
    constexpr std::uint32_t gx2_unorm_numeric_format = 0U;
    constexpr std::uint32_t rgba8_bytes_per_pixel = 4U;
    constexpr std::uint8_t alpha_selector = 3U;
    constexpr std::uint8_t one_selector = 5U;

    std::uint32_t source_channels = 0U;
    std::uint32_t expected_endian_swap = 0U;
    switch (descriptor.format)
    {
    case gx2_unorm_r8_format:
        source_channels = 1U;
        expected_endian_swap = 0U;
        break;
    case gx2_unorm_rg8_format:
        source_channels = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_rgba8_format:
        source_channels = 4U;
        expected_endian_swap = 2U;
        break;
    default:
        break;
    }
    if (descriptor.dimension != gx2_2d_dimension ||
        descriptor.tile_mode != gx2_linear_aligned_tile_mode ||
        descriptor.depth_or_layers != 1U || source_channels == 0U ||
        descriptor.numeric_format != gx2_unorm_numeric_format ||
        descriptor.endian_swap != expected_endian_swap || descriptor.force_degamma ||
        descriptor.base_level != 0U || descriptor.last_level != 0U ||
        descriptor.base_array != 0U || descriptor.last_array != 0U ||
        descriptor.width == 0U || descriptor.height == 0U ||
        descriptor.pitch_pixels < descriptor.width)
    {
        return std::nullopt;
    }

    const std::uint64_t source_row_bytes =
        static_cast<std::uint64_t>(descriptor.pitch_pixels) * source_channels;
    const std::uint64_t required_source_bytes = source_row_bytes * descriptor.height;
    const std::uint64_t output_bytes =
        static_cast<std::uint64_t>(descriptor.width) * descriptor.height *
        rgba8_bytes_per_pixel;
    if (source_row_bytes > std::numeric_limits<std::size_t>::max() ||
        required_source_bytes > guest_bytes.size() ||
        output_bytes > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }

    for (const std::uint8_t selector : descriptor.component_selectors)
    {
        if (selector > one_selector)
        {
            return std::nullopt;
        }
    }

    Rgba8TextureStaging result{};
    result.width = descriptor.width;
    result.height = descriptor.height;
    result.pixels.resize(static_cast<std::size_t>(output_bytes));

    // Guest storage already contains the channel bytes in GX2 format order.
    // The resource's endian mode describes how Latte fetches those bytes; it
    // does not require swapping the CPU-side snapshot used for Vulkan upload.
    const std::array<std::uint8_t, 6U> constants{0U, 0U, 0U, 0U, 0U, 0xFFU};
    for (std::uint32_t y = 0U; y < descriptor.height; ++y)
    {
        const std::size_t source_row = static_cast<std::size_t>(
            static_cast<std::uint64_t>(y) * source_row_bytes);
        const std::size_t destination_row = static_cast<std::size_t>(
            static_cast<std::uint64_t>(y) * descriptor.width * rgba8_bytes_per_pixel);
        for (std::uint32_t x = 0U; x < descriptor.width; ++x)
        {
            const std::size_t source_offset = source_row +
                static_cast<std::size_t>(x) * source_channels;
            const std::size_t destination_offset = destination_row +
                static_cast<std::size_t>(x) * rgba8_bytes_per_pixel;
            const std::uint8_t* source = guest_bytes.data() + source_offset;
            const std::array<std::uint8_t, 4U> logical_rgba{
                source[0U],
                static_cast<std::uint8_t>(source_channels >= 2U ? source[1U] : 0U),
                static_cast<std::uint8_t>(source_channels == 4U ? source[2U] : 0U),
                static_cast<std::uint8_t>(source_channels == 4U ? source[3U] : 0xFFU)};
            for (std::size_t component = 0U; component < rgba8_bytes_per_pixel; ++component)
            {
                const std::uint8_t selector = descriptor.component_selectors[component];
                result.pixels[destination_offset + component] =
                    selector <= alpha_selector
                        ? logical_rgba[selector]
                        : constants[selector];
            }
        }
    }
    return result;
}

}
