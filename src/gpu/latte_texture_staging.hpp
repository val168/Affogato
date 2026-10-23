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

// Converts one explicitly supported linear GX2 texture layout to tightly
// packed RGBA8 bytes suitable for a Vulkan staging buffer. GuestMemory range
// validation and snapshotting are the caller's responsibility; guest_bytes
// starts at descriptor.image_address.
[[nodiscard]] inline std::optional<Rgba8TextureStaging>
stage_linear_rgba8_texture(
    const LatteTextureResourceDescriptor& descriptor,
    std::span<const std::uint8_t> guest_bytes)
{
    constexpr std::uint32_t gx2_2d_dimension = 2U;
    constexpr std::uint32_t gx2_linear_aligned_tile_mode = 1U;
    constexpr std::uint32_t gx2_unorm_rgba8_format = 0x1AU;
    constexpr std::uint32_t gx2_unorm_numeric_format = 0U;
    constexpr std::uint32_t rgba8_bytes_per_pixel = 4U;
    constexpr std::uint8_t alpha_selector = 3U;
    constexpr std::uint8_t one_selector = 5U;

    if (descriptor.dimension != gx2_2d_dimension ||
        descriptor.tile_mode != gx2_linear_aligned_tile_mode ||
        descriptor.depth_or_layers != 1U || descriptor.format != gx2_unorm_rgba8_format ||
        descriptor.numeric_format != gx2_unorm_numeric_format ||
        descriptor.endian_swap != 0U || descriptor.force_degamma ||
        descriptor.base_level != 0U || descriptor.last_level != 0U ||
        descriptor.base_array != 0U || descriptor.last_array != 0U ||
        descriptor.width == 0U || descriptor.height == 0U ||
        descriptor.pitch_pixels < descriptor.width)
    {
        return std::nullopt;
    }

    const std::uint64_t source_row_bytes =
        static_cast<std::uint64_t>(descriptor.pitch_pixels) * rgba8_bytes_per_pixel;
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

    const std::array<std::uint8_t, 6U> constants{0U, 0U, 0U, 0U, 0U, 0xFFU};
    const auto select_component = [&](const std::uint8_t* source, std::uint8_t selector) {
        return selector <= alpha_selector ? source[selector] : constants[selector];
    };
    for (std::uint32_t y = 0U; y < descriptor.height; ++y)
    {
        const std::size_t source_row = static_cast<std::size_t>(
            static_cast<std::uint64_t>(y) * source_row_bytes);
        const std::size_t destination_row = static_cast<std::size_t>(
            static_cast<std::uint64_t>(y) * descriptor.width * rgba8_bytes_per_pixel);
        for (std::uint32_t x = 0U; x < descriptor.width; ++x)
        {
            const std::size_t source_offset = source_row +
                static_cast<std::size_t>(x) * rgba8_bytes_per_pixel;
            const std::size_t destination_offset = destination_row +
                static_cast<std::size_t>(x) * rgba8_bytes_per_pixel;
            const std::uint8_t* source = guest_bytes.data() + source_offset;
            for (std::size_t component = 0U; component < rgba8_bytes_per_pixel; ++component)
            {
                result.pixels[destination_offset + component] =
                    select_component(source, descriptor.component_selectors[component]);
            }
        }
    }
    return result;
}

}
