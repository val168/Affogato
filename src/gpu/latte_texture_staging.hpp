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
    constexpr std::uint32_t gx2_2d_dimension = 1U;
    constexpr std::uint32_t gx2_linear_aligned_tile_mode = 1U;
    constexpr std::uint32_t gx2_unorm_r8_format = 0x01U;
    constexpr std::uint32_t gx2_unorm_rg44_format = 0x02U;
    constexpr std::uint32_t gx2_unorm_rg8_format = 0x07U;
    constexpr std::uint32_t gx2_unorm_rgb565_format = 0x08U;
    constexpr std::uint32_t gx2_unorm_rgba5551_format = 0x0AU;
    constexpr std::uint32_t gx2_unorm_rgba4444_format = 0x0BU;
    constexpr std::uint32_t gx2_unorm_argb1555_format = 0x0CU;
    constexpr std::uint32_t gx2_unorm_rgba8_format = 0x1AU;
    constexpr std::uint32_t gx2_unorm_numeric_format = 0U;
    constexpr std::uint32_t rgba8_bytes_per_pixel = 4U;
    constexpr std::uint8_t alpha_selector = 3U;
    constexpr std::uint8_t one_selector = 5U;

    std::uint32_t source_channels = 0U;
    std::uint32_t source_bytes_per_pixel = 0U;
    std::uint32_t expected_endian_swap = 0U;
    enum class PackedFormat
    {
        none,
        rg44,
        rgb565,
        rgba5551,
        rgba4444,
        argb1555,
    };
    PackedFormat packed_format = PackedFormat::none;
    switch (descriptor.format)
    {
    case gx2_unorm_r8_format:
        source_channels = 1U;
        source_bytes_per_pixel = 1U;
        expected_endian_swap = 0U;
        break;
    case gx2_unorm_rg44_format:
        packed_format = PackedFormat::rg44;
        source_bytes_per_pixel = 1U;
        expected_endian_swap = 0U;
        break;
    case gx2_unorm_rg8_format:
        source_channels = 2U;
        source_bytes_per_pixel = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_rgb565_format:
        packed_format = PackedFormat::rgb565;
        source_bytes_per_pixel = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_rgba5551_format:
        packed_format = PackedFormat::rgba5551;
        source_bytes_per_pixel = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_rgba4444_format:
        packed_format = PackedFormat::rgba4444;
        source_bytes_per_pixel = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_argb1555_format:
        packed_format = PackedFormat::argb1555;
        source_bytes_per_pixel = 2U;
        expected_endian_swap = 1U;
        break;
    case gx2_unorm_rgba8_format:
        source_channels = 4U;
        source_bytes_per_pixel = 4U;
        expected_endian_swap = 2U;
        break;
    default:
        break;
    }
    if (descriptor.dimension != gx2_2d_dimension ||
        descriptor.tile_mode != gx2_linear_aligned_tile_mode ||
        descriptor.depth_or_layers != 1U || source_bytes_per_pixel == 0U ||
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
        static_cast<std::uint64_t>(descriptor.pitch_pixels) * source_bytes_per_pixel;
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
                static_cast<std::size_t>(x) * source_bytes_per_pixel;
            const std::size_t destination_offset = destination_row +
                static_cast<std::size_t>(x) * rgba8_bytes_per_pixel;
            const std::uint8_t* source = guest_bytes.data() + source_offset;
            std::array<std::uint8_t, 4U> logical_rgba{};
            if (packed_format != PackedFormat::none)
            {
                const std::uint32_t packed =
                    (static_cast<std::uint32_t>(source[0U]) << 8U) | source[1U];
                const auto expand5 = [](std::uint32_t value) {
                    return static_cast<std::uint8_t>((value << 3U) | (value >> 2U));
                };
                const auto expand4 = [](std::uint32_t value) {
                    return static_cast<std::uint8_t>((value << 4U) | value);
                };
                switch (packed_format)
                {
                case PackedFormat::rg44:
                    logical_rgba = {
                        expand4((source[0U] >> 4U) & 0xFU),
                        expand4(source[0U] & 0xFU), 0U, 0xFFU};
                    break;
                case PackedFormat::rgb565:
                {
                    const std::uint32_t red = (packed >> 11U) & 0x1FU;
                    const std::uint32_t green = (packed >> 5U) & 0x3FU;
                    const std::uint32_t blue = packed & 0x1FU;
                    logical_rgba = {
                        expand5(red),
                        static_cast<std::uint8_t>((green << 2U) | (green >> 4U)),
                        expand5(blue), 0xFFU};
                    break;
                }
                case PackedFormat::rgba5551:
                    logical_rgba = {
                        expand5((packed >> 11U) & 0x1FU),
                        expand5((packed >> 6U) & 0x1FU),
                        expand5((packed >> 1U) & 0x1FU),
                        static_cast<std::uint8_t>(
                            (packed & 1U) != 0U ? 0xFFU : 0U)};
                    break;
                case PackedFormat::rgba4444:
                    logical_rgba = {
                        expand4((packed >> 12U) & 0xFU),
                        expand4((packed >> 8U) & 0xFU),
                        expand4((packed >> 4U) & 0xFU),
                        expand4(packed & 0xFU)};
                    break;
                case PackedFormat::argb1555:
                    logical_rgba = {
                        expand5(packed & 0x1FU),
                        expand5((packed >> 5U) & 0x1FU),
                        expand5((packed >> 10U) & 0x1FU),
                        static_cast<std::uint8_t>(
                            (packed & 0x8000U) != 0U ? 0xFFU : 0U)};
                    break;
                case PackedFormat::none:
                    break;
                }
            }
            else
            {
                logical_rgba = {
                    source[0U],
                    static_cast<std::uint8_t>(source_channels >= 2U ? source[1U] : 0U),
                    static_cast<std::uint8_t>(source_channels == 4U ? source[2U] : 0U),
                    static_cast<std::uint8_t>(source_channels == 4U ? source[3U] : 0xFFU)};
            }
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
