#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

namespace affogato::gpu
{

inline constexpr std::uint32_t gx2_tile_mode_linear_aligned = 1U;
inline constexpr std::uint32_t gx2_tile_mode_1d_tiled_thin1 = 2U;
inline constexpr std::uint32_t gx2_tile_mode_2d_tiled_thin1 = 4U;

[[nodiscard]] inline std::optional<std::size_t> gx2_1d_thin1_element_offset(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t pitch_elements,
    std::uint32_t bits_per_element)
{
    constexpr std::uint32_t micro_tile_width = 8U;
    constexpr std::uint64_t micro_tile_height = 8U;
    if ((bits_per_element != 8U && bits_per_element != 16U &&
         bits_per_element != 32U) ||
        (pitch_elements % std::max(
            micro_tile_width, 256U / bits_per_element)) != 0U)
    {
        return std::nullopt;
    }

    const std::uint64_t x_in_tile = x % micro_tile_width;
    const std::uint64_t y_in_tile = y % micro_tile_height;
    std::uint64_t micro_tile_element = 0U;
    if (bits_per_element == 8U)
    {
        micro_tile_element =
            ((x_in_tile >> 0U) & 1U) |
            (((x_in_tile >> 1U) & 1U) << 1U) |
            (((x_in_tile >> 2U) & 1U) << 2U) |
            (((y_in_tile >> 1U) & 1U) << 3U) |
            (((y_in_tile >> 0U) & 1U) << 4U) |
            (((y_in_tile >> 2U) & 1U) << 5U);
    }
    else if (bits_per_element == 16U)
    {
        micro_tile_element =
            ((x_in_tile >> 0U) & 1U) |
            (((x_in_tile >> 1U) & 1U) << 1U) |
            (((x_in_tile >> 2U) & 1U) << 2U) |
            (((y_in_tile >> 0U) & 1U) << 3U) |
            (((y_in_tile >> 1U) & 1U) << 4U) |
            (((y_in_tile >> 2U) & 1U) << 5U);
    }
    else
    {
        micro_tile_element =
            ((x_in_tile >> 0U) & 1U) |
            (((x_in_tile >> 1U) & 1U) << 1U) |
            (((y_in_tile >> 0U) & 1U) << 2U) |
            (((x_in_tile >> 2U) & 1U) << 3U) |
            (((y_in_tile >> 1U) & 1U) << 4U) |
            (((y_in_tile >> 2U) & 1U) << 5U);
    }

    const std::uint64_t bytes_per_element = bits_per_element / 8U;
    const std::uint64_t micro_tile_bytes = 64U * bytes_per_element;
    const std::uint64_t micro_tiles_per_row = pitch_elements / micro_tile_width;
    const std::uint64_t micro_tile_index =
        static_cast<std::uint64_t>(x / micro_tile_width) +
        static_cast<std::uint64_t>(y / micro_tile_height) *
            micro_tiles_per_row;
    if (micro_tile_index >
        std::numeric_limits<std::uint64_t>::max() / micro_tile_bytes)
    {
        return std::nullopt;
    }

    const std::uint64_t micro_tile_offset =
        micro_tile_index * micro_tile_bytes;
    const std::uint64_t element_offset =
        micro_tile_element * bytes_per_element;
    if (micro_tile_offset >
        std::numeric_limits<std::uint64_t>::max() - element_offset)
    {
        return std::nullopt;
    }
    const std::uint64_t byte_offset = micro_tile_offset + element_offset;
    if (byte_offset > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(byte_offset);
}

[[nodiscard]] inline std::optional<std::size_t>
gx2_2d_thin1_element_offset(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t pitch_pixels,
    std::uint32_t bits_per_element,
    std::uint32_t pipe_swizzle,
    std::uint32_t bank_swizzle)
{
    constexpr std::uint32_t micro_tile_width = 8U;
    constexpr std::uint32_t macro_tile_width = 32U;
    constexpr std::uint32_t macro_tile_height = 16U;
    constexpr std::uint32_t pipe_count = 2U;
    constexpr std::uint32_t bank_count = 4U;
    constexpr std::uint32_t pipe_bits_count = 1U;
    constexpr std::uint32_t bank_bits_count = 2U;
    constexpr std::uint32_t pipe_bank_bits_count =
        pipe_bits_count + bank_bits_count;
    constexpr std::uint32_t pipe_interleave_bytes = 256U;
    constexpr std::uint32_t group_bits = 8U;

    if ((bits_per_element != 8U && bits_per_element != 16U &&
         bits_per_element != 32U) || pipe_swizzle >= pipe_count ||
        bank_swizzle >= bank_count)
    {
        return std::nullopt;
    }

    const std::uint32_t pitch_alignment = macro_tile_width *
        std::max(1U, 256U / bits_per_element / 8U);
    if (pitch_pixels % pitch_alignment != 0U)
    {
        return std::nullopt;
    }

    const std::uint32_t x_in_micro_tile = x % micro_tile_width;
    const std::uint32_t y_in_micro_tile = y % micro_tile_width;
    std::uint32_t micro_tile_element = 0U;
    if (bits_per_element == 8U)
    {
        micro_tile_element =
            ((x_in_micro_tile >> 0U) & 1U) |
            (((x_in_micro_tile >> 1U) & 1U) << 1U) |
            (((x_in_micro_tile >> 2U) & 1U) << 2U) |
            (((y_in_micro_tile >> 1U) & 1U) << 3U) |
            (((y_in_micro_tile >> 0U) & 1U) << 4U) |
            (((y_in_micro_tile >> 2U) & 1U) << 5U);
    }
    else if (bits_per_element == 16U)
    {
        micro_tile_element =
            ((x_in_micro_tile >> 0U) & 1U) |
            (((x_in_micro_tile >> 1U) & 1U) << 1U) |
            (((x_in_micro_tile >> 2U) & 1U) << 2U) |
            (((y_in_micro_tile >> 0U) & 1U) << 3U) |
            (((y_in_micro_tile >> 1U) & 1U) << 4U) |
            (((y_in_micro_tile >> 2U) & 1U) << 5U);
    }
    else
    {
        micro_tile_element =
            ((x_in_micro_tile >> 0U) & 1U) |
            (((x_in_micro_tile >> 1U) & 1U) << 1U) |
            (((y_in_micro_tile >> 0U) & 1U) << 2U) |
            (((x_in_micro_tile >> 2U) & 1U) << 3U) |
            (((y_in_micro_tile >> 1U) & 1U) << 4U) |
            (((y_in_micro_tile >> 2U) & 1U) << 5U);
    }
    const std::uint64_t element_offset =
        static_cast<std::uint64_t>(micro_tile_element) *
        (bits_per_element / 8U);

    const std::uint32_t x3 = (x >> 3U) & 1U;
    const std::uint32_t x4 = (x >> 4U) & 1U;
    const std::uint32_t y3 = (y >> 3U) & 1U;
    const std::uint32_t y4 = (y >> 4U) & 1U;
    const std::uint32_t y5 = (y >> 5U) & 1U;
    const std::uint32_t pipe = y3 ^ x3;
    const std::uint32_t bank = (y5 ^ x3) | ((y4 ^ x4) << 1U);
    const std::uint32_t bank_pipe_swizzle =
        pipe_swizzle + pipe_count * bank_swizzle;
    const std::uint32_t bank_pipe =
        (pipe + pipe_count * bank) ^ bank_pipe_swizzle;
    const std::uint32_t resolved_pipe = bank_pipe % pipe_count;
    const std::uint32_t resolved_bank = bank_pipe / pipe_count;

    const std::uint64_t macro_tile_bytes =
        static_cast<std::uint64_t>(macro_tile_width) * macro_tile_height *
        (bits_per_element / 8U);
    const std::uint64_t macro_tiles_per_row = pitch_pixels / macro_tile_width;
    const std::uint64_t macro_tile_index =
        static_cast<std::uint64_t>(x / macro_tile_width) +
        static_cast<std::uint64_t>(y / macro_tile_height) * macro_tiles_per_row;
    if (macro_tile_index >
        std::numeric_limits<std::uint64_t>::max() / macro_tile_bytes)
    {
        return std::nullopt;
    }

    const std::uint64_t macro_tile_offset = macro_tile_index * macro_tile_bytes;
    const std::uint64_t interleaved_macro_tile_offset =
        macro_tile_offset >> pipe_bank_bits_count;
    if (interleaved_macro_tile_offset >
        std::numeric_limits<std::uint64_t>::max() - element_offset)
    {
        return std::nullopt;
    }
    const std::uint64_t total_offset =
        element_offset + interleaved_macro_tile_offset;
    if (total_offset >
        (std::numeric_limits<std::uint64_t>::max() >> pipe_bank_bits_count))
    {
        return std::nullopt;
    }
    const std::uint64_t offset_high =
        (total_offset & ~std::uint64_t{pipe_interleave_bytes - 1U}) <<
        pipe_bank_bits_count;
    const std::uint64_t offset_low =
        total_offset & (pipe_interleave_bytes - 1U);
    const std::uint64_t bank_bits =
        static_cast<std::uint64_t>(resolved_bank) <<
        (pipe_bits_count + group_bits);
    const std::uint64_t pipe_bits =
        static_cast<std::uint64_t>(resolved_pipe) << group_bits;
    const std::uint64_t byte_offset =
        bank_bits | pipe_bits | offset_low | offset_high;
    if (byte_offset > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }
    return static_cast<std::size_t>(byte_offset);
}

[[nodiscard]] inline std::optional<std::size_t>
gx2_2d_thin1_rgba8_element_offset(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t pitch_pixels,
    std::uint32_t pipe_swizzle,
    std::uint32_t bank_swizzle)
{
    return gx2_2d_thin1_element_offset(
        x, y, pitch_pixels, 32U, pipe_swizzle, bank_swizzle);
}

struct Rgba8SurfaceStaging
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::vector<std::uint8_t> pixels;

    bool operator==(const Rgba8SurfaceStaging&) const = default;
};

// Thin1 modes use 8x8 displayable microtiles. In-tile element bit order
// depends on element size; for 32-bit elements R600 addrlib orders X0, X1,
// Y0, X2, Y1, Y2. Surface pitches respect the mode's pipe-interleave layout.
[[nodiscard]] inline std::optional<std::uint64_t> gx2_surface_storage_size(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t pitch_pixels,
    std::uint32_t bits_per_pixel,
    std::uint32_t tile_mode)
{
    if (width == 0U || height == 0U || pitch_pixels < width ||
        (bits_per_pixel != 8U && bits_per_pixel != 16U &&
         bits_per_pixel != 32U))
    {
        return std::nullopt;
    }

    if (tile_mode == gx2_tile_mode_1d_tiled_thin1)
    {
        const std::uint32_t pitch_alignment =
            std::max(8U, 256U / bits_per_pixel);
        if (pitch_pixels % pitch_alignment != 0U || height % 8U != 0U)
        {
            return std::nullopt;
        }
    }
    else if (tile_mode == gx2_tile_mode_2d_tiled_thin1)
    {
        constexpr std::uint32_t macro_tile_width = 32U;
        constexpr std::uint32_t macro_tile_height = 16U;
        const std::uint32_t pitch_alignment = macro_tile_width *
            std::max(1U, 256U / bits_per_pixel / 8U);
        if (pitch_pixels % pitch_alignment != 0U ||
            height % macro_tile_height != 0U)
        {
            return std::nullopt;
        }
    }
    if (tile_mode != gx2_tile_mode_linear_aligned &&
        tile_mode != gx2_tile_mode_1d_tiled_thin1 &&
        tile_mode != gx2_tile_mode_2d_tiled_thin1)
    {
        return std::nullopt;
    }

    const std::uint64_t row_size =
        static_cast<std::uint64_t>(pitch_pixels) * (bits_per_pixel / 8U);
    if (height > std::numeric_limits<std::uint64_t>::max() / row_size)
    {
        return std::nullopt;
    }

    return row_size * height;
}

[[nodiscard]] inline std::optional<std::uint64_t> rgba8_surface_storage_size(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t pitch_pixels,
    std::uint32_t tile_mode)
{
    return gx2_surface_storage_size(
        width, height, pitch_pixels, 32U, tile_mode);
}

[[nodiscard]] inline std::optional<std::size_t> rgba8_surface_pixel_offset(
    std::uint32_t x,
    std::uint32_t y,
    std::uint32_t pitch_pixels,
    std::uint32_t tile_mode)
{
    constexpr std::uint64_t bytes_per_pixel = 4U;
    std::uint64_t pixel_index = 0U;

    if (tile_mode == gx2_tile_mode_linear_aligned)
    {
        pixel_index = static_cast<std::uint64_t>(y) * pitch_pixels + x;
    }
    else if (tile_mode == gx2_tile_mode_1d_tiled_thin1)
    {
        const auto byte_offset =
            gx2_1d_thin1_element_offset(x, y, pitch_pixels, 32U);
        if (!byte_offset)
        {
            return std::nullopt;
        }
        return byte_offset;
    }
    else if (tile_mode == gx2_tile_mode_2d_tiled_thin1)
    {
        return gx2_2d_thin1_rgba8_element_offset(
            x, y, pitch_pixels, 0U, 0U);
    }
    else
    {
        return std::nullopt;
    }

    if (pixel_index > std::numeric_limits<std::uint64_t>::max() / bytes_per_pixel)
    {
        return std::nullopt;
    }
    const std::uint64_t byte_offset = pixel_index * bytes_per_pixel;
    if (byte_offset > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }

    return static_cast<std::size_t>(byte_offset);
}

[[nodiscard]] inline bool clear_rgba8_surface(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t pitch_pixels,
    std::uint32_t tile_mode,
    std::span<std::uint8_t> guest_bytes,
    const std::array<std::uint8_t, 4U>& color)
{
    const auto storage_size = rgba8_surface_storage_size(
        width, height, pitch_pixels, tile_mode);
    if (!storage_size || *storage_size > guest_bytes.size())
    {
        return false;
    }

    if (tile_mode == gx2_tile_mode_linear_aligned)
    {
        for (std::uint32_t y = 0U; y < height; ++y)
        {
            const std::size_t row_offset = static_cast<std::size_t>(
                static_cast<std::uint64_t>(y) * pitch_pixels * color.size());
            for (std::uint32_t x = 0U; x < width; ++x)
            {
                const std::size_t pixel_offset = row_offset +
                    static_cast<std::size_t>(x) * color.size();
                std::copy_n(
                    color.begin(), color.size(),
                    guest_bytes.begin() +
                        static_cast<std::ptrdiff_t>(pixel_offset));
            }
        }
        return true;
    }

    for (std::uint32_t y = 0U; y < height; ++y)
    {
        for (std::uint32_t x = 0U; x < width; ++x)
        {
            const auto destination_offset = rgba8_surface_pixel_offset(
                x, y, pitch_pixels, tile_mode);
            if (!destination_offset ||
                *destination_offset > guest_bytes.size() - color.size())
            {
                return false;
            }

            std::copy_n(
                color.begin(),
                color.size(),
                guest_bytes.begin() +
                    static_cast<std::ptrdiff_t>(*destination_offset));
        }
    }

    return true;
}

[[nodiscard]] inline std::optional<Rgba8SurfaceStaging> stage_rgba8_surface(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t pitch_pixels,
    std::uint32_t tile_mode,
    std::span<const std::uint8_t> guest_bytes)
{
    constexpr std::uint64_t bytes_per_pixel = 4U;
    const auto storage_size = rgba8_surface_storage_size(
        width, height, pitch_pixels, tile_mode);
    if (!storage_size || *storage_size > guest_bytes.size())
    {
        return std::nullopt;
    }

    const std::uint64_t output_pixel_count =
        static_cast<std::uint64_t>(width) * height;
    if (output_pixel_count >
        std::numeric_limits<std::uint64_t>::max() / bytes_per_pixel)
    {
        return std::nullopt;
    }
    const std::uint64_t output_size = output_pixel_count * bytes_per_pixel;
    if (output_size > std::numeric_limits<std::size_t>::max())
    {
        return std::nullopt;
    }

    Rgba8SurfaceStaging result{};
    result.width = width;
    result.height = height;
    result.pixels.resize(static_cast<std::size_t>(output_size));
    if (tile_mode == gx2_tile_mode_linear_aligned)
    {
        const std::size_t output_row_size =
            static_cast<std::size_t>(width) * bytes_per_pixel;
        for (std::uint32_t y = 0U; y < height; ++y)
        {
            const std::size_t source_row = static_cast<std::size_t>(
                static_cast<std::uint64_t>(y) * pitch_pixels * bytes_per_pixel);
            const std::size_t destination_row =
                static_cast<std::size_t>(y) * output_row_size;
            std::copy_n(
                guest_bytes.begin() + static_cast<std::ptrdiff_t>(source_row),
                output_row_size,
                result.pixels.begin() +
                    static_cast<std::ptrdiff_t>(destination_row));
        }
        return result;
    }

    for (std::uint32_t y = 0U; y < height; ++y)
    {
        for (std::uint32_t x = 0U; x < width; ++x)
        {
            const auto source_offset = rgba8_surface_pixel_offset(
                x, y, pitch_pixels, tile_mode);
            if (!source_offset ||
                *source_offset > guest_bytes.size() - bytes_per_pixel)
            {
                return std::nullopt;
            }

            const std::size_t destination_offset =
                (static_cast<std::size_t>(y) * width + x) * bytes_per_pixel;
            std::copy_n(
                guest_bytes.begin() + static_cast<std::ptrdiff_t>(*source_offset),
                bytes_per_pixel,
                result.pixels.begin() +
                    static_cast<std::ptrdiff_t>(destination_offset));
        }
    }

    return result;
}

[[nodiscard]] inline bool write_rgba8_surface(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t pitch_pixels,
    std::uint32_t tile_mode,
    std::span<std::uint8_t> guest_bytes,
    std::span<const std::uint8_t> pixels)
{
    constexpr std::uint64_t bytes_per_pixel = 4U;
    const auto storage_size = rgba8_surface_storage_size(
        width, height, pitch_pixels, tile_mode);
    const std::uint64_t pixel_count = static_cast<std::uint64_t>(width) * height;
    if (pixel_count > std::numeric_limits<std::uint64_t>::max() / bytes_per_pixel)
    {
        return false;
    }
    const std::uint64_t pixel_size = pixel_count * bytes_per_pixel;
    if (!storage_size || *storage_size > guest_bytes.size() ||
        pixel_size != pixels.size())
    {
        return false;
    }

    for (std::uint32_t y = 0U; y < height; ++y)
    {
        for (std::uint32_t x = 0U; x < width; ++x)
        {
            const auto destination_offset = rgba8_surface_pixel_offset(
                x, y, pitch_pixels, tile_mode);
            if (!destination_offset ||
                *destination_offset > guest_bytes.size() - bytes_per_pixel)
            {
                return false;
            }

            const std::size_t source_offset =
                (static_cast<std::size_t>(y) * width + x) * bytes_per_pixel;
            std::copy_n(
                pixels.begin() + static_cast<std::ptrdiff_t>(source_offset),
                bytes_per_pixel,
                guest_bytes.begin() +
                    static_cast<std::ptrdiff_t>(*destination_offset));
        }
    }

    return true;
}

}
