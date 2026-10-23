#pragma once

#include <array>
#include <cstdint>
#include <optional>

namespace affogato::gpu
{

using LatteSamplerWords = std::array<std::uint32_t, 3U>;

struct LatteSamplerDescriptor
{
    LatteSamplerWords raw_words{};
    std::array<std::uint8_t, 3U> wrap_modes{};
    std::uint8_t mag_filter{};
    std::uint8_t min_filter{};
    std::uint8_t z_filter{};
    std::uint8_t mip_filter{};
    std::uint8_t max_anisotropy{};
    std::uint8_t border_color_type{};
    bool point_sampling_clamp{};
    bool texture_array_override{};
    std::uint8_t depth_compare{};
    std::uint16_t min_lod_fixed{};
    std::uint16_t max_lod_fixed{};
    std::int16_t lod_bias_fixed{};
    std::uint16_t secondary_lod_bias_fixed{};
    bool force_degamma{};
    bool high_precision_filter{};
    std::uint8_t anisotropy_bias{};
    bool sampler_type{};

    bool operator==(const LatteSamplerDescriptor&) const = default;
};

[[nodiscard]] inline std::optional<LatteSamplerDescriptor>
decode_latte_sampler(const LatteSamplerWords& words)
{
    const std::uint32_t word0 = words[0];
    const std::uint32_t word1 = words[1];
    const std::uint32_t word2 = words[2];
    const std::uint8_t mag_filter =
        static_cast<std::uint8_t>((word0 >> 9U) & 7U);
    const std::uint8_t min_filter =
        static_cast<std::uint8_t>((word0 >> 12U) & 7U);
    const std::uint8_t z_filter =
        static_cast<std::uint8_t>((word0 >> 15U) & 3U);
    const std::uint8_t mip_filter =
        static_cast<std::uint8_t>((word0 >> 17U) & 3U);
    const auto valid_xy_filter = [](std::uint8_t filter) {
        // Value 3 is emitted for anisotropic bilinear filtering by GX2.
        return filter <= 5U;
    };
    if (!valid_xy_filter(mag_filter) || !valid_xy_filter(min_filter) ||
        z_filter > 2U || mip_filter > 2U)
    {
        return std::nullopt;
    }

    const std::uint32_t signed_lod_bias = (word1 >> 20U) & 0xFFFU;
    const std::int32_t lod_bias = (signed_lod_bias & 0x800U) != 0U
        ? static_cast<std::int32_t>(signed_lod_bias) - 0x1000
        : static_cast<std::int32_t>(signed_lod_bias);

    LatteSamplerDescriptor result{};
    result.raw_words = words;
    result.wrap_modes = {
        static_cast<std::uint8_t>(word0 & 7U),
        static_cast<std::uint8_t>((word0 >> 3U) & 7U),
        static_cast<std::uint8_t>((word0 >> 6U) & 7U)};
    result.mag_filter = mag_filter;
    result.min_filter = min_filter;
    result.z_filter = z_filter;
    result.mip_filter = mip_filter;
    result.max_anisotropy = static_cast<std::uint8_t>((word0 >> 19U) & 7U);
    result.border_color_type = static_cast<std::uint8_t>((word0 >> 22U) & 3U);
    result.point_sampling_clamp = ((word0 >> 24U) & 1U) != 0U;
    result.texture_array_override = ((word0 >> 25U) & 1U) != 0U;
    result.depth_compare = static_cast<std::uint8_t>((word0 >> 26U) & 7U);
    result.min_lod_fixed = static_cast<std::uint16_t>(word1 & 0x3FFU);
    result.max_lod_fixed = static_cast<std::uint16_t>((word1 >> 10U) & 0x3FFU);
    result.lod_bias_fixed = static_cast<std::int16_t>(lod_bias);
    result.secondary_lod_bias_fixed = static_cast<std::uint16_t>(word2 & 0xFFFU);
    result.force_degamma = ((word2 >> 13U) & 1U) != 0U;
    result.high_precision_filter = ((word2 >> 14U) & 1U) != 0U;
    result.anisotropy_bias = static_cast<std::uint8_t>((word2 >> 22U) & 0x3FU);
    result.sampler_type = (word2 >> 31U) != 0U;
    return result;
}

}
