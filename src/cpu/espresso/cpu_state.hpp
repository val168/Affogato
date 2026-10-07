#pragma once

#include <array>
#include <cstdint>

namespace affogato::cpu::espresso
{

namespace fpscr
{
inline constexpr std::uint32_t fpcc_mask = 0x0000F000U;
inline constexpr std::uint32_t fprf_mask = 0x0001F000U;
inline constexpr std::uint32_t fi_mask = 0x00020000U;
inline constexpr std::uint32_t fr_mask = 0x00040000U;
inline constexpr std::uint32_t xe_mask = 0x00000008U;
inline constexpr std::uint32_t ze_mask = 0x00000010U;
inline constexpr std::uint32_t ve_mask = 0x00000080U;
inline constexpr std::uint32_t vxzdz_mask = 0x00200000U;
inline constexpr std::uint32_t vxidi_mask = 0x00400000U;
inline constexpr std::uint32_t vxisi_mask = 0x00800000U;
inline constexpr std::uint32_t vxsnan_mask = 0x01000000U;
inline constexpr std::uint32_t xx_mask = 0x02000000U;
inline constexpr std::uint32_t zx_mask = 0x04000000U;
inline constexpr std::uint32_t vx_mask = 0x20000000U;
inline constexpr std::uint32_t fex_mask = 0x40000000U;
inline constexpr std::uint32_t fx_mask = 0x80000000U;
}

struct CpuState
{
    std::array<std::uint32_t, 32> gpr{};
    std::array<std::uint64_t, 32> fpr{};
    std::array<std::uint64_t, 32> fpr_ps1{};

    std::uint32_t cr{};
    std::uint32_t xer{};
    std::uint32_t fpscr{};
    std::uint32_t lr{};
    std::uint32_t ctr{};
    std::uint32_t cia{};

    void reset() noexcept
    {
        *this = {};
    }
};

}
