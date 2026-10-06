#pragma once

#include <cstdint>

namespace affogato::cpu::espresso
{

class GuestMemory;
class HleDispatcher;

inline constexpr std::uint32_t downloaded_data_base_size = 0xC008U;
inline constexpr std::uint32_t downloaded_data_base_vtable_offset = 0xC000U;
inline constexpr std::uint32_t downloaded_post_data_size = 0xC208U;
inline constexpr std::uint32_t downloaded_post_data_community_id_offset = 0xC008U;
inline constexpr std::uint32_t downloaded_post_data_empathy_count_offset = 0xC00CU;
inline constexpr std::uint32_t downloaded_post_data_comment_count_offset = 0xC010U;
inline constexpr std::uint32_t downloaded_post_data_initialized_size = 0xC014U;

// Initializes only the DownloadedDataBase subobject; its GHS vtable remains
// zero until Affogato has a real guest-visible RTTI/vtable implementation.
void initialize_downloaded_data_base(GuestMemory& memory, std::uint32_t self);

void register_nn_olv_hle(HleDispatcher& dispatcher);

}
