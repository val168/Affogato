#include "cpu/espresso/gx2_hle.hpp"

#include "cpu/espresso/cafe_os_hle.hpp"
#include "cpu/espresso/guest_memory.hpp"
#include "cpu/espresso/hle_dispatcher.hpp"
#include "cpu/espresso/interpreter.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string>
#include <thread>
#include <utility>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::uint32_t gx2_init_end = 0U;
constexpr std::uint32_t gx2_init_command_buffer_base = 1U;
constexpr std::uint32_t gx2_init_command_buffer_pool_size = 2U;
constexpr std::uint32_t gx2_init_argc = 7U;
constexpr std::uint32_t gx2_init_argv = 8U;
constexpr std::uint32_t gx2_init_profile_mode = 9U;
constexpr std::uint32_t gx2_init_toss_stage = 10U;
constexpr std::uint32_t gx2_init_app_io_thread_stack_size = 11U;

constexpr std::uint32_t default_command_pool_size = 0x00400000U;
constexpr std::uint32_t minimum_command_pool_size = 0x2000U;
constexpr std::uint32_t command_pool_alignment = 0x100U;
constexpr std::uint32_t command_buffer_alignment = 0x40U;
constexpr std::uint32_t default_app_io_stack_size = 0x1000U;
constexpr std::size_t maximum_attribute_pairs = 64U;
constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
constexpr std::uint32_t db_depth_control_register = 0x00028800U;
constexpr std::uint32_t db_stencilrefmask_register = 0x00028430U;
constexpr std::uint32_t db_stencilrefmask_bf_register = 0x00028434U;
constexpr std::uint32_t pa_su_sc_mode_cntl_register = 0x00028814U;
constexpr std::uint32_t cb_color_control_register = 0x00028808U;
constexpr std::uint32_t cb_blend0_control_register = 0x00028780U;
constexpr std::uint32_t cb_blend_red_register = 0x00028414U;
constexpr std::uint32_t cb_blend_green_register = 0x00028418U;
constexpr std::uint32_t cb_blend_blue_register = 0x0002841CU;
constexpr std::uint32_t cb_blend_alpha_register = 0x00028420U;
constexpr std::uint32_t sx_alpha_test_control_register = 0x00028410U;
constexpr std::uint32_t sx_alpha_ref_register = 0x00028438U;
constexpr std::uint32_t cb_target_mask_register = 0x00028238U;
constexpr std::uint32_t pa_sc_aa_mask_register = 0x00028C48U;
constexpr std::uint32_t db_alpha_to_mask_register = 0x00028D44U;
constexpr std::uint32_t pa_sc_generic_scissor_tl_register = 0x00028240U;
constexpr std::uint32_t pa_sc_generic_scissor_br_register = 0x00028244U;
constexpr std::uint32_t pa_sc_window_offset_register = 0x00028200U;
constexpr std::uint32_t pa_su_point_size_register = 0x00028A00U;
constexpr std::uint32_t pa_su_point_minmax_register = 0x00028A04U;
constexpr std::uint32_t pa_su_line_cntl_register = 0x00028A08U;
constexpr std::uint32_t pa_su_poly_offset_front_scale_register = 0x00028B80U;
constexpr std::uint32_t pa_su_poly_offset_front_offset_register = 0x00028B84U;
constexpr std::uint32_t pa_su_poly_offset_back_scale_register = 0x00028B88U;
constexpr std::uint32_t pa_su_poly_offset_back_offset_register = 0x00028B8CU;
constexpr std::uint32_t pa_su_poly_offset_clamp_register = 0x00028B7CU;
constexpr std::uint32_t pa_cl_vport_xscale_register = 0x0002843CU;
constexpr std::uint32_t pa_cl_gb_vert_clip_adj_register = 0x00028BE8U;
constexpr std::uint32_t pa_cl_gb_vert_disc_adj_register = 0x00028BECU;
constexpr std::uint32_t pa_cl_gb_horz_clip_adj_register = 0x00028BF0U;
constexpr std::uint32_t pa_cl_gb_horz_disc_adj_register = 0x00028BF4U;
constexpr std::uint32_t pa_sc_vport_zmin_register = 0x000282D0U;
constexpr std::uint32_t pa_sc_vport_zmax_register = 0x000282D4U;
constexpr std::uint32_t gx2_context_state_size = 0xA100U;
constexpr std::uint32_t gx2_context_shadow_state_size = 0x9800U;
constexpr std::uint32_t gx2_context_profiling_offset = 0x9800U;
constexpr std::uint32_t gx2_context_display_list_size_offset = 0x9804U;
constexpr std::uint32_t gx2_context_display_list_offset = 0x9E00U;
constexpr std::uint32_t gx2_context_display_list_capacity = 0x300U;
constexpr std::uint32_t gx2_color_buffer_size = 0x9CU;
constexpr std::uint32_t gx2_depth_buffer_size = 0xACU;
constexpr std::uint32_t gx2_render_target_count = 8U;
static_assert(gx2_context_shadow_state_size == gx2_context_profiling_offset);
static_assert(gx2_context_display_list_offset +
                  gx2_context_display_list_capacity ==
              gx2_context_state_size);

struct Gx2ContextRegisterLoadRange
{
    std::uint16_t first_register{};
    std::uint16_t register_count{};
};

constexpr Gx2ContextRegisterLoadRange gx2_context_config_load_ranges[]{
    {0x300U, 0x6U},
    {0x900U, 0x48U}, {0x980U, 0x48U}, {0xA00U, 0x48U},
    {0x310U, 0xCU}, {0x542U, 0x1U}, {0x235U, 0x1U},
    {0x232U, 0x2U}, {0x23AU, 0x1U}, {0x256U, 0x1U},
    {0x60CU, 0x1U}, {0x5C5U, 0x1U}, {0x2C8U, 0x1U},
    {0x363U, 0x1U}, {0x404U, 0x2U},
};
constexpr Gx2ContextRegisterLoadRange gx2_context_context_load_ranges[]{
    {0x0U, 0x2U}, {0x3U, 0x3U},
    {0xAU, 0x4U}, {0x10U, 0x38U}, {0x50U, 0x34U},
    {0x8EU, 0x4U}, {0x94U, 0x40U}, {0x100U, 0x9U},
    {0x10CU, 0x3U}, {0x10FU, 0x60U}, {0x185U, 0xAU},
    {0x191U, 0x27U}, {0x1E0U, 0x9U}, {0x200U, 0x1U},
    {0x202U, 0x7U}, {0xE0U, 0x20U}, {0x210U, 0x29U},
    {0x250U, 0x34U}, {0x290U, 0x1U}, {0x292U, 0x2U},
    {0x2A1U, 0x1U}, {0x2A5U, 0x1U}, {0x2A8U, 0x2U},
    {0x2ACU, 0x3U}, {0x2CAU, 0x1U}, {0x2CCU, 0x1U},
    {0x2CEU, 0x1U}, {0x300U, 0x9U}, {0x30CU, 0x1U},
    {0x312U, 0x1U}, {0x316U, 0x2U}, {0x343U, 0x2U},
    {0x349U, 0x3U}, {0x34CU, 0x2U}, {0x351U, 0x1U},
    {0x37EU, 0x6U}, {0x2B4U, 0x3U}, {0x2B8U, 0x3U},
    {0x2BCU, 0x3U}, {0x2C0U, 0x3U}, {0x2C8U, 0x1U},
    {0x29BU, 0x1U}, {0x8CU, 0x1U}, {0xD5U, 0x1U},
    {0x284U, 0xCU},
};
constexpr Gx2ContextRegisterLoadRange gx2_context_alu_load_ranges[]{
    {0U, 0x800U},
};
constexpr Gx2ContextRegisterLoadRange gx2_context_loop_load_ranges[]{
    {0U, 0x60U},
};
constexpr Gx2ContextRegisterLoadRange gx2_context_resource_load_ranges[]{
    {0U, 0x70U}, {0x380U, 0x70U},
    {0x460U, 0x70U}, {0x7E0U, 0x70U}, {0x8B9U, 0x7U},
    {0x8C0U, 0x70U}, {0x930U, 0x70U}, {0xCB0U, 0x70U},
    {0xD89U, 0x7U},
};
constexpr Gx2ContextRegisterLoadRange gx2_context_sampler_load_ranges[]{
    {0U, 0x36U},
    {0x36U, 0x36U}, {0x6CU, 0x36U},
};

constexpr std::uint32_t context_shadow_area_size(std::uint32_t register_count)
{
    const std::uint32_t byte_size = register_count * sizeof(std::uint32_t);
    return (byte_size + 0xFFU) & ~0xFFU;
}

static_assert(
    context_shadow_area_size(0xB00U) + context_shadow_area_size(0x400U) +
        context_shadow_area_size(0x800U) + context_shadow_area_size(0x60U) +
        context_shadow_area_size(0xD9EU) + context_shadow_area_size(0xA2U) ==
    gx2_context_shadow_state_size);
constexpr std::uint32_t gx2_tv_render_mode_wide_720p = 3U;
constexpr std::uint32_t gx2_tv_render_mode_wide_1080p = 5U;
constexpr std::uint32_t gx2_drc_render_mode_single = 1U;
constexpr std::uint32_t gx2_drc_render_mode_double = 2U;
constexpr std::uint32_t gx2_surface_format_unorm_r8_g8_b8_a8 = 0x1AU;
constexpr std::uint32_t gx2_buffering_mode_single = 1U;
constexpr std::uint32_t gx2_buffering_mode_double = 2U;
constexpr std::uint32_t gx2_buffering_mode_triple = 3U;
constexpr std::uint32_t gx2_event_type_vsync = 2U;
constexpr std::uint32_t gx2_event_type_flip = 3U;
constexpr std::uint32_t gx2_event_type_display_list_overrun = 4U;
// The Wii U GFD/GX2 resource format identifies this platform as GPU version 2.
constexpr std::uint32_t gx2_gpu_version = 2U;
constexpr auto gx2_virtual_vsync_period = std::chrono::nanoseconds(16'666'667);
constexpr std::uint32_t gx2_fetch_shader_instruction_size = 0x10U;
constexpr std::uint32_t gx2_fetch_shader_struct_size = 0x20U;
// GX2VertexShader ends with a 16-byte GX2RBuffer at offset 0x124.
constexpr std::uint32_t gx2_vertex_shader_struct_size = 0x134U;
constexpr std::uint32_t gx2_vertex_shader_program_size_offset = 0xD0U;
constexpr std::uint32_t gx2_vertex_shader_program_address_offset = 0xD4U;
// GX2PixelShader ends with a 16-byte GX2RBuffer at offset 0xD8.
constexpr std::uint32_t gx2_pixel_shader_struct_size = 0xE8U;
constexpr std::uint32_t gx2_pixel_shader_program_size_offset = 0xA4U;
constexpr std::uint32_t gx2_pixel_shader_program_address_offset = 0xA8U;
constexpr std::uint32_t gx2_geometry_shader_struct_size = 0xB0U;
constexpr std::uint32_t gx2_geometry_shader_program_size_offset = 0x4CU;
constexpr std::uint32_t gx2_geometry_shader_program_address_offset = 0x50U;
constexpr std::uint32_t gx2_geometry_shader_copy_size_offset = 0x54U;
constexpr std::uint32_t gx2_geometry_shader_copy_address_offset = 0x58U;
constexpr std::uint32_t gx2_max_shader_program_size = 0x01000000U;
constexpr std::uint32_t gx2_attrib_stream_size = 0x20U;
constexpr std::uint32_t gx2_max_fetch_attributes = 64U;
constexpr std::uint32_t gx2_sampler_size = 0x0CU;
constexpr std::uint32_t gx2_surface_size = 0x74U;
constexpr std::uint32_t gx2r_resflag_allocated_by_gx2r = 1U << 29U;
constexpr std::uint32_t gx2r_resflag_locked = 1U << 30U;
constexpr std::uint32_t gx2_texture_size = 0x9CU;
constexpr std::uint32_t gx2_surface_use_color_buffer = 0x02U;
constexpr std::uint32_t gx2_surface_use_depth_buffer = 0x04U;
constexpr std::uint32_t gx2_surface_dim_1d = 0U;
constexpr std::uint32_t gx2_surface_dim_3d = 2U;
constexpr std::uint32_t gx2_tile_mode_default = 0U;
constexpr std::uint32_t gx2_tile_mode_linear_aligned = 1U;
constexpr std::uint32_t gx2_tile_mode_1d_thin1 = 2U;
constexpr std::uint32_t gx2_tile_mode_1d_thick = 3U;
constexpr std::uint32_t gx2_tile_mode_2d_thin1 = 4U;
constexpr std::uint32_t gx2_tile_mode_2d_thick = 7U;
constexpr std::uint32_t gx2_tile_mode_linear_special = 16U;
constexpr std::uint32_t gx2_texture_regs_offset = 0x88U;
constexpr std::uint32_t gx2_pixel_texture_register_base = 0x00038000U;
constexpr std::uint32_t gx2_vertex_texture_register_base = 0x00039180U;
constexpr std::uint32_t gx2_geometry_texture_register_base = 0x0003A4C0U;
constexpr std::uint32_t gx2_pixel_sampler_register_base = 0x0003C000U;
constexpr std::uint32_t gx2_vertex_sampler_register_base = 0x0003C0D8U;
constexpr std::uint32_t gx2_geometry_sampler_register_base = 0x0003C1B0U;
constexpr std::uint32_t gx2_texture_register_count = 7U;
constexpr std::uint32_t gx2_texture_unit_count = 16U;
constexpr std::uint32_t gx2_vertex_buffer_count = 16U;
constexpr std::uint32_t gx2_sampler_register_count = 3U;
constexpr std::uint32_t gx2_sampler_unit_count = 16U;
constexpr std::uint32_t gx2_sampler_border_color_unit_count = 12U;
constexpr std::uint32_t gx2_pixel_sampler_border_register_base = 0x0000A400U;
constexpr std::uint32_t gx2_vertex_sampler_border_register_base = 0x0000A600U;
constexpr std::uint32_t gx2_geometry_sampler_border_register_base = 0x0000A800U;
constexpr std::uint32_t latte_pipe_interleave_bytes = 256U;
constexpr std::uint32_t latte_pipe_count = 2U;
constexpr std::uint32_t latte_bank_count = 4U;
constexpr std::uint32_t latte_split_size = 2048U;

Gx2ProgramShaderBinding capture_program_shader(
    EspressoCore& core,
    Gx2ProgramStage stage,
    std::uint32_t descriptor_size,
    std::uint32_t program_size_offset,
    std::uint32_t program_address_offset,
    const char* api_name,
    std::optional<std::pair<std::uint32_t, std::uint32_t>> secondary_offsets =
        std::nullopt)
{
    const std::uint32_t descriptor_address = core.state.gpr[3];
    if (descriptor_address == 0U || (descriptor_address & 3U) != 0U)
    {
        throw HleExecutionError(
            std::string(api_name) + ": descriptor is null or misaligned");
    }
    if (static_cast<std::uint64_t>(descriptor_address) + descriptor_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            std::string(api_name) + ": descriptor range wraps guest address space");
    }

    Gx2ProgramShaderBinding binding{};
    binding.stage = stage;
    binding.descriptor_address = descriptor_address;
    const std::size_t descriptor_word_count =
        descriptor_size / sizeof(std::uint32_t);
    binding.descriptor_words.reserve(descriptor_word_count);
    for (std::uint32_t offset = 0U; offset < descriptor_size;
         offset += sizeof(std::uint32_t))
    {
        binding.descriptor_words.push_back(
            core.memory.read32_be(descriptor_address + offset));
    }
    const std::size_t program_size_word =
        program_size_offset / sizeof(std::uint32_t);
    const std::size_t program_address_word =
        program_address_offset / sizeof(std::uint32_t);
    binding.program_size = binding.descriptor_words[program_size_word];
    binding.program_address = binding.descriptor_words[program_address_word];
    if (binding.program_size > gx2_max_shader_program_size ||
        (binding.program_size % sizeof(std::uint32_t)) != 0U)
    {
        throw HleExecutionError(
            std::string(api_name) + ": invalid shader program size");
    }
    if (binding.program_size != 0U &&
        (binding.program_address == 0U ||
         (binding.program_address & 3U) != 0U))
    {
        throw HleExecutionError(
            std::string(api_name) + ": program is null or misaligned");
    }
    if (static_cast<std::uint64_t>(binding.program_address) +
            binding.program_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            std::string(api_name) + ": program range wraps guest address space");
    }

    binding.program_words.reserve(
        binding.program_size / sizeof(std::uint32_t));
    for (std::uint32_t offset = 0U; offset < binding.program_size;
         offset += sizeof(std::uint32_t))
    {
        binding.program_words.push_back(
            core.memory.read32_be(binding.program_address + offset));
    }

    if (secondary_offsets)
    {
        const std::size_t secondary_size_word =
            secondary_offsets->first / sizeof(std::uint32_t);
        const std::size_t secondary_address_word =
            secondary_offsets->second / sizeof(std::uint32_t);
        binding.secondary_program_size =
            binding.descriptor_words[secondary_size_word];
        binding.secondary_program_address =
            binding.descriptor_words[secondary_address_word];
        if (binding.secondary_program_size > gx2_max_shader_program_size ||
            (binding.secondary_program_size % sizeof(std::uint32_t)) != 0U)
        {
            throw HleExecutionError(
                std::string(api_name) + ": invalid secondary shader program size");
        }
        if (binding.secondary_program_size != 0U &&
            (binding.secondary_program_address == 0U ||
             (binding.secondary_program_address & 3U) != 0U))
        {
            throw HleExecutionError(
                std::string(api_name) + ": secondary program is null or misaligned");
        }
        if (static_cast<std::uint64_t>(binding.secondary_program_address) +
                binding.secondary_program_size >
            guest_address_space_end)
        {
            throw HleExecutionError(
                std::string(api_name) + ": secondary program range wraps guest address space");
        }
        binding.secondary_program_words.reserve(
            binding.secondary_program_size / sizeof(std::uint32_t));
        for (std::uint32_t offset = 0U;
             offset < binding.secondary_program_size;
             offset += sizeof(std::uint32_t))
        {
            binding.secondary_program_words.push_back(
                core.memory.read32_be(binding.secondary_program_address + offset));
        }
    }
    return binding;
}

struct Gx2AttribStream
{
    std::uint32_t location{};
    std::uint32_t buffer{};
    std::uint32_t offset{};
    std::uint32_t format{};
    std::uint32_t type{};
    std::uint32_t alu_divisor{};
    std::uint32_t mask{};
    std::uint32_t endian_swap{};
};

struct TvDimensions
{
    std::uint32_t width{};
    std::uint32_t height{};
};

struct DrcScanGeometry
{
    std::uint32_t width{};
    std::uint32_t height{};
    std::uint32_t pitch{};
};

TvDimensions get_tv_dimensions(std::uint32_t render_mode, const char* api_name)
{
    switch (render_mode)
    {
    case gx2_tv_render_mode_wide_720p:
        return {1280U, 720U};
    case gx2_tv_render_mode_wide_1080p:
        return {1920U, 1080U};
    default:
        throw HleExecutionError(
            std::string("gx2::") + api_name + ": unsupported TV render mode " +
            std::to_string(render_mode));
    }
}

void validate_scan_surface_format(
    std::uint32_t surface_format,
    const char* api_name)
{
    if (surface_format != gx2_surface_format_unorm_r8_g8_b8_a8)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name + ": unsupported surface format " +
            std::to_string(surface_format));
    }
}

std::uint32_t gx2_attrib_format_bytes(std::uint32_t format)
{
    constexpr std::array<std::uint8_t, 20U> byte_sizes{
        1U, 1U, 2U, 2U, 2U, 4U, 4U, 4U, 4U, 4U,
        4U, 4U, 8U, 8U, 8U, 8U, 12U, 12U, 16U, 16U};
    const std::uint32_t data_format = format & 0x3FU;
    if (data_format >= byte_sizes.size())
    {
        return 0U;
    }
    return byte_sizes[data_format];
}

std::uint32_t gx2_attrib_endian_swap(std::uint32_t format)
{
    switch (format & 0x3FU)
    {
    case 0x00U: // 8
    case 0x01U: // 4_4
        return 0U;
    case 0x02U: // 16
    case 0x03U: // 16_FLOAT
    case 0x04U: // 8_8
    case 0x07U: // 16_16
    case 0x08U: // 16_16_FLOAT
    case 0x0EU: // 16_16_16_16
    case 0x0FU: // 16_16_16_16_FLOAT
        return 1U;
    default:
        return 2U;
    }
}

Gx2AttribStream read_attrib_stream(
    EspressoCore& core,
    std::uint32_t address)
{
    Gx2AttribStream result{};
    result.location = core.memory.read32_be(address);
    result.buffer = core.memory.read32_be(address + 0x04U);
    result.offset = core.memory.read32_be(address + 0x08U);
    result.format = core.memory.read32_be(address + 0x0CU);
    result.type = core.memory.read32_be(address + 0x10U);
    result.alu_divisor = core.memory.read32_be(address + 0x14U);
    result.mask = core.memory.read32_be(address + 0x18U);
    result.endian_swap = core.memory.read32_be(address + 0x1CU);
    return result;
}

Gx2ColorBufferState read_color_buffer(
    EspressoCore& core,
    std::uint32_t address)
{
    if (address == 0U || (address & 3U) != 0U)
    {
        throw HleExecutionError(
            "gx2::GX2SetColorBuffer: color buffer pointer is null or misaligned");
    }
    if (static_cast<std::uint64_t>(address) + gx2_color_buffer_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            "gx2::GX2SetColorBuffer: color buffer range wraps guest address space");
    }

    std::array<std::uint32_t, gx2_color_buffer_size / sizeof(std::uint32_t)> words{};
    for (std::size_t index = 0U; index < words.size(); ++index)
    {
        words[index] = core.memory.read32_be(
            address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)));
    }

    Gx2ColorBufferState result{};
    result.guest_address = address;
    result.dimension = words[0];
    result.width = words[1];
    result.height = words[2];
    result.depth = words[3];
    result.mip_levels = words[4];
    result.format = words[5];
    result.aa_mode = words[6];
    result.use = words[7];
    result.image_size = words[8];
    result.image_address = words[9];
    result.mipmap_size = words[10];
    result.mipmap_address = words[11];
    result.tile_mode = words[12];
    result.swizzle = words[13];
    result.alignment = words[14];
    result.pitch = words[15];
    std::copy_n(words.begin() + 16, result.mip_level_offsets.size(),
        result.mip_level_offsets.begin());
    result.view_mip = words[29];
    result.view_first_slice = words[30];
    result.view_num_slices = words[31];
    result.aa_buffer_address = words[32];
    result.aa_size = words[33];
    std::copy_n(words.begin() + 34, result.registers.size(),
        result.registers.begin());
    return result;
}

Gx2DepthBufferState read_depth_buffer(
    EspressoCore& core,
    std::uint32_t address)
{
    if (address == 0U || (address & 3U) != 0U)
    {
        throw HleExecutionError(
            "gx2::GX2SetDepthBuffer: depth buffer pointer is null or misaligned");
    }
    if (static_cast<std::uint64_t>(address) + gx2_depth_buffer_size >
        guest_address_space_end)
    {
        throw HleExecutionError(
            "gx2::GX2SetDepthBuffer: depth buffer range wraps guest address space");
    }

    std::array<std::uint32_t, gx2_depth_buffer_size / sizeof(std::uint32_t)> words{};
    for (std::size_t index = 0U; index < words.size(); ++index)
    {
        words[index] = core.memory.read32_be(
            address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)));
    }

    Gx2DepthBufferState result{};
    result.guest_address = address;
    result.dimension = words[0];
    result.width = words[1];
    result.height = words[2];
    result.depth = words[3];
    result.mip_levels = words[4];
    result.format = words[5];
    result.aa_mode = words[6];
    result.use = words[7];
    result.image_size = words[8];
    result.image_address = words[9];
    result.mipmap_size = words[10];
    result.mipmap_address = words[11];
    result.tile_mode = words[12];
    result.swizzle = words[13];
    result.alignment = words[14];
    result.pitch = words[15];
    std::copy_n(words.begin() + 16, result.mip_level_offsets.size(),
        result.mip_level_offsets.begin());
    result.view_mip = words[29];
    result.view_first_slice = words[30];
    result.view_num_slices = words[31];
    result.hi_z_address = words[32];
    result.hi_z_size = words[33];
    result.depth_clear_bits = words[34];
    result.stencil_clear = words[35];
    std::copy_n(words.begin() + 36, result.registers.size(),
        result.registers.begin());
    return result;
}

std::uint32_t get_scan_buffer_count(
    std::uint32_t buffering_mode,
    const char* api_name)
{
    switch (buffering_mode)
    {
    case gx2_buffering_mode_single: return 1U;
    case gx2_buffering_mode_double: return 2U;
    case gx2_buffering_mode_triple: return 3U;
    default:
        throw HleExecutionError(
            std::string("gx2::") + api_name + ": invalid buffering mode " +
            std::to_string(buffering_mode));
    }
}

DrcScanGeometry get_drc_scan_geometry(
    std::uint32_t render_mode,
    const char* api_name)
{
    if (render_mode != gx2_drc_render_mode_single &&
        render_mode != gx2_drc_render_mode_double)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name +
            ": unsupported DRC render mode " + std::to_string(render_mode));
    }

    return {854U, 480U, 864U};
}

void validate_gx2_u32_output(
    EspressoCore& core,
    std::uint32_t address,
    const char* api_name,
    const char* output_name)
{
    if (address == 0U)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name + ": " + output_name +
            " pointer is null");
    }
    if (static_cast<std::uint64_t>(address) + sizeof(std::uint32_t) >
        guest_address_space_end)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name + ": " + output_name +
            " pointer range wraps the guest address space");
    }
    core.memory.validate_write_range(address, sizeof(std::uint32_t));
}

void validate_gx2_u64_output(
    EspressoCore& core,
    std::uint32_t address,
    const char* api_name)
{
    if (address == 0U)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name +
            ": timestamp output pointer is null");
    }
    if (static_cast<std::uint64_t>(address) + sizeof(std::uint64_t) >
        guest_address_space_end)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name +
            ": timestamp output range wraps the guest address space");
    }
    try
    {
        core.memory.validate_write_range(address, sizeof(std::uint64_t));
    }
    catch (const GuestMemoryFault&)
    {
        throw HleExecutionError(
            std::string("gx2::") + api_name +
            ": timestamp output is outside mapped guest memory");
    }
}

template<std::size_t RangeCount>
void append_gx2_context_register_load(
    std::span<const std::uint8_t> context_bytes,
    std::uint32_t shadow_area_offset,
    std::uint32_t register_index_base,
    const Gx2ContextRegisterLoadRange (&ranges)[RangeCount],
    Gx2CommandType command_type,
    std::vector<Gx2Command>& commands)
{
    for (const Gx2ContextRegisterLoadRange range : ranges)
    {
        for (std::uint32_t index = 0U; index < range.register_count; ++index)
        {
            const std::uint32_t register_index =
                static_cast<std::uint32_t>(range.first_register) + index;
            const std::uint32_t value_offset = shadow_area_offset +
                register_index * sizeof(std::uint32_t);
            const std::uint32_t register_address =
                (register_index_base + register_index) * sizeof(std::uint32_t);
            const std::uint32_t value =
                (static_cast<std::uint32_t>(context_bytes[value_offset]) << 24U) |
                (static_cast<std::uint32_t>(context_bytes[value_offset + 1U]) << 16U) |
                (static_cast<std::uint32_t>(context_bytes[value_offset + 2U]) << 8U) |
                static_cast<std::uint32_t>(context_bytes[value_offset + 3U]);
            commands.push_back({
                command_type,
                register_address,
                value});
        }
    }
}

std::vector<Gx2Command> read_gx2_context_restore_commands(
    std::span<const std::uint8_t> context_bytes)
{
    std::vector<Gx2Command> commands;
    // Exact shadow-memory region sizes and load ranges follow the Cafe GX2
    // context-state layout. Preserve packet order because overlapping or
    // aliased Latte register ranges are order-sensitive.
    append_gx2_context_register_load(
        context_bytes, 0U, 0x2000U, gx2_context_config_load_ranges,
        Gx2CommandType::set_context_register, commands);
    const std::uint32_t context_area = context_shadow_area_size(0xB00U);
    append_gx2_context_register_load(
        context_bytes, context_area, 0xA000U,
        gx2_context_context_load_ranges,
        Gx2CommandType::set_context_register, commands);
    const std::uint32_t alu_area =
        context_area + context_shadow_area_size(0x400U);
    append_gx2_context_register_load(
        context_bytes, alu_area, 0xC000U, gx2_context_alu_load_ranges,
        Gx2CommandType::set_context_register, commands);
    const std::uint32_t loop_area =
        alu_area + context_shadow_area_size(0x800U);
    append_gx2_context_register_load(
        context_bytes, loop_area, 0xF880U, gx2_context_loop_load_ranges,
        Gx2CommandType::set_context_register, commands);
    const std::uint32_t resource_area =
        loop_area + context_shadow_area_size(0x60U);
    append_gx2_context_register_load(
        context_bytes, resource_area, 0xE000U,
        gx2_context_resource_load_ranges,
        Gx2CommandType::set_shader_resource, commands);
    const std::uint32_t sampler_area =
        resource_area + context_shadow_area_size(0xD9EU);
    append_gx2_context_register_load(
        context_bytes, sampler_area, 0xF000U,
        gx2_context_sampler_load_ranges,
        Gx2CommandType::set_shader_sampler, commands);
    return commands;
}

std::uint32_t fpr_float_argument_bits(
    const EspressoCore& core,
    std::size_t register_index)
{
    const double value = std::bit_cast<double>(core.state.fpr[register_index]);
    const float single_value = static_cast<float>(value);
    return std::bit_cast<std::uint32_t>(single_value);
}

std::uint16_t pack_gx2_scaled_i16_field(float value, float scale)
{
    // GX2 converts scaled float register inputs toward zero, then uses the low
    // 16 bits of the signed integer word. Each caller supplies scaling based
    // on the documented fixed-point or subpixel units of its register.
    const float fixed_value = value * scale;
    std::int32_t fixed_integer{};
    if (std::isnan(fixed_value) || fixed_value < -2147483648.0F)
    {
        // Match fctiwz's invalid/negative-saturation integer word.
        fixed_integer = std::numeric_limits<std::int32_t>::min();
    }
    else if (fixed_value >= 2147483648.0F)
    {
        fixed_integer = std::numeric_limits<std::int32_t>::max();
    }
    else
    {
        fixed_integer = static_cast<std::int32_t>(fixed_value);
    }
    return static_cast<std::uint16_t>(
        static_cast<std::uint32_t>(fixed_integer));
}

std::uint32_t pack_alpha_test_control(
    std::uint32_t alpha_test,
    std::uint32_t compare_function)
{
    const std::uint32_t enabled = alpha_test != 0U ? 1U : 0U;
    return (compare_function & 0x7U) | (enabled << 3U);
}

std::uint32_t pack_target_channel_masks(
    const std::array<std::uint32_t, 8U>& masks)
{
    std::uint32_t packed = 0U;
    for (std::size_t index = 0; index < masks.size(); ++index)
    {
        packed |= (masks[index] & 0xFU) << static_cast<unsigned>(index * 4U);
    }
    return packed;
}

std::uint32_t pack_aa_mask(
    std::uint32_t upper_left,
    std::uint32_t upper_right,
    std::uint32_t lower_left,
    std::uint32_t lower_right)
{
    return (upper_left & 0xFFU) |
           ((upper_right & 0xFFU) << 8U) |
           ((lower_left & 0xFFU) << 16U) |
           ((lower_right & 0xFFU) << 24U);
}

std::uint32_t pack_gx2_point_size(float width, float height)
{
    return (static_cast<std::uint32_t>(
                pack_gx2_scaled_i16_field(width, 8.0F))
            << 16U) |
           static_cast<std::uint32_t>(
               pack_gx2_scaled_i16_field(height, 8.0F));
}

std::uint32_t pack_alpha_to_mask(
    std::uint32_t enabled,
    std::uint32_t mode)
{
    std::uint32_t offsets = 0U;
    switch (mode)
    {
    case 0U: offsets = 0xAAU; break;
    case 1U: offsets = 0x78U; break;
    case 2U: offsets = 0xC6U; break;
    case 3U: offsets = 0x2DU; break;
    case 4U: offsets = 0x93U; break;
    default: break;
    }

    return (offsets << 8U) | (enabled != 0U ? 1U : 0U);
}

std::uint32_t calculate_tv_size(
    std::uint32_t render_mode,
    std::uint32_t surface_format,
    std::uint32_t buffering_mode)
{
    constexpr const char* api_name = "GX2CalcTVSize";
    const TvDimensions dimensions = get_tv_dimensions(render_mode, api_name);
    validate_scan_surface_format(surface_format, api_name);
    const std::uint32_t buffer_count =
        get_scan_buffer_count(buffering_mode, api_name);

    constexpr std::uint64_t bytes_per_pixel = 4U;
    const std::uint64_t width = dimensions.width;
    const std::uint64_t height = dimensions.height;
    const std::uint64_t size =
        width * height * bytes_per_pixel * buffer_count;
    if (size > std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError(
            "gx2::GX2CalcTVSize: calculated size overflows uint32_t");
    }
    return static_cast<std::uint32_t>(size);
}

std::uint32_t calculate_drc_size(
    std::uint32_t render_mode,
    std::uint32_t surface_format,
    std::uint32_t buffering_mode)
{
    constexpr const char* api_name = "GX2CalcDRCSize";
    const DrcScanGeometry geometry =
        get_drc_scan_geometry(render_mode, api_name);

    validate_scan_surface_format(surface_format, api_name);
    const std::uint32_t buffer_count =
        get_scan_buffer_count(buffering_mode, api_name);

    // The logical DRC image is 854x480, but scan-buffer storage uses an
    // 864-pixel row pitch. Size the allocation from that padded pitch.
    const std::uint64_t scan_pitch = geometry.pitch;
    const std::uint64_t height = geometry.height;
    constexpr std::uint64_t bytes_per_pixel = 4U;
    const std::uint64_t size =
        scan_pitch * height * bytes_per_pixel * buffer_count;
    if (size > std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError(
            "gx2::GX2CalcDRCSize: calculated size overflows uint32_t");
    }
    return static_cast<std::uint32_t>(size);
}

struct Gx2SurfaceMipLayout
{
    std::uint32_t pitch{};
    std::uint32_t height{};
    std::uint32_t depth{};
    std::uint32_t base_alignment{};
    std::uint32_t pitch_alignment{};
    std::uint32_t height_alignment{};
    std::uint64_t size{};
};

std::uint32_t gx2_surface_format_bits(std::uint32_t format)
{
    // GX2 formats share the low six-bit component layout across UNORM, UINT,
    // SNORM, SINT, SRGB, and floating-point variants.
    switch (format & 0x3FU)
    {
    case 0x01U: return 8U;   // R8
    case 0x02U: return 8U;   // R4_G4
    case 0x05U: return 16U;  // R16
    case 0x07U: return 16U;  // R8_G8
    case 0x06U: return 16U;  // FLOAT_R16
    case 0x08U:             // R5_G6_B5
    case 0x0AU:             // R5_G5_B5_A1
    case 0x0BU:             // R4_G4_B4_A4
    case 0x0CU: return 16U; // A1_B5_G5_R5
    case 0x0DU:             // UINT_R32
    case 0x0EU: return 32U; // FLOAT_R32
    case 0x0FU: return 32U; // R16_G16
    case 0x10U: return 32U; // FLOAT_R16_G16
    case 0x11U: return 32U; // R24_X8
    case 0x16U: return 32U; // R11_G11_B10
    case 0x19U:             // R10_G10_B10_A2
    case 0x1AU:             // R8_G8_B8_A8
    case 0x1BU: return 32U; // A2_B10_G10_R10
    case 0x1DU:             // UINT_R32_G32
    case 0x1EU: return 64U; // FLOAT_R32_G32
    case 0x1FU:             // R16_G16_B16_A16
    case 0x20U: return 64U; // FLOAT_R16_G16_B16_A16
    case 0x22U:             // R32_G32_B32_A32
    case 0x23U: return 128U;// FLOAT_R32_G32_B32_A32
    default:
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: unsupported surface format " +
            std::to_string(format));
    }
}

std::uint32_t align_gx2_dimension(std::uint32_t value, std::uint32_t alignment)
{
    if (alignment == 0U)
    {
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: invalid zero layout alignment");
    }
    const std::uint64_t aligned =
        (static_cast<std::uint64_t>(value) + alignment - 1U) / alignment * alignment;
    if (aligned > std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: aligned dimension overflows uint32_t");
    }
    return static_cast<std::uint32_t>(aligned);
}

Gx2SurfaceMipLayout calculate_gx2_surface_mip_layout(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t depth,
    std::uint32_t dimension,
    std::uint32_t format_bits,
    std::uint32_t samples,
    std::uint32_t tile_mode,
    std::uint32_t mip_level,
    bool display_surface)
{
    std::uint32_t base_alignment = 1U;
    std::uint32_t pitch_alignment = 1U;
    std::uint32_t height_alignment = 1U;
    std::uint32_t depth_alignment = 1U;
    std::uint32_t thickness = 1U;
    std::uint32_t normalized_bits = format_bits;

    if (normalized_bits == 24U || normalized_bits == 48U ||
        normalized_bits == 96U)
    {
        normalized_bits /= 3U;
    }

    switch (tile_mode)
    {
    case gx2_tile_mode_linear_special:
        break;
    case gx2_tile_mode_linear_aligned:
        base_alignment = latte_pipe_interleave_bytes;
        pitch_alignment = std::max(
            64U, (8U * latte_pipe_interleave_bytes) / normalized_bits);
        break;
    case gx2_tile_mode_1d_thin1:
    case gx2_tile_mode_1d_thick:
        thickness = tile_mode == gx2_tile_mode_1d_thick ? 4U : 1U;
        depth_alignment = thickness;
        base_alignment = latte_pipe_interleave_bytes;
        pitch_alignment = std::max(
            8U, latte_pipe_interleave_bytes / normalized_bits / samples / thickness);
        height_alignment = 8U;
        break;
    case gx2_tile_mode_2d_thin1:
    case gx2_tile_mode_2d_thick:
    {
        thickness = tile_mode == gx2_tile_mode_2d_thick ? 4U : 1U;
        depth_alignment = thickness;
        constexpr std::uint32_t macro_width = 8U * latte_bank_count;
        constexpr std::uint32_t macro_height = 8U * latte_pipe_count;
        pitch_alignment = std::max(
            macro_width,
            macro_width * (latte_pipe_interleave_bytes / normalized_bits /
                           (8U * thickness) / samples));
        height_alignment = macro_height;
        const std::uint64_t macro_tile_bytes =
            static_cast<std::uint64_t>(samples) * normalized_bits * macro_height *
            macro_width / 8U;
        const std::uint64_t aligned_row_bytes =
            static_cast<std::uint64_t>(samples) * macro_height * normalized_bits *
            pitch_alignment / 8U;
        std::uint64_t base = std::max(macro_tile_bytes, aligned_row_bytes);
        const std::uint64_t micro_tile_bytes =
            static_cast<std::uint64_t>(thickness) * samples * normalized_bits * 64U / 8U;
        const std::uint64_t splits = micro_tile_bytes < latte_split_size
            ? 1U
            : micro_tile_bytes / latte_split_size;
        base /= splits;
        if (base > std::numeric_limits<std::uint32_t>::max())
        {
            throw HleExecutionError(
                "gx2::GX2CalcSurfaceSizeAndAlignment: base alignment overflows uint32_t");
        }
        base_alignment = static_cast<std::uint32_t>(base);
        break;
    }
    default:
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: unsupported tile mode " +
            std::to_string(tile_mode));
    }

    // Scan/display surfaces require pitch alignment to at least 32 pixels.
    // This is applied after the tile-mode alignment is derived, matching GX2's
    // display-surface adjustment without changing non-display layouts.
    if (display_surface)
    {
        pitch_alignment = align_gx2_dimension(pitch_alignment, 32U);
    }

    const std::uint32_t mip_width = std::max(width >> mip_level, 1U);
    const std::uint32_t mip_height =
        dimension == gx2_surface_dim_1d ? 1U : std::max(height >> mip_level, 1U);
    const std::uint32_t mip_depth = dimension == gx2_surface_dim_3d
        ? std::max(depth >> mip_level, 1U)
        : (dimension == 4U || dimension == 5U || dimension == 7U ? depth : 1U);

    if (mip_level != 0U && tile_mode != gx2_tile_mode_linear_special)
    {
        // Non-base mip levels are power-of-two padded before the tile's own
        // pitch/height/depth alignment is applied.
        pitch_alignment = std::bit_ceil(pitch_alignment);
        height_alignment = std::bit_ceil(height_alignment);
        depth_alignment = std::bit_ceil(depth_alignment);
    }

    const std::uint32_t padded_mip_width =
        mip_level != 0U && tile_mode != gx2_tile_mode_linear_special
            ? std::bit_ceil(mip_width)
            : mip_width;
    const std::uint32_t padded_mip_height =
        mip_level != 0U && tile_mode != gx2_tile_mode_linear_special
            ? std::bit_ceil(mip_height)
            : mip_height;
    const std::uint32_t padded_mip_depth =
        mip_level != 0U && tile_mode != gx2_tile_mode_linear_special
            ? std::bit_ceil(mip_depth)
            : mip_depth;

    Gx2SurfaceMipLayout layout{};
    layout.pitch = align_gx2_dimension(padded_mip_width, pitch_alignment);
    layout.height = align_gx2_dimension(padded_mip_height, height_alignment);
    layout.depth = align_gx2_dimension(padded_mip_depth, depth_alignment);
    layout.base_alignment = base_alignment;
    layout.pitch_alignment = pitch_alignment;
    layout.height_alignment = height_alignment;
    const std::uint64_t maximum_texels =
        (static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()) * 8U) /
        (static_cast<std::uint64_t>(format_bits) * samples);
    std::uint64_t texels = layout.pitch;
    if (layout.height != 0U && texels > maximum_texels / layout.height)
    {
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: surface level size overflows uint32_t");
    }
    texels *= layout.height;
    if (layout.depth != 0U && texels > maximum_texels / layout.depth)
    {
        throw HleExecutionError(
            "gx2::GX2CalcSurfaceSizeAndAlignment: surface level size overflows uint32_t");
    }
    texels *= layout.depth;
    layout.size = texels * format_bits * samples / 8U;
    return layout;
}

void calculate_gx2_surface_size_and_alignment(
    EspressoCore& core,
    std::uint32_t surface_address)
{
    constexpr const char* api_name = "gx2::GX2CalcSurfaceSizeAndAlignment";
    if (surface_address == 0U || (surface_address & 3U) != 0U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": surface pointer is null or misaligned");
    }

    std::array<std::uint32_t, gx2_surface_size / sizeof(std::uint32_t)> words{};
    for (std::size_t index = 0U; index < words.size(); ++index)
    {
        words[index] = core.memory.read32_be(
            surface_address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)));
    }

    const std::uint32_t dimension = words[0];
    const std::uint32_t width = words[1];
    const std::uint32_t height = words[2];
    const std::uint32_t depth = words[3];
    std::uint32_t mip_levels = words[4];
    const std::uint32_t format = words[5];
    const std::uint32_t aa_mode = words[6];
    const std::uint32_t surface_use = words[7];
    std::uint32_t tile_mode = words[12];
    std::uint32_t swizzle = words[13];

    if (dimension > 7U || width == 0U || height == 0U || depth == 0U ||
        width >= 0x01000000U || height >= 0x01000000U ||
        depth >= 0x01000000U || aa_mode > 3U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": invalid surface dimensions, dimension, or AA mode");
    }

    const std::uint32_t format_bits = gx2_surface_format_bits(format);
    const std::uint32_t samples = 1U << aa_mode;
    const bool is_3d = dimension == gx2_surface_dim_3d;
    const bool color_buffer = (surface_use & gx2_surface_use_color_buffer) != 0U;
    const bool depth_buffer = (surface_use & gx2_surface_use_depth_buffer) != 0U;
    const bool display_surface = (surface_use & 0x08U) != 0U;

    if (tile_mode == gx2_tile_mode_default)
    {
        if (dimension != gx2_surface_dim_1d || depth_buffer || aa_mode != 0U)
        {
            tile_mode = (!is_3d || color_buffer)
                ? gx2_tile_mode_2d_thin1
                : gx2_tile_mode_2d_thick;
        }
        else
        {
            tile_mode = gx2_tile_mode_linear_aligned;
        }
    }

    if (mip_levels == 0U)
    {
        mip_levels = 1U;
    }
    std::uint32_t max_levels = 1U;
    for (std::uint32_t extent = std::max(width, height); extent > 1U; extent >>= 1U)
    {
        ++max_levels;
    }
    if (is_3d)
    {
        std::uint32_t depth_levels = 1U;
        for (std::uint32_t extent = depth; extent > 1U; extent >>= 1U)
            ++depth_levels;
        max_levels = std::max(max_levels, depth_levels);
    }
    mip_levels = std::min(mip_levels, max_levels);

    if (tile_mode != gx2_tile_mode_linear_special &&
        tile_mode != gx2_tile_mode_linear_aligned &&
        tile_mode != gx2_tile_mode_1d_thin1 &&
        tile_mode != gx2_tile_mode_1d_thick &&
        tile_mode != gx2_tile_mode_2d_thin1 &&
        tile_mode != gx2_tile_mode_2d_thick)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": unsupported effective tile mode " +
                                std::to_string(tile_mode));
    }

    if (tile_mode == gx2_tile_mode_2d_thin1 ||
        tile_mode == gx2_tile_mode_2d_thick)
    {
        const Gx2SurfaceMipLayout first = calculate_gx2_surface_mip_layout(
            width, height, depth, dimension, format_bits, samples, tile_mode, 0U,
            display_surface);
        if (width < first.pitch_alignment && height < first.height_alignment)
        {
            tile_mode = tile_mode == gx2_tile_mode_2d_thick
                ? gx2_tile_mode_1d_thick
                : gx2_tile_mode_1d_thin1;
        }
    }

    swizzle = (swizzle & 0xFF00FFFFU) |
              ((tile_mode == gx2_tile_mode_2d_thin1 ||
                tile_mode == gx2_tile_mode_2d_thick) ? 0x000D0000U : 0U);

    std::array<std::uint32_t, 13U> mip_offsets{};
    std::uint64_t previous_size = 0U;
    std::uint64_t current_mip_offset = 0U;
    std::uint32_t first_mip_offset = 0U;
    std::uint32_t image_size = 0U;
    std::uint32_t alignment = 0U;
    std::uint32_t pitch = 0U;

    for (std::uint32_t level = 0U; level < mip_levels; ++level)
    {
        std::uint32_t level_tile_mode = tile_mode;
        if (level != 0U &&
            (tile_mode == gx2_tile_mode_2d_thin1 ||
             tile_mode == gx2_tile_mode_2d_thick))
        {
            const std::uint32_t mip_width = std::max(width >> level, 1U);
            const std::uint32_t mip_height = std::max(height >> level, 1U);
            const Gx2SurfaceMipLayout macro = calculate_gx2_surface_mip_layout(
                width, height, depth, dimension, format_bits, samples, tile_mode,
                level, display_surface);
            if (mip_width < macro.pitch_alignment ||
                mip_height < macro.height_alignment)
            {
                level_tile_mode = tile_mode == gx2_tile_mode_2d_thick
                    ? gx2_tile_mode_1d_thick
                    : gx2_tile_mode_1d_thin1;
            }
        }

        const Gx2SurfaceMipLayout layout = calculate_gx2_surface_mip_layout(
            width, height, depth, dimension, format_bits, samples,
            level_tile_mode, level, display_surface);
        if (layout.size > std::numeric_limits<std::uint32_t>::max())
        {
            throw HleExecutionError(std::string(api_name) +
                                    ": surface level size overflows uint32_t");
        }

        if (level == 0U)
        {
            image_size = static_cast<std::uint32_t>(layout.size);
            alignment = layout.base_alignment;
            pitch = layout.pitch;
        }
        else
        {
            const std::uint64_t align = layout.base_alignment;
            const std::uint64_t padding =
                (align - current_mip_offset % align) % align;
            current_mip_offset += padding + previous_size;
            if (current_mip_offset > std::numeric_limits<std::uint32_t>::max())
            {
                throw HleExecutionError(std::string(api_name) +
                                        ": mip offset overflows uint32_t");
            }
            if (level == 1U)
            {
                first_mip_offset = static_cast<std::uint32_t>(current_mip_offset);
            }
            else
            {
                mip_offsets[level - 1U] = static_cast<std::uint32_t>(current_mip_offset);
            }
        }
        previous_size = layout.size;
    }

    const std::uint64_t mipmap_size = mip_levels > 1U
        ? previous_size + current_mip_offset
        : 0U;
    if (mipmap_size > std::numeric_limits<std::uint32_t>::max())
    {
        throw HleExecutionError(std::string(api_name) +
                                ": mipmap size overflows uint32_t");
    }
    mip_offsets[0] = first_mip_offset;

    core.memory.validate_write_range(surface_address, gx2_surface_size);
    core.memory.write32_be(surface_address + 0x10U, mip_levels);
    core.memory.write32_be(surface_address + 0x20U, image_size);
    core.memory.write32_be(surface_address + 0x28U,
                           static_cast<std::uint32_t>(mipmap_size));
    core.memory.write32_be(surface_address + 0x30U, tile_mode);
    core.memory.write32_be(surface_address + 0x34U, swizzle);
    core.memory.write32_be(surface_address + 0x38U, alignment);
    core.memory.write32_be(surface_address + 0x3CU, pitch);
    for (std::size_t index = 0U; index < mip_offsets.size(); ++index)
    {
        core.memory.write32_be(
            surface_address + 0x40U + static_cast<std::uint32_t>(index * 4U),
            mip_offsets[index]);
    }
}

void create_gx2_surface_user_memory(
    EspressoCore& core,
    std::uint32_t surface_address,
    std::uint32_t image_address,
    std::uint32_t mipmap_address,
    std::uint32_t resource_flags)
{
    constexpr const char* api_name = "gx2::GX2RCreateSurfaceUserMemory";
    if (surface_address == 0U || (surface_address & 3U) != 0U ||
        surface_address > std::numeric_limits<std::uint32_t>::max() -
                              gx2_surface_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": surface pointer is null, misaligned, or wraps");
    }

    // This API stores guest virtual addresses supplied by the caller; the
    // image allocations are not accessed while creating the descriptor.
    core.memory.validate_write_range(surface_address, gx2_surface_size);
    const std::uint32_t previous_use = core.memory.read32_be(surface_address + 0x1CU);
    const std::uint32_t user_memory_flags = resource_flags &
        ~(gx2r_resflag_locked | gx2r_resflag_allocated_by_gx2r);

    // GX2CalcSurfaceSizeAndAlignment uses the resource-use bits in this same
    // field. Restore the original word if its format/layout validation fails.
    core.memory.write32_be(surface_address + 0x1CU, user_memory_flags);
    try
    {
        calculate_gx2_surface_size_and_alignment(core, surface_address);
    }
    catch (...)
    {
        core.memory.write32_be(surface_address + 0x1CU, previous_use);
        throw;
    }

    core.memory.write32_be(surface_address + 0x24U, image_address);
    core.memory.write32_be(surface_address + 0x2CU, mipmap_address);
}

std::uint32_t gx2r_lock_surface(
    EspressoCore& core,
    std::uint32_t surface_address)
{
    constexpr const char* api_name = "gx2::GX2RLockSurfaceEx";
    if (surface_address == 0U || (surface_address & 3U) != 0U ||
        surface_address > std::numeric_limits<std::uint32_t>::max() -
                              gx2_surface_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": surface pointer is null, misaligned, or wraps");
    }

    core.memory.validate_write_range(surface_address, gx2_surface_size);
    const std::uint32_t resource_flags = core.memory.read32_be(surface_address + 0x1CU);
    const std::uint32_t image_address = core.memory.read32_be(surface_address + 0x24U);
    core.memory.write32_be(
        surface_address + 0x1CU,
        resource_flags | gx2r_resflag_locked);
    // Current GX2 semantics return the base image pointer even for nonzero
    // mip levels; per-mip pointer adjustment/invalidation is not modeled yet.
    return image_address;
}

bool gx2r_surface_exists(EspressoCore& core, std::uint32_t surface_address)
{
    if (surface_address == 0U)
    {
        return false;
    }
    constexpr std::uint32_t cpu_gpu_usage_mask = 0x00007800U;
    if ((surface_address & 3U) != 0U ||
        surface_address > std::numeric_limits<std::uint32_t>::max() -
                              gx2_surface_size)
    {
        throw HleExecutionError(
            "gx2::GX2RSurfaceExists: surface pointer is misaligned or wraps");
    }

    const std::uint32_t resource_flags = core.memory.read32_be(surface_address + 0x1CU);
    const std::uint32_t image_address = core.memory.read32_be(surface_address + 0x24U);
    return image_address != 0U && (resource_flags & cpu_gpu_usage_mask) != 0U;
}

void gx2r_unlock_surface(EspressoCore& core, std::uint32_t surface_address)
{
    constexpr const char* api_name = "gx2::GX2RUnlockSurfaceEx";
    if (surface_address == 0U || (surface_address & 3U) != 0U ||
        surface_address > std::numeric_limits<std::uint32_t>::max() -
                              gx2_surface_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": surface pointer is null, misaligned, or wraps");
    }

    core.memory.validate_write_range(surface_address, gx2_surface_size);
    const std::uint32_t resource_flags = core.memory.read32_be(surface_address + 0x1CU);
    core.memory.write32_be(
        surface_address + 0x1CU,
        resource_flags & ~gx2r_resflag_locked);
}

void create_gx2_buffer_user_memory(
    EspressoCore& core,
    std::uint32_t buffer_address,
    std::uint32_t data_address)
{
    constexpr const char* api_name = "gx2::GX2RCreateBufferUserMemory";
    constexpr std::uint32_t gx2r_buffer_size = 0x10U;
    if (buffer_address == 0U || (buffer_address & 3U) != 0U ||
        buffer_address > std::numeric_limits<std::uint32_t>::max() -
                             gx2r_buffer_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": buffer pointer is null, misaligned, or wraps");
    }

    // GX2RCreateBufferUserMemory records an already allocated guest pointer;
    // it does not access the allocation or use its legacy third argument.
    core.memory.validate_write_range(buffer_address, gx2r_buffer_size);
    const std::uint32_t resource_flags = core.memory.read32_be(buffer_address);
    const std::uint32_t user_memory_flags = resource_flags &
        ~(gx2r_resflag_locked | gx2r_resflag_allocated_by_gx2r);
    core.memory.write32_be(buffer_address, user_memory_flags);
    core.memory.write32_be(buffer_address + 0x0CU, data_address);
}

bool gx2r_buffer_exists(EspressoCore& core, std::uint32_t buffer_address)
{
    if (buffer_address == 0U)
    {
        return false;
    }
    constexpr std::uint32_t gx2r_buffer_size = 0x10U;
    constexpr std::uint32_t cpu_gpu_usage_mask = 0x00007800U;
    constexpr const char* api_name = "gx2::GX2RBufferExists";
    if ((buffer_address & 3U) != 0U ||
        buffer_address > std::numeric_limits<std::uint32_t>::max() -
                             gx2r_buffer_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": buffer pointer is misaligned or wraps");
    }

    const std::uint32_t resource_flags = core.memory.read32_be(buffer_address);
    const std::uint32_t data_address =
        core.memory.read32_be(buffer_address + 0x0CU);
    return data_address != 0U && (resource_flags & cpu_gpu_usage_mask) != 0U;
}

std::uint32_t gx2r_get_buffer_allocation_size(
    EspressoCore& core,
    std::uint32_t buffer_address)
{
    constexpr const char* api_name = "gx2::GX2RGetBufferAllocationSize";
    constexpr std::uint32_t gx2r_buffer_size = 0x10U;
    if (buffer_address == 0U || (buffer_address & 3U) != 0U ||
        buffer_address > std::numeric_limits<std::uint32_t>::max() -
                             gx2r_buffer_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": buffer pointer is null, misaligned, or wraps");
    }

    const std::uint32_t element_size = core.memory.read32_be(buffer_address + 4U);
    const std::uint32_t element_count = core.memory.read32_be(buffer_address + 8U);
    // GX2R's guest GetSize operation is uint32_t arithmetic, then pads that
    // wrapped byte count upward to the next 0x40-byte allocation boundary.
    const std::uint32_t size = element_size * element_count;
    return (size + 0x3FU) & ~0x3FU;
}

std::uint32_t gx2r_get_buffer_alignment(std::uint32_t resource_flags)
{
    constexpr std::uint32_t alignment_0x100_flags =
        (1U << 8U) | (1U << 6U) | (1U << 7U) | (1U << 10U);
    constexpr std::uint32_t alignment_0x40_flags =
        (1U << 4U) | (1U << 5U) | (1U << 9U);
    if ((resource_flags & alignment_0x100_flags) != 0U)
    {
        return 0x100U;
    }
    if ((resource_flags & alignment_0x40_flags) != 0U)
    {
        return 0x40U;
    }
    return 0x100U;
}

std::uint32_t gx2r_lock_buffer(EspressoCore& core, std::uint32_t buffer_address)
{
    constexpr const char* api_name = "gx2::GX2RLockBufferEx";
    constexpr std::uint32_t gx2r_buffer_size = 0x10U;
    if (buffer_address == 0U || (buffer_address & 3U) != 0U ||
        buffer_address > std::numeric_limits<std::uint32_t>::max() -
                             gx2r_buffer_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": buffer pointer is null, misaligned, or wraps");
    }

    // GX2RLockBufferEx returns the descriptor's guest pointer. The current
    // memory model is coherent and this operation does not modify the buffer.
    return core.memory.read32_be(buffer_address + 0x0CU);
}

void notify_gx2r_buffer_coherent(
    EspressoCore& core,
    std::uint32_t buffer_address,
    const char* api_name)
{
    constexpr std::uint32_t gx2r_buffer_size = 0x10U;
    if (buffer_address == 0U || (buffer_address & 3U) != 0U ||
        buffer_address > std::numeric_limits<std::uint32_t>::max() -
                             gx2r_buffer_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": buffer pointer is null, misaligned, or wraps");
    }

    const std::uint32_t element_size = core.memory.read32_be(buffer_address + 4U);
    const std::uint32_t element_count = core.memory.read32_be(buffer_address + 8U);
    const std::uint32_t data_address = core.memory.read32_be(buffer_address + 0x0CU);
    const std::uint32_t byte_size = element_size * element_count;
    // The reference reports this range to the GPU buffer cache. GuestMemory
    // is already coherent with GPU consumers in the current model, so there
    // is no separate cache state to flush or invalidate.
    static_cast<void>(data_address);
    static_cast<void>(byte_size);
}

std::uint32_t gx2_texture_endian_swap(std::uint32_t format)
{
    switch (format & 0x3FU)
    {
    case 0x01U: // R8
    case 0x02U: // R4_G4
    case 0x0DU: // R32
    case 0x0EU: // FLOAT_R32
    case 0x1DU: // R32_G32
    case 0x1EU: // FLOAT_R32_G32
    case 0x22U: // R32_G32_B32_A32
    case 0x23U: // FLOAT_R32_G32_B32_A32
    case 0x31U: // BC1
    case 0x32U: // BC2
    case 0x33U: // BC3
    case 0x34U: // BC4
    case 0x35U: // BC5
        return 0U;
    case 0x05U: // R16
    case 0x06U: // FLOAT_R16
    case 0x07U: // R8_G8
    case 0x08U: // R5_G6_B5
    case 0x0AU: // R5_G5_B5_A1
    case 0x0BU: // R4_G4_B4_A4
    case 0x0CU: // A1_B5_G5_R5
        return 1U;
    case 0x0FU: // R16_G16
    case 0x10U: // FLOAT_R16_G16
    case 0x11U: // R24_X8 / D24_S8
    case 0x16U: // R11_G11_B10
    case 0x19U: // R10_G10_B10_A2
    case 0x1AU: // R8_G8_B8_A8
    case 0x1BU: // A2_B10_G10_R10
        return 2U;
    case 0x1FU: // R16_G16_B16_A16
    case 0x20U: // FLOAT_R16_G16_B16_A16
        return 3U;
    default:
        throw HleExecutionError(
            "gx2::GX2InitTextureRegs: unsupported format endian mode for " +
            std::to_string(format));
    }
}

void initialize_gx2_texture_registers(
    EspressoCore& core,
    std::uint32_t texture_address)
{
    constexpr const char* api_name = "gx2::GX2InitTextureRegs";
    if (texture_address == 0U || (texture_address & 3U) != 0U ||
        texture_address > std::numeric_limits<std::uint32_t>::max() -
                              gx2_texture_size)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": texture pointer is null or misaligned");
    }

    std::array<std::uint32_t, gx2_texture_size / sizeof(std::uint32_t)> words{};
    for (std::size_t index = 0U; index < words.size(); ++index)
    {
        words[index] = core.memory.read32_be(
            texture_address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)));
    }

    constexpr std::uint32_t surface_width_offset = 0x04U;
    constexpr std::uint32_t surface_height_offset = 0x08U;
    constexpr std::uint32_t surface_depth_offset = 0x0CU;
    constexpr std::uint32_t surface_levels_offset = 0x10U;
    constexpr std::uint32_t surface_format_offset = 0x14U;
    constexpr std::uint32_t surface_aa_offset = 0x18U;
    constexpr std::uint32_t surface_use_offset = 0x1CU;
    constexpr std::uint32_t surface_tile_mode_offset = 0x30U;
    constexpr std::uint32_t surface_pitch_offset = 0x3CU;
    constexpr std::uint32_t view_first_mip_offset = 0x74U;
    constexpr std::uint32_t view_num_mips_offset = 0x78U;
    constexpr std::uint32_t view_first_slice_offset = 0x7CU;
    constexpr std::uint32_t view_num_slices_offset = 0x80U;
    constexpr std::uint32_t component_map_offset = 0x84U;

    const std::uint32_t dimension = words[0];
    const std::uint32_t width = words[surface_width_offset / 4U];
    const std::uint32_t height = std::max(words[surface_height_offset / 4U], 1U);
    const std::uint32_t depth = std::max(words[surface_depth_offset / 4U], 1U);
    const std::uint32_t surface_levels =
        std::max(words[surface_levels_offset / 4U], 1U);
    const std::uint32_t format = words[surface_format_offset / 4U];
    const std::uint32_t aa_mode = words[surface_aa_offset / 4U];
    const std::uint32_t surface_use = words[surface_use_offset / 4U];
    const std::uint32_t tile_mode = words[surface_tile_mode_offset / 4U];
    const std::uint32_t pitch = words[surface_pitch_offset / 4U];
    const std::uint32_t view_first_mip = words[view_first_mip_offset / 4U];
    const std::uint32_t view_num_mips =
        std::max(words[view_num_mips_offset / 4U], 1U);
    const std::uint32_t view_first_slice =
        words[view_first_slice_offset / 4U];
    const std::uint32_t view_num_slices =
        std::max(words[view_num_slices_offset / 4U], 1U);
    const std::uint32_t component_map = words[component_map_offset / 4U];

    if (dimension > 7U || width > 0x2000U || height > 0x2000U ||
        depth > 0x2000U || aa_mode > 3U || surface_levels > 16U ||
        view_first_mip > 15U || view_num_mips > 16U ||
        view_first_mip + view_num_mips > surface_levels ||
        view_first_slice > 0x1FFFU || view_num_slices > 0x2000U ||
        view_first_slice + view_num_slices > 0x2000U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": texture dimensions or view range are invalid");
    }

    // The register data-format field uses the low six bits of GX2SurfaceFormat;
    // validate that the format is one whose layout and byte ordering we model.
    const std::uint32_t base_format = format & 0x3FU;
    const bool compressed = base_format >= 0x31U && base_format <= 0x35U;
    if (!compressed)
        (void)gx2_surface_format_bits(format);
    if ((format & ~0x0F3FU) != 0U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": unsupported surface format flags");
    }
    if (tile_mode > 15U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": unsupported hardware tile mode " +
                                std::to_string(tile_mode));
    }
    const std::uint32_t component_x = (component_map >> 24U) & 7U;
    const std::uint32_t component_y = (component_map >> 16U) & 7U;
    const std::uint32_t component_z = (component_map >> 8U) & 7U;
    const std::uint32_t component_w = component_map & 7U;
    if (component_x > 5U || component_y > 5U ||
        component_z > 5U || component_w > 5U)
    {
        throw HleExecutionError(std::string(api_name) +
                                ": invalid texture component selector");
    }

    if (compressed && pitch > std::numeric_limits<std::uint32_t>::max() / 4U)
    {
        throw HleExecutionError(std::string(api_name) + ": texture pitch overflows");
    }
    const std::uint32_t pixel_pitch = compressed ? pitch * 4U : pitch;
    const std::uint32_t encoded_pitch = pixel_pitch == 0U
        ? 0x7FFU
        : ((pixel_pitch >> 3U) - 1U) & 0x7FFU;
    const std::uint32_t encoded_width = width == 0U ? 0x1FFFU : width - 1U;
    const std::uint32_t tile_type = (surface_use & gx2_surface_use_depth_buffer) != 0U
        ? 1U
        : 0U;

    // Latte SQ_TEX_RESOURCE_WORD0/1 field positions.
    const std::uint32_t word0 =
        (dimension & 7U) |
        ((tile_mode & 0xFU) << 3U) |
        (tile_type << 7U) |
        (encoded_pitch << 8U) |
        ((encoded_width & 0x1FFFU) << 19U);
    const std::uint32_t word1_depth =
        dimension == 3U
            ? (depth / 6U == 0U ? 0U : depth / 6U - 1U)
            : (dimension == 2U || dimension == 4U || dimension == 5U ||
                       dimension == 7U
                   ? depth - 1U
                   : 0U);
    if (dimension == 3U && (depth < 6U || depth % 6U != 0U))
    {
        throw HleExecutionError(std::string(api_name) +
                                ": cubemap depth must be a multiple of six");
    }
    const std::uint32_t word1 =
        ((height - 1U) & 0x1FFFU) |
        ((word1_depth & 0x1FFFU) << 13U) |
        ((format & 0x3FU) << 26U);

    const std::uint32_t signed_format = (format & 0x200U) != 0U ? 1U : 0U;
    const std::uint32_t numeric_format = (format & 0x800U) != 0U
        ? 2U
        : ((format & 0x100U) != 0U ? 1U : 0U);
    const std::uint32_t force_degamma = (format & 0x400U) != 0U ? 1U : 0U;
    const std::uint32_t word4 =
        signed_format * 0x55U |
        (numeric_format << 8U) |
        (force_degamma << 11U) |
        (gx2_texture_endian_swap(format) << 12U) |
        (2U << 14U) |
        (component_x << 16U) |
        (component_y << 19U) |
        (component_z << 22U) |
        (component_w << 25U) |
        ((view_first_mip & 0xFU) << 28U);

    const std::uint32_t view_last_mip =
        view_first_mip + view_num_mips - 1U;
    const std::uint32_t last_mip = aa_mode >= 1U && aa_mode <= 3U
        ? aa_mode
        : view_last_mip;
    const std::uint32_t view_last_slice =
        view_first_slice + view_num_slices - 1U;
    const std::uint32_t word5 =
        (last_mip & 0xFU) |
        ((view_first_slice & 0x1FFFU) << 4U) |
        ((view_last_slice & 0x1FFFU) << 17U) |
        ((dimension == 3U && word1_depth != 0U) ? (1U << 30U) : 0U);
    constexpr std::uint32_t word6 =
        4U | (7U << 5U) | (2U << 30U);

    const std::array<std::uint32_t, 5U> registers{
        word0, word1, word4, word5, word6};
    core.memory.validate_write_range(
        texture_address + gx2_texture_regs_offset,
        static_cast<std::uint32_t>(registers.size() * sizeof(std::uint32_t)));
    for (std::size_t index = 0U; index < registers.size(); ++index)
    {
        core.memory.write32_be(
            texture_address + gx2_texture_regs_offset +
                static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
            registers[index]);
    }
}

void set_gx2_texture(
    EspressoCore& core,
    std::uint32_t texture_address,
    std::uint32_t unit,
    const char* api_name,
    std::uint32_t register_base,
    std::array<std::optional<Gx2TextureBinding>, gx2_texture_unit_count>& bindings)
{
    if (!core.gx2.initialized)
    {
        throw HleExecutionError(
            std::string(api_name) + ": GX2 has not been initialized");
    }
    if (texture_address == 0U || (texture_address & 3U) != 0U ||
        static_cast<std::uint64_t>(texture_address) + gx2_texture_size >
            guest_address_space_end)
    {
        throw HleExecutionError(
            std::string(api_name) + ": texture pointer is null, misaligned, or wraps");
    }
    if (unit >= gx2_texture_unit_count)
    {
        throw HleExecutionError(
            std::string(api_name) + ": pixel texture unit is out of range");
    }

    // Snapshot all guest-owned descriptor fields before mutating runtime state.
    // GX2Texture::regs stores resource words 0, 1, 4, 5, and 6; words 2 and 3
    // are derived from the image and mipmap addresses when the texture is bound.
    const std::uint32_t image_address = core.memory.read32_be(texture_address + 0x24U);
    // GX2Surface stores mipmapSize at +0x28 and the mipmap pointer at +0x2C.
    // The size is not an address and must never be emitted as RESOURCE_WORD3.
    std::uint32_t mipmap_address = core.memory.read32_be(texture_address + 0x2CU);
    const std::uint32_t tile_mode = core.memory.read32_be(texture_address + 0x30U);
    const std::uint32_t swizzle = core.memory.read32_be(texture_address + 0x34U);
    const std::uint32_t word0 = core.memory.read32_be(texture_address + 0x88U);
    const std::uint32_t word1 = core.memory.read32_be(texture_address + 0x8CU);
    const std::uint32_t word4 = core.memory.read32_be(texture_address + 0x90U);
    const std::uint32_t word5 = core.memory.read32_be(texture_address + 0x94U);
    const std::uint32_t word6 = core.memory.read32_be(texture_address + 0x98U);

    if ((swizzle & 0xFFU) != 0U)
    {
        throw HleExecutionError(
            std::string(api_name) + ": unsupported low-byte swizzle bits");
    }
    if (mipmap_address == 0U)
        mipmap_address = image_address;

    // Latte resource words store 256-byte address units. Cafe guest addresses
    // in Affogato's RAM map are the GPU-visible address values, so retain the
    // address bits above the 256-byte alignment boundary as the hardware does.
    const std::uint32_t swizzle_stop_level = (swizzle >> 16U) & 0xFFU;
    const bool macro_tiled = tile_mode >= 4U && tile_mode <= 15U;
    const std::uint32_t resource_image_address =
        macro_tiled && swizzle_stop_level > 0U
            ? image_address ^ (swizzle & 0xFFFFU)
            : image_address;
    if (macro_tiled && swizzle_stop_level > 1U)
        mipmap_address ^= swizzle & 0xFFFFU;

    Gx2TextureBinding binding{};
    binding.texture_address = texture_address;
    binding.registers = {
        word0,
        word1,
        resource_image_address >> 8U,
        mipmap_address >> 8U,
        word4,
        word5,
        word6};

    const std::uint32_t first_register =
        register_base +
        unit * gx2_texture_register_count * sizeof(std::uint32_t);
    auto& commands = core.gx2.pending_commands;
    commands.reserve(commands.size() + binding.registers.size());
    for (std::size_t index = 0U; index < binding.registers.size(); ++index)
    {
        commands.push_back({
            Gx2CommandType::set_shader_resource,
            first_register + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
            binding.registers[index]});
    }
    bindings[unit] = binding;
}

void set_gx2_sampler(
    EspressoCore& core,
    std::uint32_t sampler_address,
    std::uint32_t unit,
    const char* api_name,
    std::uint32_t register_base,
    std::array<std::optional<Gx2SamplerBinding>, gx2_sampler_unit_count>& bindings)
{
    if (!core.gx2.initialized)
    {
        throw HleExecutionError(
            std::string(api_name) + ": GX2 has not been initialized");
    }
    if (sampler_address == 0U || (sampler_address & 3U) != 0U ||
        static_cast<std::uint64_t>(sampler_address) + gx2_sampler_size >
            guest_address_space_end)
    {
        throw HleExecutionError(
            std::string(api_name) + ": sampler pointer is null, misaligned, or wraps");
    }
    if (unit >= gx2_sampler_unit_count)
    {
        throw HleExecutionError(
            std::string(api_name) + ": pixel sampler unit is out of range");
    }

    Gx2SamplerBinding binding{};
    binding.sampler_address = sampler_address;
    for (std::size_t index = 0U; index < binding.registers.size(); ++index)
    {
        binding.registers[index] = core.memory.read32_be(
            sampler_address + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)));
    }

    const std::uint32_t first_register =
        register_base +
        unit * gx2_sampler_register_count * sizeof(std::uint32_t);
    auto& commands = core.gx2.pending_commands;
    commands.reserve(commands.size() + binding.registers.size());
    for (std::size_t index = 0U; index < binding.registers.size(); ++index)
    {
        commands.push_back({
            Gx2CommandType::set_shader_sampler,
            first_register + static_cast<std::uint32_t>(index * sizeof(std::uint32_t)),
            binding.registers[index]});
    }
    bindings[unit] = binding;
}

void set_gx2_sampler_border_color(
    EspressoCore& core,
    std::uint32_t unit,
    const char* api_name,
    std::uint32_t register_base,
    std::array<std::optional<std::array<std::uint32_t, 4U>>,
               gx2_sampler_border_color_unit_count>& border_colors)
{
    if (!core.gx2.initialized)
    {
        throw HleExecutionError(
            std::string(api_name) + ": GX2 has not been initialized");
    }
    if (unit >= gx2_sampler_border_color_unit_count)
    {
        throw HleExecutionError(
            std::string(api_name) + ": sampler border-color unit is out of range");
    }

    const std::array<std::uint32_t, 4U> color{
        fpr_float_argument_bits(core, 1U),
        fpr_float_argument_bits(core, 2U),
        fpr_float_argument_bits(core, 3U),
        fpr_float_argument_bits(core, 4U),
    };
    const std::uint32_t first_register = register_base + unit * 4U * 4U;
    auto& commands = core.gx2.pending_commands;
    commands.reserve(commands.size() + color.size());
    for (std::size_t channel = 0U; channel < color.size(); ++channel)
    {
        commands.push_back({
            Gx2CommandType::set_context_register,
            first_register + static_cast<std::uint32_t>(channel * 4U),
            color[channel]});
    }
    border_colors[unit] = color;
}

std::uint32_t pack_stencil_ref_mask(
    std::uint32_t mask,
    std::uint32_t write_mask,
    std::uint32_t reference)
{
    return (reference & 0xFFU) |
           ((mask & 0xFFU) << 8U) |
           ((write_mask & 0xFFU) << 16U);
}

std::uint32_t pack_polygon_control(
    std::uint32_t front_face,
    std::uint32_t cull_front,
    std::uint32_t cull_back,
    std::uint32_t polygon_mode,
    std::uint32_t polygon_mode_front,
    std::uint32_t polygon_mode_back,
    std::uint32_t polygon_offset_front_enable,
    std::uint32_t polygon_offset_back_enable,
    std::uint32_t polygon_offset_parallel_enable)
{
    const auto boolean_bit = [](std::uint32_t value) {
        return value != 0U ? 1U : 0U;
    };

    return boolean_bit(cull_front) |
           (boolean_bit(cull_back) << 1U) |
           (boolean_bit(front_face) << 2U) |
           ((polygon_mode & 0x3U) << 3U) |
           ((polygon_mode_front & 0x7U) << 5U) |
           ((polygon_mode_back & 0x7U) << 8U) |
           (boolean_bit(polygon_offset_front_enable) << 11U) |
           (boolean_bit(polygon_offset_back_enable) << 12U) |
           (boolean_bit(polygon_offset_parallel_enable) << 13U);
}

std::uint32_t pack_color_control(
    std::uint32_t rop3,
    std::uint32_t target_blend_enable,
    std::uint32_t multi_write_enable,
    std::uint32_t color_write_enable)
{
    const std::uint32_t multi_write = multi_write_enable != 0U ? 1U : 0U;
    // Cafe maps disabled color writes to SPECIAL_OP::DISABLE; enabled writes
    // select NORMAL. CB_COLOR_CONTROL has no separate color-write bit.
    const std::uint32_t special_op = color_write_enable != 0U ? 0U : 1U;

    return (multi_write << 1U) |
           (special_op << 4U) |
           ((target_blend_enable & 0xFFU) << 8U) |
           ((rop3 & 0xFFU) << 16U);
}

std::uint32_t pack_blend_control(
    std::uint32_t color_source_blend,
    std::uint32_t color_destination_blend,
    std::uint32_t color_combine,
    std::uint32_t use_alpha_blend,
    std::uint32_t alpha_source_blend,
    std::uint32_t alpha_destination_blend,
    std::uint32_t alpha_combine)
{
    const std::uint32_t separate_alpha = use_alpha_blend != 0U ? 1U : 0U;

    return (color_source_blend & 0x1FU) |
           ((color_combine & 0x7U) << 5U) |
           ((color_destination_blend & 0x1FU) << 8U) |
           ((alpha_source_blend & 0x1FU) << 16U) |
           ((alpha_combine & 0x7U) << 21U) |
           ((alpha_destination_blend & 0x1FU) << 24U) |
           (separate_alpha << 29U);
}

struct ParsedGx2Attributes
{
    std::uint32_t command_pool_base{};
    std::uint32_t command_pool_size{default_command_pool_size};
    bool command_pool_size_present{};
    std::uint32_t argc{};
    std::uint32_t argv{};
    std::uint32_t profile_mode{};
    std::uint32_t toss_stage{};
    std::uint32_t app_io_thread_stack_size{default_app_io_stack_size};
};

[[noreturn]] void gx2_init_error(const std::string& message)
{
    throw HleExecutionError("gx2::GX2Init: " + message);
}

ParsedGx2Attributes parse_attributes(GuestMemory& memory, std::uint32_t address)
{
    ParsedGx2Attributes parsed;
    if (address == 0U)
    {
        return parsed;
    }

    std::uint64_t cursor = address;
    std::size_t pairs = 0U;
    while (true)
    {
        if (cursor > guest_address_space_end - sizeof(std::uint32_t))
        {
            gx2_init_error("attribute stream address wraps the guest address space");
        }

        std::uint32_t id = 0U;
        try
        {
            id = memory.read32_be(static_cast<std::uint32_t>(cursor));
        }
        catch (const GuestMemoryFault&)
        {
            gx2_init_error("could not read an attribute ID from guest memory");
        }
        if (id == gx2_init_end)
        {
            return parsed;
        }
        if (pairs >= maximum_attribute_pairs)
        {
            gx2_init_error("attribute stream exceeded the 64-pair safety limit without GX2_INIT_END");
        }

        cursor += sizeof(std::uint32_t);
        if (cursor > guest_address_space_end - sizeof(std::uint32_t))
        {
            gx2_init_error("attribute value address wraps the guest address space");
        }

        std::uint32_t value = 0U;
        try
        {
            value = memory.read32_be(static_cast<std::uint32_t>(cursor));
        }
        catch (const GuestMemoryFault&)
        {
            gx2_init_error("attribute ID was readable but its value was not mapped");
        }

        switch (id)
        {
        case gx2_init_command_buffer_base:
            parsed.command_pool_base = value;
            break;
        case gx2_init_command_buffer_pool_size:
            parsed.command_pool_size = value;
            parsed.command_pool_size_present = true;
            break;
        case gx2_init_argc:
            parsed.argc = value;
            break;
        case gx2_init_argv:
            parsed.argv = value;
            break;
        case gx2_init_profile_mode:
            parsed.profile_mode = value;
            break;
        case gx2_init_toss_stage:
            parsed.toss_stage = value;
            break;
        case gx2_init_app_io_thread_stack_size:
            parsed.app_io_thread_stack_size = value;
            break;
        default:
            // Cafe/Decaf consume unknown ID/value pairs and continue.
            break;
        }

        ++pairs;
        cursor += sizeof(std::uint32_t);
    }
}

Gx2RuntimeState initialize_gx2(EspressoCore& core, std::uint32_t attribute_address)
{
    ParsedGx2Attributes parsed = parse_attributes(core.memory, attribute_address);
    parsed.command_pool_size = parsed.command_pool_size < minimum_command_pool_size
        ? minimum_command_pool_size
        : parsed.command_pool_size;

    const bool owned = parsed.command_pool_base == 0U;
    const std::uint32_t old_heap_cursor = core.guest_heap_cursor;
    std::uint32_t pool_base = parsed.command_pool_base;
    if (owned)
    {
        pool_base = core.allocate_guest_memory(
            parsed.command_pool_size, command_pool_alignment);
        if (pool_base == 0U)
        {
            core.guest_heap_cursor = old_heap_cursor;
            gx2_init_error("could not allocate the command-buffer pool from the default guest heap");
        }
    }

    const std::uint64_t pool_end =
        static_cast<std::uint64_t>(pool_base) + parsed.command_pool_size;
    if (pool_end > guest_address_space_end)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool range wraps the guest address space");
    }
    if ((pool_base & (command_buffer_alignment - 1U)) != 0U)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool base is not 0x40-byte aligned");
    }

    try
    {
        core.memory.validate_write_range(pool_base, parsed.command_pool_size);
    }
    catch (const GuestMemoryFault&)
    {
        if (owned)
        {
            core.guest_heap_cursor = old_heap_cursor;
        }
        gx2_init_error("command-buffer pool range is not fully mapped and writable");
    }

    // The command pool is registered, not initialized: command emission and
    // Latte/PM4 execution are intentionally deferred to later GX2 work.
    Gx2RuntimeState state;
    state.initialized = true;
    // GX2Init installs GX2's built-in GX2R adapter. The adapter itself is
    // emulator-internal, so only its mode is guest-visible in this state
    // model; explicit GX2RSetAllocator calls retain their guest callback PCs.
    state.gx2r_uses_default_allocator = true;
    // Affogato currently has one Espresso execution lane corresponding to CPU1.
    // This becomes dynamic when multiple Espresso cores are modeled.
    state.main_core_id = 1U;
    state.command_buffer_pool_base = pool_base;
    state.command_buffer_pool_size = parsed.command_pool_size;
    state.command_buffer_pool_owned = owned;
    state.argc = parsed.argc;
    state.argv = parsed.argv;
    state.profile_mode = parsed.profile_mode;
    state.toss_stage = parsed.toss_stage;
    // The AppIO stack is recorded but not carved from the pool until its
    // thread/event machinery exists.
    state.app_io_thread_stack_size = parsed.app_io_thread_stack_size;
    state.gpu_timeout_ms = 10'000U;
    state.hang_response = 1U;
    state.hang_reset_swap_timeout = 1000U;
    state.hang_reset_swaps_outstanding = 3U;
    state.swap_interval = 1U;
    return state;
}

} // namespace

void register_gx2_hle(HleDispatcher& dispatcher)
{
    dispatcher.register_function(
        "gx2",
        "GX2TempGetGPUVersion",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2_gpu_version;
        });

    dispatcher.register_function(
        "gx2",
        "GX2Invalidate",
        [](EspressoCore&) {
            // Affogato currently has one coherent guest-memory backing store
            // and no modeled Espresso data cache, Latte shader cache, or
            // deferred GPU reads. The invalidation is therefore already
            // satisfied by the memory model; keep this void API from
            // fabricating a return value or changing guest state.
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcFetchShaderSizeEx",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2CalcFetchShaderSizeEx";
            const std::uint32_t attribute_count = core.state.gpr[3];
            const std::uint32_t fetch_shader_type = core.state.gpr[4];
            const std::uint32_t tessellation_mode = core.state.gpr[5];

            // In the non-tessellated mode each input attribute produces one
            // 16-byte Latte VFETCH instruction. The supported start-up path
            // uses GX2_FETCH_SHADER_TESSELLATION_NONE / DISCRETE; other modes
            // need the additional tessellation fetch-program generation and
            // must not be silently approximated.
            if (fetch_shader_type != 0U || tessellation_mode != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) +
                    ": only non-tessellated fetch shaders are implemented");
            }

            if (attribute_count >
                std::numeric_limits<std::uint32_t>::max() /
                    gx2_fetch_shader_instruction_size)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": calculated size overflows uint32_t");
            }

            core.state.gpr[3] =
                attribute_count * gx2_fetch_shader_instruction_size;
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcGeometryShaderInputRingBufferSize",
        [](EspressoCore& core) {
            // Cemu's GX2 export computes (ringItemSize * 4) * 0x1000 in
            // uint32 arithmetic. Keep the guest ABI's modulo-2^32 behavior.
            core.state.gpr[3] = core.state.gpr[3] * 4U * 0x1000U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcGeometryShaderOutputRingBufferSize",
        [](EspressoCore& core) {
            // The GX2 output-ring calculator uses the same uint32 formula as
            // its input-ring counterpart.
            core.state.gpr[3] = core.state.gpr[3] * 4U * 0x1000U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitFetchShaderEx",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitFetchShaderEx";
            constexpr std::uint32_t maximum_buffer_index = 15U;
            const std::uint32_t fetch_shader_address = core.state.gpr[3];
            const std::uint32_t program_address = core.state.gpr[4];
            const std::uint32_t attribute_count = core.state.gpr[5];
            const std::uint32_t attributes_address = core.state.gpr[6];
            const std::uint32_t fetch_shader_type = core.state.gpr[7];
            const std::uint32_t tessellation_mode = core.state.gpr[8];

            if (fetch_shader_type != 0U || tessellation_mode != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) +
                    ": only non-tessellated fetch shaders are implemented");
            }
            if (attribute_count > gx2_max_fetch_attributes)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": attribute count exceeds safety limit");
            }
            if (fetch_shader_address == 0U || (fetch_shader_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": fetch-shader descriptor is null or misaligned");
            }
            if (static_cast<std::uint64_t>(fetch_shader_address) +
                    gx2_fetch_shader_struct_size >
                guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": descriptor range wraps guest address space");
            }

            const std::uint64_t program_size =
                static_cast<std::uint64_t>(attribute_count) *
                gx2_fetch_shader_instruction_size;
            const std::uint64_t attribute_bytes_size =
                static_cast<std::uint64_t>(attribute_count) * gx2_attrib_stream_size;
            if (program_size > std::numeric_limits<std::uint32_t>::max() ||
                (program_size != 0U &&
                 (program_address == 0U || (program_address & 3U) != 0U)) ||
                (attribute_count != 0U &&
                 (attributes_address == 0U || (attributes_address & 3U) != 0U)))
            {
                throw HleExecutionError(
                    std::string(api_name) + ": null, misaligned, or oversized argument");
            }
            if (static_cast<std::uint64_t>(program_address) + program_size >
                    guest_address_space_end ||
                static_cast<std::uint64_t>(attributes_address) +
                        attribute_bytes_size >
                    guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": buffer range wraps guest address space");
            }

            // Read all source descriptors and validate both destinations
            // before writing either output, so a bad guest pointer cannot
            // leave a half-initialized fetch shader.
            std::vector<std::uint8_t> attribute_bytes(
                static_cast<std::size_t>(attribute_bytes_size));
            if (!attribute_bytes.empty())
            {
                core.memory.read_bytes(attributes_address, attribute_bytes);
            }
            core.memory.validate_write_range(
                fetch_shader_address, gx2_fetch_shader_struct_size);
            if (program_size != 0U)
            {
                core.memory.validate_write_range(
                    program_address, static_cast<std::size_t>(program_size));
            }

            std::vector<std::uint32_t> program_words;
            program_words.reserve(static_cast<std::size_t>(attribute_count) * 4U);
            for (std::uint32_t index = 0U; index < attribute_count; ++index)
            {
                const std::uint32_t address = attributes_address +
                    index * gx2_attrib_stream_size;
                const Gx2AttribStream attribute = read_attrib_stream(core, address);
                if (attribute.type != 0U)
                {
                    throw HleExecutionError(
                        std::string(api_name) +
                        ": per-instance attributes are not implemented");
                }
                if (attribute.buffer > maximum_buffer_index)
                {
                    throw HleExecutionError(
                        std::string(api_name) + ": unsupported attribute buffer index " +
                        std::to_string(attribute.buffer));
                }
                const std::uint32_t fetch_bytes =
                    gx2_attrib_format_bytes(attribute.format);
                if (fetch_bytes == 0U)
                {
                    throw HleExecutionError(
                        std::string(api_name) + ": unsupported attribute format " +
                        std::to_string(attribute.format));
                }

                const std::uint32_t endian_swap =
                    attribute.endian_swap == 3U
                        ? gx2_attrib_endian_swap(attribute.format)
                        : attribute.endian_swap;
                if (endian_swap > 2U)
                {
                    throw HleExecutionError(
                        std::string(api_name) + ": invalid attribute endian mode");
                }

                // Latte VTX instruction word 0: semantic fetch, vertex-index
                // source GPR 0.X, attribute buffer slot, and byte fetch span.
                const std::uint32_t word0 =
                    1U |
                    (attribute.buffer << 8U) |
                    ((fetch_bytes - 1U) << 26U);

                const std::uint32_t selector_x = (attribute.mask >> 24U) & 0x7U;
                const std::uint32_t selector_y = (attribute.mask >> 16U) & 0x7U;
                const std::uint32_t selector_z = (attribute.mask >> 8U) & 0x7U;
                const std::uint32_t selector_w = attribute.mask & 0x7U;
                const std::uint32_t data_format = attribute.format & 0x3FU;
                const std::uint32_t number_format =
                    (attribute.format & 0x800U) != 0U ? 2U :
                    (attribute.format & 0x100U) != 0U ? 1U : 0U;
                // VTX_WORD1_GPR.DST_GPR is a 7-bit field. GX2 keeps location
                // in a 32-bit slot (including UINT32_MAX sentinels), so encode
                // exactly the bits the Latte instruction format retains.
                const std::uint32_t word1 =
                    ((attribute.location & 0x7FU)) |
                    (selector_x << 9U) |
                    (selector_y << 12U) |
                    (selector_z << 15U) |
                    (selector_w << 18U) |
                    (data_format << 22U) |
                    (number_format << 28U) |
                    ((attribute.format & 0x200U) != 0U ? 1U << 30U : 0U);

                const std::uint32_t word2 =
                    (attribute.offset & 0xFFFFU) |
                    (endian_swap << 16U) |
                    (1U << 19U); // MEGA_FETCH

                program_words.push_back(word0);
                program_words.push_back(word1);
                program_words.push_back(word2);
                program_words.push_back(0U); // VTX word 3 is reserved padding.
            }

            for (std::size_t index = 0U; index < program_words.size(); ++index)
            {
                core.memory.write32_be(
                    program_address + static_cast<std::uint32_t>(index * 4U),
                    program_words[index]);
            }

            const std::array<std::uint32_t, 8U> descriptor{
                fetch_shader_type,
                0U, // SQ_PGM_RESOURCES_FS: fetch-only program uses no ALU GPRs.
                static_cast<std::uint32_t>(program_size),
                program_address,
                attribute_count,
                0U, // numDivisors
                0U,
                0U};
            for (std::size_t index = 0U; index < descriptor.size(); ++index)
            {
                core.memory.write32_be(
                    fetch_shader_address + static_cast<std::uint32_t>(index * 4U),
                    descriptor[index]);
            }
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSampler",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSampler";
            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t clamp_mode = core.state.gpr[4];
            const std::uint32_t filter_mode = core.state.gpr[5];

            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (clamp_mode > 7U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid clamp mode " +
                    std::to_string(clamp_mode));
            }
            if (filter_mode > 2U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid XY filter mode " +
                    std::to_string(filter_mode));
            }

            core.memory.validate_write_range(sampler_address, gx2_sampler_size);

            // GX2Sampler stores three Latte SQ_TEX_SAMPLER words in guest
            // big-endian order. XY/Z/mip filtering is encoded in word 0;
            // word 1 carries the maximum default LOD and word 2 marks the
            // descriptor as a valid sampler.
            const std::uint32_t clamp_fields =
                clamp_mode | (clamp_mode << 3U) | (clamp_mode << 6U);
            const std::uint32_t filter_fields =
                (filter_mode << 9U) | (filter_mode << 12U) |
                (1U << 15U) | (1U << 17U) | (1U << 25U);
            core.memory.write32_be(sampler_address + 0x00U, clamp_fields | filter_fields);
            core.memory.write32_be(sampler_address + 0x04U, 0x000FFC00U);
            core.memory.write32_be(sampler_address + 0x08U, 0x80000000U);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerClamping",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerClamping";
            constexpr std::uint32_t clamp_field_mask = 0x7U;
            constexpr std::uint32_t clamp_x_shift = 0U;
            constexpr std::uint32_t clamp_y_shift = 3U;
            constexpr std::uint32_t clamp_z_shift = 6U;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t clamp_x = core.state.gpr[4];
            const std::uint32_t clamp_y = core.state.gpr[5];
            const std::uint32_t clamp_z = core.state.gpr[6];

            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (clamp_x > clamp_field_mask ||
                clamp_y > clamp_field_mask ||
                clamp_z > clamp_field_mask)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": clamp mode is outside the GX2 enum range");
            }

            // Validate the complete guest structure before reading or changing
            // it. The API updates only CLAMP_X/Y/Z in sampler word 0, so all
            // filter, LOD, and descriptor fields must be preserved.
            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t old_word0 =
                core.memory.read32_be(sampler_address);
            constexpr std::uint32_t clamp_fields_mask =
                (clamp_field_mask << clamp_x_shift) |
                (clamp_field_mask << clamp_y_shift) |
                (clamp_field_mask << clamp_z_shift);
            const std::uint32_t new_word0 =
                (old_word0 & ~clamp_fields_mask) |
                (clamp_x << clamp_x_shift) |
                (clamp_y << clamp_y_shift) |
                (clamp_z << clamp_z_shift);
            core.memory.write32_be(sampler_address, new_word0);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerXYFilter",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerXYFilter";
            constexpr std::uint32_t max_aniso_max = 4U;
            constexpr std::uint32_t xy_filter_max = 2U;
            constexpr std::uint32_t max_aniso_shift = 19U;
            constexpr std::uint32_t mag_filter_shift = 9U;
            constexpr std::uint32_t min_filter_shift = 12U;
            constexpr std::uint32_t field_mask = 0x7U;
            constexpr std::uint32_t xy_filter_mask = 0x7U;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t filter_mag = core.state.gpr[4];
            const std::uint32_t filter_min = core.state.gpr[5];
            const std::uint32_t max_aniso = core.state.gpr[6];

            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (filter_mag > xy_filter_max || filter_min > xy_filter_max)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid XY filter mode");
            }
            if (max_aniso > max_aniso_max)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid anisotropy ratio");
            }
            if (max_aniso != 0U && (filter_mag == 2U || filter_min == 2U))
            {
                throw HleExecutionError(
                    std::string(api_name) +
                    ": bicubic filtering cannot be combined with anisotropy");
            }

            // The descriptor is three BE register words. Validate the entire
            // object before changing either word, then preserve every field
            // unrelated to anisotropy and XY filtering.
            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t word0 =
                core.memory.read32_be(sampler_address + 0x00U);
            const std::uint32_t max_aniso_mask = field_mask << max_aniso_shift;
            const std::uint32_t xy_filter_fields_mask =
                (xy_filter_mask << mag_filter_shift) |
                (xy_filter_mask << min_filter_shift);
            const std::uint32_t encoded_mag_filter =
                max_aniso == 0U ? filter_mag : (filter_mag == 0U ? 2U : 3U);
            const std::uint32_t encoded_min_filter =
                max_aniso == 0U ? filter_min : (filter_min == 0U ? 2U : 3U);
            const std::uint32_t new_word0 =
                (word0 & ~(max_aniso_mask | xy_filter_fields_mask)) |
                (max_aniso << max_aniso_shift) |
                (encoded_mag_filter << mag_filter_shift) |
                (encoded_min_filter << min_filter_shift);
            core.memory.write32_be(sampler_address + 0x00U, new_word0);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerZMFilter",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerZMFilter";
            constexpr std::uint32_t z_filter_max = 2U;
            constexpr std::uint32_t mip_filter_max = 2U;
            constexpr std::uint32_t z_filter_shift = 15U;
            constexpr std::uint32_t mip_filter_shift = 17U;
            constexpr std::uint32_t filter_mask = 0x3U;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t filter_z = core.state.gpr[4];
            const std::uint32_t filter_mip = core.state.gpr[5];

            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (filter_z > z_filter_max)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid Z filter mode");
            }
            if (filter_mip > mip_filter_max)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid mip filter mode");
            }

            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t word0 = core.memory.read32_be(sampler_address);
            constexpr std::uint32_t fields_mask =
                (filter_mask << z_filter_shift) |
                (filter_mask << mip_filter_shift);
            const std::uint32_t new_word0 =
                (word0 & ~fields_mask) |
                (filter_z << z_filter_shift) |
                (filter_mip << mip_filter_shift);
            core.memory.write32_be(sampler_address, new_word0);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerLOD",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerLOD";
            constexpr std::uint32_t min_lod_mask = 0x3FFU;
            constexpr std::uint32_t max_lod_shift = 10U;
            constexpr std::uint32_t lod_bias_shift = 20U;
            constexpr std::uint32_t lod_bias_mask = 0xFFFU;
            constexpr double lod_fractional_scale = 64.0;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const double min_lod_argument = std::bit_cast<double>(core.state.fpr[1]);
            const double max_lod_argument = std::bit_cast<double>(core.state.fpr[2]);
            const double lod_bias_argument = std::bit_cast<double>(core.state.fpr[3]);

            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (std::isnan(min_lod_argument) || std::isnan(max_lod_argument) ||
                std::isnan(lod_bias_argument))
            {
                throw HleExecutionError(
                    std::string(api_name) + ": LOD arguments must not be NaN");
            }

            // Latte stores min/max LOD as unsigned u4.6 and bias as signed
            // s6.6. Saturate before conversion so infinities and out-of-range
            // arguments do not invoke undefined floating-to-integer casts.
            const double min_lod = std::clamp(min_lod_argument, 0.0, 16.0);
            const double max_lod = std::clamp(max_lod_argument, 0.0, 16.0);
            const double lod_bias = std::clamp(lod_bias_argument, -32.0, 32.0);
            const std::uint32_t min_lod_fixed = static_cast<std::uint32_t>(
                std::clamp(
                    std::floor(min_lod * lod_fractional_scale),
                    0.0,
                    static_cast<double>(min_lod_mask)));
            const std::uint32_t max_lod_fixed = static_cast<std::uint32_t>(
                std::clamp(
                    std::floor(max_lod * lod_fractional_scale),
                    0.0,
                    static_cast<double>(min_lod_mask)));
            const std::int32_t lod_bias_fixed = static_cast<std::int32_t>(
                std::clamp(
                    std::floor(lod_bias * lod_fractional_scale),
                    -2048.0,
                    2047.0));
            const std::uint32_t encoded_lod_bias =
                static_cast<std::uint32_t>(lod_bias_fixed) & lod_bias_mask;

            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t old_word1 =
                core.memory.read32_be(sampler_address + 0x04U);
            constexpr std::uint32_t lod_fields_mask =
                min_lod_mask | (min_lod_mask << max_lod_shift) |
                (lod_bias_mask << lod_bias_shift);
            const std::uint32_t new_word1 =
                (old_word1 & ~lod_fields_mask) |
                min_lod_fixed |
                (max_lod_fixed << max_lod_shift) |
                (encoded_lod_bias << lod_bias_shift);
            core.memory.write32_be(sampler_address + 0x04U, new_word1);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerDepthCompare",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerDepthCompare";
            constexpr std::uint32_t depth_compare_mask = 0x7U;
            constexpr std::uint32_t depth_compare_shift = 26U;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t depth_compare = core.state.gpr[4];
            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (depth_compare > depth_compare_mask)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid depth compare function");
            }

            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t word0 = core.memory.read32_be(sampler_address);
            constexpr std::uint32_t depth_compare_field_mask =
                depth_compare_mask << depth_compare_shift;
            const std::uint32_t new_word0 =
                (word0 & ~depth_compare_field_mask) |
                (depth_compare << depth_compare_shift);
            core.memory.write32_be(sampler_address, new_word0);
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitSamplerBorderType",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2InitSamplerBorderType";
            constexpr std::uint32_t border_type_mask = 0x3U;
            constexpr std::uint32_t border_type_shift = 22U;

            const std::uint32_t sampler_address = core.state.gpr[3];
            const std::uint32_t border_type = core.state.gpr[4];
            if (sampler_address == 0U || (sampler_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": sampler pointer is null or misaligned");
            }
            if (border_type > border_type_mask)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid border color type");
            }

            core.memory.validate_write_range(sampler_address, gx2_sampler_size);
            const std::uint32_t word0 = core.memory.read32_be(sampler_address);
            constexpr std::uint32_t border_type_field_mask =
                border_type_mask << border_type_shift;
            const std::uint32_t new_word0 =
                (word0 & ~border_type_field_mask) |
                (border_type << border_type_shift);
            core.memory.write32_be(sampler_address, new_word0);
        });

    dispatcher.register_function(
        "gx2",
        "GX2Init",
        [](EspressoCore& core) {
            // Cemu treats repeated GX2Init calls as successful no-ops. Check
            // this before touching r3 so even an invalid later pointer is safe.
            if (core.gx2.initialized)
            {
                return;
            }

            const std::uint32_t attributes = core.state.gpr[3];
            const bool host_vsync_pacing_enabled =
                core.gx2.host_vsync_pacing_enabled;
            Gx2RuntimeState initialized = initialize_gx2(core, attributes);
            // This host execution policy is configured by the launch path,
            // not by guest GX2 initialization; preserve it across state setup.
            initialized.host_vsync_pacing_enabled =
                host_vsync_pacing_enabled;
            core.gx2 = initialized;
            // GX2Init is void: preserve all guest GPRs including r3.
        });

    dispatcher.register_function(
        "gx2",
        "GX2RSetAllocator",
        [](EspressoCore& core) {
            core.gx2.gx2r_allocate_function = core.state.gpr[3];
            core.gx2.gx2r_free_function = core.state.gpr[4];
            core.gx2.gx2r_uses_default_allocator = false;
        });

    dispatcher.register_function(
        "gx2",
        "GX2RCreateBufferUserMemory",
        [](EspressoCore& core) {
            create_gx2_buffer_user_memory(
                core, core.state.gpr[3], core.state.gpr[4]);
            core.state.gpr[3] = 1U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2RBufferExists",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_buffer_exists(core, core.state.gpr[3])
                ? 1U
                : 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2RGetBufferAllocationSize",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_get_buffer_allocation_size(
                core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2RGetBufferAlignment",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_get_buffer_alignment(core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2RLockBufferEx",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_lock_buffer(core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2RUnlockBufferEx",
        [](EspressoCore& core) {
            notify_gx2r_buffer_coherent(
                core, core.state.gpr[3], "gx2::GX2RUnlockBufferEx");
        });

    dispatcher.register_function(
        "gx2",
        "GX2RInvalidateBuffer",
        [](EspressoCore& core) {
            notify_gx2r_buffer_coherent(
                core, core.state.gpr[3], "gx2::GX2RInvalidateBuffer");
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDefaultAllocator",
        [](EspressoCore& core) {
            core.gx2.default_allocate_function = core.state.gpr[3];
            core.gx2.default_free_function = core.state.gpr[4];
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDepthOnlyControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthOnlyControl: GX2 has not been initialized");
            }

            const auto boolean_bit = [](std::uint32_t value) {
                return value != 0U ? 1U : 0U;
            };
            const std::uint32_t packed =
                (boolean_bit(core.state.gpr[3]) << 1U) |
                (boolean_bit(core.state.gpr[4]) << 2U) |
                ((core.state.gpr[5] & 0x7U) << 4U);

            core.gx2.db_depth_control = packed;
            core.gx2.db_depth_control_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_depth_control_register,
                packed});
            // This convenience API disables stencil fields and, like the
            // full depth/stencil setter, changes only Latte state.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDepthStencilControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthStencilControl: GX2 has not been initialized");
            }

            const std::uint32_t depth_test = core.state.gpr[3];
            const std::uint32_t depth_write = core.state.gpr[4];
            const std::uint32_t depth_compare = core.state.gpr[5];
            const std::uint32_t stencil_test = core.state.gpr[6];
            const std::uint32_t backface_stencil = core.state.gpr[7];
            const std::uint32_t front_stencil_func = core.state.gpr[8];
            const std::uint32_t front_stencil_zpass = core.state.gpr[9];
            const std::uint32_t front_stencil_zfail = core.state.gpr[10];

            constexpr std::uint32_t stack_argument_base_offset = 0x08U;
            constexpr std::uint32_t stack_argument_last_offset = 0x18U;
            const std::uint64_t stack_pointer = core.state.gpr[1];
            const std::uint64_t last_stack_argument =
                stack_pointer + stack_argument_last_offset;
            if (last_stack_argument + sizeof(std::uint32_t) > guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthStencilControl: stack arguments wrap guest address space");
            }

            // Read every stack-passed argument before changing GX2 state. A
            // failed guest read therefore leaves the register mirror/queue intact.
            const std::uint32_t front_stencil_fail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + stack_argument_base_offset));
            const std::uint32_t back_stencil_func = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x0CU));
            const std::uint32_t back_stencil_zpass = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x10U));
            const std::uint32_t back_stencil_zfail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + 0x14U));
            const std::uint32_t back_stencil_fail = core.memory.read32_be(
                static_cast<std::uint32_t>(stack_pointer + stack_argument_last_offset));

            const auto boolean_bit = [](std::uint32_t value) {
                return value != 0U ? 1U : 0U;
            };
            const std::uint32_t packed =
                boolean_bit(stencil_test) |
                (boolean_bit(depth_test) << 1U) |
                (boolean_bit(depth_write) << 2U) |
                ((depth_compare & 0x7U) << 4U) |
                (boolean_bit(backface_stencil) << 7U) |
                ((front_stencil_func & 0x7U) << 8U) |
                ((front_stencil_fail & 0x7U) << 11U) |
                ((front_stencil_zpass & 0x7U) << 14U) |
                ((front_stencil_zfail & 0x7U) << 17U) |
                ((back_stencil_func & 0x7U) << 20U) |
                ((back_stencil_fail & 0x7U) << 23U) |
                ((back_stencil_zpass & 0x7U) << 26U) |
                ((back_stencil_zfail & 0x7U) << 29U);

            core.gx2.db_depth_control = packed;
            core.gx2.db_depth_control_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_depth_control_register,
                packed});
            // GX2SetDepthStencilControl is void. Guest arguments and all CPU
            // state are preserved; the dispatcher returns through LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetStencilMask",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetStencilMask: GX2 has not been initialized");
            }

            const std::uint32_t front_mask = core.state.gpr[3] & 0xFFU;
            const std::uint32_t front_write_mask = core.state.gpr[4] & 0xFFU;
            const std::uint32_t front_reference = core.state.gpr[5] & 0xFFU;
            const std::uint32_t back_mask = core.state.gpr[6] & 0xFFU;
            const std::uint32_t back_write_mask = core.state.gpr[7] & 0xFFU;
            const std::uint32_t back_reference = core.state.gpr[8] & 0xFFU;

            // Compute the complete pair before mutating the current register
            // mirrors or appending either command.
            const std::uint32_t front_value = pack_stencil_ref_mask(
                front_mask, front_write_mask, front_reference);
            const std::uint32_t back_value = pack_stencil_ref_mask(
                back_mask, back_write_mask, back_reference);

            core.gx2.db_stencilrefmask = front_value;
            core.gx2.db_stencilrefmask_valid = true;
            core.gx2.db_stencilrefmask_bf = back_value;
            core.gx2.db_stencilrefmask_bf_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_stencilrefmask_register,
                front_value});
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_stencilrefmask_bf_register,
                back_value});
            // This API is void: preserve all guest CPU state and return via LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPolygonControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPolygonControl: GX2 has not been initialized");
            }

            const std::uint32_t front_face = core.state.gpr[3];
            const std::uint32_t cull_front = core.state.gpr[4];
            const std::uint32_t cull_back = core.state.gpr[5];
            const std::uint32_t polygon_mode = core.state.gpr[6];
            const std::uint32_t polygon_mode_front = core.state.gpr[7];
            const std::uint32_t polygon_mode_back = core.state.gpr[8];
            const std::uint32_t polygon_offset_front_enable = core.state.gpr[9];
            const std::uint32_t polygon_offset_back_enable = core.state.gpr[10];

            constexpr std::uint32_t stack_argument_offset = 0x08U;
            const std::uint64_t stack_pointer = core.state.gpr[1];
            const std::uint64_t stack_argument_address =
                stack_pointer + stack_argument_offset;
            if (stack_argument_address + sizeof(std::uint32_t) > guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPolygonControl: stack argument wraps guest address space");
            }

            // Gather the ninth ABI argument before touching the register mirror
            // or command queue. GPR11 is not an argument register here.
            const std::uint32_t polygon_offset_parallel_enable =
                core.memory.read32_be(static_cast<std::uint32_t>(stack_argument_address));
            const std::uint32_t packed = pack_polygon_control(
                front_face,
                cull_front,
                cull_back,
                polygon_mode,
                polygon_mode_front,
                polygon_mode_back,
                polygon_offset_front_enable,
                polygon_offset_back_enable,
                polygon_offset_parallel_enable);

            core.gx2.pa_su_sc_mode_cntl = packed;
            core.gx2.pa_su_sc_mode_cntl_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_sc_mode_cntl_register,
                packed});
            // This API is void: preserve all guest CPU state and return via LR.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetColorControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetColorControl: GX2 has not been initialized");
            }

            const std::uint32_t rop3 = core.state.gpr[3];
            const std::uint32_t target_blend_enable = core.state.gpr[4];
            const std::uint32_t multi_write_enable = core.state.gpr[5];
            const std::uint32_t color_write_enable = core.state.gpr[6];
            const std::uint32_t packed = pack_color_control(
                rop3,
                target_blend_enable,
                multi_write_enable,
                color_write_enable);

            core.gx2.cb_color_control = packed;
            core.gx2.cb_color_control_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                cb_color_control_register,
                packed});
            // This void setter records logical Latte state only. PM4 encoding,
            // command-pool writes, and renderer translation remain deferred.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetBlendControl",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetBlendControl: GX2 has not been initialized");
            }

            const std::uint32_t target = core.state.gpr[3];
            if (target > 7U)
            {
                throw HleExecutionError(
                    "gx2::GX2SetBlendControl: invalid render target");
            }

            const std::uint32_t packed = pack_blend_control(
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6],
                core.state.gpr[7],
                core.state.gpr[8],
                core.state.gpr[9],
                core.state.gpr[10]);
            const std::size_t target_index = static_cast<std::size_t>(target);
            const std::uint32_t register_address =
                cb_blend0_control_register + target * sizeof(std::uint32_t);

            core.gx2.cb_blend_control[target_index] = packed;
            core.gx2.cb_blend_control_valid[target_index] = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                register_address,
                packed});
            // This void setter records logical per-target Latte state only.
            // Raw PM4 and renderer blending are intentionally deferred.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetBlendConstantColor",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetBlendConstantColor: GX2 has not been initialized");
            }

            // The Cafe floating-point ABI passes these float arguments in
            // f1-f4. Affogato stores PS0 as binary64, so narrow the numeric
            // value to binary32 and retain its IEEE-754 payload for Latte.
            const std::array<std::uint32_t, 4U> values{
                fpr_float_argument_bits(core, 1U),
                fpr_float_argument_bits(core, 2U),
                fpr_float_argument_bits(core, 3U),
                fpr_float_argument_bits(core, 4U),
            };
            constexpr std::array<std::uint32_t, 4U> registers{
                cb_blend_red_register,
                cb_blend_green_register,
                cb_blend_blue_register,
                cb_blend_alpha_register,
            };

            core.gx2.cb_blend_constant = values;
            core.gx2.cb_blend_constant_valid.fill(true);
            for (std::size_t index = 0; index < values.size(); ++index)
            {
                core.gx2.pending_commands.push_back({
                    Gx2CommandType::set_context_register,
                    registers[index],
                    values[index]});
            }
            // This void setter records logical Latte state only. The current
            // coherent GuestMemory model has no cache or renderer side effect.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetAlphaTest",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetAlphaTest: GX2 has not been initialized");
            }

            const std::uint32_t control = pack_alpha_test_control(
                core.state.gpr[3], core.state.gpr[4]);
            const std::uint32_t reference = fpr_float_argument_bits(core, 1U);

            core.gx2.sx_alpha_test_control = control;
            core.gx2.sx_alpha_test_control_valid = true;
            core.gx2.sx_alpha_ref = reference;
            core.gx2.sx_alpha_ref_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                sx_alpha_test_control_register,
                control});
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                sx_alpha_ref_register,
                reference});
            // Void setter: alphaTest/func come from r3/r4 and ref from f1.
            // It records Latte state without changing guest CPU or memory.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetTargetChannelMasks",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetTargetChannelMasks: GX2 has not been initialized");
            }

            const std::array<std::uint32_t, 8U> masks{
                core.state.gpr[3],
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6],
                core.state.gpr[7],
                core.state.gpr[8],
                core.state.gpr[9],
                core.state.gpr[10],
            };
            const std::uint32_t packed = pack_target_channel_masks(masks);

            core.gx2.cb_target_mask = packed;
            core.gx2.cb_target_mask_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                cb_target_mask_register,
                packed});
            // Void setter: r3-r10 are the eight mask arguments. No stack,
            // FPR, or guest-memory state participates in this register write.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetAAMask",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetAAMask: GX2 has not been initialized");
            }

            // GX2 accepts one 8-bit mask per pixel in a 2x2 quad. Latte's
            // PA_SC_AA_MASK register retains those four bytes in UL, UR, LL,
            // LR order; do not expand them into the later 16-sample register
            // pair used by newer AMD GPU generations.
            const std::uint32_t packed = pack_aa_mask(
                core.state.gpr[3],
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6]);

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_sc_aa_mask_register,
                packed});
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitAAMaskReg",
        [](EspressoCore& core) {
            const std::uint32_t packed = pack_aa_mask(
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6],
                core.state.gpr[7]);
            core.memory.write32_be(core.state.gpr[3], packed);
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetAAMaskReg",
        [](EspressoCore& core) {
            const std::uint32_t packed = core.memory.read32_be(core.state.gpr[3]);
            const std::array<std::uint32_t, 4U> output_addresses{
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6],
                core.state.gpr[7],
            };
            for (const std::uint32_t address : output_addresses)
            {
                core.memory.validate_write_range(address, sizeof(std::uint8_t));
            }
            for (std::size_t index = 0; index < output_addresses.size(); ++index)
            {
                const auto mask = static_cast<std::uint8_t>(
                    packed >> static_cast<unsigned>(index * 8U));
                core.memory.write8(output_addresses[index], mask);
            }
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetAAMaskReg",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetAAMaskReg: GX2 has not been initialized");
            }

            // GX2AAMaskReg is exactly one big-endian pa_sc_aa_mask word. Read
            // it before queuing the write so a bad guest pointer is atomic.
            const std::uint32_t mask = core.memory.read32_be(core.state.gpr[3]);
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_sc_aa_mask_register,
                mask});
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetScissor",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetScissor: GX2 has not been initialized");
            }

            const std::uint32_t x = core.state.gpr[3];
            const std::uint32_t y = core.state.gpr[4];
            const std::uint32_t width = core.state.gpr[5];
            const std::uint32_t height = core.state.gpr[6];

            // The Latte generic scissor uses an inclusive top-left and an
            // exclusive bottom-right. Setting WINDOW_OFFSET_DISABLE keeps
            // the API's x/y coordinates literal, independent of other state.
            constexpr std::uint32_t coordinate_mask = 0x7FFFU;
            const std::uint32_t top_left =
                0x80000000U |
                (x & coordinate_mask) |
                ((y & coordinate_mask) << 16U);
            const std::uint32_t bottom_right =
                ((x + width) & coordinate_mask) |
                (((y + height) & coordinate_mask) << 16U);

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_sc_generic_scissor_tl_register,
                top_left});
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_sc_generic_scissor_br_register,
                bottom_right});
            // Void GX2 setter: only the ordered Latte register stream changes.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetWindowOffset",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetWindowOffset: GX2 has not been initialized");
            }

            // PA_SC_WINDOW_OFFSET stores signed 16-bit X and Y offsets in the
            // low and high halfwords, respectively. Preserve their two's-
            // complement representation and let the register define wrapping.
            const std::uint32_t packed =
                (core.state.gpr[3] & 0xFFFFU) |
                ((core.state.gpr[4] & 0xFFFFU) << 16U);
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_sc_window_offset_register,
                packed});
            // This setter records the offset only; activation remains governed
            // by the independent PA_SU_SC_MODE_CNTL window-offset enable bit.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPolygonOffset",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPolygonOffset: GX2 has not been initialized");
            }

            // The ABI order is offset, scale for front and back; the register
            // structure is scale, offset. All five float arguments arrive in
            // FPR1-FPR5 and are stored as binary32 register words.
            const std::uint32_t front_offset =
                fpr_float_argument_bits(core, 1U);
            const std::uint32_t front_scale =
                fpr_float_argument_bits(core, 2U);
            const std::uint32_t back_offset =
                fpr_float_argument_bits(core, 3U);
            const std::uint32_t back_scale =
                fpr_float_argument_bits(core, 4U);
            const std::uint32_t clamp =
                fpr_float_argument_bits(core, 5U);
            constexpr std::array<std::uint32_t, 5U> registers{
                pa_su_poly_offset_front_scale_register,
                pa_su_poly_offset_front_offset_register,
                pa_su_poly_offset_back_scale_register,
                pa_su_poly_offset_back_offset_register,
                pa_su_poly_offset_clamp_register,
            };
            const std::array<std::uint32_t, 5U> values{
                front_scale, front_offset, back_scale, back_offset, clamp};

            for (std::size_t index = 0U; index < registers.size(); ++index)
            {
                core.gx2.pending_commands.push_back({
                    Gx2CommandType::set_context_register,
                    registers[index],
                    values[index]});
            }
            // As a void setter, this records only the ordered Latte state.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPointSize",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPointSize: GX2 has not been initialized");
            }

            const float width = std::bit_cast<float>(
                fpr_float_argument_bits(core, 1U));
            const float height = std::bit_cast<float>(
                fpr_float_argument_bits(core, 2U));
            const std::uint32_t packed = pack_gx2_point_size(width, height);

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_point_size_register,
                packed});
            // Width and height are full point dimensions; Latte's register
            // carries the corresponding half-width/half-height radii.
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitPointSizeReg",
        [](EspressoCore& core) {
            const float width = std::bit_cast<float>(
                fpr_float_argument_bits(core, 1U));
            const float height = std::bit_cast<float>(
                fpr_float_argument_bits(core, 2U));
            core.memory.write32_be(
                core.state.gpr[3], pack_gx2_point_size(width, height));
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPointSizeReg",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPointSizeReg: GX2 has not been initialized");
            }

            // GX2PointSizeReg contains one already-packed pa_su_point_size word.
            const std::uint32_t point_size =
                core.memory.read32_be(core.state.gpr[3]);
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_point_size_register,
                point_size});
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPointLimits",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetPointLimits: GX2 has not been initialized");
            }

            const float minimum = std::bit_cast<float>(
                fpr_float_argument_bits(core, 1U));
            const float maximum = std::bit_cast<float>(
                fpr_float_argument_bits(core, 2U));
            const std::uint32_t packed =
                static_cast<std::uint32_t>(pack_gx2_scaled_i16_field(
                    maximum, 16.0F)) << 16U |
                static_cast<std::uint32_t>(pack_gx2_scaled_i16_field(
                    minimum, 16.0F));

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_point_minmax_register,
                packed});
            // The min/max parameters are the radius limits stored by Latte.
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetLineWidthReg",
        [](EspressoCore& core) {
            const std::uint32_t line_control =
                core.memory.read32_be(core.state.gpr[3]);
            const std::uint32_t output_address = core.state.gpr[4];
            core.memory.validate_write_range(
                output_address, sizeof(std::uint32_t));

            const float width =
                static_cast<float>(line_control & 0xFFFFU) * 0.125F;
            core.memory.write32_be(
                output_address, std::bit_cast<std::uint32_t>(width));
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitLineWidthReg",
        [](EspressoCore& core) {
            const float width = std::bit_cast<float>(
                fpr_float_argument_bits(core, 1U));
            const std::uint32_t packed =
                pack_gx2_scaled_i16_field(width, 8.0F);
            core.memory.write32_be(core.state.gpr[3], packed);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetLineWidth",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetLineWidth: GX2 has not been initialized");
            }

            const float width = std::bit_cast<float>(
                fpr_float_argument_bits(core, 1U));
            // PA_SU_LINE_CNTL stores half of the full width in 1/16-pixel
            // subpixels, which is equivalent to multiplying by eight.
            const std::uint32_t packed =
                pack_gx2_scaled_i16_field(width, 8.0F);
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_line_cntl_register,
                packed});
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetLineWidthReg",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetLineWidthReg: GX2 has not been initialized");
            }

            // GX2LineWidthReg contains one already-packed pa_su_line_cntl word.
            const std::uint32_t line_control =
                core.memory.read32_be(core.state.gpr[3]);
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                pa_su_line_cntl_register,
                line_control});
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetViewport",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetViewport: GX2 has not been initialized");
            }

            // The Cafe ABI passes these six float arguments in f1-f6. Convert
            // to binary32 before doing the viewport arithmetic, as the GX2
            // register structure stores IEEE-754 single-precision values.
            const float x = std::bit_cast<float>(fpr_float_argument_bits(core, 1U));
            const float y = std::bit_cast<float>(fpr_float_argument_bits(core, 2U));
            const float width = std::bit_cast<float>(fpr_float_argument_bits(core, 3U));
            const float height = std::bit_cast<float>(fpr_float_argument_bits(core, 4U));
            const float near_z = std::bit_cast<float>(fpr_float_argument_bits(core, 5U));
            const float far_z = std::bit_cast<float>(fpr_float_argument_bits(core, 6U));

            const float half_width = width * 0.5F;
            const float half_height = height * 0.5F;
            const float half_depth = (far_z - near_z) * 0.5F;
            const std::array<std::uint32_t, 12U> values{
                std::bit_cast<std::uint32_t>(half_width),
                std::bit_cast<std::uint32_t>(x + half_width),
                std::bit_cast<std::uint32_t>(-half_height),
                std::bit_cast<std::uint32_t>(y + half_height),
                std::bit_cast<std::uint32_t>(half_depth),
                std::bit_cast<std::uint32_t>((near_z + far_z) * 0.5F),
                0x3F800000U, // no vertical clip guard-band adjustment
                0x3F800000U, // no vertical discard guard-band adjustment
                0x3F800000U, // no horizontal clip guard-band adjustment
                0x3F800000U, // no horizontal discard guard-band adjustment
                0x00000000U, // depth clamp minimum
                0x3F800000U, // depth clamp maximum
            };
            constexpr std::array<std::uint32_t, 12U> registers{
                pa_cl_vport_xscale_register,
                pa_cl_vport_xscale_register + 0x04U,
                pa_cl_vport_xscale_register + 0x08U,
                pa_cl_vport_xscale_register + 0x0CU,
                pa_cl_vport_xscale_register + 0x10U,
                pa_cl_vport_xscale_register + 0x14U,
                pa_cl_gb_vert_clip_adj_register,
                pa_cl_gb_vert_disc_adj_register,
                pa_cl_gb_horz_clip_adj_register,
                pa_cl_gb_horz_disc_adj_register,
                pa_sc_vport_zmin_register,
                pa_sc_vport_zmax_register,
            };

            for (std::size_t index = 0U; index < registers.size(); ++index)
            {
                core.gx2.pending_commands.push_back({
                    Gx2CommandType::set_context_register,
                    registers[index],
                    values[index]});
            }
            // Void GX2 setter: only the ordered Latte register stream changes.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetAlphaToMask",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetAlphaToMask: GX2 has not been initialized");
            }

            const std::uint32_t value = pack_alpha_to_mask(
                core.state.gpr[3], core.state.gpr[4]);
            core.gx2.db_alpha_to_mask = value;
            core.gx2.db_alpha_to_mask_valid = true;
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_context_register,
                db_alpha_to_mask_register,
                value});
            // The mode selects offset fields even while alpha-to-mask is
            // disabled. The void setter does not access guest memory or FPRs.
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetupContextStateEx",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetupContextStateEx: GX2 has not been initialized");
            }

            const std::uint32_t context_state = core.state.gpr[3];
            const std::uint32_t flags = core.state.gpr[4];
            if (context_state == 0U)
            {
                throw HleExecutionError(
                    "gx2::GX2SetupContextStateEx: context state pointer is null");
            }

            const std::uint64_t end =
                static_cast<std::uint64_t>(context_state) +
                gx2_context_state_size;
            if (end > guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetupContextStateEx: context state range wraps "
                    "the guest address space");
            }

            // Validate before touching either guest bytes or runtime metadata.
            // zero_fill validates again and performs an all-or-nothing range
            // write under GuestMemory's mapped-region contract.
            core.memory.validate_write_range(
                context_state, gx2_context_state_size);

            const bool profiling_enabled = (flags & 1U) != 0U;
            const bool shadow_display_list_requested = (flags & 2U) == 0U;
            core.memory.zero_fill(context_state, gx2_context_state_size);
            core.memory.write32_be(
                context_state + gx2_context_profiling_offset,
                profiling_enabled ? 1U : 0U);
            // Real Cafe creates a restore display list unless suppressed.
            // Affogato has no authentic PM4/display-list generator yet, so a
            // zero size intentionally selects the future direct-shadow restore
            // path instead of exposing fabricated command bytes.
            core.memory.write32_be(
                context_state + gx2_context_display_list_size_offset, 0U);
            // Context setup is modeled logically for now. The full Cafe
            // default-register stream requires GPU shadow execution; preserve
            // existing register mirrors and pending ordered commands.
            core.gx2.current_context_state = context_state;
            core.gx2.current_context_state_flags = flags;
            core.gx2.context_state_shadowing_enabled = true;
            core.gx2.context_state_profiling_enabled = profiling_enabled;
            core.gx2.context_state_shadow_display_list_requested =
                shadow_display_list_requested;
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetContextStateDisplayList",
        [](EspressoCore& core) {
            constexpr const char* api_name = "GX2GetContextStateDisplayList";
            const std::uint32_t context_state = core.state.gpr[3];
            const std::uint32_t out_display_list = core.state.gpr[4];
            const std::uint32_t out_size = core.state.gpr[5];

            if (context_state == 0U)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": context state pointer is null");
            }
            const std::uint64_t context_end =
                static_cast<std::uint64_t>(context_state) +
                gx2_context_state_size;
            if (context_end > guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": context state range wraps guest address space");
            }

            // Cemu returns the embedded Cafe load-display-list buffer and its
            // guest-stored size. Setup currently leaves this size zero because
            // Affogato does not yet generate authentic GX2/PM4 restore lists.
            const std::uint32_t display_list =
                context_state + gx2_context_display_list_offset;
            const std::uint32_t display_list_size = core.memory.read32_be(
                context_state + gx2_context_display_list_size_offset);

            validate_gx2_u32_output(
                core, out_display_list, api_name, "display-list output");
            validate_gx2_u32_output(
                core, out_size, api_name, "size output");
            core.memory.write32_be(out_display_list, display_list);
            core.memory.write32_be(out_size, display_list_size);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetContextState",
        [](EspressoCore& core) {
            constexpr const char* api_name = "GX2SetContextState";
            const std::uint32_t context_state = core.state.gpr[3];
            if (context_state == 0U)
            {
                // A null state disables shadow tracking. Affogato does not
                // model the PM4 context-control packet itself; the register
                // shadow and ordered register commands remain intact.
                core.gx2.context_state_shadowing_enabled = false;
                return;
            }

            const std::uint64_t context_end =
                static_cast<std::uint64_t>(context_state) +
                gx2_context_state_size;
            if (context_end > guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": context state range wraps guest address space");
            }
            std::vector<std::uint8_t> context_bytes(gx2_context_state_size);
            try
            {
                core.memory.read_bytes(context_state, context_bytes);
            }
            catch (const GuestMemoryFault&)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": context state is outside mapped guest memory");
            }

            const std::uint32_t size_offset =
                gx2_context_display_list_size_offset;
            const std::uint32_t display_list_size =
                (static_cast<std::uint32_t>(context_bytes[size_offset]) << 24U) |
                (static_cast<std::uint32_t>(context_bytes[size_offset + 1U]) << 16U) |
                (static_cast<std::uint32_t>(context_bytes[size_offset + 2U]) << 8U) |
                static_cast<std::uint32_t>(context_bytes[size_offset + 3U]);
            if (display_list_size > gx2_context_display_list_capacity)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": embedded display-list size exceeds its context buffer");
            }
            if (display_list_size != 0U)
            {
                throw HleExecutionError(
                    std::string("gx2::") + api_name +
                    ": PM4 context display-list playback is not implemented");
            }

            const std::vector<Gx2Command> restore_commands =
                read_gx2_context_restore_commands(context_bytes);

            core.gx2.pending_commands.insert(
                core.gx2.pending_commands.end(),
                restore_commands.begin(), restore_commands.end());
            core.gx2.current_context_state = context_state;
            core.gx2.context_state_shadowing_enabled = true;
        });

    dispatcher.register_function(
        "gx2",
        "GX2InitTextureRegs",
        [](EspressoCore& core) {
            initialize_gx2_texture_registers(core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPixelTexture",
        [](EspressoCore& core) {
            set_gx2_texture(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetPixelTexture",
                gx2_pixel_texture_register_base,
                core.gx2.pixel_textures);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetVertexTexture",
        [](EspressoCore& core) {
            set_gx2_texture(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetVertexTexture",
                gx2_vertex_texture_register_base,
                core.gx2.vertex_textures);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetComputeTexture",
        [](EspressoCore& core) {
            // Cafe's compute texture path shares the vertex-stage Latte bank.
            set_gx2_texture(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetComputeTexture",
                gx2_vertex_texture_register_base,
                core.gx2.vertex_textures);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetGeometryTexture",
        [](EspressoCore& core) {
            set_gx2_texture(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetGeometryTexture",
                gx2_geometry_texture_register_base,
                core.gx2.geometry_textures);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPixelSampler",
        [](EspressoCore& core) {
            set_gx2_sampler(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetPixelSampler",
                gx2_pixel_sampler_register_base,
                core.gx2.pixel_samplers);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetVertexSampler",
        [](EspressoCore& core) {
            set_gx2_sampler(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetVertexSampler",
                gx2_vertex_sampler_register_base,
                core.gx2.vertex_samplers);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetComputeSampler",
        [](EspressoCore& core) {
            // Cafe's compute sampler path shares the vertex-stage sampler bank.
            set_gx2_sampler(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetComputeSampler",
                gx2_vertex_sampler_register_base,
                core.gx2.vertex_samplers);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetGeometrySampler",
        [](EspressoCore& core) {
            set_gx2_sampler(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                "gx2::GX2SetGeometrySampler",
                gx2_geometry_sampler_register_base,
                core.gx2.geometry_samplers);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPixelSamplerBorderColor",
        [](EspressoCore& core) {
            set_gx2_sampler_border_color(
                core,
                core.state.gpr[3],
                "gx2::GX2SetPixelSamplerBorderColor",
                gx2_pixel_sampler_border_register_base,
                core.gx2.pixel_sampler_border_colors);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetVertexSamplerBorderColor",
        [](EspressoCore& core) {
            set_gx2_sampler_border_color(
                core,
                core.state.gpr[3],
                "gx2::GX2SetVertexSamplerBorderColor",
                gx2_vertex_sampler_border_register_base,
                core.gx2.vertex_sampler_border_colors);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetComputeSamplerBorderColor",
        [](EspressoCore& core) {
            // Compute samplers use the vertex-stage Latte border-color bank.
            set_gx2_sampler_border_color(
                core,
                core.state.gpr[3],
                "gx2::GX2SetComputeSamplerBorderColor",
                gx2_vertex_sampler_border_register_base,
                core.gx2.vertex_sampler_border_colors);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetGeometrySamplerBorderColor",
        [](EspressoCore& core) {
            set_gx2_sampler_border_color(
                core,
                core.state.gpr[3],
                "gx2::GX2SetGeometrySamplerBorderColor",
                gx2_geometry_sampler_border_register_base,
                core.gx2.geometry_sampler_border_colors);
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcSurfaceSizeAndAlignment",
        [](EspressoCore& core) {
            calculate_gx2_surface_size_and_alignment(core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2RCreateSurfaceUserMemory",
        [](EspressoCore& core) {
            create_gx2_surface_user_memory(
                core,
                core.state.gpr[3],
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6]);
            core.state.gpr[3] = 1U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2RLockSurfaceEx",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_lock_surface(core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2RSurfaceExists",
        [](EspressoCore& core) {
            core.state.gpr[3] = gx2r_surface_exists(core, core.state.gpr[3])
                ? 1U
                : 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2RUnlockSurfaceEx",
        [](EspressoCore& core) {
            gx2r_unlock_surface(core, core.state.gpr[3]);
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcTVSize",
        [](EspressoCore& core) {
            const std::uint32_t render_mode = core.state.gpr[3];
            const std::uint32_t surface_format = core.state.gpr[4];
            const std::uint32_t buffering_mode = core.state.gpr[5];
            const std::uint32_t out_size = core.state.gpr[6];
            const std::uint32_t out_unk = core.state.gpr[7];
            const std::uint32_t size = calculate_tv_size(
                render_mode, surface_format, buffering_mode);

            // Preflight both outputs before either is written. If pointers
            // alias, the following writes intentionally retain API order.
            validate_gx2_u32_output(
                core, out_size, "GX2CalcTVSize", "size output");
            validate_gx2_u32_output(
                core, out_unk, "GX2CalcTVSize", "unkOut output");
            core.memory.write32_be(out_size, size);
            core.memory.write32_be(out_unk, 0U);
        });

    dispatcher.register_function(
        "gx2",
        "GX2CalcDRCSize",
        [](EspressoCore& core) {
            const std::uint32_t render_mode = core.state.gpr[3];
            const std::uint32_t surface_format = core.state.gpr[4];
            const std::uint32_t buffering_mode = core.state.gpr[5];
            const std::uint32_t out_size = core.state.gpr[6];
            const std::uint32_t out_unk = core.state.gpr[7];
            const std::uint32_t size = calculate_drc_size(
                render_mode, surface_format, buffering_mode);

            // Validate both destinations before writing either one, matching
            // GX2CalcTVSize's transactional output behavior.
            validate_gx2_u32_output(
                core, out_size, "GX2CalcDRCSize", "size output");
            validate_gx2_u32_output(
                core, out_unk, "GX2CalcDRCSize", "unkOut output");
            core.memory.write32_be(out_size, size);
            core.memory.write32_be(out_unk, 0U);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetTVEnable",
        [](EspressoCore& core) {
            // Record the display-service request independently of buffer
            // setup. Guest scanout to the Vulkan window is not implemented.
            core.gx2.tv_output_enabled = core.state.gpr[3] != 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDRCEnable",
        [](EspressoCore& core) {
            // This is state/API coverage only; it does not claim a DRC frame
            // was rendered or presented.
            core.gx2.drc_output_enabled = core.state.gpr[3] != 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetTVBuffer",
        [](EspressoCore& core) {
            const std::uint32_t buffer = core.state.gpr[3];
            const std::uint32_t size = core.state.gpr[4];
            const std::uint32_t render_mode = core.state.gpr[5];
            const std::uint32_t surface_format = core.state.gpr[6];
            const std::uint32_t buffering_mode = core.state.gpr[7];

            constexpr const char* api_name = "GX2SetTVBuffer";
            const TvDimensions dimensions =
                get_tv_dimensions(render_mode, api_name);
            validate_scan_surface_format(surface_format, api_name);
            (void)get_scan_buffer_count(buffering_mode, api_name);

            // This API only records display metadata. The guest owns the buffer;
            // no memory access, allocation, cache action, or PM4 emission occurs.
            core.gx2.tv_scan_buffer_configured = true;
            core.gx2.tv_scan_buffer_address = buffer;
            core.gx2.tv_scan_buffer_size = size;
            core.gx2.tv_render_mode = render_mode;
            core.gx2.tv_surface_format = surface_format;
            core.gx2.tv_buffering_mode = buffering_mode;
            core.gx2.tv_scan_width = dimensions.width;
            core.gx2.tv_scan_height = dimensions.height;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetTVScale",
        [](EspressoCore& core) {
            const std::uint32_t width = core.state.gpr[3];
            const std::uint32_t height = core.state.gpr[4];

            // Retain the requested logical output scale independently from
            // scan-buffer dimensions. No Cafe display/AVM or renderer effects
            // are modeled here.
            core.gx2.tv_scale_configured = true;
            core.gx2.tv_scale_width = width;
            core.gx2.tv_scale_height = height;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDRCBuffer",
        [](EspressoCore& core) {
            const std::uint32_t buffer = core.state.gpr[3];
            const std::uint32_t size = core.state.gpr[4];
            const std::uint32_t render_mode = core.state.gpr[5];
            const std::uint32_t surface_format = core.state.gpr[6];
            const std::uint32_t buffering_mode = core.state.gpr[7];

            constexpr const char* api_name = "GX2SetDRCBuffer";
            const DrcScanGeometry geometry =
                get_drc_scan_geometry(render_mode, api_name);
            validate_scan_surface_format(surface_format, api_name);
            (void)get_scan_buffer_count(buffering_mode, api_name);

            // Registration only: the guest owns this image buffer. Affogato
            // does not access its pixels or emit GPU/backend commands here.
            core.gx2.drc_scan_buffer_configured = true;
            core.gx2.drc_scan_buffer_address = buffer;
            core.gx2.drc_scan_buffer_size = size;
            core.gx2.drc_render_mode = render_mode;
            core.gx2.drc_surface_format = surface_format;
            core.gx2.drc_buffering_mode = buffering_mode;
            core.gx2.drc_scan_width = geometry.width;
            core.gx2.drc_scan_height = geometry.height;
            core.gx2.drc_scan_pitch = geometry.pitch;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDRCScale",
        [](EspressoCore& core) {
            const std::uint32_t width = core.state.gpr[3];
            const std::uint32_t height = core.state.gpr[4];

            // Retain requested DRC output dimensions independently from scan
            // geometry and its padded pitch. AVM/renderer effects are deferred.
            core.gx2.drc_scale_configured = true;
            core.gx2.drc_scale_width = width;
            core.gx2.drc_scale_height = height;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetSwapInterval",
        [](EspressoCore& core) {
            const std::uint32_t interval = core.state.gpr[3];

            // Retain the raw uint32 request (Decaf semantics). VSync waits,
            // flip scheduling, and host presentation remain deferred.
            core.gx2.swap_interval = interval;
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetSwapInterval",
        [](EspressoCore& core) {
            core.state.gpr[3] = core.gx2.swap_interval;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetFetchShader",
        [](EspressoCore& core) {
            constexpr const char* api_name = "gx2::GX2SetFetchShader";
            constexpr std::uint32_t maximum_program_size =
                gx2_max_fetch_attributes * gx2_fetch_shader_instruction_size;
            const std::uint32_t descriptor_address = core.state.gpr[3];
            if (descriptor_address == 0U || (descriptor_address & 3U) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": descriptor is null or misaligned");
            }
            if (static_cast<std::uint64_t>(descriptor_address) +
                    gx2_fetch_shader_struct_size >
                guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": descriptor range wraps guest address space");
            }

            Gx2FetchShaderBinding binding{};
            binding.descriptor_address = descriptor_address;
            binding.type = core.memory.read32_be(descriptor_address + 0x00U);
            binding.program_resources = core.memory.read32_be(descriptor_address + 0x04U);
            binding.program_size = core.memory.read32_be(descriptor_address + 0x08U);
            binding.program_address = core.memory.read32_be(descriptor_address + 0x0CU);
            binding.attribute_count = core.memory.read32_be(descriptor_address + 0x10U);
            binding.divisor_count = core.memory.read32_be(descriptor_address + 0x14U);
            binding.divisors[0] = core.memory.read32_be(descriptor_address + 0x18U);
            binding.divisors[1] = core.memory.read32_be(descriptor_address + 0x1CU);

            if (binding.type != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": only non-tessellated fetch shaders are supported");
            }
            if (binding.attribute_count > gx2_max_fetch_attributes ||
                binding.divisor_count > binding.divisors.size() ||
                binding.program_size > maximum_program_size ||
                (binding.program_size % gx2_fetch_shader_instruction_size) != 0U)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": invalid fetch-shader descriptor dimensions");
            }
            if (binding.program_size != 0U &&
                (binding.program_address == 0U ||
                 (binding.program_address & 3U) != 0U))
            {
                throw HleExecutionError(
                    std::string(api_name) + ": program is null or misaligned");
            }
            if (static_cast<std::uint64_t>(binding.program_address) +
                    binding.program_size >
                guest_address_space_end)
            {
                throw HleExecutionError(
                    std::string(api_name) + ": program range wraps guest address space");
            }

            binding.program_words.reserve(binding.program_size / sizeof(std::uint32_t));
            for (std::uint32_t offset = 0U; offset < binding.program_size;
                 offset += sizeof(std::uint32_t))
            {
                binding.program_words.push_back(
                    core.memory.read32_be(binding.program_address + offset));
            }

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_fetch_shader,
                0U,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                std::nullopt,
                binding});
            core.gx2.fetch_shader = std::move(binding);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetVertexShader",
        [](EspressoCore& core) {
            Gx2ProgramShaderBinding binding = capture_program_shader(
                core,
                Gx2ProgramStage::vertex,
                gx2_vertex_shader_struct_size,
                gx2_vertex_shader_program_size_offset,
                gx2_vertex_shader_program_address_offset,
                "gx2::GX2SetVertexShader");

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_program_shader,
                0U,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                binding});
            core.gx2.program_shaders[
                static_cast<std::size_t>(Gx2ProgramStage::vertex)] =
                std::move(binding);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetPixelShader",
        [](EspressoCore& core) {
            Gx2ProgramShaderBinding binding = capture_program_shader(
                core,
                Gx2ProgramStage::pixel,
                gx2_pixel_shader_struct_size,
                gx2_pixel_shader_program_size_offset,
                gx2_pixel_shader_program_address_offset,
                "gx2::GX2SetPixelShader");

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_program_shader,
                0U,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                binding});
            core.gx2.program_shaders[
                static_cast<std::size_t>(Gx2ProgramStage::pixel)] =
                std::move(binding);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetGeometryShader",
        [](EspressoCore& core) {
            Gx2ProgramShaderBinding binding = capture_program_shader(
                core,
                Gx2ProgramStage::geometry,
                gx2_geometry_shader_struct_size,
                gx2_geometry_shader_program_size_offset,
                gx2_geometry_shader_program_address_offset,
                "gx2::GX2SetGeometryShader",
                std::pair{
                    gx2_geometry_shader_copy_size_offset,
                    gx2_geometry_shader_copy_address_offset});

            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_program_shader,
                0U,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                std::nullopt,
                std::nullopt,
                binding});
            core.gx2.program_shaders[
                static_cast<std::size_t>(Gx2ProgramStage::geometry)] =
                std::move(binding);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetAttribBuffer",
        [](EspressoCore& core) {
            const std::uint32_t index = core.state.gpr[3];
            if (index >= gx2_vertex_buffer_count)
            {
                throw HleExecutionError(
                    "GX2SetAttribBuffer: attribute buffer index is out of range");
            }

            const Gx2VertexBufferBinding binding{
                index,
                core.state.gpr[4],
                core.state.gpr[5],
                core.state.gpr[6]};
            core.gx2.pending_commands.push_back({
                Gx2CommandType::set_vertex_buffer,
                0U,
                0U,
                0U,
                0U,
                0U,
                std::nullopt,
                binding});
            core.gx2.vertex_buffers[index] = binding;
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawEx",
        [](EspressoCore& core) {
            // Preserve the guest draw request in the ordered GX2 stream for
            // the backend. Vertex fetch, shader execution, and rasterization
            // remain renderer work and are not claimed by this capture path.
            core.gx2.pending_commands.push_back({
                Gx2CommandType::draw,
                0U,
                0U,
                0U,
                0U,
                0U,
                Gx2DrawCall{
                    core.state.gpr[3], core.state.gpr[4], false, 0U, 0U,
                    core.state.gpr[5], core.state.gpr[6], 0U}});
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawIndexedEx",
        [](EspressoCore& core) {
            // Index-buffer contents are consumed by the renderer later; keep
            // the guest address and all draw parameters in stream order.
            core.gx2.pending_commands.push_back({
                Gx2CommandType::draw,
                0U,
                0U,
                0U,
                0U,
                0U,
                Gx2DrawCall{
                    core.state.gpr[3], core.state.gpr[4], true,
                    core.state.gpr[5], core.state.gpr[6], core.state.gpr[7],
                    core.state.gpr[8], 0U}});
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawEx2",
        [](EspressoCore& core) {
            core.gx2.pending_commands.push_back({
                Gx2CommandType::draw,
                0U,
                0U,
                0U,
                0U,
                0U,
                Gx2DrawCall{
                    core.state.gpr[3], core.state.gpr[4], false, 0U, 0U,
                    core.state.gpr[5], core.state.gpr[6], core.state.gpr[7]}});
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawIndexedEx2",
        [](EspressoCore& core) {
            core.gx2.pending_commands.push_back({
                Gx2CommandType::draw,
                0U,
                0U,
                0U,
                0U,
                0U,
                Gx2DrawCall{
                    core.state.gpr[3], core.state.gpr[4], true,
                    core.state.gpr[5], core.state.gpr[6], core.state.gpr[7],
                    core.state.gpr[8], core.state.gpr[9]}});
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawIndexedImmediateEx",
        [](EspressoCore& core) {
            // Immediate index data is still guest memory and remains owned by
            // the asynchronous renderer after this ordered request is queued.
            core.gx2.pending_commands.push_back({
                Gx2CommandType::draw,
                0U,
                0U,
                0U,
                0U,
                0U,
                Gx2DrawCall{
                    core.state.gpr[3], core.state.gpr[4], true,
                    core.state.gpr[5], core.state.gpr[6], core.state.gpr[7],
                    core.state.gpr[8], 0U}});
        });

    dispatcher.register_function(
        "gx2",
        "GX2SwapScanBuffers",
        [](EspressoCore& core) {
            // Cemu increments the shared request counter only on the GX2 main
            // core and records the CPU-side swap time on every call. Affogato
            // currently executes one guest lane, initialized on core 1.
            core.gx2.last_flip_time = core.time_base_ticks();
            if (core.gx2.main_core_id == 1U)
            {
                ++core.gx2.flip_request_count;
            }

            // Keep the swap in command order. flip_execute_count remains
            // renderer-owned; queueing is not evidence of a completed flip.
            core.gx2.pending_commands.push_back({
                Gx2CommandType::scan_buffer_swap,
                0U,
                core.gx2.flip_request_count});
        });

    dispatcher.register_function(
        "gx2",
        "GX2Flush",
        [](EspressoCore& core) {
            // Publish an ordered submission boundary. The frontend drains the
            // accumulated command stream after each bounded guest run; GX2Flush
            // is asynchronous and does not wait for GPU retirement.
            core.gx2.pending_commands.push_back(
                {Gx2CommandType::flush, 0U, 0U});
        });

    dispatcher.register_function(
        "gx2",
        "GX2DrawDone",
        [](EspressoCore& core) {
            // Preserve the Cafe flush-then-draw-done ordering. This emulator
            // does not submit guest draw work to Vulkan yet, so the fence is
            // already retired when recorded. Once guest draws are submitted,
            // this boundary must wait for renderer-side completion.
            core.gx2.pending_commands.push_back(
                {Gx2CommandType::flush, 0U, 0U});
            core.gx2.pending_commands.push_back(
                {Gx2CommandType::draw_done, 0U, 0U});
            core.state.gpr[3] = 1U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetGPUFence",
        [](EspressoCore& core) {
            const std::uint32_t fence_address = core.state.gpr[3];
            const std::uint32_t mask = core.state.gpr[4];
            const std::uint32_t compare_op = core.state.gpr[5] & 7U;
            const std::uint32_t compare_value = core.state.gpr[6];

            if (fence_address == 0U ||
                static_cast<std::uint64_t>(fence_address) +
                        sizeof(std::uint32_t) >
                    guest_address_space_end)
            {
                throw HleExecutionError(
                    "gx2::GX2SetGPUFence received an invalid fence address");
            }
            try
            {
                // Validate that the GPU wait target is backed by guest RAM;
                // do not read/modify it here because the wait is asynchronous.
                static_cast<void>(core.memory.read32_be(fence_address));
            }
            catch (const GuestMemoryFault&)
            {
                throw HleExecutionError(
                    "gx2::GX2SetGPUFence fence address is outside mapped guest memory");
            }

            // Preserve the GPU-side WAIT_REG_MEM operation in command order.
            // The current renderer records this synchronization command but
            // has not yet implemented guest GPU command execution.
            core.gx2.pending_commands.push_back({
                Gx2CommandType::gpu_fence,
                0U,
                compare_value,
                fence_address,
                mask,
                compare_op});
        });

    dispatcher.register_function(
        "gx2",
        "GX2GPUTimeToCPUTime",
        [](EspressoCore& core) {
            // Affogato has no GPU timestamp clock. Return zero as the
            // unavailable-time sentinel used by Cemu; do not fabricate a
            // conversion from host time or the unrelated OS timer.
            core.state.gpr[3] = 0U;
            core.state.gpr[4] = 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SampleTopGPUCycle",
        [](EspressoCore& core) {
            const std::uint32_t timestamp_output = core.state.gpr[3];
            validate_gx2_u64_output(
                core, timestamp_output, "GX2SampleTopGPUCycle");

            // Until GPU command execution can place a timestamp marker in
            // submission order, follow the established GX2 CPU-clock fallback.
            core.memory.write64_be(timestamp_output, get_os_system_time(core));
        });

    dispatcher.register_function(
        "gx2",
        "GX2SampleBottomGPUCycle",
        [](EspressoCore& core) {
            const std::uint32_t timestamp_output = core.state.gpr[3];
            validate_gx2_u64_output(
                core, timestamp_output, "GX2SampleBottomGPUCycle");

            // GPU command-order timestamps are not available without executing
            // a marker on the backend. Return the platform invalid-counter
            // sentinel, matching Cemu's fallback instead of inventing a time.
            core.memory.write64_be(
                timestamp_output, std::numeric_limits<std::uint64_t>::max());
        });

    dispatcher.register_function(
        "gx2",
        "GX2WaitForVsync",
        [](EspressoCore& core) {
            if (core.gx2.host_vsync_pacing_enabled)
            {
                // Interactive execution has a virtual 60 Hz boundary until
                // scanout and guest event scheduling are integrated.
                auto now = std::chrono::steady_clock::now();
                if (!core.gx2.vsync_deadline_initialized)
                {
                    core.gx2.next_vsync_deadline = now + gx2_virtual_vsync_period;
                    core.gx2.vsync_deadline_initialized = true;
                }
                else
                {
                    while (core.gx2.next_vsync_deadline <= now)
                    {
                        core.gx2.next_vsync_deadline += gx2_virtual_vsync_period;
                    }
                }

                std::this_thread::sleep_until(core.gx2.next_vsync_deadline);
                core.gx2.next_vsync_deadline += gx2_virtual_vsync_period;
            }

            // Headless diagnostics have no display queue to wait on. Preserve
            // tick/status state without throttling on host sleep calls.
            ++core.gx2.vsync_count;
            core.gx2.last_vsync_time = core.time_base_ticks();
            // GX2WaitForVsync has a void ABI: preserve all PPC argument GPRs.
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetSwapStatus",
        [](EspressoCore& core) {
            const std::uint32_t swap_count_output = core.state.gpr[3];
            const std::uint32_t flip_count_output = core.state.gpr[4];
            const std::uint32_t last_flip_output = core.state.gpr[5];
            const std::uint32_t last_vsync_output = core.state.gpr[6];

            // Validate every destination first so a bad later output cannot
            // leave the earlier guest words partially updated.
            validate_gx2_u32_output(
                core, swap_count_output, "GX2GetSwapStatus", "swapCount");
            validate_gx2_u32_output(
                core, flip_count_output, "GX2GetSwapStatus", "flipCount");
            validate_gx2_u64_output(
                core, last_flip_output, "GX2GetSwapStatus");
            validate_gx2_u64_output(
                core, last_vsync_output, "GX2GetSwapStatus");

            const std::uint32_t swap_count = core.gx2.flip_request_count;
            const std::uint32_t flip_count = core.gx2.flip_execute_count;
            const std::uint64_t last_flip = core.gx2.last_flip_time;
            const std::uint64_t last_vsync = core.gx2.last_vsync_time;
            core.memory.write32_be(swap_count_output, swap_count);
            core.memory.write32_be(flip_count_output, flip_count);
            core.memory.write64_be(last_flip_output, last_flip);
            core.memory.write64_be(last_vsync_output, last_vsync);
            core.state.gpr[3] = 0U;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetEventCallback",
        [](EspressoCore& core) {
            const std::uint32_t event_type = core.state.gpr[3];
            if (event_type < gx2_event_type_vsync ||
                event_type > gx2_event_type_display_list_overrun)
            {
                throw HleExecutionError(
                    "gx2::GX2SetEventCallback: unsupported event type " +
                    std::to_string(event_type));
            }

            Gx2EventCallback callback{};
            callback.function = core.state.gpr[4];
            callback.user_data = core.state.gpr[5];
            core.gx2.event_callbacks[event_type - gx2_event_type_vsync] =
                callback;

            // This records the guest registration only. Callback delivery
            // requires GX2 event scheduling, which is not modeled yet.
        });

    dispatcher.register_function(
        "gx2",
        "GX2GetEventCallback",
        [](EspressoCore& core) {
            const std::uint32_t event_type = core.state.gpr[3];
            if (event_type < gx2_event_type_vsync ||
                event_type > gx2_event_type_display_list_overrun)
            {
                throw HleExecutionError(
                    "gx2::GX2GetEventCallback: unsupported event type " +
                    std::to_string(event_type));
            }

            const std::uint32_t function_output = core.state.gpr[4];
            const std::uint32_t user_data_output = core.state.gpr[5];
            validate_gx2_u32_output(
                core, function_output, "GX2GetEventCallback", "funcOut");
            validate_gx2_u32_output(
                core, user_data_output, "GX2GetEventCallback", "userDataOut");

            const Gx2EventCallback callback =
                core.gx2.event_callbacks[event_type - gx2_event_type_vsync];
            // Both destinations are validated before either is written. If
            // the guest aliases them, preserve ABI write order.
            core.memory.write32_be(function_output, callback.function);
            core.memory.write32_be(user_data_output, callback.user_data);
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetColorBuffer",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetColorBuffer: GX2 has not been initialized");
            }

            const std::uint32_t color_buffer = core.state.gpr[3];
            const std::uint32_t target = core.state.gpr[4];
            if (target >= gx2_render_target_count)
            {
                throw HleExecutionError(
                    "gx2::GX2SetColorBuffer: invalid render-target index " +
                    std::to_string(target));
            }

            // Read the complete WUT GX2ColorBuffer (0x9C bytes) into a local
            // value before changing bound state, so guest-memory faults leave
            // the previous target binding intact.
            std::optional<Gx2ColorBufferState> descriptor;
            if (color_buffer != 0U)
            {
                descriptor = read_color_buffer(core, color_buffer);
            }
            Gx2ColorBufferBinding binding{
                target,
                color_buffer,
                descriptor};
            Gx2Command command{};
            command.type = Gx2CommandType::set_color_buffer;
            command.color_buffer = std::move(binding);
            core.gx2.pending_commands.push_back(std::move(command));
            core.gx2.color_buffer_descriptors[target] = std::move(descriptor);
            core.gx2.bound_color_buffers[target] = color_buffer;
        });

    dispatcher.register_function(
        "gx2",
        "GX2SetDepthBuffer",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2SetDepthBuffer: GX2 has not been initialized");
            }

            const std::uint32_t depth_buffer = core.state.gpr[3];
            std::optional<Gx2DepthBufferState> descriptor;
            if (depth_buffer != 0U)
            {
                descriptor = read_depth_buffer(core, depth_buffer);
            }

            Gx2DepthBufferBinding binding{depth_buffer, descriptor};
            Gx2Command command{};
            command.type = Gx2CommandType::set_depth_buffer;
            command.depth_buffer = std::move(binding);
            core.gx2.pending_commands.push_back(std::move(command));
            core.gx2.depth_buffer_descriptor = std::move(descriptor);
            core.gx2.bound_depth_buffer = depth_buffer;
        });

    dispatcher.register_function(
        "gx2",
        "GX2ClearColor",
        [](EspressoCore& core) {
            if (!core.gx2.initialized)
            {
                throw HleExecutionError(
                    "gx2::GX2ClearColor: GX2 has not been initialized");
            }

            // WUT's ABI places these float arguments in FPR1-FPR4. The HLE
            // records the guest operation; the frontend forwards the latest
            // clear request to its Vulkan presentation thread.
            Gx2ClearColorRequest request{};
            request.color_buffer = core.state.gpr[3];
            for (std::size_t index = 0U; index < request.rgba.size(); ++index)
            {
                const double value = std::bit_cast<double>(core.state.fpr[index + 1U]);
                request.rgba[index] = static_cast<float>(value);
            }
            core.gx2.latest_clear_color = request;
            ++core.gx2.clear_color_sequence;
        });
}

std::vector<Gx2Command> take_gx2_commands(Gx2RuntimeState& state)
{
    return std::exchange(state.pending_commands, std::vector<Gx2Command>{});
}

} // namespace affogato::cpu::espresso
