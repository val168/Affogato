#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace affogato::gpu
{

inline constexpr std::uint32_t latte_vgt_primitive_type_register_address =
    0x8958U;
inline constexpr std::uint32_t latte_vgt_dma_index_type_register_address =
    0x28A7CU;
inline constexpr std::uint32_t latte_vgt_dma_num_instances_register_address =
    0x28A88U;
inline constexpr std::uint32_t latte_sq_vtx_base_vtx_loc_register_address =
    0x3CFF0U;
inline constexpr std::uint32_t latte_sq_vtx_start_inst_loc_register_address =
    0x3CFF4U;

enum class LattePm4DisplayListError
{
    none,
    unsupported_packet_type,
    invalid_type2_header,
    invalid_type3_header,
    truncated_packet,
    unsupported_opcode,
    missing_register_data,
    invalid_packet_payload_length,
    register_range_out_of_bounds,
};

enum class LatteRegisterSpace
{
    config,
    context,
    control,
};

enum class LattePm4DisplayListCommandType
{
    register_write,
    draw_index_auto,
    draw_index_2,
};

struct LattePm4RegisterWrite
{
    LatteRegisterSpace space{};
    std::uint32_t address{};
    std::uint32_t value{};

    bool operator==(const LattePm4RegisterWrite&) const = default;
};

struct LattePm4DisplayListCommand
{
    LattePm4DisplayListCommandType type{};
    LattePm4RegisterWrite register_write{};
    std::uint32_t index_count{};
    std::uint32_t draw_initiator{};
    std::uint32_t max_index_count{};
    std::uint64_t index_address{};

    bool operator==(const LattePm4DisplayListCommand&) const = default;
};

struct LattePm4DisplayListDecodeResult
{
    std::vector<LattePm4DisplayListCommand> commands;
    LattePm4DisplayListError error{LattePm4DisplayListError::none};
    std::size_t error_word_offset{};

    [[nodiscard]] bool succeeded() const noexcept
    {
        return error == LattePm4DisplayListError::none;
    }
};

// Input words are host-order values decoded from guest big-endian memory.
// Only bounded config/context/control register writes, Type-2 filler,
// Type-3 NOP, register-set packets, INDEX_TYPE, NUM_INSTANCES, and
// unpredicated DRAW_INDEX_AUTO and DRAW_INDEX_2 packets are accepted.
[[nodiscard]] inline LattePm4DisplayListDecodeResult
decode_latte_pm4_display_list(
    std::span<const std::uint32_t> words)
{
    constexpr std::uint32_t packet_type_mask = 0xC0000000U;
    constexpr std::uint32_t packet_type0 = 0x00000000U;
    constexpr std::uint32_t packet_type2 = 0x80000000U;
    constexpr std::uint32_t packet_type3 = 0xC0000000U;
    constexpr std::uint32_t type2_reserved_mask = 0x3FFFFFFFU;
    constexpr std::uint32_t type3_reserved_mask = 0x000000FEU;
    constexpr std::uint32_t type3_predicate_mask = 0x00000001U;
    constexpr std::uint32_t type3_opcode_shift = 8U;
    constexpr std::uint32_t type3_opcode_mask = 0x0000FF00U;
    constexpr std::uint32_t type3_nop_opcode = 0x10U;
    constexpr std::uint32_t index_type_opcode = 0x2AU;
    constexpr std::uint32_t draw_index_auto_opcode = 0x2DU;
    constexpr std::uint32_t draw_index_2_opcode = 0x27U;
    constexpr std::uint32_t num_instances_opcode = 0x2FU;
    constexpr std::uint32_t packet_count_shift = 16U;
    constexpr std::uint32_t packet_count_mask = 0x3FFFU;
    // R7xx SET_* register packets use dword offsets.
    constexpr std::uint32_t set_config_register_opcode = 0x68U;
    constexpr std::uint32_t set_context_register_opcode = 0x69U;
    constexpr std::uint32_t set_control_constant_opcode = 0x6FU;
    // Register windows use byte addresses in captured GX2 state.
    constexpr std::uint32_t config_register_base = 0x8000U;
    constexpr std::uint32_t config_register_end = 0xAC00U;
    constexpr std::uint32_t context_register_base = 0x28000U;
    constexpr std::uint32_t context_register_end = 0x29000U;
    // Decaf's Latte map identifies base-vertex and base-instance control
    // registers as the complete currently modeled control-constant range.
    constexpr std::uint32_t control_constant_base = 0x3CFF0U;
    constexpr std::uint32_t control_constant_end = 0x3CFF8U;
    constexpr std::uint32_t register_stride = sizeof(std::uint32_t);

    LattePm4DisplayListDecodeResult result;
    std::size_t word_offset = 0U;

    while (word_offset < words.size())
    {
        const std::size_t packet_header_offset = word_offset;
        const std::uint32_t header = words[word_offset++];
        const std::uint32_t packet_type = header & packet_type_mask;

        if (packet_type == packet_type2)
        {
            if ((header & type2_reserved_mask) != 0U)
            {
                result.error = LattePm4DisplayListError::invalid_type2_header;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            continue;
        }

        if (packet_type != packet_type0 && packet_type != packet_type3)
        {
            result.error = LattePm4DisplayListError::unsupported_packet_type;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        if (packet_type == packet_type3 &&
            ((header & type3_reserved_mask) != 0U ||
             (header & type3_predicate_mask) != 0U))
        {
            result.error = LattePm4DisplayListError::invalid_type3_header;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        const std::uint32_t payload_word_count =
            ((header >> packet_count_shift) & packet_count_mask) + 1U;
        if (payload_word_count > words.size() - word_offset)
        {
            result.error = LattePm4DisplayListError::truncated_packet;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        if (packet_type == packet_type0)
        {
            // Type-0 BASE_INDEX is a register dword index. Accept only a
            // complete range inside one register window modeled by Affogato.
            const std::uint32_t register_address =
                (header & 0x0000FFFFU) * register_stride;
            LatteRegisterSpace register_space{};
            std::uint32_t register_window_end{};
            if (register_address >= config_register_base &&
                register_address < config_register_end)
            {
                register_space = LatteRegisterSpace::config;
                register_window_end = config_register_end;
            }
            else if (register_address >= context_register_base &&
                     register_address < context_register_end)
            {
                register_space = LatteRegisterSpace::context;
                register_window_end = context_register_end;
            }
            else if (register_address >= control_constant_base &&
                     register_address < control_constant_end)
            {
                register_space = LatteRegisterSpace::control;
                register_window_end = control_constant_end;
            }
            else
            {
                result.error =
                    LattePm4DisplayListError::register_range_out_of_bounds;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            if (payload_word_count >
                (register_window_end - register_address) / register_stride)
            {
                result.error =
                    LattePm4DisplayListError::register_range_out_of_bounds;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            for (std::uint32_t register_index = 0U;
                 register_index < payload_word_count;
                 ++register_index)
            {
                result.commands.push_back({
                    LattePm4DisplayListCommandType::register_write,
                    {
                        register_space,
                        register_address + register_index * register_stride,
                        words[word_offset + register_index]},
                    0U,
                    0U});
            }

            word_offset += payload_word_count;
            continue;
        }

        const std::uint32_t opcode =
            (header & type3_opcode_mask) >> type3_opcode_shift;
        if (opcode == type3_nop_opcode)
        {
            // A PM4 NOP consumes its declared payload but changes no state.
            word_offset += payload_word_count;
            continue;
        }

        if (opcode == index_type_opcode || opcode == num_instances_opcode)
        {
            if (payload_word_count != 1U)
            {
                result.error =
                    LattePm4DisplayListError::invalid_packet_payload_length;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            const std::uint32_t register_address =
                opcode == index_type_opcode
                    ? latte_vgt_dma_index_type_register_address
                    : latte_vgt_dma_num_instances_register_address;
            if (register_address < context_register_base ||
                register_address >= context_register_end)
            {
                result.error =
                    LattePm4DisplayListError::register_range_out_of_bounds;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            result.commands.push_back({
                LattePm4DisplayListCommandType::register_write,
                {
                    LatteRegisterSpace::context,
                    register_address,
                    words[word_offset]},
                0U,
                0U});
            ++word_offset;
            continue;
        }

        if (opcode == draw_index_auto_opcode)
        {
            if (payload_word_count != 2U)
            {
                result.error =
                    LattePm4DisplayListError::invalid_packet_payload_length;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            result.commands.push_back({
                LattePm4DisplayListCommandType::draw_index_auto,
                {},
                words[word_offset],
                words[word_offset + 1U]});
            word_offset += payload_word_count;
            continue;
        }

        if (opcode == draw_index_2_opcode)
        {
            if (payload_word_count != 5U)
            {
                result.error =
                    LattePm4DisplayListError::invalid_packet_payload_length;
                result.error_word_offset = packet_header_offset;
                result.commands.clear();
                return result;
            }

            const std::uint64_t index_address =
                static_cast<std::uint64_t>(words[word_offset + 1U]) |
                (static_cast<std::uint64_t>(words[word_offset + 2U] & 0xFFU)
                 << 32U);
            result.commands.push_back({
                LattePm4DisplayListCommandType::draw_index_2,
                {},
                words[word_offset + 3U],
                words[word_offset + 4U],
                words[word_offset],
                index_address});
            word_offset += payload_word_count;
            continue;
        }

        if (opcode != set_config_register_opcode &&
            opcode != set_context_register_opcode &&
            opcode != set_control_constant_opcode)
        {
            result.error = LattePm4DisplayListError::unsupported_opcode;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        if (payload_word_count < 2U)
        {
            result.error = LattePm4DisplayListError::missing_register_data;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        const std::uint32_t register_offset = words[word_offset];
        const std::uint32_t register_count = payload_word_count - 1U;
        LatteRegisterSpace register_space = LatteRegisterSpace::context;
        std::uint32_t register_window_base = context_register_base;
        std::uint32_t register_window_end = context_register_end;
        if (opcode == set_config_register_opcode)
        {
            register_space = LatteRegisterSpace::config;
            register_window_base = config_register_base;
            register_window_end = config_register_end;
        }
        else if (opcode == set_control_constant_opcode)
        {
            register_space = LatteRegisterSpace::control;
            register_window_base = control_constant_base;
            register_window_end = control_constant_end;
        }
        const std::uint32_t register_window_count =
            (register_window_end - register_window_base) / register_stride;
        if (register_offset >= register_window_count ||
            register_count > register_window_count - register_offset)
        {
            result.error =
                LattePm4DisplayListError::register_range_out_of_bounds;
            result.error_word_offset = packet_header_offset;
            result.commands.clear();
            return result;
        }

        for (std::uint32_t register_index = 0U;
             register_index < register_count;
             ++register_index)
        {
            const std::size_t value_index =
                word_offset + 1U + register_index;
            result.commands.push_back({
                LattePm4DisplayListCommandType::register_write,
                {
                    register_space,
                    register_window_base +
                        (register_offset + register_index) * register_stride,
                    words[value_index]},
                0U,
                0U});
        }

        word_offset += payload_word_count;
    }

    return result;
}

} // namespace affogato::gpu
