#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace affogato::gpu
{

enum class LatteContextDisplayListError
{
    none,
    unsupported_packet_type,
    invalid_type2_header,
    invalid_type3_header,
    truncated_packet,
    unsupported_opcode,
    missing_register_data,
    register_range_out_of_bounds,
};

struct LatteContextRegisterWrite
{
    std::uint32_t address{};
    std::uint32_t value{};

    bool operator==(const LatteContextRegisterWrite&) const = default;
};

struct LatteContextDisplayListDecodeResult
{
    std::vector<LatteContextRegisterWrite> register_writes;
    LatteContextDisplayListError error{LatteContextDisplayListError::none};
    std::size_t error_word_offset{};

    [[nodiscard]] bool succeeded() const noexcept
    {
        return error == LatteContextDisplayListError::none;
    }
};

// The input contains host-order values previously read from guest big-endian
// memory. This intentionally supports Type-2 filler, Type-0 writes to the
// context-register window, and Type-3 SET_CONTEXT_REG packets captured from
// bounded GX2 context display lists.
[[nodiscard]] inline LatteContextDisplayListDecodeResult
decode_latte_context_display_list(
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
    constexpr std::uint32_t packet_count_shift = 16U;
    constexpr std::uint32_t packet_count_mask = 0x3FFFU;
    // R7xx IT_SET_CONTEXT_REG; its offset is measured in register dwords.
    constexpr std::uint32_t set_context_register_opcode = 0x69U;
    // Context register byte range from the Latte register map.
    constexpr std::uint32_t context_register_base = 0x28000U;
    constexpr std::uint32_t context_register_end = 0x29000U;
    constexpr std::uint32_t register_stride = sizeof(std::uint32_t);

    LatteContextDisplayListDecodeResult result;
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
                result.error = LatteContextDisplayListError::invalid_type2_header;
                result.error_word_offset = packet_header_offset;
                result.register_writes.clear();
                return result;
            }

            continue;
        }

        if (packet_type != packet_type0 && packet_type != packet_type3)
        {
            result.error = LatteContextDisplayListError::unsupported_packet_type;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        if (packet_type == packet_type3 &&
            ((header & type3_reserved_mask) != 0U ||
             (header & type3_predicate_mask) != 0U))
        {
            result.error = LatteContextDisplayListError::invalid_type3_header;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        const std::uint32_t payload_word_count =
            ((header >> packet_count_shift) & packet_count_mask) + 1U;
        if (payload_word_count > words.size() - word_offset)
        {
            result.error = LatteContextDisplayListError::truncated_packet;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        if (packet_type == packet_type0)
        {
            // Type-0 BASE_INDEX is a register dword index. Restrict this
            // context restore path to the window it can safely model.
            const std::uint32_t register_address =
                (header & 0x0000FFFFU) * register_stride;
            if (register_address < context_register_base ||
                register_address >= context_register_end ||
                payload_word_count >
                    (context_register_end - register_address) /
                        register_stride)
            {
                result.error =
                    LatteContextDisplayListError::register_range_out_of_bounds;
                result.error_word_offset = packet_header_offset;
                result.register_writes.clear();
                return result;
            }

            for (std::uint32_t register_index = 0U;
                 register_index < payload_word_count;
                 ++register_index)
            {
                result.register_writes.push_back({
                    register_address + register_index * register_stride,
                    words[word_offset + register_index]});
            }

            word_offset += payload_word_count;
            continue;
        }

        const std::uint32_t opcode =
            (header & type3_opcode_mask) >> type3_opcode_shift;
        if (opcode != set_context_register_opcode)
        {
            result.error = LatteContextDisplayListError::unsupported_opcode;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        if (payload_word_count < 2U)
        {
            result.error = LatteContextDisplayListError::missing_register_data;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        const std::uint32_t register_offset = words[word_offset];
        const std::uint32_t register_count = payload_word_count - 1U;
        const std::uint32_t context_register_count =
            (context_register_end - context_register_base) / register_stride;
        if (register_offset >= context_register_count ||
            register_count > context_register_count - register_offset)
        {
            result.error =
                LatteContextDisplayListError::register_range_out_of_bounds;
            result.error_word_offset = packet_header_offset;
            result.register_writes.clear();
            return result;
        }

        for (std::uint32_t register_index = 0U;
             register_index < register_count;
             ++register_index)
        {
            const std::size_t value_index =
                word_offset + 1U + register_index;
            result.register_writes.push_back({
                context_register_base +
                    (register_offset + register_index) * register_stride,
                words[value_index]});
        }

        word_offset += payload_word_count;
    }

    return result;
}

} // namespace affogato::gpu
