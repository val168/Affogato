#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace affogato::gpu
{

enum class LatteControlFlowOpcode : std::uint8_t
{
    call_fetch_shader = 0x13U,
    export_data = 0x27U,
    export_done = 0x28U
};

struct LatteControlFlowRecord
{
    std::size_t word_offset{}; // Index of this record's first captured DWORD.
    std::uint32_t word0{};
    std::uint32_t word1{};
    LatteControlFlowOpcode opcode{};
    bool end_of_program{};

    bool operator==(const LatteControlFlowRecord&) const = default;
};

// Decode the verified non-ALU control-flow records in a shader program.
// GX2 shader memory is captured as big-endian CPU words; Latte instruction
// DWORDs use the little-endian word values emitted in GFD shader programs.
[[nodiscard]] inline std::vector<LatteControlFlowRecord>
decode_latte_control_flow(
    std::span<const std::uint32_t> guest_big_endian_words)
{
    constexpr std::size_t words_per_record = 2U;
    constexpr std::uint32_t opcode_shift = 23U;
    constexpr std::uint32_t opcode_mask = 0x7FU;
    constexpr std::uint32_t end_of_program_shift = 21U;

    if ((guest_big_endian_words.size() % words_per_record) != 0U)
    {
        throw std::runtime_error(
            "Latte shader control-flow program has an incomplete DWORD pair");
    }

    std::vector<LatteControlFlowRecord> records;

    for (std::size_t word_offset = 0U;
         word_offset < guest_big_endian_words.size();
         word_offset += words_per_record)
    {
        // The GFD fixture stores each Latte DWORD little-endian, while the
        // guest CPU helper captured it as a big-endian integer.
        const std::uint32_t word0 =
            std::byteswap(guest_big_endian_words[word_offset]);
        const std::uint32_t word1 =
            std::byteswap(guest_big_endian_words[word_offset + 1U]);
        const std::uint32_t raw_opcode =
            (word1 >> opcode_shift) & opcode_mask;

        LatteControlFlowOpcode opcode{};
        switch (raw_opcode)
        {
        case static_cast<std::uint32_t>(
            LatteControlFlowOpcode::call_fetch_shader):
            opcode = LatteControlFlowOpcode::call_fetch_shader;
            break;
        case static_cast<std::uint32_t>(LatteControlFlowOpcode::export_data):
            opcode = LatteControlFlowOpcode::export_data;
            break;
        case static_cast<std::uint32_t>(LatteControlFlowOpcode::export_done):
            opcode = LatteControlFlowOpcode::export_done;
            break;
        default:
            throw std::runtime_error(
                "Unsupported Latte shader control-flow opcode " +
                std::to_string(raw_opcode) + " at DWORD " +
                std::to_string(word_offset));
        }

        const bool end_of_program =
            ((word1 >> end_of_program_shift) & 1U) != 0U;
        records.push_back({word_offset, word0, word1, opcode, end_of_program});

        if (end_of_program)
        {
            return records;
        }
    }

    throw std::runtime_error(
        "Latte shader control-flow program is missing its end marker");
}

} // namespace affogato::gpu
