#include "cpu/espresso/elf_loader.hpp"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::size_t elf32_header_size = 52;
constexpr std::size_t elf32_program_header_size = 32;
constexpr std::uint32_t loadable_segment = 1;
constexpr std::uint32_t executable_segment_flag = 1;

struct LoadSegment
{
    std::uint32_t file_offset{};
    std::uint32_t virtual_address{};
    std::uint32_t file_size{};
    std::uint32_t memory_size{};
    std::uint32_t flags{};
};

void require_file_range(
    std::span<const std::uint8_t> file,
    std::size_t offset,
    std::size_t length)
{
    if (offset > file.size() || length > file.size() - offset)
    {
        throw std::invalid_argument("ELF file contains a truncated structure or segment");
    }
}

[[nodiscard]] std::uint16_t read16_be(
    std::span<const std::uint8_t> file,
    std::size_t offset)
{
    require_file_range(file, offset, 2);
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(file[offset]) << 8U) |
        static_cast<std::uint16_t>(file[offset + 1]));
}

[[nodiscard]] std::uint32_t read32_be(
    std::span<const std::uint8_t> file,
    std::size_t offset)
{
    require_file_range(file, offset, 4);
    return (static_cast<std::uint32_t>(file[offset]) << 24U) |
           (static_cast<std::uint32_t>(file[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(file[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(file[offset + 3]);
}

[[nodiscard]] bool is_power_of_two(std::uint32_t value) noexcept
{
    return value != 0 && (value & (value - 1U)) == 0;
}

}

ElfLoadResult load_elf32_powerpc(
    EspressoCore& core,
    std::span<const std::uint8_t> file)
{
    require_file_range(file, 0, elf32_header_size);

    if (file[0] != 0x7FU || file[1] != 'E' || file[2] != 'L' || file[3] != 'F')
    {
        throw std::invalid_argument("input is not an ELF file");
    }
    if (file[4] != 1 || file[5] != 2 || file[6] != 1)
    {
        throw std::invalid_argument("ELF must be 32-bit, big-endian, and current-version");
    }

    const std::uint16_t type = read16_be(file, 16);
    const std::uint16_t machine = read16_be(file, 18);
    const std::uint32_t version = read32_be(file, 20);
    const std::uint32_t entry_point = read32_be(file, 24);
    const std::uint32_t program_header_offset = read32_be(file, 28);
    const std::uint16_t header_size = read16_be(file, 40);
    const std::uint16_t program_header_entry_size = read16_be(file, 42);
    const std::uint16_t program_header_count = read16_be(file, 44);

    if (type != 2 || machine != 20 || version != 1)
    {
        throw std::invalid_argument("ELF must be an executable for 32-bit PowerPC");
    }
    if (header_size < elf32_header_size ||
        program_header_entry_size < elf32_program_header_size ||
        program_header_count == 0)
    {
        throw std::invalid_argument("ELF has an invalid header or no program headers");
    }
    require_file_range(file, 0, header_size);

    const std::size_t table_offset = program_header_offset;
    const std::size_t entry_size = program_header_entry_size;
    if (table_offset < header_size || table_offset > file.size() ||
        program_header_count > (file.size() - table_offset) / entry_size)
    {
        throw std::invalid_argument("ELF program-header table is truncated");
    }

    std::vector<LoadSegment> segments;
    segments.reserve(program_header_count);
    bool entry_is_executable = false;

    for (std::size_t index = 0; index < program_header_count; ++index)
    {
        const std::size_t offset = table_offset + index * entry_size;
        if (read32_be(file, offset) != loadable_segment)
        {
            continue;
        }

        const std::uint32_t file_offset = read32_be(file, offset + 4);
        const std::uint32_t virtual_address = read32_be(file, offset + 8);
        const std::uint32_t file_size = read32_be(file, offset + 16);
        const std::uint32_t memory_size = read32_be(file, offset + 20);
        const std::uint32_t flags = read32_be(file, offset + 24);
        const std::uint32_t alignment = read32_be(file, offset + 28);

        if (file_size > memory_size)
        {
            throw std::invalid_argument("ELF load segment has p_filesz larger than p_memsz");
        }
        require_file_range(file, file_offset, file_size);

        const std::uint64_t guest_end =
            static_cast<std::uint64_t>(virtual_address) + memory_size;
        if (guest_end > (std::uint64_t{1} << 32U))
        {
            throw std::invalid_argument("ELF load segment wraps the 32-bit guest address space");
        }
        const std::size_t guest_offset = virtual_address;
        if (guest_offset > core.memory.size() ||
            memory_size > core.memory.size() - guest_offset)
        {
            throw std::invalid_argument("ELF load segment is outside guest memory");
        }
        if (alignment > 1 &&
            (!is_power_of_two(alignment) || file_offset % alignment != virtual_address % alignment))
        {
            throw std::invalid_argument("ELF load segment has invalid alignment");
        }

        if ((flags & executable_segment_flag) != 0 &&
            entry_point >= virtual_address &&
            static_cast<std::uint64_t>(entry_point) < guest_end)
        {
            entry_is_executable = true;
        }
        segments.push_back({file_offset, virtual_address, file_size, memory_size, flags});
    }

    if (segments.empty())
    {
        throw std::invalid_argument("ELF contains no loadable segments");
    }
    if ((entry_point & 3U) != 0 || !entry_is_executable)
    {
        throw std::invalid_argument("ELF entry point is not in an aligned executable segment");
    }

    for (const LoadSegment& segment : segments)
    {
        core.memory.zero_fill(segment.virtual_address, segment.memory_size);
        const auto data = file.subspan(segment.file_offset, segment.file_size);
        core.memory.write_bytes(segment.virtual_address, data);
    }

    core.state.cia = entry_point;
    return {entry_point, segments.size()};
}

}
