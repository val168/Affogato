#include "cpu/espresso/rpx_loader.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <zlib.h>

namespace affogato::cpu::espresso
{
namespace
{

constexpr std::size_t elf32_header_size = 52;
constexpr std::size_t elf32_section_header_size = 40;
constexpr std::size_t elf32_symbol_size = 16;
constexpr std::size_t elf32_rela_size = 12;
constexpr std::uint32_t section_alloc = 0x2;
constexpr std::uint32_t section_execute = 0x4;
constexpr std::uint32_t section_deflated = 0x08000000;
constexpr std::uint32_t section_nobits = 8;
constexpr std::uint32_t section_symtab = 2;
constexpr std::uint32_t section_strtab = 3;
constexpr std::uint32_t section_dynsym = 11;
constexpr std::uint32_t section_rela = 4;
constexpr std::uint32_t section_rpl_imports = 0x80000002;
constexpr std::uint32_t section_rpl_fileinfo = 0x80000004;
constexpr std::uint32_t relocation_none = 0;
constexpr std::uint32_t relocation_addr32 = 1;
constexpr std::uint32_t relocation_addr16_lo = 4;
constexpr std::uint32_t relocation_addr16_hi = 5;
constexpr std::uint32_t relocation_addr16_ha = 6;
constexpr std::uint32_t relocation_rel24 = 10;
constexpr std::size_t max_inflated_section_size = 64U * 1024U * 1024U;

struct Section
{
    std::uint32_t type{};
    std::uint32_t flags{};
    std::uint32_t address{};
    std::uint32_t offset{};
    std::uint32_t size{};
    std::uint32_t link{};
    std::uint32_t info{};
    std::uint32_t entry_size{};
    std::vector<std::uint8_t> contents;
};

struct LoadedSection
{
    std::uint32_t address{};
    bool executable{};
    std::vector<std::uint8_t> contents;
};

void require_range(std::span<const std::uint8_t> bytes, std::size_t offset, std::size_t size)
{
    if (offset > bytes.size() || size > bytes.size() - offset)
    {
        throw std::invalid_argument("RPX contains a truncated structure or section");
    }
}

[[nodiscard]] std::uint16_t read16_be(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    require_range(bytes, offset, 2);
    return static_cast<std::uint16_t>(
        (static_cast<std::uint16_t>(bytes[offset]) << 8U) | bytes[offset + 1]);
}

[[nodiscard]] std::uint32_t read32_be(std::span<const std::uint8_t> bytes, std::size_t offset)
{
    require_range(bytes, offset, 4);
    return (static_cast<std::uint32_t>(bytes[offset]) << 24U) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 16U) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 8U) |
           static_cast<std::uint32_t>(bytes[offset + 3]);
}

[[nodiscard]] std::vector<std::uint8_t> read_section(
    std::span<const std::uint8_t> file,
    const Section& section);

[[nodiscard]] std::string read_string(
    std::span<const std::uint8_t> bytes,
    std::size_t offset,
    const char* description)
{
    if (offset >= bytes.size())
    {
        throw std::invalid_argument(std::string("RPX ") + description + " has an invalid string offset");
    }
    const auto begin = bytes.begin() + static_cast<std::ptrdiff_t>(offset);
    const auto end = std::find(begin, bytes.end(), std::uint8_t{});
    if (end == bytes.end())
    {
        throw std::invalid_argument(std::string("RPX ") + description + " contains an unterminated string");
    }
    return {begin, end};
}

[[nodiscard]] std::string import_symbol_name(
    std::span<const std::uint8_t> file,
    const std::vector<Section>& sections,
    std::size_t import_section_index)
{
    for (const Section& relocation_section : sections)
    {
        if (relocation_section.type != section_rela || relocation_section.link >= sections.size() ||
            relocation_section.entry_size != elf32_rela_size)
        {
            continue;
        }

        const Section& symbol_table = sections[relocation_section.link];
        if (symbol_table.type != section_symtab || symbol_table.entry_size != elf32_symbol_size ||
            symbol_table.link >= sections.size() ||
            sections[symbol_table.link].type != section_strtab)
        {
            continue;
        }

        const std::vector<std::uint8_t> relocations = read_section(file, relocation_section);
        const std::vector<std::uint8_t> symbols = read_section(file, symbol_table);
        const std::vector<std::uint8_t> strings = read_section(file, sections[symbol_table.link]);
        if (relocations.size() % elf32_rela_size != 0 || symbols.size() % elf32_symbol_size != 0)
        {
            continue;
        }
        for (std::size_t offset = 0; offset < relocations.size(); offset += elf32_rela_size)
        {
            const std::uint32_t symbol_index = read32_be(relocations, offset + 4) >> 8U;
            const std::size_t symbol_offset =
                static_cast<std::size_t>(symbol_index) * elf32_symbol_size;
            if (symbol_offset >= symbols.size() ||
                read16_be(symbols, symbol_offset + 14) != import_section_index)
            {
                continue;
            }
            const std::uint32_t name_offset = read32_be(symbols, symbol_offset);
            if (name_offset != 0)
            {
                return read_string(strings, name_offset, "import symbol table");
            }
        }
    }

    for (const Section& symbol_table : sections)
    {
        if (symbol_table.type != section_symtab || symbol_table.entry_size != elf32_symbol_size ||
            symbol_table.link >= sections.size() ||
            sections[symbol_table.link].type != section_strtab)
        {
            continue;
        }

        const std::vector<std::uint8_t> symbols = read_section(file, symbol_table);
        const std::vector<std::uint8_t> strings = read_section(file, sections[symbol_table.link]);
        if (symbols.size() % elf32_symbol_size != 0)
        {
            continue;
        }
        for (std::size_t offset = 0; offset < symbols.size(); offset += elf32_symbol_size)
        {
            const std::uint32_t name_offset = read32_be(symbols, offset);
            if (name_offset != 0 && read16_be(symbols, offset + 14) == import_section_index)
            {
                return read_string(strings, name_offset, "import symbol table");
            }
        }
    }
    return {};
}

[[nodiscard]] std::string import_library_name(
    std::span<const std::uint8_t> file,
    const Section& section)
{
    const std::vector<std::uint8_t> contents = read_section(file, section);
    if (contents.size() < 9)
    {
        throw std::invalid_argument("RPX import section is truncated");
    }
    return read_string(contents, 8, "import library name");
}

[[nodiscard]] std::vector<std::uint8_t> read_section(
    std::span<const std::uint8_t> file,
    const Section& section)
{
    if (section.type == section_nobits)
    {
        return std::vector<std::uint8_t>(section.size, 0);
    }

    require_range(file, section.offset, section.size);
    const auto source = file.subspan(section.offset, section.size);
    if ((section.flags & section_deflated) == 0)
    {
        return {source.begin(), source.end()};
    }
    if (source.size() < 4)
    {
        throw std::invalid_argument("compressed RPX section is missing its inflated-size prefix");
    }

    const std::uint32_t inflated_size = read32_be(source, 0);
    if (inflated_size > max_inflated_section_size)
    {
        throw std::invalid_argument("compressed RPX section exceeds the 64 MiB safety limit");
    }

    std::vector<std::uint8_t> result(inflated_size);
    uLongf output_size = static_cast<uLongf>(result.size());
    const int status = uncompress(
        result.data(),
        &output_size,
        source.data() + 4,
        static_cast<uLong>(source.size() - 4));
    if (status != Z_OK || output_size != inflated_size)
    {
        throw std::invalid_argument("RPX section contains invalid or size-mismatched zlib data");
    }
    return result;
}

[[nodiscard]] std::uint32_t addend_value(std::uint32_t symbol, std::int32_t addend)
{
    return symbol + static_cast<std::uint32_t>(addend);
}

void apply_relocations(std::span<const std::uint8_t> file, const std::vector<Section>& sections,
                       std::vector<LoadedSection>& loaded)
{
    for (const Section& relocation_section : sections)
    {
        if (relocation_section.type != section_rela)
        {
            continue;
        }
        if (relocation_section.info >= sections.size() || relocation_section.link >= sections.size() ||
            (sections[relocation_section.link].type != section_symtab &&
             sections[relocation_section.link].type != section_dynsym) ||
            relocation_section.entry_size != elf32_rela_size ||
            relocation_section.size % elf32_rela_size != 0)
        {
            throw std::invalid_argument("RPX has an invalid RELA section or symbol-table link");
        }

        const Section& symbol_table = sections[relocation_section.link];
        if (symbol_table.entry_size != elf32_symbol_size ||
            symbol_table.size % elf32_symbol_size != 0)
        {
            throw std::invalid_argument("RPX has an invalid ELF32 symbol table");
        }
        const LoadedSection& target_section = loaded[relocation_section.info];
        require_range(file, relocation_section.offset, relocation_section.size);
        require_range(file, symbol_table.offset, symbol_table.size);

        for (std::size_t offset = 0; offset < relocation_section.size; offset += elf32_rela_size)
        {
            const std::size_t relocation_offset = relocation_section.offset + offset;
            const std::uint32_t target_address = read32_be(file, relocation_offset);
            const std::uint32_t info = read32_be(file, relocation_offset + 4);
            const std::int32_t addend = static_cast<std::int32_t>(read32_be(file, relocation_offset + 8));
            const std::uint32_t symbol_index = info >> 8U;
            const std::uint32_t type = info & 0xFFU;
            if (type == relocation_none)
            {
                continue;
            }

            std::uint32_t symbol_value = 0;
            if (symbol_index != 0)
            {
                const std::size_t symbol_offset = symbol_table.offset +
                    static_cast<std::size_t>(symbol_index) * elf32_symbol_size;
                if (symbol_index >= symbol_table.size / elf32_symbol_size)
                {
                    throw std::invalid_argument("RPX relocation references a missing symbol");
                }
                const std::uint16_t symbol_section = read16_be(file, symbol_offset + 14);
                if (symbol_section == 0)
                {
                    throw std::invalid_argument(
                        "RPX relocation references an unresolved import; Cafe OS/HLE is not available");
                }
                symbol_value = read32_be(file, symbol_offset + 4);
            }

            if (target_address < target_section.address)
            {
                throw std::invalid_argument("RPX relocation target precedes its target section");
            }
            const std::size_t patch_offset = target_address - target_section.address;
            auto& target = loaded[relocation_section.info].contents;
            const std::uint32_t value = addend_value(symbol_value, addend);
            switch (type)
            {
            case relocation_addr32:
                if (patch_offset > target.size() || 4 > target.size() - patch_offset)
                {
                    throw std::invalid_argument("RPX 32-bit relocation is outside its target section");
                }
                target[patch_offset] = static_cast<std::uint8_t>(value >> 24U);
                target[patch_offset + 1] = static_cast<std::uint8_t>(value >> 16U);
                target[patch_offset + 2] = static_cast<std::uint8_t>(value >> 8U);
                target[patch_offset + 3] = static_cast<std::uint8_t>(value);
                break;
            case relocation_addr16_lo:
            case relocation_addr16_hi:
            case relocation_addr16_ha:
            {
                if (patch_offset > target.size() || 2 > target.size() - patch_offset)
                {
                    throw std::invalid_argument("RPX 16-bit relocation is outside its target section");
                }
                const std::uint32_t adjusted = type == relocation_addr16_ha ? value + 0x8000U : value;
                const unsigned shift = type == relocation_addr16_lo ? 0U : 16U;
                const std::uint16_t half = static_cast<std::uint16_t>(adjusted >> shift);
                target[patch_offset] = static_cast<std::uint8_t>(half >> 8U);
                target[patch_offset + 1] = static_cast<std::uint8_t>(half);
                break;
            }
            case relocation_rel24:
            {
                if (patch_offset > target.size() || 4 > target.size() - patch_offset)
                {
                    throw std::invalid_argument("RPX REL24 relocation is outside its target section");
                }
                const std::int64_t displacement = static_cast<std::int64_t>(value) - target_address;
                if ((displacement & 3) != 0 || displacement < -0x02000000LL || displacement > 0x01FFFFFCLL)
                {
                    throw std::invalid_argument("RPX REL24 relocation is unaligned or out of range");
                }
                const std::uint32_t instruction =
                    (static_cast<std::uint32_t>(target[patch_offset]) << 24U) |
                    (static_cast<std::uint32_t>(target[patch_offset + 1]) << 16U) |
                    (static_cast<std::uint32_t>(target[patch_offset + 2]) << 8U) |
                    target[patch_offset + 3];
                const std::uint32_t patched = (instruction & 0xFC000003U) |
                    (static_cast<std::uint32_t>(displacement) & 0x03FFFFFCU);
                target[patch_offset] = static_cast<std::uint8_t>(patched >> 24U);
                target[patch_offset + 1] = static_cast<std::uint8_t>(patched >> 16U);
                target[patch_offset + 2] = static_cast<std::uint8_t>(patched >> 8U);
                target[patch_offset + 3] = static_cast<std::uint8_t>(patched);
                break;
            }
            default:
                throw std::invalid_argument("RPX contains an unsupported PowerPC relocation type " +
                                            std::to_string(type));
            }
        }
    }
}

}

RpxLoadResult load_rpx32_powerpc(EspressoCore& core, std::span<const std::uint8_t> file)
{
    require_range(file, 0, elf32_header_size);
    if (file[0] != 0x7F || file[1] != 'E' || file[2] != 'L' || file[3] != 'F' ||
        file[4] != 1 || file[5] != 2 || file[6] != 1)
    {
        throw std::invalid_argument("RPX must be a current-version, big-endian ELF32 file");
    }
    if (read16_be(file, 7) != 0xCAFE || read16_be(file, 18) != 20 ||
        read16_be(file, 16) != 0xFE01)
    {
        throw std::invalid_argument("ELF header is not a Cafe OS PowerPC RPX");
    }

    const std::uint32_t entry_point = read32_be(file, 24);
    const std::uint32_t section_table_offset = read32_be(file, 32);
    const std::uint16_t header_size = read16_be(file, 40);
    const std::uint16_t section_entry_size = read16_be(file, 46);
    const std::uint16_t section_count = read16_be(file, 48);
    const std::uint16_t string_table_index = read16_be(file, 50);
    if (header_size < elf32_header_size || section_entry_size < elf32_section_header_size ||
        section_count == 0 || string_table_index >= section_count)
    {
        throw std::invalid_argument("RPX has invalid section-header metadata");
    }
    require_range(file, 0, header_size);
    if (section_table_offset < header_size || section_table_offset > file.size() ||
        section_count > (file.size() - section_table_offset) / section_entry_size)
    {
        throw std::invalid_argument("RPX section-header table is truncated");
    }

    std::vector<Section> sections(section_count);
    bool has_file_info = false;
    for (std::size_t i = 0; i < sections.size(); ++i)
    {
        const std::size_t offset = section_table_offset + i * section_entry_size;
        Section& section = sections[i];
        section.type = read32_be(file, offset + 4);
        section.flags = read32_be(file, offset + 8);
        section.address = read32_be(file, offset + 12);
        section.offset = read32_be(file, offset + 16);
        section.size = read32_be(file, offset + 20);
        section.link = read32_be(file, offset + 24);
        section.info = read32_be(file, offset + 28);
        section.entry_size = read32_be(file, offset + 36);
    }

    for (std::size_t i = 0; i < sections.size(); ++i)
    {
        const Section& section = sections[i];
        if (section.type == section_rpl_fileinfo)
        {
            if (section.size < 4)
            {
                throw std::invalid_argument("RPX FILEINFO section is truncated");
            }
            require_range(file, section.offset, section.size);
            if (read32_be(file, section.offset) != 0xCAFE0402U)
            {
                throw std::invalid_argument("RPX FILEINFO section has an invalid Cafe magic");
            }
            has_file_info = true;
        }
        if (section.type == section_rpl_imports && section.size != 0)
        {
            const std::vector<std::uint8_t> import_contents = read_section(file, section);
            if (import_contents.size() < 4)
            {
                throw std::invalid_argument("RPX import section is truncated");
            }
            if (read32_be(import_contents, 0) != 0)
            {
                const std::string library = import_library_name(file, section);
                const std::string symbol = import_symbol_name(file, sections, i);
                std::string message = "RPX import from '" + library + "'";
                if (!symbol.empty())
                {
                    message += " requires unresolved symbol '" + symbol + "'";
                }
                message += "; dynamic linking/Cafe OS HLE is not implemented yet";
                throw std::invalid_argument(message);
            }
        }
    }
    if (!has_file_info)
    {
        throw std::invalid_argument("RPX is missing its Cafe FILEINFO section");
    }

    std::vector<LoadedSection> loaded(sections.size());
    bool entry_is_executable = false;
    std::size_t loaded_count = 0;
    for (std::size_t i = 0; i < sections.size(); ++i)
    {
        const Section& section = sections[i];
        if ((section.flags & section_alloc) == 0 || section.size == 0 ||
            section.type == section_rpl_fileinfo)
        {
            continue;
        }
        LoadedSection& destination = loaded[i];
        destination.address = section.address;
        destination.executable = (section.flags & section_execute) != 0;
        destination.contents = read_section(file, section);
        const std::uint64_t end = static_cast<std::uint64_t>(destination.address) + destination.contents.size();
        if (end > (std::uint64_t{1} << 32U) || destination.address > core.memory.size() ||
            destination.contents.size() > core.memory.size() - destination.address)
        {
            throw std::invalid_argument("RPX section lies outside the current flat guest memory");
        }
        if (destination.executable && entry_point >= destination.address &&
            static_cast<std::uint64_t>(entry_point) < end)
        {
            entry_is_executable = true;
        }
        ++loaded_count;
    }
    if (loaded_count == 0 || (entry_point & 3U) != 0 || !entry_is_executable)
    {
        throw std::invalid_argument("RPX entry point is not inside a loaded executable section");
    }

    apply_relocations(file, sections, loaded);
    for (std::size_t i = 0; i < loaded.size(); ++i)
    {
        if (!loaded[i].contents.empty())
        {
            core.memory.write_bytes(loaded[i].address, loaded[i].contents);
        }
    }
    core.state.cia = entry_point;
    return {entry_point, loaded_count};
}

}
