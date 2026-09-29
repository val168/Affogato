#include "cpu/espresso/rpx_loader.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
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
constexpr std::uint32_t trampoline_size = 16;
constexpr std::uint16_t symbol_section_absolute = 0xFFF1;
constexpr std::uint16_t symbol_section_undefined = 0;
constexpr std::uint8_t symbol_binding_weak = 2;
constexpr std::uint8_t symbol_binding_global = 1;
constexpr std::uint8_t symbol_type_object = 1;
constexpr std::uint32_t rpl_loader_metadata_base = 0xC0000000U;
constexpr std::uint32_t relocation_none = 0;
constexpr std::uint32_t relocation_addr32 = 1;
constexpr std::uint32_t relocation_ghs_rel16_hi = 252;
constexpr std::uint32_t relocation_ghs_rel16_lo = 253;
constexpr std::uint32_t relocation_addr16_lo = 4;
constexpr std::uint32_t relocation_addr16_hi = 5;
constexpr std::uint32_t relocation_addr16_ha = 6;
constexpr std::uint32_t relocation_rel24 = 10;
constexpr std::size_t max_inflated_section_size = 64U * 1024U * 1024U;
constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;

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

struct RpxFileInfo
{
    std::uint32_t text_size{};
    std::uint32_t tramp_adjust{};
    // Present in v4.2 FILEINFO, but Decaf's Cafe trampoline allocator does
    // not consume this field; keep it parsed without inventing semantics.
    std::uint32_t tramp_addition{};
};

struct TrampolineAllocator
{
    std::uint64_t pre_begin{};
    std::uint64_t pre_cursor{};
    std::size_t pre_slots{};
    std::uint64_t post_cursor{};
    std::uint64_t post_end{};
    std::size_t post_slots{};
};

[[nodiscard]] std::uint64_t align_up_16(std::uint64_t value)
{
    return (value + 15U) & ~std::uint64_t{15U};
}

[[nodiscard]] TrampolineAllocator make_trampoline_allocator(
    const RpxFileInfo& file_info,
    const std::vector<LoadedSection>& loaded)
{
    std::uint64_t text_begin = guest_address_space_end;
    std::uint64_t text_end = 0;
    for (const LoadedSection& section : loaded)
    {
        if (section.contents.empty() || !section.executable ||
            section.address >= rpl_loader_metadata_base)
        {
            continue;
        }
        text_begin = std::min<std::uint64_t>(text_begin, section.address);
        text_end = std::max<std::uint64_t>(
            text_end, static_cast<std::uint64_t>(section.address) + section.contents.size());
    }

    if (text_begin == guest_address_space_end || text_begin < file_info.tramp_adjust)
    {
        return {};
    }

    const std::uint64_t reservation_begin = text_begin - file_info.tramp_adjust;
    const std::uint64_t reservation_end = reservation_begin + file_info.text_size;
    if (file_info.text_size < file_info.tramp_adjust ||
        reservation_end > guest_address_space_end || text_end > reservation_end)
    {
        return {};
    }

    const std::uint64_t pre_end = text_begin & ~std::uint64_t{15U};
    const std::uint64_t post_begin = align_up_16(text_end);
    TrampolineAllocator allocator;
    allocator.pre_begin = reservation_begin;
    allocator.pre_cursor = pre_end;
    if (pre_end >= reservation_begin)
    {
        allocator.pre_slots = static_cast<std::size_t>(
            (pre_end - reservation_begin) / trampoline_size);
    }
    allocator.post_cursor = post_begin;
    allocator.post_end = reservation_end;
    if (reservation_end >= post_begin)
    {
        allocator.post_slots = static_cast<std::size_t>(
            (reservation_end - post_begin) / trampoline_size);
    }
    return allocator;
}

[[nodiscard]] bool ranges_overlap(
    std::uint64_t address,
    std::uint64_t size,
    std::uint64_t other_address,
    std::uint64_t other_size)
{
    return address < other_address + other_size && other_address < address + size;
}

[[nodiscard]] bool trampoline_slot_available(
    std::uint32_t address,
    const std::vector<LoadedSection>& loaded,
    const std::vector<LoadedSection>& trampolines)
{
    const std::uint64_t end = static_cast<std::uint64_t>(address) + trampoline_size;
    if (end > guest_address_space_end ||
        (address < HleDispatcher::import_address_limit &&
         end > HleDispatcher::first_import_address))
    {
        return false;
    }
    const auto overlaps_section = [address](const LoadedSection& section) {
        return !section.contents.empty() && ranges_overlap(
            address, trampoline_size, section.address, section.contents.size());
    };
    return std::none_of(loaded.begin(), loaded.end(), overlaps_section) &&
           std::none_of(trampolines.begin(), trampolines.end(), overlaps_section);
}

[[nodiscard]] std::uint32_t allocate_trampoline(
    TrampolineAllocator& allocator,
    std::uint32_t target_address,
    const std::vector<LoadedSection>& loaded,
    const std::vector<LoadedSection>& trampolines)
{
    const auto in_rel24_range = [target_address](std::uint32_t address) {
        const std::int64_t displacement = static_cast<std::int64_t>(address) - target_address;
        return (displacement & 3) == 0 && displacement >= -0x02000000LL &&
               displacement <= 0x01FFFFFCLL;
    };

    while (allocator.post_slots != 0)
    {
        const std::uint64_t candidate = allocator.post_cursor;
        allocator.post_cursor += trampoline_size;
        --allocator.post_slots;
        if (candidate + trampoline_size > allocator.post_end ||
            candidate > UINT32_MAX || !in_rel24_range(static_cast<std::uint32_t>(candidate)))
        {
            break;
        }
        if (trampoline_slot_available(static_cast<std::uint32_t>(candidate), loaded, trampolines))
        {
            return static_cast<std::uint32_t>(candidate);
        }
    }

    while (allocator.pre_slots != 0)
    {
        if (allocator.pre_cursor < allocator.pre_begin + trampoline_size)
        {
            break;
        }
        allocator.pre_cursor -= trampoline_size;
        --allocator.pre_slots;
        const std::uint64_t candidate = allocator.pre_cursor;
        if (!in_rel24_range(static_cast<std::uint32_t>(candidate)))
        {
            break;
        }
        if (trampoline_slot_available(static_cast<std::uint32_t>(candidate), loaded, trampolines))
        {
            return static_cast<std::uint32_t>(candidate);
        }
    }

    throw std::invalid_argument(
        "RPX REL24 relocation has no reachable free FILEINFO trampoline slot");
}

void write_trampoline(std::vector<std::uint8_t>& bytes, std::uint32_t target)
{
    const std::array<std::uint32_t, 4> instructions{
        0x3D600000U | (target >> 16U), // lis r11, target@h
        0x616B0000U | (target & 0xFFFFU), // ori r11, r11, target@l
        0x7D6903A6U, // mtctr r11
        0x4E800420U, // bctr
    };
    bytes.resize(trampoline_size);
    for (std::size_t i = 0; i < instructions.size(); ++i)
    {
        const std::uint32_t instruction = instructions[i];
        bytes[i * 4] = static_cast<std::uint8_t>(instruction >> 24U);
        bytes[i * 4 + 1] = static_cast<std::uint8_t>(instruction >> 16U);
        bytes[i * 4 + 2] = static_cast<std::uint8_t>(instruction >> 8U);
        bytes[i * 4 + 3] = static_cast<std::uint8_t>(instruction);
    }
}

[[nodiscard]] std::string section_range_error(
    std::size_t index,
    std::uint32_t address,
    std::uint64_t size,
    std::uint64_t end,
    const char* reason)
{
    std::ostringstream message;
    message << "RPX section " << index << ' ' << reason << " (guest address 0x"
            << std::hex << std::uppercase << address << ", section size 0x" << size
            << ", section end 0x" << end << ')';
    return message.str();
}

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

[[nodiscard]] std::string symbol_name(
    std::span<const std::uint8_t> strings,
    std::span<const std::uint8_t> symbols,
    std::uint32_t symbol_index)
{
    const std::size_t offset = static_cast<std::size_t>(symbol_index) * elf32_symbol_size;
    if (offset > symbols.size() || elf32_symbol_size > symbols.size() - offset)
    {
        throw std::invalid_argument("RPX relocation references a missing symbol");
    }

    const std::uint32_t name_offset = read32_be(symbols, offset);
    return name_offset == 0 ? std::string{} : read_string(strings, name_offset, "symbol table");
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

void apply_relocations(
    EspressoCore& core,
    std::span<const std::uint8_t> file,
    const std::vector<Section>& sections,
    std::vector<LoadedSection>& loaded,
    const RpxFileInfo& file_info,
    std::vector<LoadedSection>& trampolines)
{
    TrampolineAllocator trampoline_allocator = make_trampoline_allocator(file_info, loaded);
    for (const Section& relocation_section : sections)
    {
        if (relocation_section.type != section_rela)
        {
            continue;
        }
        if (relocation_section.info >= sections.size() || relocation_section.link >= sections.size() ||
            (sections[relocation_section.link].type != section_symtab &&
             sections[relocation_section.link].type != section_dynsym) ||
            relocation_section.entry_size != elf32_rela_size)
        {
            throw std::invalid_argument("RPX has an invalid RELA section or symbol-table link");
        }

        const Section& target = sections[relocation_section.info];
        if (target.address >= rpl_loader_metadata_base && target.type != section_rpl_imports)
        {
            continue;
        }

        const Section& symbol_table = sections[relocation_section.link];
        if (symbol_table.entry_size != elf32_symbol_size || symbol_table.link >= sections.size() ||
            sections[symbol_table.link].type != section_strtab)
        {
            throw std::invalid_argument("RPX has an invalid ELF32 symbol table");
        }
        const LoadedSection& target_section = loaded[relocation_section.info];
        if (target_section.contents.empty())
        {
            continue;
        }
        const std::vector<std::uint8_t> relocations = read_section(file, relocation_section);
        const std::vector<std::uint8_t> symbols = read_section(file, symbol_table);
        const std::vector<std::uint8_t> strings = read_section(file, sections[symbol_table.link]);
        if (relocations.size() % elf32_rela_size != 0 || symbols.size() % elf32_symbol_size != 0)
        {
            throw std::invalid_argument("RPX has malformed inflated relocation or symbol data");
        }

        for (std::size_t offset = 0; offset < relocations.size(); offset += elf32_rela_size)
        {
            const std::uint32_t target_address = read32_be(relocations, offset);
            const std::uint32_t info = read32_be(relocations, offset + 4);
            const std::int32_t addend = static_cast<std::int32_t>(read32_be(relocations, offset + 8));
            const std::uint32_t symbol_index = info >> 8U;
            const std::uint32_t type = info & 0xFFU;
            if (type == relocation_none)
            {
                continue;
            }

            std::uint32_t symbol_value = 0;
            std::string relocation_symbol;
            if (symbol_index != 0)
            {
                const std::size_t symbol_offset =
                    static_cast<std::size_t>(symbol_index) * elf32_symbol_size;
                if (symbol_offset > symbols.size() ||
                    elf32_symbol_size > symbols.size() - symbol_offset)
                {
                    throw std::invalid_argument("RPX relocation references a missing symbol");
                }
                const std::uint16_t symbol_section = read16_be(symbols, symbol_offset + 14);
                const std::string name = symbol_name(strings, symbols, symbol_index);
                relocation_symbol = name;
                const std::uint8_t binding = symbols[symbol_offset + 12] >> 4U;
                const std::uint8_t symbol_type = symbols[symbol_offset + 12] & 0x0FU;
                const std::uint32_t symbol_entry_value = read32_be(symbols, symbol_offset + 4);
                const bool zero_valued_weak_absolute =
                    symbol_section == symbol_section_absolute && binding == symbol_binding_weak &&
                    symbol_entry_value == 0;
                const bool synthetic_undefined_symbol =
                    symbol_section == symbol_section_undefined &&
                    name == "$UNDEF" && symbol_entry_value == 0 &&
                    binding == symbol_binding_global && symbol_type == symbol_type_object;
                if (symbol_section == symbol_section_undefined || zero_valued_weak_absolute)
                {
                    if (binding != symbol_binding_weak && !synthetic_undefined_symbol)
                    {
                        throw std::invalid_argument("RPX relocation references unresolved symbol '" +
                                                    name + "'");
                    }
                    if (synthetic_undefined_symbol)
                    {
                        // WUT elf2rpl emits this global object as a synthetic
                        // zero-valued relocation symbol, not as a library import.
                        symbol_value = 0;
                    }
                    else if (symbol_type == 2U || type == relocation_rel24) // STT_FUNC or call relocation
                    {
                        // Preserve a callable address for optional weak
                        // functions. Without an HLE registration this becomes
                        // a named runtime trap if the guest actually calls it.
                        symbol_value = core.hle.bind_import("ELF weak", name);
                    }
                    // Non-function undefined weak symbols retain ELF's zero
                    // value semantics.
                }
                else if (symbol_section < sections.size() &&
                         sections[symbol_section].type == section_rpl_imports)
                {
                    const std::string library = import_library_name(file, sections[symbol_section]);
                    const std::uint32_t import_address = core.hle.bind_import(library, name);
                    symbol_value = symbol_type == 2U ? import_address : symbol_entry_value;
                    LoadedSection& import_table = loaded[symbol_section];
                    if (symbol_entry_value < import_table.address ||
                        symbol_entry_value - import_table.address > import_table.contents.size() ||
                        sizeof(std::uint32_t) >
                            import_table.contents.size() - (symbol_entry_value - import_table.address))
                    {
                        throw std::invalid_argument(
                            "RPX import symbol slot is outside its import section");
                    }
                    const std::size_t slot = symbol_entry_value - import_table.address;
                    import_table.contents[slot] = static_cast<std::uint8_t>(import_address >> 24U);
                    import_table.contents[slot + 1] = static_cast<std::uint8_t>(import_address >> 16U);
                    import_table.contents[slot + 2] = static_cast<std::uint8_t>(import_address >> 8U);
                    import_table.contents[slot + 3] = static_cast<std::uint8_t>(import_address);
                }
                else
                {
                    symbol_value = symbol_entry_value;
                }
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
            case relocation_ghs_rel16_hi:
            case relocation_ghs_rel16_lo:
            {
                if (patch_offset > target.size() || 2 > target.size() - patch_offset)
                {
                    throw std::invalid_argument("RPX GHS REL16 relocation is outside its target section");
                }
                const std::uint32_t displacement = value - target_address;
                const std::uint16_t half = static_cast<std::uint16_t>(
                    type == relocation_ghs_rel16_hi ? displacement >> 16U : displacement);
                target[patch_offset] = static_cast<std::uint8_t>(half >> 8U);
                target[patch_offset + 1] = static_cast<std::uint8_t>(half);
                break;
            }
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
                if ((target_address & 3U) != 0 || (value & 3U) != 0)
                {
                    throw std::invalid_argument("RPX REL24 relocation for '" + relocation_symbol +
                                                "' at " + std::to_string(target_address) +
                                                " to " + std::to_string(value) + " (symbol " +
                                                std::to_string(symbol_value) + ", addend " +
                                                std::to_string(addend) + ")" +
                                                " is unaligned");
                }
                const std::uint32_t instruction =
                    (static_cast<std::uint32_t>(target[patch_offset]) << 24U) |
                    (static_cast<std::uint32_t>(target[patch_offset + 1]) << 16U) |
                    (static_cast<std::uint32_t>(target[patch_offset + 2]) << 8U) |
                    target[patch_offset + 3];
                std::int64_t displacement = static_cast<std::int64_t>(value) - target_address;
                if (displacement < -0x02000000LL || displacement > 0x01FFFFFCLL)
                {
                    if ((instruction & 2U) != 0)
                    {
                        throw std::invalid_argument(
                            "RPX out-of-range REL24 with AA set cannot use a relative trampoline");
                    }
                    std::uint32_t trampoline_address = 0;
                    try
                    {
                        trampoline_address = allocate_trampoline(
                            trampoline_allocator, target_address, loaded, trampolines);
                    }
                    catch (const std::invalid_argument& error)
                    {
                        throw std::invalid_argument(
                            "RPX REL24 relocation for '" + relocation_symbol + "' at " +
                            std::to_string(target_address) + " to " + std::to_string(value) +
                            " (symbol " + std::to_string(symbol_value) + ", addend " +
                            std::to_string(addend) + "): " + error.what());
                    }

                    LoadedSection trampoline;
                    trampoline.address = trampoline_address;
                    trampoline.executable = true;
                    write_trampoline(trampoline.contents, value);
                    trampolines.push_back(std::move(trampoline));
                    displacement = static_cast<std::int64_t>(trampoline_address) - target_address;
                }
                if (displacement < -0x02000000LL || displacement > 0x01FFFFFCLL)
                {
                    throw std::invalid_argument("RPX REL24 trampoline is outside branch range");
                }
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
    RpxFileInfo file_info;
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
            if (has_file_info)
            {
                throw std::invalid_argument("RPX contains multiple Cafe FILEINFO sections");
            }
            if (section.size < 0x60)
            {
                throw std::invalid_argument("RPX v4.2 FILEINFO section is truncated");
            }
            require_range(file, section.offset, section.size);
            if (read32_be(file, section.offset) != 0xCAFE0402U)
            {
                throw std::invalid_argument("RPX FILEINFO section has an invalid Cafe magic");
            }
            file_info.text_size = read32_be(file, section.offset + 4);
            file_info.tramp_adjust = read32_be(file, section.offset + 0x20);
            file_info.tramp_addition = read32_be(file, section.offset + 0x48);
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
                static_cast<void>(import_library_name(file, section));
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
        const bool import_section = section.type == section_rpl_imports;
        if ((section.flags & section_alloc) == 0 || section.size == 0 ||
            section.type == section_rpl_fileinfo)
        {
            continue;
        }
        LoadedSection& destination = loaded[i];
        destination.address = section.address;
        destination.executable = (section.flags & section_execute) != 0;

        const std::uint64_t declared_end =
            static_cast<std::uint64_t>(destination.address) + section.size;
        if ((section.flags & section_deflated) == 0)
        {
            if (declared_end > guest_address_space_end)
            {
                throw std::invalid_argument(section_range_error(
                    i, destination.address, section.size, declared_end,
                    "exceeds the 32-bit guest address space"));
            }
            if (destination.address < rpl_loader_metadata_base &&
                declared_end > rpl_loader_metadata_base)
            {
                throw std::invalid_argument(section_range_error(
                    i, destination.address, section.size, declared_end,
                    "overlaps reserved RPL loader metadata"));
            }
            if (destination.address >= rpl_loader_metadata_base && !import_section)
            {
                // Cafe's high-address loader metadata is not mapped as a
                // regular section. Validate its range before excluding it.
                continue;
            }
        }
        destination.contents = read_section(file, section);
        const std::uint64_t end = static_cast<std::uint64_t>(destination.address) + destination.contents.size();
        if (end > guest_address_space_end)
        {
            throw std::invalid_argument(section_range_error(
                i, destination.address, destination.contents.size(), end,
                "exceeds the 32-bit guest address space"));
        }
        if (destination.address < rpl_loader_metadata_base &&
            end > rpl_loader_metadata_base)
        {
            throw std::invalid_argument(section_range_error(
                i, destination.address, destination.contents.size(), end,
                "overlaps reserved RPL loader metadata"));
        }
        if (destination.address >= rpl_loader_metadata_base && !import_section)
        {
            // Deflated metadata needs its inflated guest size checked before
            // it can safely be excluded from normal section loading.
            destination.contents.clear();
            continue;
        }
        if (destination.executable && entry_point >= destination.address &&
            static_cast<std::uint64_t>(entry_point) < end)
        {
            entry_is_executable = true;
        }
        ++loaded_count;
    }

    for (std::size_t i = 0; i < loaded.size(); ++i)
    {
        if (loaded[i].contents.empty())
        {
            continue;
        }
        const std::uint64_t first_end =
            static_cast<std::uint64_t>(loaded[i].address) + loaded[i].contents.size();
        for (std::size_t j = i + 1; j < loaded.size(); ++j)
        {
            if (loaded[j].contents.empty())
            {
                continue;
            }
            const std::uint64_t second_end =
                static_cast<std::uint64_t>(loaded[j].address) + loaded[j].contents.size();
            if (loaded[i].address < second_end && loaded[j].address < first_end)
            {
                throw std::invalid_argument(
                    section_range_error(i, loaded[i].address, loaded[i].contents.size(), first_end,
                                        "overlaps another allocated RPX section") +
                    " (section " + std::to_string(j) + ")");
            }
        }
    }
    if (loaded_count == 0 || (entry_point & 3U) != 0 || !entry_is_executable)
    {
        throw std::invalid_argument("RPX entry point is not inside a loaded executable section");
    }

    for (const LoadedSection& section : loaded)
    {
        const std::uint64_t section_end =
            static_cast<std::uint64_t>(section.address) + section.contents.size();
        if (section.contents.empty())
        {
            continue;
        }
        if (section.address < HleDispatcher::import_address_limit &&
            section_end > HleDispatcher::first_import_address)
        {
            throw std::invalid_argument("RPX section overlaps the reserved HLE import address range");
        }
    }

    std::vector<LoadedSection> trampolines;
    apply_relocations(core, file, sections, loaded, file_info, trampolines);
    std::uint64_t highest_flat_guest_address = 0;
    for (std::size_t i = 0; i < loaded.size(); ++i)
    {
        if (!loaded[i].contents.empty())
        {
            const std::uint64_t section_end =
                static_cast<std::uint64_t>(loaded[i].address) + loaded[i].contents.size();
            if (section_end > core.memory.size())
            {
                const std::uint64_t sparse_begin =
                    std::max<std::uint64_t>(loaded[i].address, core.memory.size());
                core.memory.map_region(
                    static_cast<std::uint32_t>(sparse_begin),
                    static_cast<std::size_t>(section_end - sparse_begin));
            }
            core.memory.write_bytes(loaded[i].address, loaded[i].contents);
            if (loaded[i].address < core.memory.size())
            {
                highest_flat_guest_address = std::max(
                    highest_flat_guest_address,
                    std::min<std::uint64_t>(section_end, core.memory.size()));
            }
        }
    }
    for (const LoadedSection& trampoline : trampolines)
    {
        const std::uint64_t section_end =
            static_cast<std::uint64_t>(trampoline.address) + trampoline.contents.size();
        if (section_end > core.memory.size())
        {
            const std::uint64_t sparse_begin =
                std::max<std::uint64_t>(trampoline.address, core.memory.size());
            try
            {
                core.memory.map_region(
                    static_cast<std::uint32_t>(sparse_begin),
                    static_cast<std::size_t>(section_end - sparse_begin));
            }
            catch (const std::invalid_argument& error)
            {
                throw std::invalid_argument(
                    "RPX REL24 trampoline at " + std::to_string(trampoline.address) +
                    " could not be mapped: " + error.what());
            }
        }
        try
        {
            core.memory.write_bytes(trampoline.address, trampoline.contents);
        }
        catch (const std::out_of_range& error)
        {
            throw std::invalid_argument(
                "RPX REL24 trampoline at " + std::to_string(trampoline.address) +
                " is not backed by guest memory: " + error.what());
        }
        if (trampoline.address < core.memory.size())
        {
            highest_flat_guest_address = std::max(
                highest_flat_guest_address,
                std::min<std::uint64_t>(section_end, core.memory.size()));
        }
    }
    const std::uint64_t heap_limit = core.memory.size() > 0x10000U
        ? core.memory.size() - 0x10000U
        : 0U;

    if (heap_limit > 0x1000U && heap_limit <= UINT32_MAX)
    {
        using AddressRange = std::pair<std::uint64_t, std::uint64_t>;
        std::vector<AddressRange> occupied_ranges;
        const auto add_occupied_range = [&](std::uint64_t address, std::uint64_t size) {
            const std::uint64_t end = address + size;
            const std::uint64_t begin_in_heap = std::max<std::uint64_t>(address, 0x1000U);
            const std::uint64_t end_in_heap = std::min(end, heap_limit);
            if (begin_in_heap < end_in_heap)
            {
                occupied_ranges.emplace_back(begin_in_heap, end_in_heap);
            }
        };
        for (const LoadedSection& section : loaded)
        {
            if (!section.contents.empty())
            {
                add_occupied_range(section.address, section.contents.size());
            }
        }
        for (const LoadedSection& trampoline : trampolines)
        {
            add_occupied_range(trampoline.address, trampoline.contents.size());
        }
        add_occupied_range(
            HleDispatcher::first_import_address,
            HleDispatcher::import_address_limit - HleDispatcher::first_import_address);
        std::sort(occupied_ranges.begin(), occupied_ranges.end());

        std::uint64_t selected_begin = 0;
        std::uint64_t selected_end = 0;
        const auto consider_gap = [&](std::uint64_t begin, std::uint64_t end) {
            begin = (begin + 0xFU) & ~std::uint64_t{0xFU};
            end &= ~std::uint64_t{0xFU};
            if (end > begin && end - begin > selected_end - selected_begin)
            {
                selected_begin = begin;
                selected_end = end;
            }
        };

        // Preserve the existing placement after the image whenever that gap
        // remains available; otherwise choose the largest free flat-memory gap.
        const std::uint64_t preferred_begin = std::max<std::uint64_t>(
            0x1000U, (highest_flat_guest_address + 0xFFFU) & ~std::uint64_t{0xFFFU});
        std::uint64_t preferred_end = heap_limit;
        bool preferred_is_free = preferred_begin < heap_limit;
        if (preferred_is_free)
        {
            for (const AddressRange& range : occupied_ranges)
            {
                if (range.first <= preferred_begin && preferred_begin < range.second)
                {
                    preferred_is_free = false;
                    break;
                }
                if (range.first > preferred_begin)
                {
                    preferred_end = std::min(preferred_end, range.first);
                }
            }
        }
        if (preferred_is_free)
        {
            consider_gap(preferred_begin, preferred_end);
        }

        if (selected_begin == 0)
        {
            std::uint64_t cursor = 0x1000U;
            for (const AddressRange& range : occupied_ranges)
            {
                consider_gap(cursor, range.first);
                cursor = std::max(cursor, range.second);
            }
            consider_gap(cursor, heap_limit);
        }

        if (selected_begin != 0)
        {
            core.configure_guest_heap(
                static_cast<std::uint32_t>(selected_begin),
                static_cast<std::uint32_t>(selected_end));
        }
    }
    core.state.cia = entry_point;
    return {entry_point, loaded_count};
}

}
