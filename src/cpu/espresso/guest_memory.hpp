#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace affogato::cpu::espresso
{

enum class GuestMemoryAccess
{
    read,
    write,
};

class GuestMemoryFault : public std::out_of_range
{
public:
    GuestMemoryFault(std::uint32_t address, std::size_t width, GuestMemoryAccess access)
        : std::out_of_range("guest memory access is outside mapped guest memory"),
          address_(address), width_(width), access_(access)
    {
    }

    [[nodiscard]] std::uint32_t address() const noexcept { return address_; }
    [[nodiscard]] std::size_t width() const noexcept { return width_; }
    [[nodiscard]] GuestMemoryAccess access() const noexcept { return access_; }

private:
    std::uint32_t address_;
    std::size_t width_;
    GuestMemoryAccess access_;
};

class GuestMemory
{
public:
    explicit GuestMemory(std::size_t size)
        : bytes_(size),
          flat_page_write_generations_((size + guest_page_size - 1U) / guest_page_size)
    {
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return bytes_.size();
    }

    [[nodiscard]] std::uint64_t write_generation() const noexcept
    {
        return write_generation_;
    }

    [[nodiscard]] std::uint64_t page_write_generation(
        std::uint32_t address) const noexcept
    {
        const std::uint32_t page = address / guest_page_size;
        if (page < flat_page_write_generations_.size())
        {
            return flat_page_write_generations_[page];
        }
        const auto found = sparse_page_write_generations_.find(page);
        return found == sparse_page_write_generations_.end() ? 0U : found->second;
    }

    void validate_write_range(std::uint32_t address, std::size_t width) const
    {
        validate_access(address, width, GuestMemoryAccess::write);
    }

    void map_region(std::uint32_t address, std::size_t size)
    {
        constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
        if (size == 0 || size > guest_address_space_end - address ||
            overlaps(address, size, 0, bytes_.size()))
        {
            throw std::invalid_argument(
                "guest memory mapping is empty, out of range, or overlaps flat memory");
        }
        for (const MappedRegion& region : mapped_regions_)
        {
            if (overlaps(address, size, region.address, region.bytes.size()))
            {
                throw std::invalid_argument("guest memory mappings overlap");
            }
        }
        mapped_regions_.push_back({address, std::vector<std::uint8_t>(size)});
    }

    [[nodiscard]] std::uint8_t read8(std::uint32_t address) const
    {
        if (is_flat_range(address, 1U))
        {
            return bytes_[address];
        }
        validate_access(address, 1, GuestMemoryAccess::read);
        return storage_from(address)[0];
    }

    void read_bytes(std::uint32_t address, std::span<std::uint8_t> destination) const
    {
        if (destination.empty())
        {
            return;
        }
        validate_access(address, destination.size(), GuestMemoryAccess::read);
        std::size_t copied = 0;
        while (copied < destination.size())
        {
            const auto current = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(address) + copied);
            const auto source = storage_from(current);
            const std::size_t count = std::min(source.size(), destination.size() - copied);
            std::copy_n(source.begin(), count,
                        destination.begin() + static_cast<std::ptrdiff_t>(copied));
            copied += count;
        }
    }

    void write8(std::uint32_t address, std::uint8_t value)
    {
        if (is_flat_range(address, 1U))
        {
            bytes_[address] = value;
            ++write_generation_;
            mark_page_written(address / guest_page_size);
            return;
        }
        validate_access(address, 1, GuestMemoryAccess::write);
        storage_from(address)[0] = value;
        ++write_generation_;
        mark_page_written(address / guest_page_size);
    }

    void write_bytes(std::uint32_t address, std::span<const std::uint8_t> values)
    {
        validate_access(address, values.size(), GuestMemoryAccess::write);
        std::size_t copied = 0;
        while (copied < values.size())
        {
            const auto current = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(address) + copied);
            auto destination = storage_from(current);
            const std::size_t count = std::min(destination.size(), values.size() - copied);
            std::copy_n(values.begin() + static_cast<std::ptrdiff_t>(copied), count,
                        destination.begin());
            copied += count;
        }
        if (!values.empty())
        {
            ++write_generation_;
            mark_pages_written(address, values.size());
        }
    }

    // Guest ranges must not overlap, matching memcpy's contract. Both ranges
    // are validated before the destination is modified.
    void copy_bytes(std::uint32_t source, std::uint32_t destination, std::size_t length)
    {
        if (length == 0)
        {
            return;
        }
        validate_access(source, length, GuestMemoryAccess::read);
        validate_access(destination, length, GuestMemoryAccess::write);

        std::vector<std::uint8_t> temporary(length);
        read_bytes(source, temporary);
        write_bytes(destination, temporary);
    }

    // Guest ranges may overlap, matching memmove's contract. Validate both
    // complete ranges before staging and writing so faults never partially move.
    void move_bytes(std::uint32_t source, std::uint32_t destination, std::size_t length)
    {
        if (length == 0)
        {
            return;
        }
        validate_access(source, length, GuestMemoryAccess::read);
        validate_access(destination, length, GuestMemoryAccess::write);

        std::vector<std::uint8_t> temporary(length);
        read_bytes(source, temporary);
        write_bytes(destination, temporary);
    }

    void fill_bytes(std::uint32_t address, std::size_t length, std::uint8_t value)
    {
        if (length == 0)
        {
            return;
        }
        validate_access(address, length, GuestMemoryAccess::write);
        std::size_t filled = 0;
        while (filled < length)
        {
            const auto current = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(address) + filled);
            auto destination = storage_from(current);
            const std::size_t count = std::min(destination.size(), length - filled);
            std::fill_n(destination.begin(), count, value);
            filled += count;
        }
        ++write_generation_;
        mark_pages_written(address, length);
    }

    void zero_fill(std::uint32_t address, std::size_t length)
    {
        fill_bytes(address, length, 0);
    }

    [[nodiscard]] std::uint16_t read16_be(std::uint32_t address) const
    {
        if (is_flat_range(address, sizeof(std::uint16_t)))
        {
            return static_cast<std::uint16_t>(
                (static_cast<std::uint16_t>(bytes_[address]) << 8U) |
                static_cast<std::uint16_t>(bytes_[address + 1U]));
        }
        validate_access(address, sizeof(std::uint16_t), GuestMemoryAccess::read);
        const auto first_segment = storage_from(address);
        const auto high = static_cast<std::uint16_t>(first_segment[0]);
        const auto low = static_cast<std::uint16_t>(
            first_segment.size() >= sizeof(std::uint16_t)
                ? first_segment[1]
                : storage_from(address + 1U)[0]);
        return static_cast<std::uint16_t>((high << 8U) | low);
    }

    [[nodiscard]] std::uint32_t read32_be(std::uint32_t address) const
    {
        if (is_flat_range(address, sizeof(std::uint32_t)))
        {
            return (static_cast<std::uint32_t>(bytes_[address]) << 24U) |
                   (static_cast<std::uint32_t>(bytes_[address + 1U]) << 16U) |
                   (static_cast<std::uint32_t>(bytes_[address + 2U]) << 8U) |
                   static_cast<std::uint32_t>(bytes_[address + 3U]);
        }
        validate_access(address, sizeof(std::uint32_t), GuestMemoryAccess::read);
        const auto first_segment = storage_from(address);
        std::uint32_t value = 0U;
        if (first_segment.size() >= sizeof(std::uint32_t))
        {
            for (std::size_t byte = 0; byte < sizeof(std::uint32_t); ++byte)
            {
                value = (value << 8U) | first_segment[byte];
            }
            return value;
        }
        for (std::uint32_t byte = 0; byte < sizeof(std::uint32_t); ++byte)
        {
            const auto segment = byte == 0U
                ? first_segment
                : storage_from(address + byte);
            value = (value << 8U) | segment[0];
        }
        return value;
    }

    [[nodiscard]] std::uint64_t read64_be(std::uint32_t address) const
    {
        if (is_flat_range(address, sizeof(std::uint64_t)))
        {
            std::uint64_t value = 0;
            for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
            {
                value = (value << 8U) | bytes_[address + byte];
            }
            return value;
        }
        validate_access(address, sizeof(std::uint64_t), GuestMemoryAccess::read);
        const auto first_segment = storage_from(address);
        std::uint64_t value = 0;
        if (first_segment.size() >= sizeof(std::uint64_t))
        {
            for (std::size_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
            {
                value = (value << 8U) | first_segment[byte];
            }
            return value;
        }
        for (std::uint32_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
        {
            const auto segment = byte == 0U ? first_segment : storage_from(address + byte);
            value = (value << 8U) | segment[0];
        }
        return value;
    }

    void write16_be(std::uint32_t address, std::uint16_t value)
    {
        if (is_flat_range(address, sizeof(std::uint16_t)))
        {
            bytes_[address] = static_cast<std::uint8_t>(value >> 8U);
            bytes_[address + 1U] = static_cast<std::uint8_t>(value);
            ++write_generation_;
            mark_pages_written(address, sizeof(std::uint16_t));
            return;
        }
        validate_access(address, sizeof(std::uint16_t), GuestMemoryAccess::write);
        auto first_segment = storage_from(address);
        if (first_segment.size() >= sizeof(std::uint16_t))
        {
            first_segment[0] = static_cast<std::uint8_t>(value >> 8U);
            first_segment[1] = static_cast<std::uint8_t>(value);
        }
        else
        {
            first_segment[0] = static_cast<std::uint8_t>(value >> 8U);
            storage_from(address + 1U)[0] = static_cast<std::uint8_t>(value);
        }
        ++write_generation_;
        mark_pages_written(address, sizeof(std::uint16_t));
    }

    void write32_be(std::uint32_t address, std::uint32_t value)
    {
        if (is_flat_range(address, sizeof(std::uint32_t)))
        {
            bytes_[address] = static_cast<std::uint8_t>(value >> 24U);
            bytes_[address + 1U] = static_cast<std::uint8_t>(value >> 16U);
            bytes_[address + 2U] = static_cast<std::uint8_t>(value >> 8U);
            bytes_[address + 3U] = static_cast<std::uint8_t>(value);
            ++write_generation_;
            mark_pages_written(address, sizeof(std::uint32_t));
            return;
        }
        validate_access(address, sizeof(std::uint32_t), GuestMemoryAccess::write);
        auto first_segment = storage_from(address);
        if (first_segment.size() >= sizeof(std::uint32_t))
        {
            first_segment[0] = static_cast<std::uint8_t>(value >> 24U);
            first_segment[1] = static_cast<std::uint8_t>(value >> 16U);
            first_segment[2] = static_cast<std::uint8_t>(value >> 8U);
            first_segment[3] = static_cast<std::uint8_t>(value);
            ++write_generation_;
            mark_pages_written(address, sizeof(std::uint32_t));
            return;
        }
        for (std::uint32_t byte = 0; byte < sizeof(std::uint32_t); ++byte)
        {
            const auto segment = byte == 0U ? first_segment : storage_from(address + byte);
            segment[0] = static_cast<std::uint8_t>(
                value >> ((sizeof(std::uint32_t) - 1U - byte) * 8U));
        }
        ++write_generation_;
        mark_pages_written(address, sizeof(std::uint32_t));
    }

    void write64_be(std::uint32_t address, std::uint64_t value)
    {
        if (is_flat_range(address, sizeof(std::uint64_t)))
        {
            for (std::uint32_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
            {
                const unsigned shift = static_cast<unsigned>((7U - byte) * 8U);
                bytes_[address + byte] = static_cast<std::uint8_t>(value >> shift);
            }
            ++write_generation_;
            mark_pages_written(address, sizeof(std::uint64_t));
            return;
        }
        validate_access(address, sizeof(std::uint64_t), GuestMemoryAccess::write);
        auto first_segment = storage_from(address);
        if (first_segment.size() >= sizeof(std::uint64_t))
        {
            for (std::uint32_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
            {
                const unsigned shift = static_cast<unsigned>((7U - byte) * 8U);
                first_segment[byte] = static_cast<std::uint8_t>(value >> shift);
            }
            ++write_generation_;
            mark_pages_written(address, sizeof(std::uint64_t));
            return;
        }
        for (std::uint32_t byte = 0; byte < sizeof(std::uint64_t); ++byte)
        {
            const unsigned shift = static_cast<unsigned>((7U - byte) * 8U);
            const auto segment = byte == 0U ? first_segment : storage_from(address + byte);
            segment[0] = static_cast<std::uint8_t>(value >> shift);
        }
        ++write_generation_;
        mark_pages_written(address, sizeof(std::uint64_t));
    }

private:
    static constexpr std::uint32_t guest_page_size = 0x1000U;

    [[nodiscard]] bool is_flat_range(
        std::uint32_t address, std::size_t width) const noexcept
    {
        return address <= bytes_.size() && width <= bytes_.size() - address;
    }

    void mark_page_written(std::uint32_t page)
    {
        if (page < flat_page_write_generations_.size())
        {
            ++flat_page_write_generations_[page];
        }
        else
        {
            ++sparse_page_write_generations_[page];
        }
    }

    void mark_pages_written(std::uint32_t address, std::size_t width)
    {
        const std::uint64_t first_page = address / guest_page_size;
        const std::uint64_t last_page =
            (static_cast<std::uint64_t>(address) + width - 1U) / guest_page_size;
        for (std::uint64_t page = first_page; page <= last_page; ++page)
        {
            mark_page_written(static_cast<std::uint32_t>(page));
        }
    }

    struct MappedRegion
    {
        std::uint32_t address{};
        std::vector<std::uint8_t> bytes;
    };

    [[nodiscard]] static bool overlaps(
        std::uint32_t address,
        std::size_t size,
        std::uint32_t other_address,
        std::size_t other_size) noexcept
    {
        const std::uint64_t end = static_cast<std::uint64_t>(address) + size;
        const std::uint64_t other_end = static_cast<std::uint64_t>(other_address) + other_size;
        return address < other_end && other_address < end;
    }

    void validate_access(
        std::uint32_t address,
        std::size_t width,
        GuestMemoryAccess access) const
    {
        constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
        if (width > guest_address_space_end - address)
        {
            throw GuestMemoryFault(address, width, access);
        }

        if (width == 0)
        {
            if (address < bytes_.size())
            {
                return;
            }
            for (const MappedRegion& region : mapped_regions_)
            {
                if (address >= region.address &&
                    static_cast<std::uint64_t>(address) - region.address <= region.bytes.size())
                {
                    return;
                }
            }
            throw GuestMemoryFault(address, width, access);
        }

        std::uint64_t current = address;
        std::size_t remaining = width;
        while (remaining != 0)
        {
            std::span<const std::uint8_t> segment;
            try
            {
                segment = storage_from(static_cast<std::uint32_t>(current));
            }
            catch (const std::out_of_range&)
            {
                throw GuestMemoryFault(address, width, access);
            }
            const std::size_t count = std::min(segment.size(), remaining);
            if (count == 0)
            {
                throw GuestMemoryFault(address, width, access);
            }
            current += count;
            remaining -= count;
        }
    }

    [[nodiscard]] std::span<std::uint8_t> storage_from(std::uint32_t address)
    {
        if (address < bytes_.size())
        {
            return std::span<std::uint8_t>(bytes_).subspan(address);
        }
        for (MappedRegion& region : mapped_regions_)
        {
            if (address >= region.address)
            {
                const std::size_t offset = static_cast<std::size_t>(address - region.address);
                if (offset < region.bytes.size())
                {
                    return std::span<std::uint8_t>(region.bytes).subspan(offset);
                }
            }
        }
        throw std::out_of_range("guest memory access is outside mapped guest memory");
    }

    [[nodiscard]] std::span<const std::uint8_t> storage_from(std::uint32_t address) const
    {
        if (address < bytes_.size())
        {
            return std::span<const std::uint8_t>(bytes_).subspan(address);
        }
        for (const MappedRegion& region : mapped_regions_)
        {
            if (address >= region.address)
            {
                const std::size_t offset = static_cast<std::size_t>(address - region.address);
                if (offset < region.bytes.size())
                {
                    return std::span<const std::uint8_t>(region.bytes).subspan(offset);
                }
            }
        }
        throw std::out_of_range("guest memory access is outside mapped guest memory");
    }

    std::vector<std::uint8_t> bytes_;
    std::vector<MappedRegion> mapped_regions_;
    std::vector<std::uint64_t> flat_page_write_generations_;
    std::unordered_map<std::uint32_t, std::uint64_t> sparse_page_write_generations_;
    std::uint64_t write_generation_{};
};

}
