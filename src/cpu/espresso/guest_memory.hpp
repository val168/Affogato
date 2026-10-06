#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
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
        : bytes_(size)
    {
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return bytes_.size();
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
        validate_access(address, 1, GuestMemoryAccess::write);
        storage_from(address)[0] = value;
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
    }

    void zero_fill(std::uint32_t address, std::size_t length)
    {
        fill_bytes(address, length, 0);
    }

    [[nodiscard]] std::uint16_t read16_be(std::uint32_t address) const
    {
        validate_access(address, sizeof(std::uint16_t), GuestMemoryAccess::read);
        const auto high = static_cast<std::uint16_t>(read8(address));
        const auto low = static_cast<std::uint16_t>(read8(address + 1U));
        return static_cast<std::uint16_t>((high << 8U) | low);
    }

    [[nodiscard]] std::uint32_t read32_be(std::uint32_t address) const
    {
        validate_access(address, sizeof(std::uint32_t), GuestMemoryAccess::read);
        return (static_cast<std::uint32_t>(read8(address)) << 24U) |
               (static_cast<std::uint32_t>(read8(address + 1U)) << 16U) |
               (static_cast<std::uint32_t>(read8(address + 2U)) << 8U) |
               static_cast<std::uint32_t>(read8(address + 3U));
    }

    void write16_be(std::uint32_t address, std::uint16_t value)
    {
        validate_access(address, sizeof(std::uint16_t), GuestMemoryAccess::write);
        write8(address, static_cast<std::uint8_t>(value >> 8U));
        write8(address + 1U, static_cast<std::uint8_t>(value));
    }

    void write32_be(std::uint32_t address, std::uint32_t value)
    {
        validate_access(address, sizeof(std::uint32_t), GuestMemoryAccess::write);
        write8(address, static_cast<std::uint8_t>(value >> 24U));
        write8(address + 1U, static_cast<std::uint8_t>(value >> 16U));
        write8(address + 2U, static_cast<std::uint8_t>(value >> 8U));
        write8(address + 3U, static_cast<std::uint8_t>(value));
    }

private:
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
};

}
