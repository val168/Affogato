#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

namespace affogato::cpu::espresso
{

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
        validate_access(address, 1);
        return storage_from(address)[0];
    }

    void write8(std::uint32_t address, std::uint8_t value)
    {
        validate_access(address, 1);
        storage_from(address)[0] = value;
    }

    void write_bytes(std::uint32_t address, std::span<const std::uint8_t> values)
    {
        validate_access(address, values.size());
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

    void zero_fill(std::uint32_t address, std::size_t length)
    {
        validate_access(address, length);
        std::size_t filled = 0;
        while (filled < length)
        {
            const auto current = static_cast<std::uint32_t>(
                static_cast<std::uint64_t>(address) + filled);
            auto destination = storage_from(current);
            const std::size_t count = std::min(destination.size(), length - filled);
            std::fill_n(destination.begin(), count, std::uint8_t{});
            filled += count;
        }
    }

    [[nodiscard]] std::uint16_t read16_be(std::uint32_t address) const
    {
        validate_access(address, sizeof(std::uint16_t));
        const auto high = static_cast<std::uint16_t>(read8(address));
        const auto low = static_cast<std::uint16_t>(read8(address + 1U));
        return static_cast<std::uint16_t>((high << 8U) | low);
    }

    [[nodiscard]] std::uint32_t read32_be(std::uint32_t address) const
    {
        validate_access(address, sizeof(std::uint32_t));
        return (static_cast<std::uint32_t>(read8(address)) << 24U) |
               (static_cast<std::uint32_t>(read8(address + 1U)) << 16U) |
               (static_cast<std::uint32_t>(read8(address + 2U)) << 8U) |
               static_cast<std::uint32_t>(read8(address + 3U));
    }

    void write16_be(std::uint32_t address, std::uint16_t value)
    {
        validate_access(address, sizeof(std::uint16_t));
        write8(address, static_cast<std::uint8_t>(value >> 8U));
        write8(address + 1U, static_cast<std::uint8_t>(value));
    }

    void write32_be(std::uint32_t address, std::uint32_t value)
    {
        validate_access(address, sizeof(std::uint32_t));
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

    void validate_access(std::uint32_t address, std::size_t width) const
    {
        constexpr std::uint64_t guest_address_space_end = std::uint64_t{1} << 32U;
        if (width > guest_address_space_end - address)
        {
            throw std::out_of_range("guest memory access exceeds the 32-bit address space");
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
            throw std::out_of_range("guest memory access is outside mapped guest memory");
        }

        std::uint64_t current = address;
        std::size_t remaining = width;
        while (remaining != 0)
        {
            const auto segment = storage_from(static_cast<std::uint32_t>(current));
            const std::size_t count = std::min(segment.size(), remaining);
            if (count == 0)
            {
                throw std::out_of_range("guest memory access is outside mapped guest memory");
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
