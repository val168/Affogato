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
        if (size == 0 || overlaps(address, size, 0, bytes_.size()))
        {
            throw std::invalid_argument("guest memory mapping is empty or overlaps flat memory");
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
        return storage_for(address, 1)[0];
    }

    void write8(std::uint32_t address, std::uint8_t value)
    {
        storage_for(address, 1)[0] = value;
    }

    void write_bytes(std::uint32_t address, std::span<const std::uint8_t> values)
    {
        auto destination = storage_for(address, values.size());
        std::copy(values.begin(), values.end(), destination.begin());
    }

    void zero_fill(std::uint32_t address, std::size_t length)
    {
        auto destination = storage_for(address, length);
        std::fill(destination.begin(), destination.end(), std::uint8_t{});
    }

    [[nodiscard]] std::uint16_t read16_be(std::uint32_t address) const
    {
        const auto bytes = storage_for(address, sizeof(std::uint16_t));
        return (static_cast<std::uint16_t>(bytes[0]) << 8U) |
               static_cast<std::uint16_t>(bytes[1]);
    }

    [[nodiscard]] std::uint32_t read32_be(std::uint32_t address) const
    {
        const auto bytes = storage_for(address, sizeof(std::uint32_t));
        return (static_cast<std::uint32_t>(bytes[0]) << 24U) |
               (static_cast<std::uint32_t>(bytes[1]) << 16U) |
               (static_cast<std::uint32_t>(bytes[2]) << 8U) |
               static_cast<std::uint32_t>(bytes[3]);
    }

    void write16_be(std::uint32_t address, std::uint16_t value)
    {
        auto bytes = storage_for(address, sizeof(std::uint16_t));
        bytes[0] = static_cast<std::uint8_t>(value >> 8U);
        bytes[1] = static_cast<std::uint8_t>(value);
    }

    void write32_be(std::uint32_t address, std::uint32_t value)
    {
        auto bytes = storage_for(address, sizeof(std::uint32_t));
        bytes[0] = static_cast<std::uint8_t>(value >> 24U);
        bytes[1] = static_cast<std::uint8_t>(value >> 16U);
        bytes[2] = static_cast<std::uint8_t>(value >> 8U);
        bytes[3] = static_cast<std::uint8_t>(value);
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

    [[nodiscard]] std::span<std::uint8_t> storage_for(std::uint32_t address, std::size_t width)
    {
        if (address <= bytes_.size() && width <= bytes_.size() - address)
        {
            return std::span<std::uint8_t>(bytes_).subspan(address, width);
        }
        for (MappedRegion& region : mapped_regions_)
        {
            if (address >= region.address)
            {
                const std::size_t offset = static_cast<std::size_t>(address - region.address);
                if (offset <= region.bytes.size() && width <= region.bytes.size() - offset)
                {
                    return std::span<std::uint8_t>(region.bytes).subspan(offset, width);
                }
            }
        }
        throw std::out_of_range("guest memory access is outside mapped guest memory");
    }

    [[nodiscard]] std::span<const std::uint8_t> storage_for(
        std::uint32_t address,
        std::size_t width) const
    {
        if (address <= bytes_.size() && width <= bytes_.size() - address)
        {
            return std::span<const std::uint8_t>(bytes_).subspan(address, width);
        }
        for (const MappedRegion& region : mapped_regions_)
        {
            if (address >= region.address)
            {
                const std::size_t offset = static_cast<std::size_t>(address - region.address);
                if (offset <= region.bytes.size() && width <= region.bytes.size() - offset)
                {
                    return std::span<const std::uint8_t>(region.bytes).subspan(offset, width);
                }
            }
        }
        throw std::out_of_range("guest memory access is outside mapped guest memory");
    }

    std::vector<std::uint8_t> bytes_;
    std::vector<MappedRegion> mapped_regions_;
};

}
