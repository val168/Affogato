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

    [[nodiscard]] std::uint8_t read8(std::uint32_t address) const
    {
        return bytes_[checked_offset(address, 1)];
    }

    void write8(std::uint32_t address, std::uint8_t value)
    {
        bytes_[checked_offset(address, 1)] = value;
    }

    void write_bytes(std::uint32_t address, std::span<const std::uint8_t> values)
    {
        const std::size_t offset = checked_offset(address, values.size());
        std::copy(values.begin(), values.end(), bytes_.begin() + offset);
    }

    void zero_fill(std::uint32_t address, std::size_t length)
    {
        const std::size_t offset = checked_offset(address, length);
        std::fill_n(bytes_.begin() + offset, length, std::uint8_t{});
    }

    [[nodiscard]] std::uint16_t read16_be(std::uint32_t address) const
    {
        const std::size_t offset = checked_offset(address, sizeof(std::uint16_t));

        return (static_cast<std::uint16_t>(bytes_[offset]) << 8U) |
               static_cast<std::uint16_t>(bytes_[offset + 1]);
    }

    [[nodiscard]] std::uint32_t read32_be(std::uint32_t address) const
    {
        const std::size_t offset = checked_offset(address, sizeof(std::uint32_t));

        return (static_cast<std::uint32_t>(bytes_[offset]) << 24U) |
               (static_cast<std::uint32_t>(bytes_[offset + 1]) << 16U) |
               (static_cast<std::uint32_t>(bytes_[offset + 2]) << 8U) |
               static_cast<std::uint32_t>(bytes_[offset + 3]);
    }

    void write16_be(std::uint32_t address, std::uint16_t value)
    {
        const std::size_t offset = checked_offset(address, sizeof(std::uint16_t));

        bytes_[offset] = static_cast<std::uint8_t>(value >> 8U);
        bytes_[offset + 1] = static_cast<std::uint8_t>(value);
    }

    void write32_be(std::uint32_t address, std::uint32_t value)
    {
        const std::size_t offset = checked_offset(address, sizeof(std::uint32_t));

        bytes_[offset] = static_cast<std::uint8_t>(value >> 24U);
        bytes_[offset + 1] = static_cast<std::uint8_t>(value >> 16U);
        bytes_[offset + 2] = static_cast<std::uint8_t>(value >> 8U);
        bytes_[offset + 3] = static_cast<std::uint8_t>(value);
    }

private:
    [[nodiscard]] std::size_t checked_offset(
        std::uint32_t address,
        std::size_t width) const
    {
        const std::size_t offset = address;

        if (offset > bytes_.size() || width > bytes_.size() - offset)
        {
            throw std::out_of_range("guest memory access is outside the guest address space");
        }

        return offset;
    }

    std::vector<std::uint8_t> bytes_;
};

}
