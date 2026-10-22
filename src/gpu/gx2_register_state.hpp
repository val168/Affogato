#pragma once

#include "cpu/espresso/gx2_hle.hpp"

#include <cstdint>
#include <cstddef>
#include <optional>
#include <span>
#include <unordered_map>

namespace affogato::gpu
{

// CPU-side shadow of Latte register writes consumed by the GPU backend.
// It intentionally stores raw register words; decoding into Vulkan objects
// belongs to later rendering stages.
class Gx2RegisterState
{
public:
    void apply(std::span<const cpu::espresso::Gx2Command> commands)
    {
        for (const cpu::espresso::Gx2Command& command : commands)
        {
            registers_[command.register_address] = command.value;
        }
    }

    [[nodiscard]] std::optional<std::uint32_t> read(
        std::uint32_t register_address) const
    {
        const auto it = registers_.find(register_address);
        if (it == registers_.end())
        {
            return std::nullopt;
        }
        return it->second;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return registers_.size();
    }

private:
    std::unordered_map<std::uint32_t, std::uint32_t> registers_;
};

}
