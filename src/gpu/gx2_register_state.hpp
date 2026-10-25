#pragma once

#include "cpu/espresso/gx2_hle.hpp"
#include "gpu/latte_sampler.hpp"
#include "gpu/latte_texture_resource.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <unordered_map>
#include <vector>

namespace affogato::gpu
{

enum class Gx2ShaderStage
{
    pixel,
    vertex,
    geometry,
    compute,
};

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
            if (command.type == cpu::espresso::Gx2CommandType::gpu_fence)
            {
                // Retain the ordered GPU wait parameters separately from
                // Latte register state until command execution is modeled.
                gpu_fences_.push_back(command);
                continue;
            }
            if (command.type == cpu::espresso::Gx2CommandType::scan_buffer_swap)
            {
                // Retain swap requests in stream order until scan-buffer
                // presentation is connected to the renderer.
                scan_buffer_swaps_.push_back(command);
                continue;
            }
            if (command.type == cpu::espresso::Gx2CommandType::draw)
            {
                // Preserve draw-call order and arguments for the future Latte
                // command translator; recording is not GPU execution.
                draw_calls_.push_back(command);
                draw_vertex_buffers_.push_back(vertex_buffers_);
                continue;
            }
            if (command.type == cpu::espresso::Gx2CommandType::set_vertex_buffer)
            {
                if (command.vertex_buffer &&
                    command.vertex_buffer->index < vertex_buffers_.size())
                {
                    vertex_buffers_[command.vertex_buffer->index] =
                        command.vertex_buffer;
                }
                continue;
            }
            if (command.type == cpu::espresso::Gx2CommandType::flush ||
                command.type == cpu::espresso::Gx2CommandType::draw_done)
            {
                // Submission/synchronization boundaries preserve command
                // ordering but do not themselves write a Latte register.
                continue;
            }
            registers_[command.register_address] = {command.type, command.value};
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
        return it->second.value;
    }

    // Return a binding only after all words have arrived. Compute resources
    // share Cafe's vertex-stage resource register bank.
    [[nodiscard]] std::optional<LatteTextureResourceWords> texture_resource(
        Gx2ShaderStage stage,
        std::uint32_t unit) const
    {
        constexpr std::uint32_t words_per_resource = 7U;
        constexpr std::uint32_t max_units = 16U;
        if (unit >= max_units)
        {
            return std::nullopt;
        }

        std::uint32_t base{};
        switch (stage)
        {
        case Gx2ShaderStage::pixel: base = 0x00038000U; break;
        case Gx2ShaderStage::vertex:
        case Gx2ShaderStage::compute: base = 0x00039180U; break;
        case Gx2ShaderStage::geometry: base = 0x0003A4C0U; break;
        default: return std::nullopt;
        }

        LatteTextureResourceWords result{};
        const std::uint32_t first =
            base + unit * words_per_resource * sizeof(std::uint32_t);
        for (std::uint32_t index = 0U; index < words_per_resource; ++index)
        {
            const auto found = registers_.find(first + index * sizeof(std::uint32_t));
            if (found == registers_.end() ||
                found->second.type != cpu::espresso::Gx2CommandType::set_shader_resource)
            {
                return std::nullopt;
            }
            result[index] = found->second.value;
        }
        return result;
    }

    [[nodiscard]] std::optional<LatteSamplerWords> sampler(
        Gx2ShaderStage stage,
        std::uint32_t unit) const
    {
        constexpr std::uint32_t words_per_sampler = 3U;
        constexpr std::uint32_t max_units = 16U;
        if (unit >= max_units)
        {
            return std::nullopt;
        }

        std::uint32_t base{};
        switch (stage)
        {
        case Gx2ShaderStage::pixel: base = 0x0003C000U; break;
        case Gx2ShaderStage::vertex:
        case Gx2ShaderStage::compute: base = 0x0003C0D8U; break;
        case Gx2ShaderStage::geometry: base = 0x0003C1B0U; break;
        default: return std::nullopt;
        }

        LatteSamplerWords result{};
        const std::uint32_t first =
            base + unit * words_per_sampler * sizeof(std::uint32_t);
        for (std::uint32_t index = 0U; index < words_per_sampler; ++index)
        {
            const auto found = registers_.find(first + index * sizeof(std::uint32_t));
            if (found == registers_.end() ||
                found->second.type != cpu::espresso::Gx2CommandType::set_shader_sampler)
            {
                return std::nullopt;
            }
            result[index] = found->second.value;
        }
        return result;
    }

    [[nodiscard]] std::optional<LatteTextureResourceDescriptor>
    texture_descriptor(Gx2ShaderStage stage, std::uint32_t unit) const
    {
        const auto words = texture_resource(stage, unit);
        return words ? decode_latte_texture_resource(*words) : std::nullopt;
    }

    [[nodiscard]] std::optional<LatteSamplerDescriptor> sampler_descriptor(
        Gx2ShaderStage stage,
        std::uint32_t unit) const
    {
        const auto words = sampler(stage, unit);
        return words ? decode_latte_sampler(*words) : std::nullopt;
    }

    [[nodiscard]] std::size_t size() const noexcept
    {
        return registers_.size();
    }

    [[nodiscard]] std::span<const cpu::espresso::Gx2Command> gpu_fences() const noexcept
    {
        return gpu_fences_;
    }

    [[nodiscard]] std::span<const cpu::espresso::Gx2Command>
    scan_buffer_swaps() const noexcept
    {
        return scan_buffer_swaps_;
    }

    [[nodiscard]] std::span<const cpu::espresso::Gx2Command>
    draw_calls() const noexcept
    {
        return draw_calls_;
    }

    [[nodiscard]] const std::array<
        std::optional<cpu::espresso::Gx2VertexBufferBinding>, 16U>&
    vertex_buffers() const noexcept
    {
        return vertex_buffers_;
    }

    [[nodiscard]] std::span<const std::array<
        std::optional<cpu::espresso::Gx2VertexBufferBinding>, 16U>>
    draw_vertex_buffers() const noexcept
    {
        return draw_vertex_buffers_;
    }

private:
    struct RegisterValue
    {
        cpu::espresso::Gx2CommandType type{};
        std::uint32_t value{};
    };

    std::unordered_map<std::uint32_t, RegisterValue> registers_;
    std::vector<cpu::espresso::Gx2Command> gpu_fences_;
    std::vector<cpu::espresso::Gx2Command> scan_buffer_swaps_;
    std::vector<cpu::espresso::Gx2Command> draw_calls_;
    std::array<std::optional<cpu::espresso::Gx2VertexBufferBinding>, 16U>
        vertex_buffers_{};
    std::vector<std::array<
        std::optional<cpu::espresso::Gx2VertexBufferBinding>, 16U>>
        draw_vertex_buffers_;
};

}
