#pragma once

#include "gpu/gx2_register_state.hpp"
#include "gpu/latte_texture_staging.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace affogato::gpu::vulkan
{

struct DiagnosticVertex
{
    float x{};
    float y{};
};

enum class GuestDrawPolicy
{
    capture_only,
    diagnostic_float2_position,
    diagnostic_latte_vertex_position_base_zero,
    diagnostic_latte_vertex_spirv_position_base_zero,
    diagnostic_latte_vertex_fragment_spirv_position_base_zero,
    diagnostic_latte_vertex_varying_passthrough_base_zero,
};

class VulkanRenderer
{
public:
    explicit VulkanRenderer(
        bool visible = true,
        bool enable_frame_readback = false);
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void poll_events();
    void render_frame();
    void set_guest_clear_color(const std::array<float, 4U>& rgba) noexcept;
    void set_guest_diagnostic_vertices(
        std::span<const DiagnosticVertex> vertices);
    void clear_guest_diagnostic_draws() noexcept;
    [[nodiscard]] bool set_guest_scanout(Rgba8TextureStaging scanout);
    [[nodiscard]] const std::optional<Rgba8TextureStaging>& frame_readback() const noexcept
    {
        return frame_readback_;
    }
    void consume_guest_commands(
        std::span<const cpu::espresso::Gx2Command> commands,
        GuestDrawPolicy draw_policy = GuestDrawPolicy::capture_only);
    [[nodiscard]] std::vector<cpu::espresso::Gx2ColorBufferSnapshot>
    take_guest_color_buffer_updates();
    [[nodiscard]] std::size_t upload_rgba8_texture(
        const Rgba8TextureStaging& staging);
    [[nodiscard]] VkImageView uploaded_texture_view(std::size_t texture_id) const;
    [[nodiscard]] std::size_t uploaded_texture_count() const noexcept
    {
        return uploaded_textures_.size();
    }

    [[nodiscard]] bool should_close() const noexcept;
    [[nodiscard]] const std::string& device_name() const noexcept
    {
        return device_name_;
    }

private:
    static constexpr std::size_t frames_in_flight = 2U;

    struct QueueFamilies
    {
        std::uint32_t graphics{};
        std::uint32_t present{};
        bool has_graphics{};
        bool has_present{};
    };

    struct UploadedTexture
    {
        VkImage image{};
        VkDeviceMemory memory{};
        VkImageView view{};
        std::uint32_t width{};
        std::uint32_t height{};
    };

    struct ScanoutUploadBuffer
    {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        bool memory_coherent{};
    };

    struct FrameReadbackBuffer
    {
        VkBuffer buffer{};
        VkDeviceMemory memory{};
        bool memory_coherent{};
    };

    struct DiagnosticDrawRange
    {
        std::uint32_t first_vertex{};
        std::uint32_t vertex_count{};
    };

    void create_instance();
    void create_surface();
    void select_physical_device();
    void create_logical_device();
    void create_swapchain();
    void create_image_views();
    void create_render_pass();
    void create_diagnostic_pipeline();
    [[nodiscard]] VkShaderModule create_shader_module(
        std::span<const std::uint32_t> code,
        const char* operation) const;
    [[nodiscard]] VkPipeline create_diagnostic_graphics_pipeline(
        VkShaderModule vertex_module,
        VkShaderModule fragment_module,
        VkRenderPass render_pass) const;
    [[nodiscard]] VkPipeline create_latte_vertex_pipeline(
        std::span<const std::uint32_t> vertex_code) const;
    [[nodiscard]] VkPipeline create_latte_graphics_pipeline(
        std::span<const std::uint32_t> vertex_code,
        std::span<const std::uint32_t> fragment_code,
        VkRenderPass render_pass) const;
    void set_latte_vertex_pipeline(
        std::span<const std::uint32_t> vertex_code);
    void set_latte_graphics_pipeline(
        std::span<const std::uint32_t> vertex_code,
        std::span<const std::uint32_t> fragment_code);
    void create_framebuffers();
    void create_guest_color_buffer_render_pass();
    void create_command_resources();
    void create_synchronization();
    void create_scanout_upload_buffers();
    void destroy_scanout_upload_buffers() noexcept;
    void create_frame_readback_buffers();
    void destroy_frame_readback_buffers() noexcept;
    void read_frame_pixels(std::size_t frame_index);
    [[nodiscard]] cpu::espresso::Gx2ColorBufferSnapshot
    render_guest_color_buffer_diagnostic(
        const cpu::espresso::Gx2ColorBufferSnapshot& snapshot,
        std::span<const DiagnosticVertex> vertices,
        VkPipeline pipeline);
    void recreate_swapchain();
    void destroy_swapchain() noexcept;
    void destroy_uploaded_textures() noexcept;
    void cleanup() noexcept;

    [[nodiscard]] QueueFamilies find_queue_families(VkPhysicalDevice device) const;
    [[nodiscard]] bool supports_swapchain(VkPhysicalDevice device) const;
    [[nodiscard]] std::uint32_t find_memory_type(
        std::uint32_t type_bits,
        VkMemoryPropertyFlags required,
        VkMemoryPropertyFlags preferred = 0U) const;

    GLFWwindow* window_{};
    bool owns_glfw_{};
    VkInstance instance_{};
    VkDebugUtilsMessengerEXT debug_messenger_{};
    VkSurfaceKHR surface_{};
    VkPhysicalDevice physical_device_{};
    VkDevice device_{};
    VkQueue graphics_queue_{};
    VkQueue present_queue_{};
    VkSwapchainKHR swapchain_{};
    VkFormat swapchain_format_{};
    VkExtent2D swapchain_extent_{};
    std::vector<VkImage> swapchain_images_;
    std::vector<VkImageView> image_views_;
    VkRenderPass render_pass_{};
    VkRenderPass guest_color_buffer_render_pass_{};
    VkPipelineLayout diagnostic_pipeline_layout_{};
    VkPipeline diagnostic_pipeline_{};
    VkPipeline guest_color_buffer_diagnostic_pipeline_{};
    VkPipeline latte_vertex_pipeline_{};
    VkPipeline latte_guest_color_buffer_pipeline_{};
    std::vector<std::uint32_t> latte_vertex_shader_code_;
    std::vector<std::uint32_t> latte_fragment_shader_code_;
    std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool command_pool_{};
    std::array<VkCommandBuffer, frames_in_flight> command_buffers_{};
    std::array<VkSemaphore, frames_in_flight> image_available_{};
    std::vector<VkSemaphore> render_finished_;
    std::array<VkFence, frames_in_flight> in_flight_{};
    std::vector<VkFence> images_in_flight_;
    std::vector<bool> swapchain_images_initialized_;
    std::array<ScanoutUploadBuffer, frames_in_flight> scanout_upload_buffers_{};
    VkDeviceSize scanout_upload_buffer_size_{};
    std::array<FrameReadbackBuffer, frames_in_flight> frame_readback_buffers_{};
    VkDeviceSize frame_readback_buffer_size_{};
    VkBuffer diagnostic_vertex_buffer_{};
    VkDeviceMemory diagnostic_vertex_memory_{};
    bool diagnostic_vertex_memory_coherent_{};
    std::uint32_t diagnostic_vertex_count_{};
    std::vector<DiagnosticDrawRange> diagnostic_draw_ranges_;
    bool diagnostic_draw_policy_active_{};
    bool diagnostic_draw_uses_latte_vertex_pipeline_{};
    std::vector<std::uint8_t> scaled_scanout_pixels_;
    bool supports_swapchain_transfer_dst_{};
    bool supports_swapchain_transfer_src_{};
    bool frame_readback_enabled_{};
    bool guest_scanout_dirty_{};
    std::optional<Rgba8TextureStaging> guest_scanout_;
    std::optional<Rgba8TextureStaging> frame_readback_;
    std::vector<cpu::espresso::Gx2ColorBufferSnapshot>
        guest_color_buffer_updates_;
    std::vector<UploadedTexture> uploaded_textures_;
    std::size_t current_frame_{};
    std::array<float, 4U> guest_clear_color_{0.025F, 0.055F, 0.09F, 1.0F};
    affogato::gpu::Gx2RegisterState guest_register_state_;
    std::string device_name_;
};

}
