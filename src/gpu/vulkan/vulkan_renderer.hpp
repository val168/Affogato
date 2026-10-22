#pragma once

#include "gpu/gx2_register_state.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace affogato::gpu::vulkan
{

class VulkanRenderer
{
public:
    VulkanRenderer();
    ~VulkanRenderer();

    VulkanRenderer(const VulkanRenderer&) = delete;
    VulkanRenderer& operator=(const VulkanRenderer&) = delete;

    void poll_events();
    void render_frame();
    void set_guest_clear_color(const std::array<float, 4U>& rgba) noexcept;
    void consume_guest_commands(
        std::span<const cpu::espresso::Gx2Command> commands);

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

    void create_instance();
    void create_surface();
    void select_physical_device();
    void create_logical_device();
    void create_swapchain();
    void create_image_views();
    void create_render_pass();
    void create_framebuffers();
    void create_command_resources();
    void create_synchronization();
    void recreate_swapchain();
    void destroy_swapchain() noexcept;
    void cleanup() noexcept;

    [[nodiscard]] QueueFamilies find_queue_families(VkPhysicalDevice device) const;
    [[nodiscard]] bool supports_swapchain(VkPhysicalDevice device) const;

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
    std::vector<VkFramebuffer> framebuffers_;
    VkCommandPool command_pool_{};
    std::array<VkCommandBuffer, frames_in_flight> command_buffers_{};
    std::array<VkSemaphore, frames_in_flight> image_available_{};
    std::vector<VkSemaphore> render_finished_;
    std::array<VkFence, frames_in_flight> in_flight_{};
    std::vector<VkFence> images_in_flight_;
    std::size_t current_frame_{};
    std::array<float, 4U> guest_clear_color_{0.025F, 0.055F, 0.09F, 1.0F};
    affogato::gpu::Gx2RegisterState guest_register_state_;
    std::string device_name_;
};

}
