#include "gpu/vulkan/vulkan_renderer.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace affogato::gpu::vulkan
{
namespace
{

constexpr const char* window_title = "Affogato — Vulkan output (clear test)";
constexpr std::array<const char*, 1U> device_extensions{
    VK_KHR_SWAPCHAIN_EXTENSION_NAME,
};

template <typename T>
T make_info(VkStructureType type)
{
    T result{};
    result.sType = type;
    return result;
}

[[noreturn]] void fail(const char* operation, VkResult result)
{
    throw std::runtime_error(
        std::string(operation) + " failed with VkResult " + std::to_string(result));
}

void check(VkResult result, const char* operation)
{
    if (result != VK_SUCCESS)
    {
        fail(operation, result);
    }
}

bool has_instance_layer(const char* name)
{
    std::uint32_t count = 0U;
    if (vkEnumerateInstanceLayerProperties(&count, nullptr) != VK_SUCCESS)
    {
        return false;
    }
    std::vector<VkLayerProperties> layers(count);
    if (vkEnumerateInstanceLayerProperties(&count, layers.data()) != VK_SUCCESS)
    {
        return false;
    }
    return std::ranges::any_of(layers, [name](const auto& layer)
    {
        return std::strcmp(layer.layerName, name) == 0;
    });
}

bool has_instance_extension(const char* name)
{
    std::uint32_t count = 0U;
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr) != VK_SUCCESS)
    {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data()) != VK_SUCCESS)
    {
        return false;
    }
    return std::ranges::any_of(extensions, [name](const auto& extension)
    {
        return std::strcmp(extension.extensionName, name) == 0;
    });
}

VKAPI_ATTR VkBool32 VKAPI_CALL validation_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* callback_data,
    void*)
{
    if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) != 0U)
    {
        std::fprintf(stderr, "Vulkan validation error: %s\n", callback_data->pMessage);
    }
    else if ((severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) != 0U)
    {
        std::fprintf(stderr, "Vulkan validation warning: %s\n", callback_data->pMessage);
    }
    return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT debug_messenger_info()
{
    VkDebugUtilsMessengerCreateInfoEXT info = make_info<VkDebugUtilsMessengerCreateInfoEXT>(
        VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT);
    info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
        VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    info.pfnUserCallback = validation_callback;
    return info;
}

}

VulkanRenderer::VulkanRenderer()
{
    if (!glfwInit())
    {
        throw std::runtime_error("GLFW initialization failed");
    }
    owns_glfw_ = true;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(1280, 720, window_title, nullptr, nullptr);
    if (window_ == nullptr)
    {
        cleanup();
        throw std::runtime_error("Could not create the Vulkan output window");
    }

    try
    {
        create_instance();
        create_surface();
        select_physical_device();
        create_logical_device();
        create_swapchain();
        create_image_views();
        create_render_pass();
        create_framebuffers();
        create_command_resources();
        create_synchronization();
    }
    catch (...)
    {
        cleanup();
        throw;
    }
}

VulkanRenderer::~VulkanRenderer()
{
    cleanup();
}

void VulkanRenderer::poll_events()
{
    glfwPollEvents();
}

bool VulkanRenderer::should_close() const noexcept
{
    return window_ == nullptr || glfwWindowShouldClose(window_) == GLFW_TRUE;
}

void VulkanRenderer::set_guest_clear_color(
    const std::array<float, 4U>& rgba) noexcept
{
    guest_clear_color_ = rgba;
}

VulkanRenderer::QueueFamilies VulkanRenderer::find_queue_families(VkPhysicalDevice device) const
{
    std::uint32_t count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> families(count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &count, families.data());

    QueueFamilies result{};
    for (std::uint32_t index = 0U; index < count; ++index)
    {
        if ((families[index].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U)
        {
            result.graphics = index;
            result.has_graphics = true;
        }
        VkBool32 present_supported = VK_FALSE;
        if (vkGetPhysicalDeviceSurfaceSupportKHR(device, index, surface_, &present_supported) ==
                VK_SUCCESS && present_supported == VK_TRUE)
        {
            result.present = index;
            result.has_present = true;
        }
        if (result.has_graphics && result.has_present)
        {
            break;
        }
    }
    return result;
}

bool VulkanRenderer::supports_swapchain(VkPhysicalDevice device) const
{
    std::uint32_t count = 0U;
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr) != VK_SUCCESS)
    {
        return false;
    }
    std::vector<VkExtensionProperties> extensions(count);
    if (vkEnumerateDeviceExtensionProperties(device, nullptr, &count, extensions.data()) != VK_SUCCESS)
    {
        return false;
    }
    return std::ranges::any_of(extensions, [](const auto& extension)
    {
        return std::strcmp(extension.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
    });
}

void VulkanRenderer::create_instance()
{
    std::uint32_t glfw_extension_count = 0U;
    const char** glfw_extensions = glfwGetRequiredInstanceExtensions(&glfw_extension_count);
    if (glfw_extensions == nullptr || glfw_extension_count == 0U)
    {
        throw std::runtime_error("GLFW did not provide Vulkan surface extensions");
    }
    std::vector<const char*> extensions(glfw_extensions, glfw_extensions + glfw_extension_count);

    std::vector<const char*> layers;
#ifndef NDEBUG
    constexpr const char* validation_layer = "VK_LAYER_KHRONOS_validation";
    if (has_instance_layer(validation_layer))
    {
        layers.push_back(validation_layer);
    }
#endif
    const bool enable_debug_utils = has_instance_extension(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    if (enable_debug_utils)
    {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    VkApplicationInfo application = make_info<VkApplicationInfo>(VK_STRUCTURE_TYPE_APPLICATION_INFO);
    application.pApplicationName = "Affogato";
    application.applicationVersion = VK_MAKE_API_VERSION(0U, 0U, 1U, 0U);
    application.pEngineName = "Affogato";
    application.engineVersion = VK_MAKE_API_VERSION(0U, 0U, 1U, 0U);
    // Keep the backend compatible with Vulkan 1.0 implementations for now;
    // this first path uses render passes and core semaphores/fences only.
    application.apiVersion = VK_API_VERSION_1_0;

    VkDebugUtilsMessengerCreateInfoEXT debug_info = debug_messenger_info();
    VkInstanceCreateInfo create_info = make_info<VkInstanceCreateInfo>(
        VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO);
    create_info.pApplicationInfo = &application;
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount = static_cast<std::uint32_t>(layers.size());
    create_info.ppEnabledLayerNames = layers.data();
    if (enable_debug_utils && !layers.empty())
    {
        create_info.pNext = &debug_info;
    }
    check(vkCreateInstance(&create_info, nullptr, &instance_), "vkCreateInstance");

    if (enable_debug_utils && !layers.empty())
    {
        const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (create_messenger != nullptr)
        {
            check(create_messenger(instance_, &debug_info, nullptr, &debug_messenger_),
                "vkCreateDebugUtilsMessengerEXT");
        }
    }
}

void VulkanRenderer::create_surface()
{
    check(glfwCreateWindowSurface(instance_, window_, nullptr, &surface_),
        "glfwCreateWindowSurface");
}

void VulkanRenderer::select_physical_device()
{
    std::uint32_t count = 0U;
    check(vkEnumeratePhysicalDevices(instance_, &count, nullptr), "vkEnumeratePhysicalDevices");
    if (count == 0U)
    {
        throw std::runtime_error("No Vulkan physical devices are available");
    }
    std::vector<VkPhysicalDevice> devices(count);
    check(vkEnumeratePhysicalDevices(instance_, &count, devices.data()), "vkEnumeratePhysicalDevices");

    int best_score = std::numeric_limits<int>::min();
    for (VkPhysicalDevice candidate : devices)
    {
        const QueueFamilies families = find_queue_families(candidate);
        if (!families.has_graphics || !families.has_present || !supports_swapchain(candidate))
        {
            continue;
        }

        std::uint32_t format_count = 0U;
        std::uint32_t mode_count = 0U;
        if (vkGetPhysicalDeviceSurfaceFormatsKHR(candidate, surface_, &format_count, nullptr) !=
                VK_SUCCESS || format_count == 0U ||
            vkGetPhysicalDeviceSurfacePresentModesKHR(candidate, surface_, &mode_count, nullptr) !=
                VK_SUCCESS || mode_count == 0U)
        {
            continue;
        }

        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(candidate, &properties);
        const int score = properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ? 100 :
            (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU ? 50 : 0);
        if (score > best_score)
        {
            best_score = score;
            physical_device_ = candidate;
            device_name_ = properties.deviceName;
        }
    }
    if (physical_device_ == VK_NULL_HANDLE)
    {
        throw std::runtime_error("No Vulkan device supports graphics and presentation");
    }
}

void VulkanRenderer::create_logical_device()
{
    const QueueFamilies families = find_queue_families(physical_device_);
    const std::set<std::uint32_t> unique_families{families.graphics, families.present};
    constexpr float queue_priority = 1.0F;
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    for (const std::uint32_t family : unique_families)
    {
        VkDeviceQueueCreateInfo queue_info = make_info<VkDeviceQueueCreateInfo>(
            VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO);
        queue_info.queueFamilyIndex = family;
        queue_info.queueCount = 1U;
        queue_info.pQueuePriorities = &queue_priority;
        queue_infos.push_back(queue_info);
    }

    VkPhysicalDeviceFeatures features{};
    VkDeviceCreateInfo create_info = make_info<VkDeviceCreateInfo>(
        VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO);
    create_info.queueCreateInfoCount = static_cast<std::uint32_t>(queue_infos.size());
    create_info.pQueueCreateInfos = queue_infos.data();
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(device_extensions.size());
    create_info.ppEnabledExtensionNames = device_extensions.data();
    create_info.pEnabledFeatures = &features;
    check(vkCreateDevice(physical_device_, &create_info, nullptr, &device_), "vkCreateDevice");
    vkGetDeviceQueue(device_, families.graphics, 0U, &graphics_queue_);
    vkGetDeviceQueue(device_, families.present, 0U, &present_queue_);
}

void VulkanRenderer::create_swapchain()
{
    VkSurfaceCapabilitiesKHR capabilities{};
    check(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_device_, surface_, &capabilities),
        "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");

    std::uint32_t format_count = 0U;
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, nullptr),
        "vkGetPhysicalDeviceSurfaceFormatsKHR");
    std::vector<VkSurfaceFormatKHR> formats(format_count);
    check(vkGetPhysicalDeviceSurfaceFormatsKHR(physical_device_, surface_, &format_count, formats.data()),
        "vkGetPhysicalDeviceSurfaceFormatsKHR");
    if (formats.empty()) throw std::runtime_error("Surface exposes no formats");
    const auto preferred_format = std::ranges::find_if(formats, [](const auto& format)
    {
        return format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });
    const VkSurfaceFormatKHR surface_format = preferred_format != formats.end()
        ? *preferred_format
        : (formats.size() == 1U && formats.front().format == VK_FORMAT_UNDEFINED
            ? VkSurfaceFormatKHR{VK_FORMAT_B8G8R8A8_UNORM, formats.front().colorSpace}
            : formats.front());
    swapchain_format_ = surface_format.format;

    std::uint32_t mode_count = 0U;
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device_, surface_, &mode_count, nullptr), "vkGetPhysicalDeviceSurfacePresentModesKHR");
    std::vector<VkPresentModeKHR> modes(mode_count);
    check(vkGetPhysicalDeviceSurfacePresentModesKHR(
        physical_device_, surface_, &mode_count, modes.data()), "vkGetPhysicalDeviceSurfacePresentModesKHR");
    const VkPresentModeKHR present_mode = VK_PRESENT_MODE_FIFO_KHR;

    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        swapchain_extent_ = capabilities.currentExtent;
    }

    if ((capabilities.supportedUsageFlags & VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT) == 0U)
    {
        throw std::runtime_error("Surface images cannot be used as color attachments");
    }
    else
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window_, &width, &height);
        swapchain_extent_.width = std::clamp(static_cast<std::uint32_t>(std::max(width, 1)),
            capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        swapchain_extent_.height = std::clamp(static_cast<std::uint32_t>(std::max(height, 1)),
            capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    std::uint32_t image_count = capabilities.minImageCount + 1U;
    if (capabilities.maxImageCount != 0U && image_count > capabilities.maxImageCount)
    {
        image_count = capabilities.maxImageCount;
    }
    const QueueFamilies families = find_queue_families(physical_device_);
    const std::array<std::uint32_t, 2U> queue_indices{families.graphics, families.present};
    VkSwapchainCreateInfoKHR create_info = make_info<VkSwapchainCreateInfoKHR>(
        VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR);
    create_info.surface = surface_;
    create_info.minImageCount = image_count;
    create_info.imageFormat = surface_format.format;
    create_info.imageColorSpace = surface_format.colorSpace;
    create_info.imageExtent = swapchain_extent_;
    create_info.imageArrayLayers = 1U;
    create_info.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (families.graphics != families.present)
    {
        create_info.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = static_cast<std::uint32_t>(queue_indices.size());
        create_info.pQueueFamilyIndices = queue_indices.data();
    }
    else
    {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    create_info.preTransform = capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if ((capabilities.supportedCompositeAlpha & create_info.compositeAlpha) == 0U)
    {
        constexpr std::array<VkCompositeAlphaFlagBitsKHR, 4U> choices{
            VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR,
            VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR,
            VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR};
        const auto choice = std::ranges::find_if(choices, [&capabilities](const auto value)
        {
            return (capabilities.supportedCompositeAlpha & value) != 0U;
        });
        if (choice == choices.end()) throw std::runtime_error("Surface exposes no composite alpha mode");
        create_info.compositeAlpha = *choice;
    }
    create_info.presentMode = present_mode;
    create_info.clipped = VK_TRUE;
    create_info.oldSwapchain = VK_NULL_HANDLE;
    check(vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_), "vkCreateSwapchainKHR");

    std::uint32_t actual_count = 0U;
    check(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count, nullptr), "vkGetSwapchainImagesKHR");
    swapchain_images_.resize(actual_count);
    check(vkGetSwapchainImagesKHR(device_, swapchain_, &actual_count, swapchain_images_.data()),
        "vkGetSwapchainImagesKHR");
    images_in_flight_.assign(actual_count, VK_NULL_HANDLE);

    render_finished_.resize(actual_count, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkSemaphore& semaphore : render_finished_)
    {
        check(vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore),
            "vkCreateSemaphore(present ready)");
    }
}

void VulkanRenderer::create_image_views()
{
    image_views_.resize(swapchain_images_.size());
    for (std::size_t index = 0U; index < swapchain_images_.size(); ++index)
    {
        VkImageViewCreateInfo create_info = make_info<VkImageViewCreateInfo>(
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        create_info.image = swapchain_images_[index];
        create_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        create_info.format = swapchain_format_;
        create_info.components = {
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY,
            VK_COMPONENT_SWIZZLE_IDENTITY};
        create_info.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        create_info.subresourceRange.baseMipLevel = 0U;
        create_info.subresourceRange.levelCount = 1U;
        create_info.subresourceRange.baseArrayLayer = 0U;
        create_info.subresourceRange.layerCount = 1U;
        check(vkCreateImageView(device_, &create_info, nullptr, &image_views_[index]),
            "vkCreateImageView");
    }
}

void VulkanRenderer::create_render_pass()
{
    VkAttachmentDescription attachment{};
    attachment.format = swapchain_format_;
    attachment.samples = VK_SAMPLE_COUNT_1_BIT;
    attachment.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    attachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
    attachment.stencilLoadOp = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    attachment.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    attachment.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    attachment.finalLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentReference color_reference{};
    color_reference.attachment = 0U;
    color_reference.layout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount = 1U;
    subpass.pColorAttachments = &color_reference;

    VkSubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0U;
    dependency.srcStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstStageMask = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;

    VkRenderPassCreateInfo create_info = make_info<VkRenderPassCreateInfo>(
        VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO);
    create_info.attachmentCount = 1U;
    create_info.pAttachments = &attachment;
    create_info.subpassCount = 1U;
    create_info.pSubpasses = &subpass;
    create_info.dependencyCount = 1U;
    create_info.pDependencies = &dependency;
    check(vkCreateRenderPass(device_, &create_info, nullptr, &render_pass_), "vkCreateRenderPass");
}

void VulkanRenderer::create_framebuffers()
{
    framebuffers_.resize(image_views_.size());
    for (std::size_t index = 0U; index < image_views_.size(); ++index)
    {
        VkFramebufferCreateInfo create_info = make_info<VkFramebufferCreateInfo>(
            VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO);
        create_info.renderPass = render_pass_;
        create_info.attachmentCount = 1U;
        create_info.pAttachments = &image_views_[index];
        create_info.width = swapchain_extent_.width;
        create_info.height = swapchain_extent_.height;
        create_info.layers = 1U;
        check(vkCreateFramebuffer(device_, &create_info, nullptr, &framebuffers_[index]),
            "vkCreateFramebuffer");
    }
}

void VulkanRenderer::create_command_resources()
{
    const QueueFamilies families = find_queue_families(physical_device_);
    VkCommandPoolCreateInfo pool_info = make_info<VkCommandPoolCreateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO);
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pool_info.queueFamilyIndex = families.graphics;
    check(vkCreateCommandPool(device_, &pool_info, nullptr, &command_pool_), "vkCreateCommandPool");

    VkCommandBufferAllocateInfo allocate_info = make_info<VkCommandBufferAllocateInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
    allocate_info.commandPool = command_pool_;
    allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocate_info.commandBufferCount = static_cast<std::uint32_t>(command_buffers_.size());
    check(vkAllocateCommandBuffers(device_, &allocate_info, command_buffers_.data()),
        "vkAllocateCommandBuffers");
}

void VulkanRenderer::create_synchronization()
{
    VkSemaphoreCreateInfo semaphore_info = make_info<VkSemaphoreCreateInfo>(
        VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO);
    VkFenceCreateInfo fence_info = make_info<VkFenceCreateInfo>(VK_STRUCTURE_TYPE_FENCE_CREATE_INFO);
    fence_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (std::size_t index = 0U; index < frames_in_flight; ++index)
    {
        check(vkCreateSemaphore(device_, &semaphore_info, nullptr, &image_available_[index]),
            "vkCreateSemaphore(image available)");
        check(vkCreateFence(device_, &fence_info, nullptr, &in_flight_[index]), "vkCreateFence");
    }
}

void VulkanRenderer::render_frame()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width <= 0 || height <= 0)
    {
        return;
    }
    if (static_cast<std::uint32_t>(width) != swapchain_extent_.width ||
        static_cast<std::uint32_t>(height) != swapchain_extent_.height)
    {
        recreate_swapchain();
        return;
    }

    check(vkWaitForFences(device_, 1U, &in_flight_[current_frame_], VK_TRUE,
        std::numeric_limits<std::uint64_t>::max()), "vkWaitForFences");
    std::uint32_t image_index = 0U;
    const VkResult acquire_result = vkAcquireNextImageKHR(device_, swapchain_,
        std::numeric_limits<std::uint64_t>::max(), image_available_[current_frame_],
        VK_NULL_HANDLE, &image_index);
    if (acquire_result == VK_ERROR_OUT_OF_DATE_KHR)
    {
        recreate_swapchain();
        return;
    }
    if (acquire_result != VK_SUCCESS && acquire_result != VK_SUBOPTIMAL_KHR)
    {
        fail("vkAcquireNextImageKHR", acquire_result);
    }
    if (images_in_flight_[image_index] != VK_NULL_HANDLE)
    {
        check(vkWaitForFences(device_, 1U, &images_in_flight_[image_index], VK_TRUE,
            std::numeric_limits<std::uint64_t>::max()), "vkWaitForFences(image in flight)");
    }
    images_in_flight_[image_index] = in_flight_[current_frame_];
    check(vkResetFences(device_, 1U, &in_flight_[current_frame_]), "vkResetFences");

    VkCommandBuffer command_buffer = command_buffers_[current_frame_];
    check(vkResetCommandBuffer(command_buffer, 0U), "vkResetCommandBuffer");
    VkCommandBufferBeginInfo begin_info = make_info<VkCommandBufferBeginInfo>(
        VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    check(vkBeginCommandBuffer(command_buffer, &begin_info), "vkBeginCommandBuffer");

    VkClearValue clear{};
    std::copy(guest_clear_color_.begin(), guest_clear_color_.end(),
        clear.color.float32);
    VkRenderPassBeginInfo render_info = make_info<VkRenderPassBeginInfo>(
        VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO);
    render_info.renderPass = render_pass_;
    render_info.framebuffer = framebuffers_[image_index];
    render_info.renderArea.extent = swapchain_extent_;
    render_info.clearValueCount = 1U;
    render_info.pClearValues = &clear;
    vkCmdBeginRenderPass(command_buffer, &render_info, VK_SUBPASS_CONTENTS_INLINE);
    vkCmdEndRenderPass(command_buffer);
    check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    constexpr VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo submit_info = make_info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
    submit_info.waitSemaphoreCount = 1U;
    submit_info.pWaitSemaphores = &image_available_[current_frame_];
    submit_info.pWaitDstStageMask = &wait_stage;
    submit_info.commandBufferCount = 1U;
    submit_info.pCommandBuffers = &command_buffer;
    submit_info.signalSemaphoreCount = 1U;
    submit_info.pSignalSemaphores = &render_finished_[image_index];
    check(vkQueueSubmit(graphics_queue_, 1U, &submit_info, in_flight_[current_frame_]),
        "vkQueueSubmit");

    VkPresentInfoKHR present_info = make_info<VkPresentInfoKHR>(VK_STRUCTURE_TYPE_PRESENT_INFO_KHR);
    present_info.waitSemaphoreCount = 1U;
    present_info.pWaitSemaphores = &render_finished_[image_index];
    present_info.swapchainCount = 1U;
    present_info.pSwapchains = &swapchain_;
    present_info.pImageIndices = &image_index;
    const VkResult present_result = vkQueuePresentKHR(present_queue_, &present_info);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR ||
        acquire_result == VK_SUBOPTIMAL_KHR)
    {
        recreate_swapchain();
    }
    else if (present_result != VK_SUCCESS)
    {
        fail("vkQueuePresentKHR", present_result);
    }

    current_frame_ = (current_frame_ + 1U) % frames_in_flight;
}

void VulkanRenderer::recreate_swapchain()
{
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    if (width == 0 || height == 0 || should_close())
    {
        return;
    }

    check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(swapchain recreation)");
    destroy_swapchain();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_framebuffers();
}

void VulkanRenderer::destroy_swapchain() noexcept
{
    if (device_ == VK_NULL_HANDLE) return;
    for (const VkFramebuffer framebuffer : framebuffers_)
    {
        if (framebuffer != VK_NULL_HANDLE) vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    framebuffers_.clear();
    if (render_pass_ != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }
    for (const VkImageView view : image_views_)
    {
        if (view != VK_NULL_HANDLE) vkDestroyImageView(device_, view, nullptr);
    }
    image_views_.clear();
    for (const VkSemaphore semaphore : render_finished_)
    {
        if (semaphore != VK_NULL_HANDLE) vkDestroySemaphore(device_, semaphore, nullptr);
    }
    render_finished_.clear();
    swapchain_images_.clear();
    images_in_flight_.clear();
    if (swapchain_ != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::cleanup() noexcept
{
    if (device_ != VK_NULL_HANDLE)
    {
        static_cast<void>(vkDeviceWaitIdle(device_));
        destroy_swapchain();
        for (std::size_t index = 0U; index < frames_in_flight; ++index)
        {
            if (in_flight_[index] != VK_NULL_HANDLE) vkDestroyFence(device_, in_flight_[index], nullptr);
            if (image_available_[index] != VK_NULL_HANDLE)
                vkDestroySemaphore(device_, image_available_[index], nullptr);
        }
        if (command_pool_ != VK_NULL_HANDLE) vkDestroyCommandPool(device_, command_pool_, nullptr);
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }
    if (surface_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE)
    {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
        surface_ = VK_NULL_HANDLE;
    }
    if (debug_messenger_ != VK_NULL_HANDLE && instance_ != VK_NULL_HANDLE)
    {
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy_messenger != nullptr) destroy_messenger(instance_, debug_messenger_, nullptr);
        debug_messenger_ = VK_NULL_HANDLE;
    }
    if (instance_ != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
    if (window_ != nullptr)
    {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    if (owns_glfw_)
    {
        glfwTerminate();
        owns_glfw_ = false;
    }
}

}
