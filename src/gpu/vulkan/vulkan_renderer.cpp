#include "gpu/vulkan/vulkan_renderer.hpp"

#include "gpu/vulkan/diagnostic_shader_spirv.hpp"
#include "gpu/latte_vertex_fetch.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <limits>
#include <ranges>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace affogato::gpu::vulkan
{
namespace
{

constexpr const char* window_title = "Affogato — Vulkan output";
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

#ifndef NDEBUG
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
#endif

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

std::vector<DiagnosticVertex> decode_diagnostic_float2_draw(
    const cpu::espresso::Gx2Command& command,
    const cpu::espresso::Gx2FetchShaderBinding& fetch_shader)
{
    constexpr std::uint32_t gx2_primitive_mode_triangles = 4U;
    constexpr std::uint32_t diagnostic_position_location = 0U;
    constexpr std::uint32_t diagnostic_position_components = 2U;
    constexpr std::uint32_t bytes_per_float = sizeof(float);

    if (!command.draw_call)
    {
        throw std::runtime_error(
            "Vulkan diagnostic draw has no captured GX2 draw parameters");
    }
    const cpu::espresso::Gx2DrawCall& draw = *command.draw_call;
    if (draw.indexed || draw.primitive_mode != gx2_primitive_mode_triangles ||
        draw.instance_count != 1U || draw.base_instance != 0U ||
        draw.element_count == 0U || draw.element_count % 3U != 0U)
    {
        throw std::runtime_error(
            "Vulkan float2 diagnostic supports only non-indexed, single-instance triangle lists");
    }
    const std::vector<LatteVertexAttribute> attributes =
        decode_latte_vertex_fetch_shader(fetch_shader);
    const auto position_attribute = std::ranges::find_if(
        attributes,
        [](const LatteVertexAttribute& attribute)
        {
            return attribute.location == diagnostic_position_location;
        });
    if (position_attribute == attributes.end() ||
        position_attribute->component_count < diagnostic_position_components ||
        position_attribute->selectors[0U] != 0U ||
        position_attribute->selectors[1U] != 1U)
    {
        throw std::runtime_error(
            "Vulkan float2 diagnostic requires location 0 to select the first two float components");
    }

    const auto vertex_buffer = std::ranges::find_if(
        command.draw_vertex_data,
        [&position_attribute](
            const cpu::espresso::Gx2VertexBufferSnapshot& snapshot)
        {
            return snapshot.binding.index == position_attribute->buffer_index;
        });
    if (vertex_buffer == command.draw_vertex_data.end())
    {
        throw std::runtime_error(
            "Vulkan float2 diagnostic is missing the fetch shader's vertex buffer slot");
    }

    const cpu::espresso::Gx2VertexBufferSnapshot& snapshot = *vertex_buffer;
    const std::uint32_t position_byte_size =
        diagnostic_position_components * bytes_per_float;
    const std::uint64_t expected_offset =
        static_cast<std::uint64_t>(draw.element_offset) *
        snapshot.binding.stride;
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(draw.element_count) *
        snapshot.binding.stride;
    if (!snapshot.is_valid || snapshot.binding.stride == 0U ||
        position_attribute->endian_swap != 2U ||
        position_attribute->byte_offset > snapshot.binding.stride ||
        position_byte_size > snapshot.binding.stride -
            position_attribute->byte_offset ||
        snapshot.byte_offset != expected_offset ||
        snapshot.bytes.size() != expected_size)
    {
        throw std::runtime_error(
            "Vulkan float2 diagnostic received an invalid or unsupported vertex-buffer snapshot");
    }

    const auto read_be32 = [&snapshot](std::size_t offset) {
        return (static_cast<std::uint32_t>(snapshot.bytes[offset]) << 24U) |
            (static_cast<std::uint32_t>(snapshot.bytes[offset + 1U]) << 16U) |
            (static_cast<std::uint32_t>(snapshot.bytes[offset + 2U]) << 8U) |
            static_cast<std::uint32_t>(snapshot.bytes[offset + 3U]);
    };

    std::vector<DiagnosticVertex> vertices(draw.element_count);
    for (std::size_t index = 0U; index < vertices.size(); ++index)
    {
        const std::size_t vertex_offset =
            index * snapshot.binding.stride + position_attribute->byte_offset;
        vertices[index].x = std::bit_cast<float>(read_be32(vertex_offset));
        vertices[index].y = std::bit_cast<float>(
            read_be32(vertex_offset + sizeof(float)));
    }
    return vertices;
}

}

VulkanRenderer::VulkanRenderer(bool visible)
{
    if (!glfwInit())
    {
        throw std::runtime_error("GLFW initialization failed");
    }
    owns_glfw_ = true;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_VISIBLE, visible ? GLFW_TRUE : GLFW_FALSE);
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
        create_diagnostic_pipeline();
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

void VulkanRenderer::set_guest_diagnostic_vertices(
    std::span<const DiagnosticVertex> vertices)
{
    if (vertices.empty() || vertices.size() % 3U != 0U ||
        vertices.size() > std::numeric_limits<std::uint32_t>::max())
    {
        throw std::invalid_argument(
            "Guest diagnostic geometry must contain complete triangles");
    }
    for (const DiagnosticVertex& vertex : vertices)
    {
        if (!std::isfinite(vertex.x) || !std::isfinite(vertex.y) ||
            vertex.x < -1.0F || vertex.x > 1.0F ||
            vertex.y < -1.0F || vertex.y > 1.0F)
        {
            throw std::invalid_argument(
                "Guest diagnostic vertex is outside finite clip-space bounds");
        }
    }

    check(vkDeviceWaitIdle(device_), "vkDeviceWaitIdle(diagnostic geometry update)");
    if (diagnostic_vertex_buffer_ != VK_NULL_HANDLE)
    {
        vkDestroyBuffer(device_, diagnostic_vertex_buffer_, nullptr);
        diagnostic_vertex_buffer_ = VK_NULL_HANDLE;
    }
    if (diagnostic_vertex_memory_ != VK_NULL_HANDLE)
    {
        vkFreeMemory(device_, diagnostic_vertex_memory_, nullptr);
        diagnostic_vertex_memory_ = VK_NULL_HANDLE;
    }

    const VkDeviceSize buffer_size =
        static_cast<VkDeviceSize>(vertices.size_bytes());
    VkBufferCreateInfo buffer_info = make_info<VkBufferCreateInfo>(
        VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
    buffer_info.size = buffer_size;
    buffer_info.usage = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    check(vkCreateBuffer(
        device_, &buffer_info, nullptr, &diagnostic_vertex_buffer_),
        "vkCreateBuffer(guest diagnostic vertices)");

    VkMemoryRequirements requirements{};
    vkGetBufferMemoryRequirements(device_, diagnostic_vertex_buffer_, &requirements);
    const std::uint32_t memory_type_index = find_memory_type(
        requirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkPhysicalDeviceMemoryProperties memory_properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
    diagnostic_vertex_memory_coherent_ =
        (memory_properties.memoryTypes[memory_type_index].propertyFlags &
         VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
    VkMemoryAllocateInfo allocation_info = make_info<VkMemoryAllocateInfo>(
        VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
    allocation_info.allocationSize = requirements.size;
    allocation_info.memoryTypeIndex = memory_type_index;
    check(vkAllocateMemory(
        device_, &allocation_info, nullptr, &diagnostic_vertex_memory_),
        "vkAllocateMemory(guest diagnostic vertices)");
    check(vkBindBufferMemory(
        device_, diagnostic_vertex_buffer_, diagnostic_vertex_memory_, 0U),
        "vkBindBufferMemory(guest diagnostic vertices)");

    void* mapped_memory = nullptr;
    check(vkMapMemory(
        device_, diagnostic_vertex_memory_, 0U, VK_WHOLE_SIZE, 0U, &mapped_memory),
        "vkMapMemory(guest diagnostic vertices)");
    std::memcpy(mapped_memory, vertices.data(), vertices.size_bytes());
    if (!diagnostic_vertex_memory_coherent_)
    {
        VkMappedMemoryRange flush_range = make_info<VkMappedMemoryRange>(
            VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
        flush_range.memory = diagnostic_vertex_memory_;
        flush_range.offset = 0U;
        flush_range.size = VK_WHOLE_SIZE;
        check(vkFlushMappedMemoryRanges(device_, 1U, &flush_range),
            "vkFlushMappedMemoryRanges(guest diagnostic vertices)");
    }
    vkUnmapMemory(device_, diagnostic_vertex_memory_);
    diagnostic_vertex_count_ = static_cast<std::uint32_t>(vertices.size());
}

bool VulkanRenderer::set_guest_scanout(Rgba8TextureStaging scanout)
{
    constexpr std::uint64_t bytes_per_pixel = 4U;
    if (scanout.width == 0U || scanout.height == 0U ||
        scanout.height > std::numeric_limits<std::uint64_t>::max() /
            (static_cast<std::uint64_t>(scanout.width) * bytes_per_pixel))
    {
        throw std::invalid_argument("Guest scanout has invalid RGBA8 dimensions or size");
    }
    const std::uint64_t expected_size =
        static_cast<std::uint64_t>(scanout.width) * scanout.height * bytes_per_pixel;
    if (expected_size != scanout.pixels.size())
    {
        throw std::invalid_argument("Guest scanout has invalid RGBA8 dimensions or size");
    }
    if (!supports_swapchain_transfer_dst_)
    {
        return false;
    }
    guest_scanout_ = std::move(scanout);
    guest_scanout_dirty_ = true;
    return true;
}

void VulkanRenderer::consume_guest_commands(
    std::span<const cpu::espresso::Gx2Command> commands,
    GuestDrawPolicy draw_policy)
{
    const std::size_t first_draw_index = guest_register_state_.draw_calls().size();
    guest_register_state_.apply(commands);
    if (draw_policy == GuestDrawPolicy::capture_only)
    {
        return;
    }

    std::vector<DiagnosticVertex> diagnostic_vertices;
    std::size_t draw_count = 0U;
    std::size_t draw_index = first_draw_index;
    const auto draw_fetch_shaders = guest_register_state_.draw_fetch_shaders();
    for (const cpu::espresso::Gx2Command& command : commands)
    {
        if (command.type != cpu::espresso::Gx2CommandType::draw)
        {
            continue;
        }
        ++draw_count;
        if (draw_count > 1U)
        {
            throw std::runtime_error(
                "Vulkan float2 diagnostic currently accepts one draw per command batch");
        }
        if (draw_index >= draw_fetch_shaders.size() ||
            !draw_fetch_shaders[draw_index])
        {
            throw std::runtime_error(
                "Vulkan diagnostic draw has no captured GX2 fetch shader");
        }
        diagnostic_vertices = decode_diagnostic_float2_draw(
            command, *draw_fetch_shaders[draw_index]);
        ++draw_index;
    }
    if (diagnostic_vertices.empty())
    {
        throw std::runtime_error(
            "Vulkan diagnostic draw policy received no guest draw commands");
    }
    set_guest_diagnostic_vertices(diagnostic_vertices);
}

std::uint32_t VulkanRenderer::find_memory_type(
    std::uint32_t type_bits,
    VkMemoryPropertyFlags required,
    VkMemoryPropertyFlags preferred) const
{
    VkPhysicalDeviceMemoryProperties properties{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &properties);
    for (std::uint32_t index = 0U; index < properties.memoryTypeCount; ++index)
    {
        const VkMemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
        if ((type_bits & (1U << index)) != 0U && (flags & required) == required &&
            (flags & preferred) == preferred)
        {
            return index;
        }
    }
    for (std::uint32_t index = 0U; index < properties.memoryTypeCount; ++index)
    {
        const VkMemoryPropertyFlags flags = properties.memoryTypes[index].propertyFlags;
        if ((type_bits & (1U << index)) != 0U && (flags & required) == required)
        {
            return index;
        }
    }
    throw std::runtime_error("No compatible Vulkan memory type is available");
}

std::size_t VulkanRenderer::upload_rgba8_texture(const Rgba8TextureStaging& staging)
{
    if (staging.width == 0U || staging.height == 0U)
    {
        throw std::invalid_argument("RGBA8 Vulkan texture staging has invalid dimensions or size");
    }
    const std::uint64_t row_bytes = static_cast<std::uint64_t>(staging.width) * 4U;
    if (staging.height > std::numeric_limits<std::uint64_t>::max() / row_bytes)
    {
        throw std::invalid_argument("RGBA8 Vulkan texture staging size overflows");
    }
    const std::uint64_t expected_size = row_bytes * staging.height;
    if (expected_size != staging.pixels.size())
    {
        throw std::invalid_argument("RGBA8 Vulkan texture staging has invalid dimensions or size");
    }

    VkPhysicalDeviceProperties device_properties{};
    vkGetPhysicalDeviceProperties(physical_device_, &device_properties);
    if (staging.width > device_properties.limits.maxImageDimension2D ||
        staging.height > device_properties.limits.maxImageDimension2D)
    {
        throw std::invalid_argument("RGBA8 texture exceeds the Vulkan device's 2D image limit");
    }
    VkFormatProperties format_properties{};
    vkGetPhysicalDeviceFormatProperties(
        physical_device_, VK_FORMAT_R8G8B8A8_UNORM, &format_properties);
    constexpr VkFormatFeatureFlags required_format_features =
        VK_FORMAT_FEATURE_SAMPLED_IMAGE_BIT | VK_FORMAT_FEATURE_TRANSFER_DST_BIT;
    if ((format_properties.optimalTilingFeatures & required_format_features) !=
        required_format_features)
    {
        throw std::runtime_error("Vulkan device cannot sample optimal RGBA8 images");
    }

    VkBuffer staging_buffer{};
    VkDeviceMemory staging_memory{};
    VkImage image{};
    VkDeviceMemory image_memory{};
    VkImageView view{};
    VkCommandBuffer upload_command{};
    bool staging_mapped = false;
    try
    {
        VkBufferCreateInfo buffer_info = make_info<VkBufferCreateInfo>(
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        buffer_info.size = static_cast<VkDeviceSize>(staging.pixels.size());
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &buffer_info, nullptr, &staging_buffer),
            "vkCreateBuffer(texture staging)");

        VkMemoryRequirements buffer_requirements{};
        vkGetBufferMemoryRequirements(device_, staging_buffer, &buffer_requirements);
        const std::uint32_t staging_memory_type = find_memory_type(
            buffer_requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkMemoryAllocateInfo staging_allocation = make_info<VkMemoryAllocateInfo>(
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        staging_allocation.allocationSize = buffer_requirements.size;
        staging_allocation.memoryTypeIndex = staging_memory_type;
        check(vkAllocateMemory(device_, &staging_allocation, nullptr, &staging_memory),
            "vkAllocateMemory(texture staging)");
        check(vkBindBufferMemory(device_, staging_buffer, staging_memory, 0U),
            "vkBindBufferMemory(texture staging)");

        void* mapped{};
        check(vkMapMemory(device_, staging_memory, 0U, buffer_requirements.size, 0U, &mapped),
            "vkMapMemory(texture staging)");
        staging_mapped = true;
        std::memcpy(mapped, staging.pixels.data(), staging.pixels.size());
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
        const VkMemoryPropertyFlags staging_flags =
            memory_properties.memoryTypes[staging_memory_type].propertyFlags;
        if ((staging_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0U)
        {
            VkMappedMemoryRange mapped_range = make_info<VkMappedMemoryRange>(
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
            mapped_range.memory = staging_memory;
            mapped_range.offset = 0U;
            mapped_range.size = VK_WHOLE_SIZE;
            check(vkFlushMappedMemoryRanges(device_, 1U, &mapped_range),
                "vkFlushMappedMemoryRanges(texture staging)");
        }
        vkUnmapMemory(device_, staging_memory);
        staging_mapped = false;

        VkImageCreateInfo image_info = make_info<VkImageCreateInfo>(
            VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO);
        image_info.imageType = VK_IMAGE_TYPE_2D;
        image_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        image_info.extent = {staging.width, staging.height, 1U};
        image_info.mipLevels = 1U;
        image_info.arrayLayers = 1U;
        image_info.samples = VK_SAMPLE_COUNT_1_BIT;
        image_info.tiling = VK_IMAGE_TILING_OPTIMAL;
        image_info.usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        image_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        image_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        check(vkCreateImage(device_, &image_info, nullptr, &image), "vkCreateImage(texture)");

        VkMemoryRequirements image_requirements{};
        vkGetImageMemoryRequirements(device_, image, &image_requirements);
        VkMemoryAllocateInfo image_allocation = make_info<VkMemoryAllocateInfo>(
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        image_allocation.allocationSize = image_requirements.size;
        image_allocation.memoryTypeIndex = find_memory_type(
            image_requirements.memoryTypeBits, 0U, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        check(vkAllocateMemory(device_, &image_allocation, nullptr, &image_memory),
            "vkAllocateMemory(texture image)");
        check(vkBindImageMemory(device_, image, image_memory, 0U),
            "vkBindImageMemory(texture)");

        VkCommandBufferAllocateInfo command_allocation = make_info<VkCommandBufferAllocateInfo>(
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO);
        command_allocation.commandPool = command_pool_;
        command_allocation.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_allocation.commandBufferCount = 1U;
        check(vkAllocateCommandBuffers(device_, &command_allocation, &upload_command),
            "vkAllocateCommandBuffers(texture upload)");
        VkCommandBufferBeginInfo command_begin = make_info<VkCommandBufferBeginInfo>(
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO);
        command_begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        check(vkBeginCommandBuffer(upload_command, &command_begin),
            "vkBeginCommandBuffer(texture upload)");

        VkImageMemoryBarrier to_transfer = make_info<VkImageMemoryBarrier>(
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
        to_transfer.srcAccessMask = 0U;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_transfer.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = image;
        to_transfer.subresourceRange = {
            VK_IMAGE_ASPECT_COLOR_BIT, 0U, 1U, 0U, 1U};
        vkCmdPipelineBarrier(upload_command, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT, 0U, 0U, nullptr, 0U, nullptr,
            1U, &to_transfer);

        VkBufferImageCopy copy{};
        copy.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0U, 0U, 1U};
        copy.imageExtent = {staging.width, staging.height, 1U};
        vkCmdCopyBufferToImage(upload_command, staging_buffer, image,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1U, &copy);

        VkImageMemoryBarrier to_shader_read = make_info<VkImageMemoryBarrier>(
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
        to_shader_read.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_shader_read.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        to_shader_read.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_shader_read.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        to_shader_read.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_shader_read.image = image;
        to_shader_read.subresourceRange = to_transfer.subresourceRange;
        vkCmdPipelineBarrier(upload_command, VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
            0U, 0U, nullptr, 0U, nullptr, 1U, &to_shader_read);
        check(vkEndCommandBuffer(upload_command), "vkEndCommandBuffer(texture upload)");

        VkSubmitInfo submit = make_info<VkSubmitInfo>(VK_STRUCTURE_TYPE_SUBMIT_INFO);
        submit.commandBufferCount = 1U;
        submit.pCommandBuffers = &upload_command;
        check(vkQueueSubmit(graphics_queue_, 1U, &submit, VK_NULL_HANDLE),
            "vkQueueSubmit(texture upload)");
        check(vkQueueWaitIdle(graphics_queue_), "vkQueueWaitIdle(texture upload)");
        vkFreeCommandBuffers(device_, command_pool_, 1U, &upload_command);
        upload_command = VK_NULL_HANDLE;

        VkImageViewCreateInfo view_info = make_info<VkImageViewCreateInfo>(
            VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO);
        view_info.image = image;
        view_info.viewType = VK_IMAGE_VIEW_TYPE_2D;
        view_info.format = VK_FORMAT_R8G8B8A8_UNORM;
        view_info.subresourceRange = to_transfer.subresourceRange;
        check(vkCreateImageView(device_, &view_info, nullptr, &view),
            "vkCreateImageView(texture)");

        vkDestroyBuffer(device_, staging_buffer, nullptr);
        staging_buffer = VK_NULL_HANDLE;
        vkFreeMemory(device_, staging_memory, nullptr);
        staging_memory = VK_NULL_HANDLE;
        uploaded_textures_.push_back({image, image_memory, view, staging.width, staging.height});
        return uploaded_textures_.size() - 1U;
    }
    catch (...)
    {
        if (staging_mapped)
        {
            vkUnmapMemory(device_, staging_memory);
        }
        if (upload_command != VK_NULL_HANDLE)
        {
            vkFreeCommandBuffers(device_, command_pool_, 1U, &upload_command);
        }
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, view, nullptr);
        }
        if (image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device_, image, nullptr);
        }
        if (image_memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, image_memory, nullptr);
        }
        if (staging_buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_, staging_buffer, nullptr);
        }
        if (staging_memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, staging_memory, nullptr);
        }
        throw;
    }
}

VkImageView VulkanRenderer::uploaded_texture_view(std::size_t texture_id) const
{
    if (texture_id >= uploaded_textures_.size())
    {
        throw std::out_of_range("Vulkan texture handle is out of range");
    }
    return uploaded_textures_[texture_id].view;
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
    if (formats.empty())
    {
        throw std::runtime_error("Surface exposes no formats");
    }
    const auto preferred_format = std::ranges::find_if(formats, [](const auto& format)
    {
        return format.format == VK_FORMAT_B8G8R8A8_SRGB &&
            format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    });
    auto selected_format = preferred_format;
    if (selected_format == formats.end())
    {
        selected_format = std::ranges::find_if(formats, [](const auto& format)
        {
            return format.format == VK_FORMAT_B8G8R8A8_UNORM ||
                format.format == VK_FORMAT_R8G8B8A8_SRGB ||
                format.format == VK_FORMAT_R8G8B8A8_UNORM;
        });
    }
    const VkSurfaceFormatKHR surface_format = selected_format != formats.end()
        ? *selected_format
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
    supports_swapchain_transfer_dst_ =
        (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT) != 0U;
    VkFormatProperties swapchain_format_properties{};
    vkGetPhysicalDeviceFormatProperties(
        physical_device_, swapchain_format_, &swapchain_format_properties);
    const bool has_rgba8_swapchain_format =
        swapchain_format_ == VK_FORMAT_B8G8R8A8_SRGB ||
        swapchain_format_ == VK_FORMAT_B8G8R8A8_UNORM ||
        swapchain_format_ == VK_FORMAT_R8G8B8A8_SRGB ||
        swapchain_format_ == VK_FORMAT_R8G8B8A8_UNORM;
    supports_swapchain_transfer_dst_ = supports_swapchain_transfer_dst_ &&
        has_rgba8_swapchain_format &&
        (swapchain_format_properties.optimalTilingFeatures &
         VK_FORMAT_FEATURE_TRANSFER_DST_BIT) != 0U;

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
    if (supports_swapchain_transfer_dst_)
    {
        create_info.imageUsage |= VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    }
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
        if (choice == choices.end())
        {
            throw std::runtime_error("Surface exposes no composite alpha mode");
        }
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
    swapchain_images_initialized_.assign(actual_count, false);

    render_finished_.resize(actual_count, VK_NULL_HANDLE);
    VkSemaphoreCreateInfo semaphore_info{};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (VkSemaphore& semaphore : render_finished_)
    {
        check(vkCreateSemaphore(device_, &semaphore_info, nullptr, &semaphore),
            "vkCreateSemaphore(present ready)");
    }
    if (supports_swapchain_transfer_dst_)
    {
        create_scanout_upload_buffers();
    }
}

void VulkanRenderer::create_scanout_upload_buffers()
{
    const std::uint64_t pixel_count =
        static_cast<std::uint64_t>(swapchain_extent_.width) * swapchain_extent_.height;
    if (pixel_count > std::numeric_limits<VkDeviceSize>::max() / 4U)
    {
        throw std::runtime_error("Vulkan swapchain extent overflows scanout buffer size");
    }
    scanout_upload_buffer_size_ = static_cast<VkDeviceSize>(pixel_count * 4U);

    for (ScanoutUploadBuffer& upload : scanout_upload_buffers_)
    {
        VkBufferCreateInfo buffer_info = make_info<VkBufferCreateInfo>(
            VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO);
        buffer_info.size = scanout_upload_buffer_size_;
        buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        check(vkCreateBuffer(device_, &buffer_info, nullptr, &upload.buffer),
            "vkCreateBuffer(guest scanout)");

        VkMemoryRequirements requirements{};
        vkGetBufferMemoryRequirements(device_, upload.buffer, &requirements);
        const std::uint32_t memory_type_index = find_memory_type(
            requirements.memoryTypeBits,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        VkPhysicalDeviceMemoryProperties memory_properties{};
        vkGetPhysicalDeviceMemoryProperties(physical_device_, &memory_properties);
        upload.memory_coherent =
            (memory_properties.memoryTypes[memory_type_index].propertyFlags &
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0U;
        VkMemoryAllocateInfo allocation_info = make_info<VkMemoryAllocateInfo>(
            VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO);
        allocation_info.allocationSize = requirements.size;
        allocation_info.memoryTypeIndex = memory_type_index;
        check(vkAllocateMemory(device_, &allocation_info, nullptr, &upload.memory),
            "vkAllocateMemory(guest scanout)");
        check(vkBindBufferMemory(device_, upload.buffer, upload.memory, 0U),
            "vkBindBufferMemory(guest scanout)");
    }
}

void VulkanRenderer::destroy_scanout_upload_buffers() noexcept
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }
    for (ScanoutUploadBuffer& upload : scanout_upload_buffers_)
    {
        if (upload.buffer != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_, upload.buffer, nullptr);
            upload.buffer = VK_NULL_HANDLE;
        }
        if (upload.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, upload.memory, nullptr);
            upload.memory = VK_NULL_HANDLE;
        }
    }
    scanout_upload_buffer_size_ = 0U;
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

void VulkanRenderer::create_diagnostic_pipeline()
{
    const auto create_shader_module = [this](const auto& code, const char* operation) {
        VkShaderModuleCreateInfo create_info = make_info<VkShaderModuleCreateInfo>(
            VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO);
        create_info.codeSize = code.size() * sizeof(code.front());
        create_info.pCode = code.data();
        VkShaderModule module = VK_NULL_HANDLE;
        check(vkCreateShaderModule(device_, &create_info, nullptr, &module), operation);
        return module;
    };

    const VkShaderModule vertex_module = create_shader_module(
        detail::diagnostic_vertex_shader_spirv,
        "vkCreateShaderModule(guest diagnostic vertex)");
    VkShaderModule fragment_module = VK_NULL_HANDLE;
    try
    {
        fragment_module = create_shader_module(
            detail::diagnostic_fragment_shader_spirv,
            "vkCreateShaderModule(guest diagnostic fragment)");

        const VkPipelineShaderStageCreateInfo shader_stages[] = {
            [&] {
                auto info = make_info<VkPipelineShaderStageCreateInfo>(
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
                info.stage = VK_SHADER_STAGE_VERTEX_BIT;
                info.module = vertex_module;
                info.pName = "main";
                return info;
            }(),
            [&] {
                auto info = make_info<VkPipelineShaderStageCreateInfo>(
                    VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO);
                info.stage = VK_SHADER_STAGE_FRAGMENT_BIT;
                info.module = fragment_module;
                info.pName = "main";
                return info;
            }()};

        const VkVertexInputBindingDescription binding{
            0U, sizeof(DiagnosticVertex), VK_VERTEX_INPUT_RATE_VERTEX};
        const VkVertexInputAttributeDescription attribute{
            0U, 0U, VK_FORMAT_R32G32_SFLOAT,
            static_cast<std::uint32_t>(offsetof(DiagnosticVertex, x))};
        VkPipelineVertexInputStateCreateInfo vertex_input =
            make_info<VkPipelineVertexInputStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO);
        vertex_input.vertexBindingDescriptionCount = 1U;
        vertex_input.pVertexBindingDescriptions = &binding;
        vertex_input.vertexAttributeDescriptionCount = 1U;
        vertex_input.pVertexAttributeDescriptions = &attribute;

        VkPipelineInputAssemblyStateCreateInfo input_assembly =
            make_info<VkPipelineInputAssemblyStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO);
        input_assembly.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;

        VkPipelineViewportStateCreateInfo viewport =
            make_info<VkPipelineViewportStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO);
        viewport.viewportCount = 1U;
        viewport.scissorCount = 1U;

        VkPipelineRasterizationStateCreateInfo rasterization =
            make_info<VkPipelineRasterizationStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO);
        rasterization.polygonMode = VK_POLYGON_MODE_FILL;
        rasterization.cullMode = VK_CULL_MODE_NONE;
        rasterization.frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE;
        rasterization.lineWidth = 1.0F;

        VkPipelineMultisampleStateCreateInfo multisample =
            make_info<VkPipelineMultisampleStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO);
        multisample.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

        VkPipelineColorBlendAttachmentState color_attachment{};
        color_attachment.colorWriteMask = VK_COLOR_COMPONENT_R_BIT |
            VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
            VK_COLOR_COMPONENT_A_BIT;
        VkPipelineColorBlendStateCreateInfo color_blend =
            make_info<VkPipelineColorBlendStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO);
        color_blend.attachmentCount = 1U;
        color_blend.pAttachments = &color_attachment;

        const VkDynamicState dynamic_states[]{
            VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
        VkPipelineDynamicStateCreateInfo dynamic_state =
            make_info<VkPipelineDynamicStateCreateInfo>(
                VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO);
        dynamic_state.dynamicStateCount = 2U;
        dynamic_state.pDynamicStates = dynamic_states;

        VkPipelineLayoutCreateInfo layout_info = make_info<VkPipelineLayoutCreateInfo>(
            VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO);
        check(vkCreatePipelineLayout(
            device_, &layout_info, nullptr, &diagnostic_pipeline_layout_),
            "vkCreatePipelineLayout(guest diagnostic)");

        VkGraphicsPipelineCreateInfo pipeline_info =
            make_info<VkGraphicsPipelineCreateInfo>(
                VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO);
        pipeline_info.stageCount = 2U;
        pipeline_info.pStages = shader_stages;
        pipeline_info.pVertexInputState = &vertex_input;
        pipeline_info.pInputAssemblyState = &input_assembly;
        pipeline_info.pViewportState = &viewport;
        pipeline_info.pRasterizationState = &rasterization;
        pipeline_info.pMultisampleState = &multisample;
        pipeline_info.pColorBlendState = &color_blend;
        pipeline_info.pDynamicState = &dynamic_state;
        pipeline_info.layout = diagnostic_pipeline_layout_;
        pipeline_info.renderPass = render_pass_;
        pipeline_info.subpass = 0U;
        check(vkCreateGraphicsPipelines(
            device_, VK_NULL_HANDLE, 1U, &pipeline_info, nullptr,
            &diagnostic_pipeline_),
            "vkCreateGraphicsPipelines(guest diagnostic)");
    }
    catch (...)
    {
        vkDestroyShaderModule(device_, vertex_module, nullptr);
        if (fragment_module != VK_NULL_HANDLE)
        {
            vkDestroyShaderModule(device_, fragment_module, nullptr);
        }
        throw;
    }
    vkDestroyShaderModule(device_, vertex_module, nullptr);
    vkDestroyShaderModule(device_, fragment_module, nullptr);
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

    const bool present_guest_scanout =
        guest_scanout_.has_value() && supports_swapchain_transfer_dst_;
    if (present_guest_scanout)
    {
        const bool swapchain_is_bgra =
            swapchain_format_ == VK_FORMAT_B8G8R8A8_SRGB ||
            swapchain_format_ == VK_FORMAT_B8G8R8A8_UNORM;
        const bool swapchain_is_rgba =
            swapchain_format_ == VK_FORMAT_R8G8B8A8_SRGB ||
            swapchain_format_ == VK_FORMAT_R8G8B8A8_UNORM;
        if (!swapchain_is_bgra && !swapchain_is_rgba)
        {
            throw std::runtime_error(
                "Vulkan swapchain format cannot present RGBA8 guest scanout");
        }

        const std::uint32_t source_width = guest_scanout_->width;
        const std::uint32_t source_height = guest_scanout_->height;
        const std::size_t output_pixel_count =
            static_cast<std::size_t>(swapchain_extent_.width) *
            swapchain_extent_.height;
        if (guest_scanout_dirty_ ||
            scaled_scanout_pixels_.size() != output_pixel_count * 4U)
        {
            scaled_scanout_pixels_.resize(output_pixel_count * 4U);
            for (std::uint32_t output_y = 0U;
                 output_y < swapchain_extent_.height;
                 ++output_y)
            {
                const std::uint32_t source_y = static_cast<std::uint32_t>(
                    (static_cast<std::uint64_t>(output_y) * source_height) /
                    swapchain_extent_.height);
                for (std::uint32_t output_x = 0U;
                     output_x < swapchain_extent_.width;
                     ++output_x)
                {
                    const std::uint32_t source_x = static_cast<std::uint32_t>(
                        (static_cast<std::uint64_t>(output_x) * source_width) /
                        swapchain_extent_.width);
                    const std::size_t source_offset =
                        (static_cast<std::size_t>(source_y) * source_width + source_x) * 4U;
                    const std::size_t output_offset =
                        (static_cast<std::size_t>(output_y) * swapchain_extent_.width +
                         output_x) * 4U;
                    const auto& source_pixels = guest_scanout_->pixels;
                    scaled_scanout_pixels_[output_offset] =
                        source_pixels[source_offset + (swapchain_is_bgra ? 2U : 0U)];
                    scaled_scanout_pixels_[output_offset + 1U] =
                        source_pixels[source_offset + 1U];
                    scaled_scanout_pixels_[output_offset + 2U] =
                        source_pixels[source_offset + (swapchain_is_bgra ? 0U : 2U)];
                    scaled_scanout_pixels_[output_offset + 3U] =
                        source_pixels[source_offset + 3U];
                }
            }
            guest_scanout_dirty_ = false;
        }

        ScanoutUploadBuffer& upload = scanout_upload_buffers_[current_frame_];
        void* mapped_memory = nullptr;
        check(vkMapMemory(
            device_, upload.memory, 0U, scanout_upload_buffer_size_, 0U, &mapped_memory),
            "vkMapMemory(guest scanout)");
        std::memcpy(
            mapped_memory,
            scaled_scanout_pixels_.data(),
            scaled_scanout_pixels_.size());
        if (!upload.memory_coherent)
        {
            VkMappedMemoryRange flush_range = make_info<VkMappedMemoryRange>(
                VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE);
            flush_range.memory = upload.memory;
            flush_range.offset = 0U;
            flush_range.size = VK_WHOLE_SIZE;
            check(vkFlushMappedMemoryRanges(device_, 1U, &flush_range),
                "vkFlushMappedMemoryRanges(guest scanout)");
        }
        vkUnmapMemory(device_, upload.memory);

        VkImageMemoryBarrier to_transfer = make_info<VkImageMemoryBarrier>(
            VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER);
        to_transfer.srcAccessMask = 0U;
        to_transfer.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_transfer.oldLayout = swapchain_images_initialized_[image_index]
            ? VK_IMAGE_LAYOUT_PRESENT_SRC_KHR
            : VK_IMAGE_LAYOUT_UNDEFINED;
        to_transfer.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_transfer.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        to_transfer.image = swapchain_images_[image_index];
        to_transfer.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        to_transfer.subresourceRange.baseMipLevel = 0U;
        to_transfer.subresourceRange.levelCount = 1U;
        to_transfer.subresourceRange.baseArrayLayer = 0U;
        to_transfer.subresourceRange.layerCount = 1U;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0U,
            0U,
            nullptr,
            0U,
            nullptr,
            1U,
            &to_transfer);

        VkBufferImageCopy copy{};
        copy.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copy.imageSubresource.layerCount = 1U;
        copy.imageExtent = {swapchain_extent_.width, swapchain_extent_.height, 1U};
        vkCmdCopyBufferToImage(
            command_buffer,
            upload.buffer,
            swapchain_images_[image_index],
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1U,
            &copy);

        VkImageMemoryBarrier to_present = to_transfer;
        to_present.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_present.dstAccessMask = 0U;
        to_present.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_present.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        vkCmdPipelineBarrier(
            command_buffer,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
            0U,
            0U,
            nullptr,
            0U,
            nullptr,
            1U,
            &to_present);
    }
    else
    {
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
        if (diagnostic_vertex_count_ > 0U)
        {
            VkViewport viewport{};
            viewport.width = static_cast<float>(swapchain_extent_.width);
            viewport.height = static_cast<float>(swapchain_extent_.height);
            viewport.maxDepth = 1.0F;
            VkRect2D scissor{};
            scissor.extent = swapchain_extent_;
            const VkDeviceSize vertex_offset = 0U;
            vkCmdBindPipeline(
                command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS,
                diagnostic_pipeline_);
            vkCmdBindVertexBuffers(
                command_buffer, 0U, 1U, &diagnostic_vertex_buffer_, &vertex_offset);
            vkCmdSetViewport(command_buffer, 0U, 1U, &viewport);
            vkCmdSetScissor(command_buffer, 0U, 1U, &scissor);
            vkCmdDraw(command_buffer, diagnostic_vertex_count_, 1U, 0U, 0U);
        }
        vkCmdEndRenderPass(command_buffer);
    }
    check(vkEndCommandBuffer(command_buffer), "vkEndCommandBuffer");

    const VkPipelineStageFlags wait_stage = present_guest_scanout
        ? VK_PIPELINE_STAGE_TRANSFER_BIT
        : VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
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
    if (present_guest_scanout)
    {
        swapchain_images_initialized_[image_index] = true;
    }

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
    guest_scanout_dirty_ = guest_scanout_.has_value();
    create_image_views();
    create_render_pass();
    create_diagnostic_pipeline();
    create_framebuffers();
}

void VulkanRenderer::destroy_swapchain() noexcept
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }
    for (const VkFramebuffer framebuffer : framebuffers_)
    {
        if (framebuffer != VK_NULL_HANDLE)
        {
            vkDestroyFramebuffer(device_, framebuffer, nullptr);
        }
    }
    framebuffers_.clear();
    if (diagnostic_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device_, diagnostic_pipeline_, nullptr);
        diagnostic_pipeline_ = VK_NULL_HANDLE;
    }
    if (diagnostic_pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device_, diagnostic_pipeline_layout_, nullptr);
        diagnostic_pipeline_layout_ = VK_NULL_HANDLE;
    }
    if (render_pass_ != VK_NULL_HANDLE)
    {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
        render_pass_ = VK_NULL_HANDLE;
    }
    for (const VkImageView view : image_views_)
    {
        if (view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, view, nullptr);
        }
    }
    image_views_.clear();
    for (const VkSemaphore semaphore : render_finished_)
    {
        if (semaphore != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device_, semaphore, nullptr);
        }
    }
    render_finished_.clear();
    swapchain_images_initialized_.clear();
    destroy_scanout_upload_buffers();
    swapchain_images_.clear();
    images_in_flight_.clear();
    if (swapchain_ != VK_NULL_HANDLE)
    {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
}

void VulkanRenderer::destroy_uploaded_textures() noexcept
{
    if (device_ == VK_NULL_HANDLE)
    {
        return;
    }
    for (const UploadedTexture& texture : uploaded_textures_)
    {
        if (texture.view != VK_NULL_HANDLE)
        {
            vkDestroyImageView(device_, texture.view, nullptr);
        }
        if (texture.image != VK_NULL_HANDLE)
        {
            vkDestroyImage(device_, texture.image, nullptr);
        }
        if (texture.memory != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, texture.memory, nullptr);
        }
    }
    uploaded_textures_.clear();
}

void VulkanRenderer::cleanup() noexcept
{
    if (device_ != VK_NULL_HANDLE)
    {
        static_cast<void>(vkDeviceWaitIdle(device_));
        destroy_uploaded_textures();
        destroy_swapchain();
        if (diagnostic_vertex_buffer_ != VK_NULL_HANDLE)
        {
            vkDestroyBuffer(device_, diagnostic_vertex_buffer_, nullptr);
            diagnostic_vertex_buffer_ = VK_NULL_HANDLE;
        }
        if (diagnostic_vertex_memory_ != VK_NULL_HANDLE)
        {
            vkFreeMemory(device_, diagnostic_vertex_memory_, nullptr);
            diagnostic_vertex_memory_ = VK_NULL_HANDLE;
        }
        for (std::size_t index = 0U; index < frames_in_flight; ++index)
        {
            if (in_flight_[index] != VK_NULL_HANDLE)
            {
                vkDestroyFence(device_, in_flight_[index], nullptr);
            }
            if (image_available_[index] != VK_NULL_HANDLE)
            {
                vkDestroySemaphore(device_, image_available_[index], nullptr);
            }
        }
        if (command_pool_ != VK_NULL_HANDLE)
        {
            vkDestroyCommandPool(device_, command_pool_, nullptr);
        }
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
        if (destroy_messenger != nullptr)
        {
            destroy_messenger(instance_, debug_messenger_, nullptr);
        }
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
