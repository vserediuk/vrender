#pragma once
#include <vulkan/vulkan.h>
#include "vk_initializers.h"
#include <fstream>
#include <vector>
#include <optional>
#include "vk_types.h"

namespace vkutil {
	void generate_mipmaps(VkCommandBuffer cmd, VkImage image, VkExtent2D imageSize);
	void transition_image(VkCommandBuffer cmd, VkImage image, VkImageLayout currentLayout, VkImageLayout newLayout);
	void copy_image_to_image(VkCommandBuffer cmd, VkImage source, VkImage destination, VkExtent2D srcSize, VkExtent2D dstSize);
	bool load_shader_module(const char* filePath,
		VkDevice device,
		VkShaderModule* outShaderModule);
	std::optional<AllocatedImage> load_image(VulkanEngine* engine, fastgltf::Asset& asset, fastgltf::Image& image);
}
