#pragma once
#include <vulkan/vulkan.h>
namespace vkinit
{
	VkCommandPoolCreateInfo command_pool_create_info(uint32_t queueFamilyIndex,
		VkCommandPoolCreateFlags flags /*= 0*/);

	VkCommandBufferAllocateInfo command_buffer_allocate_info(
		VkCommandPool pool, uint32_t count /*= 1*/);

	VkFenceCreateInfo fence_create_info(VkFenceCreateFlags flags /*= 0*/);

	VkSemaphoreCreateInfo semaphore_create_info(VkSemaphoreCreateFlags flags = 0);

	VkCommandBufferBeginInfo command_buffer_begin_info(VkCommandBufferUsageFlags flags /*= 0*/);

	VkCommandBufferSubmitInfo command_buffer_submit_info(VkCommandBuffer cmd);
	VkSubmitInfo2 submit_info(VkCommandBufferSubmitInfo* cmd, VkSemaphoreSubmitInfo* signalSemaphoreInfo,
		VkSemaphoreSubmitInfo* waitSemaphoreInfo);

	VkRenderingAttachmentInfo attachment_info(
		VkImageView view, VkClearValue* clear, VkImageLayout layout /*= VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL*/);

	VkSemaphoreSubmitInfo semaphore_submit_info(VkPipelineStageFlags2 stageMask, VkSemaphore semaphore);

	VkImageSubresourceRange image_subresource_range(VkImageAspectFlags aspectMask);

	VkImageCreateInfo image_create_info(VkFormat format, VkImageUsageFlags usageFlags, VkExtent3D extent);

    VkImageViewCreateInfo imageview_create_info(VkFormat format, VkImage image, VkImageAspectFlags aspectFlags);

	VkPipelineLayoutCreateInfo pipeline_layout_create_info();

	VkPipelineShaderStageCreateInfo pipeline_shader_stage_create_info(VkShaderStageFlagBits stage, VkShaderModule shaderModule, const char* entry = "main");

	VkRenderingAttachmentInfo depth_attachment_info(VkImageView view, VkImageLayout layout);

	VkRenderingInfo rendering_info(VkExtent2D renderExtent, VkRenderingAttachmentInfo* colorAttachment,
		VkRenderingAttachmentInfo* depthAttachment);
}