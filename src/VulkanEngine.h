#pragma once
#include "VkBootstrap.h"
#include <SDL2/SDL.h>
#include <vulkan/vk_enum_string_helper.h>
#include <SDL2/SDL_vulkan.h>
#include "vk_initializers.h"
#include <iostream>
#include "vk_images.h"
#include <deque>
#include <thread>
#include <functional>
#include <vma/vk_mem_alloc.h>
#include "vk_descriptors.h"
#include "vk_pipelines.h"
#include "imgui.h"
#include "glm/glm.hpp"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

struct ComputePushConstants {
	glm::vec4 data1;
	glm::vec4 data2;
	glm::vec4 data3;
	glm::vec4 data4;
};

struct ComputeEffect {
	const char* name;

	VkPipeline pipeline;
	VkPipelineLayout layout;

	ComputePushConstants data;
};

#define VK_CHECK(x)                                                      \
    do {                                                                 \
        VkResult err = x;                                                \
        if (err) {                                                       \
            throw std::runtime_error(string_VkResult(err));				 \
        }                                                                \
    } while (0)

struct DeletionQueue
{
	std::deque<std::function<void()>> deletors;

	void push_function(std::function<void()>&& function) {
		deletors.push_back(function);
	}

	void flush() {
		// reverse iterate the deletion queue to execute all the functions
		for (auto it = deletors.rbegin(); it != deletors.rend(); it++) {
			(*it)(); //call functors
		}

		deletors.clear();
	}
};

struct AllocatedImage {
	VkImage image;
	VkImageView imageView;
	VmaAllocation allocation;
	VkExtent3D imageExtent;
	VkFormat imageFormat;
};

struct FrameData {

	VkCommandPool _commandPool;
	VkCommandBuffer _mainCommandBuffer;
	VkSemaphore _swapchainSemaphore, _renderSemaphore;
	VkFence _renderFence;
	DeletionQueue _deletionQueue;
};

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr bool bUseValidationLayers = false;	

class VulkanEngine
{
private:
	void init();
	void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
	void draw();
	void init_vulkan();
	void init_swapchain();
	void destroy_swapchain();
	void create_swapchain(uint32_t width, uint32_t height);
	void cleanup();
	void init_commands();
	void init_sync_structures();
	void init_descriptors();
	void init_pipelines();
	void init_background_pipelines();
	std::vector<ComputeEffect> backgroundEffects;
	int currentBackgroundEffect{ 0 };
public:
	VkDescriptorSet _drawImageDescriptors;
	VkDescriptorSetLayout _drawImageDescriptorLayout;
	VkExtent2D _windowExtent = {600, 800};
	bool _isInitialized;
	SDL_Window* _window;
	VkDebugUtilsMessengerEXT _debug_messenger;
	VkDevice _device;
	VkInstance _instance;
	VkSurfaceKHR _surface;
	VkPhysicalDevice _chosenGPU;
	VkSwapchainKHR _swapchain;
	VkFormat _swapchainImageFormat;

	std::vector<VkImage> _swapchainImages;
	std::vector<VkImageView> _swapchainImageViews;
	VkExtent2D _swapchainExtent;

	uint32_t _frameNumber;

	FrameData _frames[FRAME_OVERLAP];

	FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

	VkQueue _graphicsQueue;
	uint32_t _graphicsQueueFamily;
	DeletionQueue _mainDeletionQueue;
	VmaAllocator _allocator;
	AllocatedImage _drawImage;
	DescriptorAllocator _globalDescriptorAllocator;
	VkExtent2D _drawExtent;
	VkPipeline _gradientPipeline;
	VkPipelineLayout _gradientPipelineLayout;
	VkPipeline _trianglePipeline;
	VkPipelineLayout _trianglePipelineLayout;
	void run();
	void draw_geometry(VkCommandBuffer cmd);
	void init_imgui();
	void init_triangle_pipeline();
};