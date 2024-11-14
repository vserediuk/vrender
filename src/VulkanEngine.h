#pragma once
#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include "vk_images.h"
#include "vk_loader.h"

struct GPUSceneData {
	glm::mat4 view;
	glm::mat4 proj;
	glm::mat4 viewproj;
	glm::vec4 ambientColor;
	glm::vec4 sunlightDirection; // w for sun power
	glm::vec4 sunlightColor;
};

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
		DescriptorAllocatorGrowable _frameDescriptors;
	};

	constexpr unsigned int FRAME_OVERLAP = 2;
	constexpr bool bUseValidationLayers = false;

	class VulkanEngine
	{
	private:
		VkDescriptorSetLayout _singleImageDescriptorLayout;
		AllocatedImage _whiteImage;
		AllocatedImage _blackImage;
		AllocatedImage _greyImage;
		AllocatedImage _errorCheckerboardImage;

		VkSampler _defaultSamplerLinear;
		VkSampler _defaultSamplerNearest;
		AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
		AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
		void destroy_image(const AllocatedImage& img);
		void init();
		void draw_imgui(VkCommandBuffer cmd, VkImageView targetImageView);
		void resize_swapchain();
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
		bool resize_requested = false;
		std::vector<ComputeEffect> backgroundEffects;
		int currentBackgroundEffect{ 0 };
		std::vector<std::shared_ptr<MeshAsset>> testMeshes;
		GPUDrawPushConstants push_constants;
		GPUSceneData sceneData;
	public:
		VkDescriptorSetLayout _gpuSceneDataDescriptorLayout;
		AllocatedImage _drawImage;
		AllocatedImage _depthImage;
		VkDescriptorSet _drawImageDescriptors;
		VkDescriptorSetLayout _drawImageDescriptorLayout;
		VkExtent2D _windowExtent = { 600, 800 };
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
		DescriptorAllocatorGrowable _globalDescriptorAllocator;

		uint32_t _frameNumber;

		FrameData _frames[FRAME_OVERLAP];

		FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; };

		VkQueue _graphicsQueue;
		uint32_t _graphicsQueueFamily;
		DeletionQueue _mainDeletionQueue;
		VmaAllocator _allocator;
		VkExtent2D _drawExtent;
		float renderScale = 1.f;
		VkPipeline _gradientPipeline;
		VkPipelineLayout _gradientPipelineLayout;
		VkPipeline _trianglePipeline;
		VkPipelineLayout _trianglePipelineLayout;
		void run();
		void draw_geometry(VkCommandBuffer cmd);
		void init_imgui();
		AllocatedBuffer create_buffer(size_t allocSize, VkBufferUsageFlags usage, VmaMemoryUsage memoryUsage);
		void destroy_buffer(const AllocatedBuffer& buffer);
		void init_triangle_pipeline();
		void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
		GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
		void init_default_data();
		VkFence _immFence;
		VkCommandBuffer _immCommandBuffer;
		VkCommandPool _immCommandPool;
		VkPipelineLayout _meshPipelineLayout;
		VkPipeline _meshPipeline;

		GPUMeshBuffers rectangle;

		void init_mesh_pipeline();
	};
