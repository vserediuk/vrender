#pragma once

#include "vk_types.h"
#include "vk_descriptors.h"
#include "vk_initializers.h"
#include "vk_pipelines.h"
#include "vk_images.h"
#include "vk_loader.h"
#include "camera.h"

constexpr unsigned int FRAME_OVERLAP = 2;
constexpr bool bUseValidationLayers = false;

struct EngineStats {
    float frametime;
    int triangle_count;
    int drawcall_count;
    float scene_update_time;
    float mesh_draw_time;
};

struct RenderObject {
    uint32_t indexCount;
    uint32_t firstIndex;
    VkBuffer indexBuffer;

    MaterialInstance* material;
    Bounds bounds;
    glm::mat4 transform;
    VkDeviceAddress vertexBufferAddress;
};

struct DrawContext {
    std::vector<RenderObject> OpaqueSurfaces;
    std::vector<RenderObject> TransparentSurfaces;
};

struct MeshNode : public Node {
    bool hasAnimation;
    std::shared_ptr<MeshAsset> mesh;

    virtual void Draw(const glm::mat4& topMatrix, DrawContext& ctx) override;
};

class VulkanEngine {
public:
    Camera mainCamera;

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
    GPUSceneData sceneData;
    EngineStats stats;
    std::unordered_map<std::string, std::shared_ptr<LoadedGLTF>> loadedScenes;

    AllocatedImage create_image(VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    AllocatedImage create_image(void* data, VkExtent3D size, VkFormat format, VkImageUsageFlags usage, bool mipmapped = false);
    void destroy_image(const AllocatedImage& img);

    VkDescriptorSetLayout _singleImageDescriptorLayout;
    VkSampler _defaultSamplerLinear;
    VkSampler _defaultSamplerNearest;
    AllocatedImage _whiteImage;
    AllocatedImage _blackImage;
    AllocatedImage _greyImage;
    AllocatedImage _errorCheckerboardImage;

    DrawContext mainDrawContext;
    std::unordered_map<std::string, std::shared_ptr<Node>> loadedNodes;
    MaterialInstance defaultData;
    GLTFMetallic_Roughness metalRoughMaterial;
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

    FrameData& get_current_frame() { return _frames[_frameNumber % FRAME_OVERLAP]; }

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
    void immediate_submit(std::function<void(VkCommandBuffer cmd)>&& function);
    GPUMeshBuffers uploadMesh(std::span<uint32_t> indices, std::span<Vertex> vertices);
    void init_default_data();
    void update_scene();

    VkFence _immFence;
    VkCommandPool _commandPool;
    VkCommandBuffer _immCommandBuffer;
    VkCommandPool _immCommandPool;
    VkPipelineLayout _meshPipelineLayout;
    VkPipeline _meshPipeline;
};