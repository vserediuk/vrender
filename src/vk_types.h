#pragma once
#include <iostream>
#include <array>
#include <deque>
#include <thread>
#include <functional>
#include <optional>

#include "imgui.h"
#include "imgui_impl_sdl2.h"
#include "imgui_impl_vulkan.h"

#include <SDL2/SDL.h>
#include <SDL2/SDL_vulkan.h>
#include "VkBootstrap.h"

#include "glm/glm.hpp"
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include <vulkan/vk_enum_string_helper.h>
#include <vma/vk_mem_alloc.h>

#include "fastgltf/core.hpp"
#include "fastgltf/tools.hpp"
#include "fastgltf/glm_element_traits.hpp"


struct AllocatedBuffer {
    VkBuffer buffer;
    VmaAllocation allocation;
    VmaAllocationInfo info;
};

struct Vertex {
    glm::vec3 position;
    float uv_x;
    glm::vec3 normal;
    float uv_y;
    glm::vec4 color;
};

// holds the resources needed for a mesh
struct GPUMeshBuffers {
    AllocatedBuffer indexBuffer;
    AllocatedBuffer vertexBuffer;
    VkDeviceAddress vertexBufferAddress;
};

// push constants for our mesh object draws
struct GPUDrawPushConstants {
    glm::mat4 worldMatrix;
    VkDeviceAddress vertexBuffer;
};

#define VK_CHECK(x)                                                      \
    do {                                                                 \
        VkResult err = x;                                                \
        if (err) {                                                       \
            throw std::runtime_error(string_VkResult(err));				 \
        }                                                                \
    } while (0)