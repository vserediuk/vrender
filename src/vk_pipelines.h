#include "vulkan/vulkan.h"
#include <fstream>
#include <vector>

namespace vkutil
{
    bool load_shader_module(const char* filePath,
        VkDevice device,
        VkShaderModule* outShaderModule);
}