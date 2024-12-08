#include <vma/vk_mem_alloc.h>
#include "vk_loader.h"
#include "VulkanEngine.h"
#include "vk_images.h"
#include <optional>
#include "fastgltf/glm_element_traits.hpp"
#include <variant>
#include <glm/gtc/quaternion.hpp>


VkFilter extract_filter(fastgltf::Filter filter)
{
    switch (filter) {
        // nearest samplers
    case fastgltf::Filter::Nearest:
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::NearestMipMapLinear:
        return VK_FILTER_NEAREST;

        // linear samplers
    case fastgltf::Filter::Linear:
    case fastgltf::Filter::LinearMipMapNearest:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_FILTER_LINEAR;
    }
}

VkSamplerMipmapMode extract_mipmap_mode(fastgltf::Filter filter)
{
    switch (filter) {
    case fastgltf::Filter::NearestMipMapNearest:
    case fastgltf::Filter::LinearMipMapNearest:
        return VK_SAMPLER_MIPMAP_MODE_NEAREST;

    case fastgltf::Filter::NearestMipMapLinear:
    case fastgltf::Filter::LinearMipMapLinear:
    default:
        return VK_SAMPLER_MIPMAP_MODE_LINEAR;
    }
}

void LoadedGLTF::updateAnimation(VulkanEngine* engine, uint32_t activeAnimation, float deltaTime)
{
    if (activeAnimation > static_cast<uint32_t>(animations.size()) - 1)
    {
        std::cout << "No animation with index " << activeAnimation << std::endl;
        return;
    }
    Animation& animation = animations[activeAnimation];
    animation.currentTime += deltaTime;
    if (animation.currentTime > animation.end)
    {
        animation.currentTime -= animation.end;
    }

    for (auto& channel : animation.channels)
    {
        AnimationSampler& sampler = animation.samplers[channel.samplerIndex];
        for (size_t i = 0; i < sampler.inputs.size() - 1; i++)
        {
            if (sampler.interpolation != fastgltf::AnimationInterpolation::Linear)
            {
                std::cout << "This sample only supports linear interpolations\n";
                continue;
            }

            // Get the input keyframe values for the current time stamp
            if ((animation.currentTime >= sampler.inputs[i]) && (animation.currentTime <= sampler.inputs[i + 1]))
            {
                float a = (animation.currentTime - sampler.inputs[i]) / (sampler.inputs[i + 1] - sampler.inputs[i]);
                if (channel.path == fastgltf::AnimationPath::Translation)
                {
                    glm::vec3 interpolatedTranslation = glm::mix(
                        glm::vec3(sampler.outputsVec4[i]), // Convert vec4 to vec3 (x, y, z)
                        glm::vec3(sampler.outputsVec4[i + 1]),
                        a
                    );

                    // Apply the translation to the transformation matrix using fastgltf::math::translate
                    channel.node->localTransform = glm::translate(glm::mat4(1.0f), interpolatedTranslation);
                }
                if (channel.path == fastgltf::AnimationPath::Rotation)
                {
                    glm::quat q1(sampler.outputsVec4[i].x, sampler.outputsVec4[i].y, sampler.outputsVec4[i].z, sampler.outputsVec4[i].w);
                    glm::quat q2(sampler.outputsVec4[i + 1].x, sampler.outputsVec4[i + 1].y, sampler.outputsVec4[i + 1].z, sampler.outputsVec4[i + 1].w);

                    // Perform spherical linear interpolation (SLERP) between q1 and q2
                    glm::quat interpolatedQuat = glm::normalize(glm::slerp(q1, q2, a));

                    // Convert the quaternion to a rotation matrix and apply it to localTransform
                    channel.node->localTransform = glm::mat4_cast(interpolatedQuat);
                }
                if (channel.path == fastgltf::AnimationPath::Scale)
                {
                    glm::vec3 interpolatedScale = glm::mix(
                        glm::vec3(sampler.outputsVec4[i]),
                        glm::vec3(sampler.outputsVec4[i + 1]),
                        a
                    );

                    channel.node->localTransform = glm::scale(glm::mat4(1.0f), interpolatedScale);
                }
            }
        }
    }
    for (auto& node : topNodes)
    {
        updateJoints(engine, node);
    }
}

void LoadedGLTF::updateJoints(VulkanEngine* engine, std::shared_ptr<Node> node)
{
    if (node->skin > -1)
    {
        // Update the joint matrices
        glm::mat4              inverseTransform = glm::inverse(node->localTransform);
        Skin                   skin = skins[node->skin];
        size_t                 numJoints = skin.joints.size();
        std::vector<glm::mat4> jointMatrices(numJoints);
        for (size_t i = 0; i < std::min(numJoints, skin.inverseBindMatrices.size()); i++)
        {
            jointMatrices[i] = skin.joints[i]->localTransform * skin.inverseBindMatrices[i];
            jointMatrices[i] = inverseTransform * jointMatrices[i];
        }
        // Update ssbo
        engine->updateSkinSSBO(node->skin, jointMatrices);
    }

    for (auto& child : node->children)
    {
        updateJoints(engine, child);
    }
}

std::optional<std::shared_ptr<LoadedGLTF>> loadGltf(VulkanEngine* engine, std::string_view filePath)
{
    std::cout << "Loading GLTF: " << filePath.data() << '\n';

    std::shared_ptr<LoadedGLTF> scene = std::make_shared<LoadedGLTF>();
    scene->creator = engine;
    LoadedGLTF& file = *scene.get();

    static constexpr auto supportedExtensions =
        fastgltf::Extensions::KHR_mesh_quantization |
        fastgltf::Extensions::KHR_texture_transform |
        fastgltf::Extensions::KHR_materials_variants;

    fastgltf::Parser parser(supportedExtensions);

    constexpr auto gltfOptions = fastgltf::Options::DontRequireValidAssetMember | fastgltf::Options::AllowDouble | fastgltf::Options::LoadGLBBuffers | fastgltf::Options::LoadExternalBuffers
        | fastgltf::Options::LoadExternalImages;

    auto result = fastgltf::GltfDataBuffer::FromPath(filePath);
    fastgltf::GltfDataBuffer data = std::move(result.get());
    fastgltf::Asset gltf;

    std::filesystem::path path = filePath;

    auto type = fastgltf::determineGltfFileType(data);
    if (type == fastgltf::GltfType::glTF) {
        auto load = parser.loadGltf(data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        }
        else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else if (type == fastgltf::GltfType::GLB) {
        auto load = parser.loadGltfBinary(data, path.parent_path(), gltfOptions);
        if (load) {
            gltf = std::move(load.get());
        }
        else {
            std::cerr << "Failed to load glTF: " << fastgltf::to_underlying(load.error()) << std::endl;
            return {};
        }
    }
    else {
        std::cerr << "Failed to determine glTF container" << std::endl;
        return {};
    }

    std::vector<DescriptorAllocatorGrowable::PoolSizeRatio> sizes = { { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 3 },
       { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 3 },
       { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1 } };

    file.descriptorPool.init(engine->_device, gltf.materials.size(), sizes);

    std::vector<std::shared_ptr<MeshAsset>> meshes;
    std::vector<std::shared_ptr<Node>> nodes;
    std::vector<AllocatedImage> images;
    std::vector<std::shared_ptr<GLTFMaterial>> materials;

    for (fastgltf::Sampler& sampler : gltf.samplers) {

        VkSamplerCreateInfo sampl = { .sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .pNext = nullptr };
        sampl.maxLod = VK_LOD_CLAMP_NONE;
        sampl.minLod = 0;

        sampl.magFilter = extract_filter(sampler.magFilter.value_or(fastgltf::Filter::Nearest));
        sampl.minFilter = extract_filter(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        sampl.mipmapMode = extract_mipmap_mode(sampler.minFilter.value_or(fastgltf::Filter::Nearest));

        VkSampler newSampler;
        vkCreateSampler(engine->_device, &sampl, nullptr, &newSampler);

        file.samplers.push_back(newSampler);
    }

    // load all textures
    for (fastgltf::Image& image : gltf.images) {
        std::optional<AllocatedImage> img = vkutil::load_image(engine, gltf, image);

        if (img.has_value()) {
            images.push_back(*img);
            file.images[image.name.c_str()] = *img;
        }
        else {
            // we failed to load, so lets give the slot a default white texture to not
            // completely break loading
            images.push_back(engine->_errorCheckerboardImage);
            std::cout << "gltf failed to load texture " << image.name << std::endl;
        }
    }

    file.materialDataBuffer = engine->create_buffer(sizeof(GLTFMetallic_Roughness::MaterialConstants) * gltf.materials.size(),
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, VMA_MEMORY_USAGE_CPU_TO_GPU);
    int data_index = 0;
    GLTFMetallic_Roughness::MaterialConstants* sceneMaterialConstants = (GLTFMetallic_Roughness::MaterialConstants*)file.materialDataBuffer.info.pMappedData;

    for (fastgltf::Material& mat : gltf.materials) {
        std::shared_ptr<GLTFMaterial> newMat = std::make_shared<GLTFMaterial>();
        materials.push_back(newMat);
        file.materials[mat.name.c_str()] = newMat;

        GLTFMetallic_Roughness::MaterialConstants constants;
        constants.colorFactors.x = mat.pbrData.baseColorFactor[0];
        constants.colorFactors.y = mat.pbrData.baseColorFactor[1];
        constants.colorFactors.z = mat.pbrData.baseColorFactor[2];
        constants.colorFactors.w = mat.pbrData.baseColorFactor[3];

        constants.metal_rough_factors.x = mat.pbrData.metallicFactor;
        constants.metal_rough_factors.y = mat.pbrData.roughnessFactor;
        // write material parameters to buffer
        sceneMaterialConstants[data_index] = constants;

        MaterialPass passType = MaterialPass::MainColor;
        if (mat.alphaMode == fastgltf::AlphaMode::Blend) {
            passType = MaterialPass::Transparent;
        }

        GLTFMetallic_Roughness::MaterialResources materialResources;
        // default the material textures
        materialResources.colorImage = engine->_whiteImage;
        materialResources.colorSampler = engine->_defaultSamplerLinear;
        materialResources.metalRoughImage = engine->_whiteImage;
        materialResources.metalRoughSampler = engine->_defaultSamplerLinear;

        // set the uniform buffer for the material data
        materialResources.dataBuffer = file.materialDataBuffer.buffer;
        materialResources.dataBufferOffset = data_index * sizeof(GLTFMetallic_Roughness::MaterialConstants);
        // grab textures from gltf file
        if (mat.pbrData.baseColorTexture.has_value()) {
            size_t img = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].imageIndex.value();
            size_t sampler = gltf.textures[mat.pbrData.baseColorTexture.value().textureIndex].samplerIndex.value();

            materialResources.colorImage = images[img];
            materialResources.colorSampler = file.samplers[sampler];
        }
        // build material
        newMat->data = engine->metalRoughMaterial.write_material(engine->_device, passType, materialResources, file.descriptorPool);

        data_index++;
    }

    // use the same vectors for all meshes so that the memory doesnt reallocate as
// often
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh : gltf.meshes) {
        std::shared_ptr<MeshAsset> newmesh = std::make_shared<MeshAsset>();
        meshes.push_back(newmesh);
        file.meshes[mesh.name.c_str()] = newmesh;
        newmesh->name = mesh.name;

        // clear the mesh arrays each mesh, we dont want to merge them by error
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = (uint32_t)indices.size();
            newSurface.count = (uint32_t)gltf.accessors[p.indicesAccessor.value()].count;

            size_t initial_vtx = vertices.size();

            // load indexes
            {
                fastgltf::Accessor& indexaccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexaccessor.count);

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexaccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initial_vtx);
                    });
            }

            // load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count);

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = { 1, 0, 0 };
                        newvtx.color = glm::vec4{ 1.f };
                        newvtx.uv_x = 0;
                        newvtx.uv_y = 0;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // load vertex normals
            auto normals = p.findAttribute("NORMAL");
            if (normals != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[(*normals).accessorIndex],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // load UVs
            auto uv = p.findAttribute("TEXCOORD_0");
            if (uv != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[(*uv).accessorIndex],
                    [&](glm::vec2 v, size_t index) {
                        vertices[initial_vtx + index].uv_x = v.x;
                        vertices[initial_vtx + index].uv_y = v.y;
                    });
            }

            // load vertex colors
            auto colors = p.findAttribute("COLOR_0");
            if (colors != p.attributes.end()) {

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(*colors).accessorIndex],
                    [&](glm::vec4 v, size_t index) {
                        vertices[initial_vtx + index].color = v;
                    });
            }

            // load skin animation
            auto joints = p.findAttribute("JOINTS_0");
            if (joints != p.attributes.end()) {
                fastgltf::Accessor& jointsAccessor = gltf.accessors[(*joints).accessorIndex];

                fastgltf::iterateAccessorWithIndex<glm::uvec4>(gltf, jointsAccessor,
                    [&](glm::uvec4 jointIdx, size_t index) {
                        vertices[initial_vtx + index].jointIndices = glm::ivec4(jointIdx); // Convert unsigned to signed if necessary
                    });
            }

            auto weights = p.findAttribute("WEIGHTS_0");
            if (weights != p.attributes.end()) {
                fastgltf::Accessor& weightsAccessor = gltf.accessors[(*weights).accessorIndex];

                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, weightsAccessor,
                    [&](glm::vec4 weight, size_t index) {
                        vertices[initial_vtx + index].jointWeights = weight;
                    });
            }

            if (p.materialIndex.has_value()) {
                newSurface.material = materials[p.materialIndex.value()];
            }
            else {
                newSurface.material = materials[0];
            }

            //loop the vertices of this surface, find min/max bounds
            glm::vec3 minpos = vertices[initial_vtx].position;
            glm::vec3 maxpos = vertices[initial_vtx].position;
            for (int i = initial_vtx; i < vertices.size(); i++) {
                minpos = glm::min(minpos, vertices[i].position);
                maxpos = glm::max(maxpos, vertices[i].position);
            }
            // calculate origin and extents from the min/max, use extent lenght for radius
            newSurface.bounds.origin = (maxpos + minpos) / 2.f;
            newSurface.bounds.extents = (maxpos - minpos) / 2.f;
            newSurface.bounds.sphereRadius = glm::length(newSurface.bounds.extents);

            newmesh->surfaces.push_back(newSurface);
        }

        newmesh->meshBuffers = engine->uploadMesh(indices, vertices);
    }

    // load all nodes and their meshes
    for (fastgltf::Node& node : gltf.nodes) {
        std::shared_ptr<Node> newNode;

        // find if the node has a mesh, and if it does hook it to the mesh pointer and allocate it with the meshnode class
        if (node.meshIndex.has_value()) {
            newNode = std::make_shared<MeshNode>();
            static_cast<MeshNode*>(newNode.get())->mesh = meshes[*node.meshIndex];
        }
        else {
            newNode = std::make_shared<Node>();
        }

        nodes.push_back(newNode);
        file.nodes[node.name.c_str()];

        std::visit(
            fastgltf::visitor{
                [&](fastgltf::math::fmat4x4 matrix) {
                // Copy the matrix data directly into the node's local transform
                static_assert(sizeof(newNode->localTransform) == sizeof(matrix), "Size mismatch in matrix copy");
                memcpy(&newNode->localTransform, matrix.data(), sizeof(matrix));
            },
            [&](fastgltf::TRS transform) {
                // Decompose transform into translation, rotation, and scale
                glm::vec3 translation(transform.translation[0], transform.translation[1], transform.translation[2]);
                fastgltf::math::fquat rotation = transform.rotation;
                glm::vec3 scale(transform.scale[0], transform.scale[1], transform.scale[2]);

                // Construct transformation matrices
                glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), translation);

                glm::quat glmQuat(rotation.w(), rotation.x(), rotation.y(), rotation.z());

                glm::mat4 rotationMatrix = glm::tmat4x4<float>(glmQuat);
                glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), scale);

                // Combine transformations
                newNode->localTransform = translationMatrix * rotationMatrix * scaleMatrix;
            } },
            node.transform);

        if (node.skinIndex.has_value()) {
            newNode->skin = node.skinIndex.value();
        }
    }

    // run loop again to setup transform hierarchy
    for (int i = 0; i < gltf.nodes.size(); i++) {
        fastgltf::Node& node = gltf.nodes[i];
        std::shared_ptr<Node>& sceneNode = nodes[i];

        for (auto& c : node.children) {
            sceneNode->children.push_back(nodes[c]);
            nodes[c]->parent = sceneNode;
        }
    }

    // find the top nodes, with no parents
    for (auto& node : nodes) {
        if (node->parent.lock() == nullptr) {
            file.topNodes.push_back(node);
            node->refreshTransform(glm::mat4{ 1.f });
        }
    }

    scene->skins.resize(gltf.skins.size());

    for (size_t i = 0; i < gltf.skins.size(); i++) {
        auto& skin = gltf.skins[i];

        scene->skins[i].name = skin.name;

        // Find the root node of the skeleton
        //scene->skins[i].skeletonRoot = nodes[skin.skeleton.value()].get();WWW

        for (int jointIndex : skin.joints) {
            if (jointIndex >= 0 && jointIndex < nodes.size()) {
                scene->skins[i].joints.push_back(nodes[jointIndex].get());
            }
        }


        if (skin.inverseBindMatrices.has_value()) {
            const auto& accessor = gltf.accessors[skin.inverseBindMatrices.value()];

            if (accessor.bufferViewIndex.has_value()) {
                const auto& bufferView = gltf.bufferViews[accessor.bufferViewIndex.value()];
                const auto& buffer = gltf.buffers[bufferView.bufferIndex];

                // Resize the inverse bind matrices storage in the scene
                scene->skins[i].inverseBindMatrices.resize(skin.inverseBindMatrices.value());

                // Copy the data from the buffer into the inverseBindMatrices vector
                auto data = std::get<fastgltf::sources::Array>(buffer.data);
                void* dataPtr = &data.bytes[accessor.byteOffset + bufferView.byteOffset];

                memcpy(scene->skins[i].inverseBindMatrices.data(), dataPtr, sizeof(glm::mat4)* scene->skins[i].inverseBindMatrices.size());

                // Create a GPU buffer for the inverse bind matrices
                scene->skins[i].ssbo = engine->create_buffer(
                    sizeof(glm::mat4) * scene->skins[i].inverseBindMatrices.size(),
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_CPU_TO_GPU
                );
            }
        }
    }

    scene->animations.resize(gltf.animations.size());

    for (size_t i = 0; i < gltf.animations.size(); i++)
    {
        auto glTFAnimation = gltf.animations[i];
        scene->animations[i].name = glTFAnimation.name;

        // Samplers
        scene->animations[i].samplers.resize(glTFAnimation.samplers.size());
        for (size_t j = 0; j < glTFAnimation.samplers.size(); j++)
        {
            auto glTFSampler = glTFAnimation.samplers[j];
            AnimationSampler& dstSampler = scene->animations[i].samplers[j];
            dstSampler.interpolation = glTFSampler.interpolation;

            // Read sampler keyframe input time values
            {
                const auto accessor = gltf.accessors[glTFSampler.inputAccessor];
                const auto& bufferView = gltf.bufferViews[accessor.bufferViewIndex.value()];
                const auto& buffer = gltf.buffers[bufferView.bufferIndex];
                auto data = std::get<fastgltf::sources::Array>(buffer.data);
                void* dataPtr = &data.bytes[accessor.byteOffset + bufferView.byteOffset];
                const float* buf = static_cast<const float*>(dataPtr);
                for (size_t index = 0; index < accessor.count; index++)
                {
                    dstSampler.inputs.push_back(buf[index]);
                }
                // Adjust animation's start and end times
                for (auto input : scene->animations[i].samplers[j].inputs)
                {
                    if (input < scene->animations[i].start)
                    {
                        scene->animations[i].start = input;
                    };
                    if (input > scene->animations[i].end)
                    {
                        scene->animations[i].end = input;
                    }
                }
            }

            // Read sampler keyframe output translate/rotate/scale values
            {
                const fastgltf::Accessor& accessor = gltf.accessors[glTFSampler.inputAccessor];
                const fastgltf::BufferView& bufferView = gltf.bufferViews[accessor.bufferViewIndex.value()];
                const fastgltf::Buffer& buffer = gltf.buffers[bufferView.bufferIndex];
                auto data = std::get<fastgltf::sources::Array>(buffer.data);
                void* dataPtr = &data.bytes[accessor.byteOffset + bufferView.byteOffset];
                switch (accessor.type)
                {
                case fastgltf::AccessorType::Vec3: {
                    const glm::vec3* buf = static_cast<const glm::vec3*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f));
                    }
                    break;
                }
                case fastgltf::AccessorType::Vec4: {
                    const glm::vec4* buf = static_cast<const glm::vec4*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        dstSampler.outputsVec4.push_back(buf[index]);
                    }
                    break;
                }
                case fastgltf::AccessorType::Scalar: {
                    const float* buf = static_cast<const float*>(dataPtr);
                    for (size_t index = 0; index < accessor.count; index++)
                    {
                        dstSampler.outputsVec4.push_back(glm::vec4(buf[index], 0.0f, 0.0f, 0.0f));
                    }
                    break;
                }
                default: {
                    std::cout << "unknown type: " << std::endl;
                    break;
                }
                }
            }
        }

        // Channels
        scene->animations[i].channels.resize(glTFAnimation.channels.size());
        for (size_t j = 0; j < glTFAnimation.channels.size(); j++)
        {
            fastgltf::AnimationChannel glTFChannel = glTFAnimation.channels[j];
            AnimationChannel& dstChannel = scene->animations[i].channels[j];
            dstChannel.path = glTFChannel.path;
            dstChannel.samplerIndex = glTFChannel.samplerIndex;
            dstChannel.node = nodes[glTFChannel.nodeIndex.value()].get();
        }
    }

    return scene;
}


void GLTFMetallic_Roughness::build_pipelines(VulkanEngine* engine)
{
    VkShaderModule meshFragShader;
    if (!vkutil::load_shader_module("../shaders/mesh.frag.spv", engine->_device, &meshFragShader)) {
        throw std::runtime_error("Error when building the triangle fragment shader module");
    }

    VkShaderModule meshVertexShader;
    if (!vkutil::load_shader_module("../shaders/mesh.vert.spv", engine->_device, &meshVertexShader)) {
        throw std::runtime_error("Error when building the triangle vertex shader module");
    }

    VkPushConstantRange matrixRange{};
    matrixRange.offset = 0;
    matrixRange.size = sizeof(GPUDrawPushConstants);
    matrixRange.stageFlags = VK_SHADER_STAGE_VERTEX_BIT;

    DescriptorLayoutBuilder layoutBuilder;
    layoutBuilder.add_binding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    layoutBuilder.add_binding(1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    layoutBuilder.add_binding(2, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    materialLayout = layoutBuilder.build(engine->_device, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT);

    VkDescriptorSetLayout layouts[] = { engine->_gpuSceneDataDescriptorLayout,
        materialLayout, engine->_skinDescriptorLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 3;
    mesh_layout_info.pSetLayouts = layouts;
    mesh_layout_info.pPushConstantRanges = &matrixRange;
    mesh_layout_info.pushConstantRangeCount = 1;

    VkPipelineLayout newLayout;
    VK_CHECK(vkCreatePipelineLayout(engine->_device, &mesh_layout_info, nullptr, &newLayout));

    opaquePipeline.layout = newLayout;
    transparentPipeline.layout = newLayout;

    // build the stage-create-info for both vertex and fragment stages. This lets
    // the pipeline know the shader modules per stage
    PipelineBuilder pipelineBuilder;
    pipelineBuilder.set_shaders(meshVertexShader, meshFragShader);
    pipelineBuilder.set_input_topology(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    pipelineBuilder.set_polygon_mode(VK_POLYGON_MODE_FILL);
    pipelineBuilder.set_cull_mode(VK_CULL_MODE_NONE, VK_FRONT_FACE_CLOCKWISE);
    pipelineBuilder.set_multisampling_none();
    pipelineBuilder.disable_blending();
    pipelineBuilder.enable_depthtest(true, VK_COMPARE_OP_GREATER_OR_EQUAL);

    //render format
    pipelineBuilder.set_color_attachment_format(engine->_drawImage.imageFormat);
    pipelineBuilder.set_depth_format(engine->_depthImage.imageFormat);

    // use the triangle layout we created
    pipelineBuilder._pipelineLayout = newLayout;

    // finally build the pipeline
    opaquePipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    // create the transparent variant
    pipelineBuilder.enable_blending_additive();

    pipelineBuilder.enable_depthtest(false, VK_COMPARE_OP_GREATER_OR_EQUAL);

    transparentPipeline.pipeline = pipelineBuilder.build_pipeline(engine->_device);

    vkDestroyShaderModule(engine->_device, meshFragShader, nullptr);
    vkDestroyShaderModule(engine->_device, meshVertexShader, nullptr);
}

MaterialInstance GLTFMetallic_Roughness::write_material(VkDevice device, MaterialPass pass, const MaterialResources& resources, DescriptorAllocatorGrowable& descriptorAllocator)
{
    MaterialInstance matData;
    matData.passType = pass;
    if (pass == MaterialPass::Transparent) {
        matData.pipeline = &transparentPipeline;
    }
    else {
        matData.pipeline = &opaquePipeline;
    }

    matData.materialSet = descriptorAllocator.allocate(device, materialLayout);


    writer.clear();
    writer.write_buffer(0, resources.dataBuffer, sizeof(MaterialConstants), resources.dataBufferOffset, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    writer.write_image(1, resources.colorImage.imageView, resources.colorSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    writer.write_image(2, resources.metalRoughImage.imageView, resources.metalRoughSampler, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);

    writer.update_set(device, matData.materialSet);

    return matData;
}

void LoadedGLTF::Draw(const glm::mat4& topMatrix, DrawContext& ctx)
{
    // create renderables from the scenenodes
    for (auto& n : topNodes) {
        n->Draw(topMatrix, ctx);
    }
}

void LoadedGLTF::clearAll()
{
    VkDevice dv = creator->_device;

    descriptorPool.destroy_pools(dv);
    creator->destroy_buffer(materialDataBuffer);

    for (auto& [k, v] : meshes) {

        creator->destroy_buffer(v->meshBuffers.indexBuffer);
        creator->destroy_buffer(v->meshBuffers.vertexBuffer);
    }

    for (auto& [k, v] : images) {

        if (v.image == creator->_errorCheckerboardImage.image) {
            //dont destroy the default images
            continue;
        }
        creator->destroy_image(v);
    }

    for (auto& sampler : samplers) {
        vkDestroySampler(dv, sampler, nullptr);
    }
}
