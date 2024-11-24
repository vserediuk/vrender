#include "vk_loader.h"
#include "VulkanEngine.h"
#include "vk_images.h"
#include <variant>
#include "fastgltf/glm_element_traits.hpp"
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
        materialLayout };

    VkPipelineLayoutCreateInfo mesh_layout_info = vkinit::pipeline_layout_create_info();
    mesh_layout_info.setLayoutCount = 2;
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
