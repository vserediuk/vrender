#include "vk_loader.h"
#include "VulkanEngine.h"

std::optional<std::vector<std::shared_ptr<MeshAsset>>> loadGltfMeshes(VulkanEngine* engine, std::filesystem::path filePath) {
    if (!std::filesystem::exists(filePath)) {
        std::cerr << "Failed to find " << filePath << '\n';
        return std::nullopt; // Return early if the file doesn't exist
    }

    std::cout << "Loading " << filePath << '\n';

    // Define supported extensions
    static constexpr auto supportedExtensions =
        fastgltf::Extensions::KHR_mesh_quantization |
        fastgltf::Extensions::KHR_texture_transform |
        fastgltf::Extensions::KHR_materials_variants;

    fastgltf::Parser parser(supportedExtensions);

    // Define options for parsing the GLTF
    constexpr auto gltfOptions =
        fastgltf::Options::DontRequireValidAssetMember |
        fastgltf::Options::AllowDouble |
        fastgltf::Options::LoadGLBBuffers |
        fastgltf::Options::LoadExternalBuffers |
        fastgltf::Options::LoadExternalImages |
        fastgltf::Options::GenerateMeshIndices;

    // Open the GLTF file and check for errors
    auto gltfFile = fastgltf::MappedGltfFile::FromPath(filePath);
    if (!gltfFile) {
        std::cerr << "Failed to open glTF file: " << fastgltf::getErrorMessage(gltfFile.error()) << '\n';
        return std::nullopt; // Return early if file opening fails
    }

    // Load the GLTF asset
    auto asset = parser.loadGltf(gltfFile.get(), filePath.parent_path(), gltfOptions);
    if (asset.error() != fastgltf::Error::None) {
        std::cerr << "Failed to load glTF: " << fastgltf::getErrorMessage(asset.error()) << '\n';
        return std::nullopt; // Return early if parsing fails
    }

    fastgltf::Asset gltf = std::move(asset.get());

    std::vector<std::shared_ptr<MeshAsset>> meshes;

    // Reserve space to minimize reallocations
    std::vector<uint32_t> indices;
    std::vector<Vertex> vertices;

    for (fastgltf::Mesh& mesh : gltf.meshes) {
        MeshAsset newMesh;
        newMesh.name = mesh.name;

        // Clear the mesh arrays for each mesh
        indices.clear();
        vertices.clear();

        for (auto&& p : mesh.primitives) {
            GeoSurface newSurface;
            newSurface.startIndex = static_cast<uint32_t>(indices.size());
            newSurface.count = static_cast<uint32_t>(gltf.accessors[p.indicesAccessor.value()].count);

            size_t initial_vtx = vertices.size();

            // Load indices
            {
                fastgltf::Accessor& indexAccessor = gltf.accessors[p.indicesAccessor.value()];
                indices.reserve(indices.size() + indexAccessor.count);  // Reserve space for indices

                fastgltf::iterateAccessor<std::uint32_t>(gltf, indexAccessor,
                    [&](std::uint32_t idx) {
                        indices.push_back(idx + initial_vtx);
                    });
            }

            // Load vertex positions
            {
                fastgltf::Accessor& posAccessor = gltf.accessors[p.findAttribute("POSITION")->accessorIndex];
                vertices.resize(vertices.size() + posAccessor.count); // Resize for new vertices

                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, posAccessor,
                    [&](glm::vec3 v, size_t index) {
                        Vertex newvtx;
                        newvtx.position = v;
                        newvtx.normal = glm::vec3(1.0f, 0.0f, 0.0f); // Default normal
                        newvtx.color = glm::vec4(1.0f); // Default color
                        newvtx.uv_x = 0.0f; // Default UV
                        newvtx.uv_y = 0.0f;
                        vertices[initial_vtx + index] = newvtx;
                    });
            }

            // Load vertex normals
            if (auto normals = p.findAttribute("NORMAL"); normals != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec3>(gltf, gltf.accessors[normals->accessorIndex],
                    [&](glm::vec3 v, size_t index) {
                        vertices[initial_vtx + index].normal = v;
                    });
            }

            // Load UVs
            if (auto uv = p.findAttribute("TEXCOORD_0"); uv != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec2>(gltf, gltf.accessors[uv->accessorIndex],
                    [&](glm::vec2 uv, size_t index) {
                        vertices[initial_vtx + index].uv_x = uv.x;
                        vertices[initial_vtx + index].uv_y = uv.y;
                    });
            }

            // Load vertex colors
            if (auto colors = p.findAttribute("COLOR_0"); colors != p.attributes.end()) {
                fastgltf::iterateAccessorWithIndex<glm::vec4>(gltf, gltf.accessors[(colors)->accessorIndex],
                    [&](glm::vec4 color, size_t index) {
                        vertices[initial_vtx + index].color = color;
                    });
            }

            newMesh.surfaces.push_back(newSurface);
        }

        // Optional: Override vertex colors with normals
        constexpr bool OverrideColors = true;
        if (OverrideColors) {
            for (Vertex& vtx : vertices) {
                vtx.color = glm::vec4(vtx.normal, 1.f); // Color override with normal
            }
        }

        // Upload the mesh data to Vulkan (replace with actual Vulkan upload)
        newMesh.meshBuffers = engine->uploadMesh(indices, vertices);

        meshes.emplace_back(std::make_shared<MeshAsset>(std::move(newMesh)));
    }

    return meshes;
}