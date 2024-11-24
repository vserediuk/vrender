cmake_minimum_required(VERSION 3.10)

project(MyVulkanApp)

set(CMAKE_CXX_STANDARD 20)

set(VULKAN_SDK "C:/VulkanSDK")
set(VULKAN_INCLUDE_DIR "${VULKAN_SDK}/1.3.296.0/Include")
set(VULKAN_LIB_DIR "${VULKAN_SDK}/1.3.296.0/Lib")
set(GLSLANG_VALIDATOR "${VULKAN_SDK}/1.3.296.0/Bin/glslangValidator.exe")

file(GLOB SOURCES "src/*.cpp")

set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/imgui)
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/stb)

add_executable(MyVulkanApp ${SOURCES} 
    ${IMGUI_DIR}/backends/imgui_impl_sdl2.cpp 
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp 
    ${IMGUI_DIR}/imgui.cpp 
    ${IMGUI_DIR}/imgui_draw.cpp 
    ${IMGUI_DIR}/imgui_demo.cpp 
    ${IMGUI_DIR}/imgui_tables.cpp 
    ${IMGUI_DIR}/imgui_widgets.cpp)

add_subdirectory(3rd-party/vk-bootstrap)
add_subdirectory(3rd-party/fastgltf)

target_include_directories(MyVulkanApp PRIVATE 
    ${VULKAN_INCLUDE_DIR}            
    ${CMAKE_CURRENT_SOURCE_DIR}       
    ${IMGUI_DIR}                       
    ${IMGUI_DIR}/backends  
    ${STB_DIR}           
)

target_link_libraries(MyVulkanApp  
    ${VULKAN_LIB_DIR}/vulkan-1.lib  
    ${VULKAN_LIB_DIR}/SDL2.lib  
    ${VULKAN_LIB_DIR}/SDL2main.lib 
    vk-bootstrap::vk-bootstrap
    fastgltf
)

file(GLOB SHADER_FILES "shaders/*.frag" "shaders/*.vert" "shaders/*.comp")

set(SPV_FILES)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME_WE)
    get_filename_component(SHADER_EXT ${SHADER} EXT)

    set(OUTPUT_FILE "${CMAKE_CURRENT_SOURCE_DIR}/shaders/${SHADER_NAME}${SHADER_EXT}.spv")

    list(FIND SPV_FILES ${OUTPUT_FILE} SPV_FILE_INDEX)
    if(SPV_FILE_INDEX EQUAL -1)
        add_custom_command(
            OUTPUT ${OUTPUT_FILE}
            COMMAND ${GLSLANG_VALIDATOR} -V ${SHADER} -o ${OUTPUT_FILE}
            DEPENDS ${SHADER}
            COMMENT "Compiling shader ${SHADER_NAME}${SHADER_EXT} to ${OUTPUT_FILE}"
        )
        list(APPEND SPV_FILES ${OUTPUT_FILE})
    endif()
endforeach()

add_custom_target(compile_shaders ALL DEPENDS ${SPV_FILES})

add_dependencies(MyVulkanApp compile_shaders)
