cmake_minimum_required(VERSION 3.10)

project(MyVulkanApp)

set(CMAKE_CXX_STANDARD 20)

# Указание пути к Vulkan SDK
set(VULKAN_SDK "C:/VulkanSDK")  # Замените на путь к вашему Vulkan SDK
set(VULKAN_INCLUDE_DIR "${VULKAN_SDK}/1.3.296.0/Include")
set(VULKAN_LIB_DIR "${VULKAN_SDK}/1.3.296.0/Lib")

# Находим все .cpp файлы в папке src и добавляем их в проект
file(GLOB SOURCES "src/*.cpp")

# Указываем путь к ImGui
set(IMGUI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/imgui)
set(STB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rd-party/stb)

# Создаем исполнимый файл с использованием найденных исходников
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


# Добавляем пути для заголовочных файлов
target_include_directories(MyVulkanApp PRIVATE 
    ${VULKAN_INCLUDE_DIR}            
    ${CMAKE_CURRENT_SOURCE_DIR}       
    ${IMGUI_DIR}                      
    ${IMGUI_DIR}/backends  
    ${STB_DIR}           
)

# Линковка с Vulkan, SDL2 и vk-bootstrap
target_link_libraries(MyVulkanApp  
    ${VULKAN_LIB_DIR}/vulkan-1.lib  
    ${VULKAN_LIB_DIR}/SDL2.lib  
    ${VULKAN_LIB_DIR}/SDL2main.lib 
    vk-bootstrap::vk-bootstrap
    fastgltf
)