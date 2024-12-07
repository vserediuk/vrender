#include "VulkanEngine.h"

int main(int argc, char* argv[])
{
    VulkanEngine engine;

    try {
        engine.init();
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    engine.run();

    engine.cleanup();

    return EXIT_SUCCESS;
}