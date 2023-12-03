#include <fstream>
#include "ComputeEngine.hpp"
#include "TaskBuilder.hpp"
#include "Log.hpp"

#define IMAGE_HEADER_OFFSET 16
#define LABEL_HEADER_OFFSET 8

std::vector<uint8_t> ImportMnist(const char* path, size_t offset) {
    if (std::ifstream ifs{path, std::ios::binary | std::ios::ate}) {
        size_t fileSize = size_t(ifs.tellg()) + offset;
        std::vector<uint8_t> contents(fileSize);

        ifs.seekg(offset).read((char*)contents.data(), fileSize);
        return contents; // RVO
    } else {
        nn::LogError("could not open", path);
        return {};
    }
}

int main() {
    try {
        nn::ComputeEngine computeEngine;
        nn::TaskBuilder taskBuilder(computeEngine);

        taskBuilder.SetShader("tests/spirv/mnist.comp.spv");

        taskBuilder.SetBuffers({
            .SrcCount = 28,
            .SrcSize = sizeof(int32_t),
            .DstCount = 28,
            .DstSize = sizeof(int32_t),
        });

        taskBuilder.SetPipeline({
            .bindings =
                {
                    {
                        .binding = 0,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eCompute,
                    },
                    {
                        .binding = 1,
                        .descriptorType = vk::DescriptorType::eStorageBuffer,
                        .descriptorCount = 1,
                        .stageFlags = vk::ShaderStageFlagBits::eCompute,
                    },
                },
        });

        auto task = taskBuilder.create();
        computeEngine.PushTask(task);

        computeEngine.ExecuteTasks();
    } catch (std::exception& e) {
        nn::LogError(e.what());
        throw e;
    }

    auto trainingImages = ImportMnist("tests/mnist/dataset/train-images-idx3-ubyte", IMAGE_HEADER_OFFSET);
    auto trainingLabels = ImportMnist("tests/mnist/dataset/train-labels-idx1-ubyte", LABEL_HEADER_OFFSET);

    return 0;
}
