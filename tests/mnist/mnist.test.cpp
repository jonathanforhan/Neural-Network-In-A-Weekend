#include <fstream>
#include "ComputeEngine.hpp"
#include "Log.hpp"

#define IMAGE_HEADER_OFFSET 16
#define LABEL_HEADER_OFFSET 8

void ImportMnist(const char* path, std::vector<uint8_t>& data, size_t offset) {
    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        nn::LogError("could not open", path);
        return;
    }

    size_t fileSize = size_t(ifs.tellg()) + offset;
    ifs.seekg(offset);
    data.resize(fileSize);
    ifs.read((char*)data.data(), fileSize);
    ifs.close();
}

int main() {
    try {
        nn::ComputeEngine computeEngine;
        computeEngine.AddBuffers();
        computeEngine.AddShader("tests/spirv/mnist.comp.spv");
        computeEngine.AddPipeline();
        computeEngine.AddCommandPool();
        computeEngine.RecordCommands();
    } catch (std::exception& e) {
        nn::LogError(e.what());
        throw e;
    }

    std::vector<uint8_t> trainingImages;
    ImportMnist("tests/mnist/dataset/train-images-idx3-ubyte", trainingImages, IMAGE_HEADER_OFFSET);

    std::vector<uint8_t> trainingLabels;
    ImportMnist("tests/mnist/dataset/train-labels-idx1-ubyte", trainingLabels, LABEL_HEADER_OFFSET);

    return 0;
}
