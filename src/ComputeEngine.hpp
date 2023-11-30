#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1
#include <vulkan/vulkan.hpp>
#include <unordered_map>

namespace nn {

class Buffer {
public:
    Buffer(vk::Device& device)
        : m_Device(device) {}

    vk::Buffer Buf;
    vk::DeviceMemory Mem;
    uint32_t Size = 256 * sizeof(int32_t);

    void Destroy() {
        m_Device.destroyBuffer(Buf);
        m_Device.freeMemory(Mem);
    }

private:
    vk::Device& m_Device;
};

class ComputeEngine {
public:
    ComputeEngine();
    ~ComputeEngine();

    void AddShader(const std::string& path);
    void RemoveShader(const std::string& path);
    void AddBuffers();
    void AddPipeline();
    void AddCommandPool();
    void RecordCommands();

private:
    vk::Instance m_Instance;
    vk::PhysicalDevice m_PhyscialDevice;
    vk::Device m_Device;
    vk::Pipeline m_ComputePipeline;
    vk::PipelineLayout m_PipelineLayout;
    vk::PipelineCache m_PipelineCache;
    vk::DescriptorSetLayout m_DescriptorSetLayout;
    vk::DescriptorSet m_DescriptorSet;
    vk::DescriptorPool m_DescriptorPool;
    vk::CommandPool m_CommandPool;
    vk::CommandBuffer m_CommandBuffer;
    vk::Fence m_Fence;
    uint32_t m_ComputeQueueIndex;
    std::unordered_map<std::string, vk::ShaderModule> m_Shaders;
    Buffer m_SrcBuffer;
    Buffer m_DstBuffer;

    const std::vector<const char*> m_ValidationLayers = {
        "VK_LAYER_KHRONOS_validation",
    };

    const std::vector<const char*> m_Extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
};

} // namespace nn