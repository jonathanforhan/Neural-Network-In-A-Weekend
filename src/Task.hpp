#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1
#include <vulkan/vulkan.hpp>

namespace nn {

class ComputeEngine;

struct BufferSpecification {
    size_t SrcCount; // elements in buffer
    size_t SrcSize;  // size of a given element
    size_t DstCount;
    size_t DstSize;
};

struct PipelineSpecification {
    std::vector<vk::DescriptorSetLayoutBinding> bindings;
};

class Task {
public:
    Task() = default;
    Task(const Task&) = delete;
    void operator=(const Task&) = delete;
    Task(Task&&) noexcept = default;
    Task& operator=(Task&&) noexcept = default;

    void Execute(const ComputeEngine& engine);

private:
    void setShader(const ComputeEngine& engine, std::string_view path);
    void setBuffers(const ComputeEngine& engine, const BufferSpecification& spec);
    void setPipeline(const ComputeEngine& engine, const PipelineSpecification& spec);
    void setCommandPool(const ComputeEngine& engine);

    struct {
        vk::UniqueBuffer Buffer;
        vk::UniqueDeviceMemory Memory;
        uint32_t Size;
        uint32_t Count;
    } m_Src;

    struct {
        vk::UniqueBuffer Buffer;
        vk::UniqueDeviceMemory Memory;
        uint32_t Size;
        uint32_t Count;
    } m_Dst;

    vk::UniquePipeline m_ComputePipeline;
    vk::UniquePipelineLayout m_PipelineLayout;
    vk::UniquePipelineCache m_PipelineCache;
    vk::UniqueDescriptorPool m_DescriptorPool;
    vk::UniqueDescriptorSetLayout m_DescriptorSetLayout;
    vk::UniqueDescriptorSet m_DescriptorSet;
    vk::UniqueCommandPool m_CommandPool;
    vk::UniqueCommandBuffer m_CommandBuffer;
    vk::UniqueShaderModule m_Shader;
    vk::UniqueFence m_Fence;

    friend class TaskBuilder;
};

} // namespace nn
