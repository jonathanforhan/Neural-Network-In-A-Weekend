#include "Task.hpp"
#include <fstream>
#include "ComputeEngine.hpp"
#include "Log.hpp"

namespace nn {

void Task::Execute(const ComputeEngine& engine) {
    vk::CommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = vk::StructureType::eCommandBufferBeginInfo,
        .pNext = nullptr,
        .flags = {},
        .pInheritanceInfo = nullptr,
    };

    m_CommandBuffer->begin(commandBufferBeginInfo);
    m_CommandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, *m_ComputePipeline);
    m_CommandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, *m_PipelineLayout, 0, {*m_DescriptorSet}, {});
    m_CommandBuffer->dispatch(64, 1, 1);
    m_CommandBuffer->end();

    vk::Queue queue = engine.Device().getQueue(engine.ComputeQueue(), 0);

    vk::FenceCreateInfo fenceCreateInfo = {
        .sType = vk::StructureType::eFenceCreateInfo,
        .pNext = nullptr,
        .flags = {},
    };
    m_Fence = engine.Device().createFenceUnique(fenceCreateInfo);

    vk::SubmitInfo submitInfo = {
        .sType = vk::StructureType::eSubmitInfo,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &*m_CommandBuffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };

    using mu = std::chrono::microseconds;
    auto start = std::chrono::high_resolution_clock::now();

    queue.submit({submitInfo}, *m_Fence);

    vk::Result result = engine.Device().waitForFences({*m_Fence}, true, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        LogWarning("fence wait result error on line", __LINE__, __FILE__);
    }

    auto finish = std::chrono::high_resolution_clock::now();

    int32_t* srcBufferPtr = (int32_t*)engine.Device().mapMemory(*m_Src.Memory, 0, m_Src.Count * m_Src.Size);
    LogInfo("//--- Source Buffer ---//");
    for (uint32_t i = 0; i < m_Src.Count; i++) {
        printf("%d : %d\n", i, srcBufferPtr[i]);
    }
    engine.Device().unmapMemory(*m_Src.Memory);

    int32_t* dstBufferPtr = (int32_t*)engine.Device().mapMemory(*m_Dst.Memory, 0, m_Dst.Count * m_Dst.Size);
    LogInfo("//--- Destination Buffer ---//");
    for (uint32_t i = 0; i < m_Dst.Count; i++) {
        printf("%d : %d\n", i, dstBufferPtr[i]);
    }
    engine.Device().unmapMemory(*m_Dst.Memory);

    LogInfo("GPU round trip:", std::chrono::duration_cast<mu>(finish - start).count(), "microseconds");
}

void Task::setShader(const ComputeEngine& engine, std::string_view path) {
    std::vector<char> contents;

    std::ifstream ifs(path.data(), std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        LogError("could not open", path);
        return;
    }

    const size_t fileSize = ifs.tellg();
    ifs.seekg(0);
    contents.resize(fileSize);
    ifs.read(contents.data(), fileSize);
    ifs.close();

    vk::ShaderModuleCreateInfo shaderModuleCreateInfo = {
        .sType = vk::StructureType::eShaderModuleCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .codeSize = fileSize,
        .pCode = reinterpret_cast<uint32_t*>(contents.data()),
    };

    m_Shader = engine.Device().createShaderModuleUnique(shaderModuleCreateInfo);
}

void Task::setBuffers(const ComputeEngine& engine, const BufferSpecification& spec) {
    uint32_t computeIndex = engine.ComputeQueue();

    m_Src.Size = uint32_t(spec.SrcSize);
    m_Src.Count = uint32_t(spec.SrcCount);

    m_Dst.Size = uint32_t(spec.DstSize);
    m_Dst.Count = uint32_t(spec.DstCount);

    vk::BufferCreateInfo srcBufferCreateInfo = {
        .sType = vk::StructureType::eBufferCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .size = m_Src.Size * m_Src.Count,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &computeIndex,
    };
    auto dstBufferCreateInfo = srcBufferCreateInfo;
    dstBufferCreateInfo.size = m_Dst.Count * m_Dst.Size;

    m_Src.Buffer = engine.Device().createBufferUnique(srcBufferCreateInfo);
    m_Dst.Buffer = engine.Device().createBufferUnique(dstBufferCreateInfo);

    vk::MemoryRequirements srcBufferMemoryRequirements = engine.Device().getBufferMemoryRequirements(*m_Src.Buffer);
    vk::MemoryRequirements dstBufferMemoryRequirements = engine.Device().getBufferMemoryRequirements(*m_Dst.Buffer);

    vk::PhysicalDeviceMemoryProperties memoryProperties = engine.GPU().getMemoryProperties();

    uint32_t memoryTypeIndex = UINT32_MAX;
    vk::DeviceSize memoryHeapSize = UINT32_MAX;
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        vk::MemoryType memoryType = memoryProperties.memoryTypes[i];
        if ((vk::MemoryPropertyFlagBits::eHostVisible & memoryType.propertyFlags) &&
            (vk::MemoryPropertyFlagBits::eHostCoherent & memoryType.propertyFlags)) {
            memoryHeapSize = memoryProperties.memoryHeaps[memoryType.heapIndex].size;
            memoryTypeIndex = i;
            break;
        }
    }

    vk::MemoryAllocateInfo srcBufferAllocInfo = {
        .sType = vk::StructureType::eMemoryAllocateInfo,
        .pNext = nullptr,
        .allocationSize = srcBufferMemoryRequirements.size,
        .memoryTypeIndex = memoryTypeIndex,
    };
    auto dstBufferAllocInfo = srcBufferAllocInfo;
    dstBufferAllocInfo.allocationSize = dstBufferMemoryRequirements.size;

    m_Src.Memory = engine.Device().allocateMemoryUnique(srcBufferAllocInfo);
    m_Dst.Memory = engine.Device().allocateMemoryUnique(dstBufferAllocInfo);

    int32_t* srcBufferPtr = (int32_t*)engine.Device().mapMemory(*m_Src.Memory, 0, m_Src.Count * m_Src.Size);
    for (uint32_t i = 0; i < m_Src.Count; i++) {
        srcBufferPtr[i] = i;
    }
    engine.Device().unmapMemory(*m_Src.Memory);

    engine.Device().bindBufferMemory(*m_Src.Buffer, *m_Src.Memory, 0);
    engine.Device().bindBufferMemory(*m_Dst.Buffer, *m_Dst.Memory, 0);
}

void Task::setPipeline(const ComputeEngine& engine, const PipelineSpecification& spec) {
    vk::DescriptorSetLayoutCreateInfo descriptSetLayoutCreateInfo = {
        .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .bindingCount = 2,
        .pBindings = spec.bindings.data(),
    };

    m_DescriptorSetLayout = engine.Device().createDescriptorSetLayoutUnique(descriptSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .setLayoutCount = 1,
        .pSetLayouts = &*m_DescriptorSetLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    m_PipelineLayout = engine.Device().createPipelineLayoutUnique(pipelineLayoutCreateInfo);
    m_PipelineCache = engine.Device().createPipelineCacheUnique({});

    vk::PipelineShaderStageCreateInfo shaderStageCreateInfo = {
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = *m_Shader,
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };

    vk::ComputePipelineCreateInfo computePipelineCreateInfo = {
        .sType = vk::StructureType::eComputePipelineCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .stage = shaderStageCreateInfo,
        .layout = *m_PipelineLayout,
        .basePipelineHandle = {},
        .basePipelineIndex = {},
    };

    vk::ResultValue result = engine.Device().createComputePipelineUnique(*m_PipelineCache, computePipelineCreateInfo);
    if (result.result != vk::Result::eSuccess) {
        LogError("could not create compute pipeline");
    }
    m_ComputePipeline = std::move(result.value);

    vk::DescriptorPoolSize descriptorPoolSize = {
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 2,
    };

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        .sType = vk::StructureType::eDescriptorPoolCreateInfo,
        .pNext = nullptr,
        .flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &descriptorPoolSize,
    };

    m_DescriptorPool = engine.Device().createDescriptorPoolUnique(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo = {
        .sType = vk::StructureType::eDescriptorSetAllocateInfo,
        .pNext = nullptr,
        .descriptorPool = *m_DescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &*m_DescriptorSetLayout,
    };
    std::vector<vk::UniqueDescriptorSet> descriptorSets =
        engine.Device().allocateDescriptorSetsUnique(descriptorSetAllocInfo);
    m_DescriptorSet = std::move(descriptorSets.front());

    vk::DescriptorBufferInfo srcBufferInfo = {
        .buffer = *m_Src.Buffer,
        .offset = 0,
        .range = m_Src.Count * m_Src.Size,
    };

    vk::DescriptorBufferInfo dstBufferInfo = {
        .buffer = *m_Dst.Buffer,
        .offset = 0,
        .range = m_Dst.Count * m_Dst.Size,
    };

    // describes write operations
    // we just use our input and output buffers for this, may expand if future
    std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {
            .sType = vk::StructureType::eWriteDescriptorSet,
            .pNext = nullptr,
            .dstSet = *m_DescriptorSet,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pImageInfo = nullptr,
            .pBufferInfo = &srcBufferInfo,
            .pTexelBufferView = nullptr,
        },
        {
            .sType = vk::StructureType::eWriteDescriptorSet,
            .pNext = nullptr,
            .dstSet = *m_DescriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pImageInfo = nullptr,
            .pBufferInfo = &dstBufferInfo,
            .pTexelBufferView = nullptr,
        },
    };

    engine.Device().updateDescriptorSets(writeDescriptorSets, {});
}

void Task::setCommandPool(const ComputeEngine& engine) {
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {
        .sType = vk::StructureType::eCommandPoolCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .queueFamilyIndex = engine.ComputeQueue(),
    };
    
    m_CommandPool = engine.Device().createCommandPoolUnique(commandPoolCreateInfo);

    vk::CommandBufferAllocateInfo commandBufferAllocInfo = {
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .pNext = nullptr,
        .commandPool = *m_CommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    std::vector<vk::UniqueCommandBuffer> commandBuffers =
        engine.Device().allocateCommandBuffersUnique(commandBufferAllocInfo);
    m_CommandBuffer = std::move(commandBuffers.front());
}

} // namespace nn