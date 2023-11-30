#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include <fstream>
#include "Log.hpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    (void)messageType;
    (void)pUserData;

    switch (messageSeverity) {
        default:
            // nn::Log(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            nn::LogWarning(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            nn::LogError(pCallbackData->pMessage);
            break;
    }

    return VK_FALSE;
}

namespace nn {

ComputeEngine::ComputeEngine()
    : m_SrcBuffer(m_Device),
      m_DstBuffer(m_Device),
      m_ComputeQueueIndex(0) {
    //--- Application
    vk::ApplicationInfo applicationInfo = {
        .sType = vk::StructureType::eApplicationInfo,
        .pNext = nullptr,
        .pApplicationName = "Neural Netword in a weekend",
        .applicationVersion = VK_VERSION_1_3,
        .pEngineName = "No Engine",
        .engineVersion = VK_VERSION_1_3,
        .apiVersion = VK_VERSION_1_3,
    };

    //--- Debug
    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfo = {
        .sType = vk::StructureType::eDebugUtilsMessengerCreateInfoEXT,
        .pNext = nullptr,
        .flags = {},
        .messageSeverity =
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo | vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback,
        .pUserData = nullptr,
    };

    //--- Instance
    vk::InstanceCreateInfo instanceCreateInfo = {
        .sType = vk::StructureType::eInstanceCreateInfo,
        .pNext = &debugUtilsMessengerCreateInfo,
        .flags = {},
        .pApplicationInfo = &applicationInfo,
        .enabledLayerCount = static_cast<uint32_t>(m_ValidationLayers.size()),
        .ppEnabledLayerNames = m_ValidationLayers.data(),
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
    };

    m_Instance = vk::createInstance(instanceCreateInfo);

    //--- Physical Device Selection
    std::vector<vk::PhysicalDevice> devices = m_Instance.enumeratePhysicalDevices();

    auto it = std::ranges::find_if(devices, [](const vk::PhysicalDevice& device) {
        vk::PhysicalDeviceProperties deviceProperties = device.getProperties();
        return deviceProperties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu;
    });
    m_PhyscialDevice = it != devices.end() ? *it : devices.front();

    //--- Compute Queue Query
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = m_PhyscialDevice.getQueueFamilyProperties();
    uint32_t i = 0;
    for (const auto& queueFamilyProperty : queueFamilyProperties) {
        if (queueFamilyProperty.queueFlags & vk::QueueFlagBits::eCompute) {
            break;
        }
        i++;
    }
    uint32_t m_ComputeQueueIndex = i;

    //--- Device Queue(s)
    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo ={
        .sType = vk::StructureType::eDeviceQueueCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .queueFamilyIndex = m_ComputeQueueIndex,
        .queueCount = 1,
        .pQueuePriorities = &queuePriority,
    };

    //--- Device
    vk::DeviceCreateInfo deviceCreateInfo = {
        .sType = vk::StructureType::eDeviceCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &deviceQueueCreateInfo,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = nullptr,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = nullptr,
        .pEnabledFeatures = nullptr,
    };

    m_Device = m_PhyscialDevice.createDevice(deviceCreateInfo);
}

ComputeEngine::~ComputeEngine() {
    m_Device.destroyFence(m_Fence);
    m_Device.resetCommandPool(m_CommandPool);
    m_Device.destroyCommandPool(m_CommandPool);
    m_Device.destroyDescriptorSetLayout(m_DescriptorSetLayout);
    m_Device.destroyDescriptorPool(m_DescriptorPool);
    m_Device.destroyPipelineLayout(m_PipelineLayout);
    m_Device.destroyPipelineCache(m_PipelineCache);
    m_Device.destroyPipeline(m_ComputePipeline);

    m_SrcBuffer.Destroy();
    m_DstBuffer.Destroy();

    for (auto& shader : m_Shaders) {
        m_Device.destroyShaderModule(shader.second);
    }
    m_Device.destroy();
    m_Instance.destroy();
}

void ComputeEngine::AddShader(const std::string& path) {
    std::vector<char> contents;

    std::ifstream ifs(path, std::ios::binary | std::ios::ate);
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

    if (!m_Shaders.contains(path)) {
        m_Shaders[path] = m_Device.createShaderModule(shaderModuleCreateInfo);
    } else {
        LogWarning("ComputeEngine shaders already contains", path, "not adding shader");
    }
}

void ComputeEngine::RemoveShader(const std::string& path) {
    if (m_Shaders.contains(path)) {
        m_Device.destroyShaderModule(m_Shaders[path]);
        m_Shaders.erase(path);
    } else {
        LogWarning("ComputeEngine shaders does not contain", path, "cannot remove shader");
    }
}

void ComputeEngine::AddBuffers() {
    uint32_t nElements = 256;

    vk::BufferCreateInfo bufferCreateInfo = {
        .sType = vk::StructureType::eBufferCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .size = m_SrcBuffer.Size,
        .usage = vk::BufferUsageFlagBits::eStorageBuffer,
        .sharingMode = vk::SharingMode::eExclusive,
        .queueFamilyIndexCount = 1,
        .pQueueFamilyIndices = &m_ComputeQueueIndex,
    };

    m_SrcBuffer.Buf = m_Device.createBuffer(bufferCreateInfo);
    m_DstBuffer.Buf = m_Device.createBuffer(bufferCreateInfo);

    vk::MemoryRequirements srcBufferMemoryRequirements = m_Device.getBufferMemoryRequirements(m_SrcBuffer.Buf);
    vk::MemoryRequirements dstBufferMemoryRequirements = m_Device.getBufferMemoryRequirements(m_DstBuffer.Buf);

    vk::PhysicalDeviceMemoryProperties memoryProperties = m_PhyscialDevice.getMemoryProperties();

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

    vk::MemoryAllocateInfo dstBufferAllocInfo = srcBufferAllocInfo;
    dstBufferAllocInfo.allocationSize = dstBufferMemoryRequirements.size;

    m_SrcBuffer.Mem = m_Device.allocateMemory(srcBufferAllocInfo);
    m_DstBuffer.Mem = m_Device.allocateMemory(dstBufferAllocInfo);

    int32_t* srcBufferPtr = (int32_t*)m_Device.mapMemory(m_SrcBuffer.Mem, 0, m_SrcBuffer.Size);
    for (uint32_t i = 0; i < 256; i++) {
        srcBufferPtr[i] = i;
    }
    m_Device.unmapMemory(m_SrcBuffer.Mem);

    m_Device.bindBufferMemory(m_SrcBuffer.Buf, m_SrcBuffer.Mem, 0);
    m_Device.bindBufferMemory(m_DstBuffer.Buf, m_DstBuffer.Mem, 0);
}

void ComputeEngine::AddPipeline() {
    const std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings = {
        {
            .binding = 0,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .pImmutableSamplers = nullptr,
        },
        {
            .binding = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .descriptorCount = 1,
            .stageFlags = vk::ShaderStageFlagBits::eCompute,
            .pImmutableSamplers = nullptr,
        },
    };

    vk::DescriptorSetLayoutCreateInfo descriptSetLayoutCreateInfo = {
        .sType = vk::StructureType::eDescriptorSetLayoutCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .bindingCount = 2,
        .pBindings = descriptorSetLayoutBindings.data(),
    };

    m_DescriptorSetLayout = m_Device.createDescriptorSetLayout(descriptSetLayoutCreateInfo);

    vk::PipelineLayoutCreateInfo pipelineLayoutCreateInfo = {
        .sType = vk::StructureType::ePipelineLayoutCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .setLayoutCount = 1,
        .pSetLayouts = &m_DescriptorSetLayout,
        .pushConstantRangeCount = 0,
        .pPushConstantRanges = nullptr,
    };

    m_PipelineLayout = m_Device.createPipelineLayout(pipelineLayoutCreateInfo);
    m_PipelineCache = m_Device.createPipelineCache({});

    vk::PipelineShaderStageCreateInfo shaderStageCreateInfo = {
        .sType = vk::StructureType::ePipelineShaderStageCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .stage = vk::ShaderStageFlagBits::eCompute,
        .module = m_Shaders["tests/spirv/mnist.comp.spv"],
        .pName = "main",
        .pSpecializationInfo = nullptr,
    };

    vk::ComputePipelineCreateInfo computePipelineCreateInfo = {
        .sType = vk::StructureType::eComputePipelineCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .stage = shaderStageCreateInfo,
        .layout = m_PipelineLayout,
        .basePipelineHandle = {},
        .basePipelineIndex = {},
    };

    vk::ResultValue result = m_Device.createComputePipeline(m_PipelineCache, computePipelineCreateInfo);
    if (result.result != vk::Result::eSuccess) {
        LogError("could not create compute pipeline");
    }
    m_ComputePipeline = result.value;

    vk::DescriptorPoolSize descriptorPoolSize = {
        .type = vk::DescriptorType::eStorageBuffer,
        .descriptorCount = 2,
    };

    vk::DescriptorPoolCreateInfo descriptorPoolCreateInfo = {
        .sType = vk::StructureType::eDescriptorPoolCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .maxSets = 1,
        .poolSizeCount = 1,
        .pPoolSizes = &descriptorPoolSize,
    };

    m_DescriptorPool = m_Device.createDescriptorPool(descriptorPoolCreateInfo);

    vk::DescriptorSetAllocateInfo descriptorSetAllocInfo = {
        .sType = vk::StructureType::eDescriptorSetAllocateInfo,
        .pNext = nullptr,
        .descriptorPool = m_DescriptorPool,
        .descriptorSetCount = 1,
        .pSetLayouts = &m_DescriptorSetLayout,
    };
    const std::vector<vk::DescriptorSet> descriptorSets = m_Device.allocateDescriptorSets(descriptorSetAllocInfo);
    m_DescriptorSet = descriptorSets.front();

    vk::DescriptorBufferInfo srcBufferInfo = {
        .buffer = m_SrcBuffer.Buf,
        .offset = 0,
        .range = 256 * sizeof(int32_t),
    };

    vk::DescriptorBufferInfo dstBufferInfo = {
        .buffer = m_DstBuffer.Buf,
        .offset = 0,
        .range = 256 * sizeof(int32_t),
    };

    const std::vector<vk::WriteDescriptorSet> writeDescriptorSets = {
        {
            .sType = vk::StructureType::eWriteDescriptorSet,
            .pNext = nullptr,
            .dstSet = m_DescriptorSet,
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
            .dstSet = m_DescriptorSet,
            .dstBinding = 1,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk::DescriptorType::eStorageBuffer,
            .pImageInfo = nullptr,
            .pBufferInfo = &dstBufferInfo,
            .pTexelBufferView = nullptr,
        },
    };

    m_Device.updateDescriptorSets(writeDescriptorSets, {});
}

void ComputeEngine::AddCommandPool() {
    vk::CommandPoolCreateInfo commandPoolCreateInfo = {
        .sType = vk::StructureType::eCommandPoolCreateInfo,
        .pNext = nullptr,
        .flags = {},
        .queueFamilyIndex = m_ComputeQueueIndex,
    };
    
    m_CommandPool = m_Device.createCommandPool(commandPoolCreateInfo);

    vk::CommandBufferAllocateInfo commandBufferAllocInfo = {
        .sType = vk::StructureType::eCommandBufferAllocateInfo,
        .pNext = nullptr,
        .commandPool = m_CommandPool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1,
    };

    const std::vector<vk::CommandBuffer> commandBuffers = m_Device.allocateCommandBuffers(commandBufferAllocInfo);
    m_CommandBuffer = commandBuffers.front();
}

void ComputeEngine::RecordCommands() {
    vk::CommandBufferBeginInfo commandBufferBeginInfo = {
        .sType = vk::StructureType::eCommandBufferBeginInfo,
        .pNext = nullptr,
        .flags = {},
        .pInheritanceInfo = nullptr,
    };

    m_CommandBuffer.begin(commandBufferBeginInfo);
    m_CommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, m_ComputePipeline);
    m_CommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_PipelineLayout, 0, {m_DescriptorSet}, {});
    m_CommandBuffer.dispatch(256, 1, 1);
    m_CommandBuffer.end();

    vk::Queue queue = m_Device.getQueue(m_ComputeQueueIndex, 0);

    vk::FenceCreateInfo fenceCreateInfo = {
        .sType = vk::StructureType::eFenceCreateInfo,
        .pNext = nullptr,
        .flags = {},
    };
    m_Fence = m_Device.createFence(fenceCreateInfo);

    vk::SubmitInfo submitInfo = {
        .sType = vk::StructureType::eSubmitInfo,
        .pNext = nullptr,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = nullptr,
        .pWaitDstStageMask = nullptr,
        .commandBufferCount = 1,
        .pCommandBuffers = &m_CommandBuffer,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = nullptr,
    };

    queue.submit({submitInfo}, m_Fence);

    vk::Result result = m_Device.waitForFences({m_Fence}, true, UINT64_MAX);
    if (result != vk::Result::eSuccess) {
        LogWarning("fence wait result error on line", __LINE__, __FILE__);
    }

    int32_t* srcBufferPtr = (int32_t*)m_Device.mapMemory(m_SrcBuffer.Mem, 0, m_SrcBuffer.Size);
    LogInfo("//--- Source Buffer ---//");
    for (uint32_t i = 0; i < 256; i++) {
        LogInfo(i, ":", srcBufferPtr[i]);
    }
    m_Device.unmapMemory(m_SrcBuffer.Mem);

    int32_t* dstBufferPtr = (int32_t*)m_Device.mapMemory(m_DstBuffer.Mem, 0, m_DstBuffer.Size);
    LogInfo("//--- Destination Buffer ---//");
    for (uint32_t i = 0; i < 256; i++) {
        LogInfo(i, ":", dstBufferPtr[i]);
    }
    m_Device.unmapMemory(m_DstBuffer.Mem);
}

} // namespace nn