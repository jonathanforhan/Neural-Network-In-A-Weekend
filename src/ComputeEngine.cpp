#include "ComputeEngine.hpp"
#include "ComputeEngine.hpp"
#include <fstream>
#include <chrono>
#include "Log.hpp"

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    (void)messageType;
    (void)pUserData;

    switch (messageSeverity) {
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
            nn::LogWarning(pCallbackData->pMessage);
            break;
        case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
            nn::LogError(pCallbackData->pMessage);
            break;
        default:
            break;
    }

    return VK_FALSE;
}

namespace nn {

ComputeEngine::ComputeEngine() {
    //--- Application
    vk::ApplicationInfo applicationInfo = {
        .sType = vk::StructureType::eApplicationInfo,
        .pNext = nullptr,
        .pApplicationName = "Neural Netword In A Weekend",
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

    m_Instance = vk::createInstanceUnique(instanceCreateInfo);

    //--- Physical Device Selection
    std::vector<vk::PhysicalDevice> devices = m_Instance->enumeratePhysicalDevices();

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

    m_Device = m_PhyscialDevice.createDeviceUnique(deviceCreateInfo);
}

void ComputeEngine::ExecuteTasks() {
    while (!m_TaskQueue.empty()) {
        auto& task = m_TaskQueue.front();
        m_TaskQueue.pop();
        task->Execute(*this);
    }
}

} // namespace nn