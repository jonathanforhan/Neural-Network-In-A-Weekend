#pragma once
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS 1
#include <vulkan/vulkan.hpp>
#include <queue>
#include "Task.hpp"

namespace nn {

class ComputeEngine {
public:
    ComputeEngine();
    ComputeEngine(const ComputeEngine&) = delete;
    void operator=(const ComputeEngine&) = delete;

    vk::Device Device() const { return *m_Device; }
    vk::PhysicalDevice GPU() const { return m_PhyscialDevice; }
    uint32_t ComputeQueue() const { return m_ComputeQueueIndex; }
    void PushTask(std::shared_ptr<Task> task) { m_TaskQueue.push(task); }
    void ExecuteTasks();

private:
    std::queue<std::shared_ptr<Task>> m_TaskQueue;
    vk::UniqueInstance m_Instance;
    vk::PhysicalDevice m_PhyscialDevice;
    vk::UniqueDevice m_Device;
    uint32_t m_ComputeQueueIndex = 0;

    const std::vector<const char*> m_ValidationLayers = {
        "VK_LAYER_KHRONOS_validation",
    };

    const std::vector<const char*> m_Extensions = {
        VK_EXT_DEBUG_UTILS_EXTENSION_NAME,
    };
};

} // namespace nn