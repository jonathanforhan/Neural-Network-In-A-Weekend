#include "TaskBuilder.hpp"

namespace nn {

std::shared_ptr<Task> TaskBuilder::create() const {
    auto task = std::make_shared<Task>();
    task->setShader(m_ComputeEngine, m_ShaderPath);
    task->setBuffers(m_ComputeEngine, m_BufferSpec);
    task->setPipeline(m_ComputeEngine, m_PipelineSpec);
    task->setCommandPool(m_ComputeEngine);
    return task;
}

} // namespace nn