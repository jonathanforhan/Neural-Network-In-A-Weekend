#pragma once
#include <memory>
#include "ComputeEngine.hpp"
#include "Task.hpp"

namespace nn {

class TaskBuilder {
public:
    explicit TaskBuilder(const ComputeEngine& computeEngine)
        : m_ComputeEngine(computeEngine),
          m_ShaderPath(),
          m_BufferSpec(),
          m_PipelineSpec() {}

    void SetShader(std::string_view shader) { m_ShaderPath = shader; }
    void SetBuffers(const BufferSpecification& spec) { m_BufferSpec = spec; }
    void SetPipeline(const PipelineSpecification& spec) { m_PipelineSpec = spec; }

    std::shared_ptr<Task> create() const;

private:
    const ComputeEngine& m_ComputeEngine;

    std::string m_ShaderPath;
    BufferSpecification m_BufferSpec;
    PipelineSpecification m_PipelineSpec;
};

} // namespace nn
