#pragma once
#include <cuda_runtime.h>
#include <array>
#include <cstdint>
#include <vector>
namespace strata::qwen35moe {
struct MoeGraphEntry {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    uint64_t kernels = 0, expert_kernels = 0;
    ~MoeGraphEntry() {
        if (exec) cudaGraphExecDestroy(exec);
        if (graph) cudaGraphDestroy(graph);
    }
    MoeGraphEntry() = default;
    MoeGraphEntry(const MoeGraphEntry&) = delete;
    MoeGraphEntry& operator=(const MoeGraphEntry&) = delete;
};
// Submission schedule only. No weight/state/residency ownership or host callback.
struct MoeGraphState {
    bool enabled = false;
    uint64_t executed_kernel_nodes = 0;
    std::vector<std::array<MoeGraphEntry,2>> layers;
    explicit MoeGraphState(size_t count) : layers(count) {}
};
}
