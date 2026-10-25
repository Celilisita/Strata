#pragma once
#include <cuda_runtime.h>
#include "strata/qwen35moe/gdn_dense_graph.hpp"
#include <cstdio>
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
    bool enabled = false, recurrent_enabled = false;
    uint64_t executed_kernel_nodes = 0;
    std::vector<std::array<MoeGraphEntry,2>> layers;
    std::vector<MoeGraphEntry> recurrent; // Same weights/state; optional single-row schedule.
    explicit MoeGraphState(size_t count, bool capture_recurrent = false)
        : layers(count), recurrent(capture_recurrent ? count : 0), dense_recurrent(count) {}
    bool dense_recurrent_enabled=false;
    uint64_t recurrent_submissions=0,recurrent_kernel_nodes=0;
    std::vector<GdnDenseGraph> dense_recurrent;
};
}
