#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace strata::qwen35moe {
struct ExpertStageTargets { uintptr_t addresses[64]{}; };
struct ExpertStageResources { int registers=0,max_threads=0,active_blocks=0;size_t local_bytes=0,shared_bytes=0; };
// Parameters are copied by the kernel launch; no asynchronous host-array lease.
void expert_stage_scatter(const void* stage,size_t stage_bytes,const ExpertStageTargets& targets,
                          size_t gate_bytes,size_t down_bytes,unsigned experts,cudaStream_t stream);
ExpertStageResources expert_stage_resources();
}
