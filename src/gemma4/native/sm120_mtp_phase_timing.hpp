#pragma once
#include <cuda_runtime_api.h>
#include <array>
#include <cstdint>
namespace strata::gemma4::native::sampled_mtp {
// Diagnostic same-stream globaltimer intervals; not isolated kernel timings.
// Only enabled controllers allocate40B and enqueue the five stamp kernels.
struct PhaseTiming {
    bool enabled=false;
    std::array<std::uint64_t,5> ticks{};
    double draft_gpu_ms=0,target_gpu_ms=0,sample_gpu_ms=0,commit_gpu_ms=0,total_gpu_ms=0;
};
void phase_stamp(std::uint64_t* ticks,int slot,cudaStream_t stream);
}
