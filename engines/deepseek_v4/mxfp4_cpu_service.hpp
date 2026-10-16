#pragma once
#include "ggml.h"
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::deepseek_v4 {
struct Mxfp4CpuBank {
    const ggml_tensor *gate = nullptr, *up = nullptr, *down = nullptr;
    float clamp = 0;
};
// Native Strata submission/row scheduling; immutable real GGUF banks and
// validated GGML quantization/dot mathematics. No per-layer private graph.
class Mxfp4CpuService {
public:
    explicit Mxfp4CpuService(int threads);
    ~Mxfp4CpuService();
    Mxfp4CpuService(const Mxfp4CpuService&) = delete;
    Mxfp4CpuService& operator=(const Mxfp4CpuService&) = delete;
    using Completion = void(*)(void*, bool error);
    void submit(const Mxfp4CpuBank&, const float* x, const int32_t* ids,
                size_t count, float* down, Completion = nullptr, void* = nullptr);
    void wait();
    void compute(const Mxfp4CpuBank&, const float* x, const int32_t* ids, size_t count, float* down);
    // Two activations, original six ranks per position, 2*6*4096 output values.
    // Union experts are processed together, so both rows reuse each weight row.
    size_t compute_pair(const Mxfp4CpuBank&, const float* x, const int32_t* ids,
                        const uint32_t* missing_masks, float* contributions);
    // Idle-pool read-only job on selected immutable experts; no activation
    // math, cache admission, new threads or affinity change.
    using IdleGuard=bool(*)(void*);
    bool idle() const;
    // Park workers during PP; restore the same spin policy at the TG boundary.
    void set_spin_active(bool active);
    // Diagnostic only: total CPU time of the persistent expert workers.
    uint64_t worker_cpu_100ns() const;
    uint64_t read_ahead_pair(const Mxfp4CpuBank&, const int32_t* ids,
                            const uint32_t* missing_masks, double window_ms=0,
                            const int32_t* selected=nullptr,IdleGuard=nullptr,void* =nullptr,
                            bool* finished_while_valid=nullptr,uint64_t* first_expert_bytes=nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
