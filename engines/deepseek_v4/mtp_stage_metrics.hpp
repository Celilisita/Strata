#pragma once
#include <array>
#include <cstdint>

namespace strata::deepseek_v4 {
struct MtpConfidence {float top1_probability=0,top1_top2_logit_gap=0;bool measured=false;};
// Host wall clocks around each MTP draft boundary. CUDA graph calls can enqueue
// asynchronously; their wall time is intentionally not labeled GPU execution.
struct MtpStageMetrics {
    uint64_t calls = 0;
    std::array<double, 5> host_ms{}; // input, front submit, finish submit, token read, KV commit
    // Input preparation detail: window, mask build/upload, hidden upload,
    // embedding prepare/upload, IDs upload, positions upload, host staging.
    std::array<double, 8> input_sub_ms{};
    uint64_t cpu_miss_jobs = 0;
    uint64_t cpu_miss_compute_ns = 0;
};
struct MtpCpuServiceSnapshot {
    uint64_t jobs = 0;
    uint64_t compute_ns = 0;
};
MtpCpuServiceSnapshot mtp_cpu_service_snapshot() noexcept;
}
