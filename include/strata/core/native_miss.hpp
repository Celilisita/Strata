#pragma once
#include "strata/core/device_miss.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include <memory>
#include <vector>

namespace strata::kernels::cpu { class ExpertPool; }
namespace strata::core {

struct NativeMissSource {
    const uint8_t* gate = nullptr;
    const uint8_t* up = nullptr;
    const uint8_t* down = nullptr;
    kernels::cpu::NativeFmt format;
    int n_expert = 0;
    const float *gate_scales=nullptr,*up_scales=nullptr,*down_scales=nullptr;
};
struct NativeMissStats {
    uint64_t requests = 0, experts = 0, expert_bytes = 0;
    uint64_t norm_bytes = 0, id_bytes = 0, result_bytes = 0;
    uint64_t jobs = 0, paired_jobs = 0;
    double pool_ms = 0, gu_ms = 0, quant_ms = 0, down_ms = 0;
};

// A delivery adapter to Strata's existing native expert pool, not another executor.
// It owns only a bounded mapped packet and activation scratch. Weight views,
// residency, pool and ordered CUDA stream remain caller-owned.
class NativeMissService {
public:
    NativeMissService(kernels::cpu::ExpertPool& pool,
                      std::vector<NativeMissSource> sources, int n_embd, int ranks, int max_tokens = 1,
                      bool route_ledger = false);
    ~NativeMissService();
    NativeMissService(const NativeMissService&) = delete;
    NativeMissService& operator=(const NativeMissService&) = delete;
    // Caller must drain the CUDA stream before stop/start/free or epoch changes.
    void start(uint32_t epoch);
    void stop();
    NativeMissArgs device_args() const;
    NativeMissStats snapshot() const;
    // Caller has completed the normal head sync before reading this mapped view.
    const NativeRouteLedger* route_ledger() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::core
