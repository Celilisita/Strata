#pragma once

#include "cpu_expert_service.hpp"
#include "ggml-backend.h"
#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>

struct llama_model;

namespace strata::deepseek_v4 {

struct ColdExpertMetrics {
    uint64_t jobs = 0, cold_experts = 0, avoided_weight_bytes = 0;
    uint64_t contribution_h2d_bytes = 0, activation_d2h_bytes = 0;
    uint64_t route_control_d2h_bytes = 0, host_continuations = 0, joins = 0;
    uint64_t placeholder_gpu_experts = 0;
    double cpu_service_ms = 0, join_wait_ms = 0;
};

// Initial TG miss provider: the original cache remains the sole residency
// owner. Free cache capacity admits misses; saturated misses can use CPU raw
// Down without evicting hot experts. Only cold route lanes are replaced.
// Canonical GPU placeholders still execute for misses in this vertical slice.
class ColdExpertProvider {
public:
    ColdExpertProvider(const llama_model& model, ggml_backend_t gpu, int threads, bool admit_free,
                       ggml_backend_sched_eval_callback observer, void* observer_data);
    ~ColdExpertProvider();
    ColdExpertProvider(const ColdExpertProvider&) = delete;
    ColdExpertProvider& operator=(const ColdExpertProvider&) = delete;
    static const int32_t* prepare(ggml_backend_t gpu, int layer, const int32_t* ids,
                                  std::size_t count, const ggml_tensor* slot_map,
                                  bool has_free_slot, ggml_cgraph* graph, void* data);
    static bool evaluate(ggml_tensor* tensor, bool ask, void* data);
    ColdExpertMetrics metrics() const noexcept { return metrics_; }
private:
    const int32_t* submit(int layer, const int32_t* ids, const ggml_tensor* slot_map, ggml_cgraph* graph);
    void join(ggml_tensor* tensor);
    void worker_loop() noexcept;
    void release() noexcept;

    ggml_backend_t gpu_;
    bool admit_free_;
    CpuExpertService service_;
    ggml_backend_buffer_t pinned_ = nullptr;
    ggml_backend_event_t device_event_ = nullptr;
    float *activation_ = nullptr, *output_ = nullptr;
    std::array<int32_t, 256> upload_map_{};
    std::array<int32_t, 6> ids_{};
    std::array<uint32_t, 6> ranks_{};
    std::size_t count_ = 0;
    int layer_ = -1;
    std::string down_name_, worker_error_;
    ggml_backend_sched_eval_callback observer_;
    void* observer_data_;
    std::mutex mutex_;
    std::condition_variable condition_;
    bool ready_ = false, done_ = false, stop_ = false;
    double work_ms_ = 0;
    std::thread worker_;
    ColdExpertMetrics metrics_{};
};

} // namespace strata::deepseek_v4
