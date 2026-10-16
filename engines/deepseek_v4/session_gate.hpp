#pragma once
#include "runtime.hpp"
#include "checkpoint.hpp"
#include "device_cache.hpp"
#include "mtp_generation.hpp"
#include <filesystem>
#include <ostream>

namespace strata::deepseek_v4 {
struct SessionGateCallback {
    CheckpointCollector* collector = nullptr;
    static bool evaluate(ggml_tensor* tensor, bool ask, void* data) noexcept {
        auto* collector = static_cast<SessionGateCallback*>(data)->collector;
        return collector ? CheckpointCollector::callback(tensor, ask, collector) : true;
    }
};
struct LayerGateResult {
    int32_t token = 0;
    uint64_t compared = 0, all_hit = 0, misses = 0, hit_routes = 0, bytes = 0;
    uint64_t service_ns = 0, memcpy_ns = 0, slab_wait_ns = 0, submit_ns = 0, commit_wait_ns = 0;
    double candidate_ms = 0, control_ms = 0;
};
struct SessionRequest {
    struct PhaseCost {
        uint64_t weight_bytes = 0, weight_calls = 0, source_cpu_copy_ns = 0;
        uint64_t lookups = 0, total_routes=0, initial_hit_routes = 0, host_services = 0;
        uint64_t native_pp_layers = 0, native_pp_groups = 0;
        uint64_t pp_cache_reused_bytes = 0;
        PpStreamMetrics pp_timing;
        uint64_t cpu_routes = 0, cpu_logical_weight_bytes = 0, cpu_compute_ns = 0, native_slot_h2d_bytes = 0;
    };
    MtpGenerationMetrics mtp;
    std::vector<llama_token> output;
    double pp_ms = 0, tg_ms = 0, request_ms = 0, ttft_ms = 0;
    int32_t pp_tokens = 0, tg_tokens = 0;
    ExpertStagingMetrics transport;
    DeviceCacheMetrics cache;
    PhaseCost pp_cost, tg_cost;
};
std::vector<LayerGateResult> session_layer_gates(Runtime&, SessionGateCallback&,
    const std::vector<llama_token>& input, const std::filesystem::path& output);
SessionRequest session_request(Runtime&, const std::vector<llama_token>&, const std::filesystem::path&);
SessionRequest session_request_count(Runtime&, const std::vector<llama_token>&, const std::filesystem::path&, int output_count,
    bool stop_on_eog = false);
struct AllHitResult {
    double median_ms = 0;
    uint64_t expert_bytes = 0, host_services = 0, gpu_lookups = 0, gpu_allhit = 0;
    llama_token output = -1;
    int warmups = 0;
    bool repeated_ids_equal = true;
};
AllHitResult session_all_hit(Runtime&, bool candidate);
void session_write_all_hit(std::ostream&, const AllHitResult&);
void session_set_source_pin(uint64_t mib);
void session_set_private_source(bool);
bool session_long_gate(Runtime&, const Options&, const std::filesystem::path& prompt,
    const std::filesystem::path& output, uint64_t source_pin_mib, bool frequency);
void session_write_request(std::ostream&, const SessionRequest&);
void session_write_layer(std::ostream&, const LayerGateResult&);
std::string session_json(const std::string&);
void session_set_device_cache(bool);
void session_set_frequency(bool);
void session_set_native(bool stream_pp, bool hybrid_tg);
void session_diagnose(Runtime&, const Options&, const std::vector<llama_token>&, const std::filesystem::path&);
}
