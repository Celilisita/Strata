#pragma once

#include "checkpoint.hpp"
#include "expert_staging.hpp"
#include "cold_expert_provider.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {

struct RequestReport {
    uint32_t index = 0;
    bool measured = true;
    std::string state, status, error, output_text;
    std::vector<int32_t> input_ids, output_ids;
    ExpertStagingMetrics expert_transport_delta;
    ExpertStagingMetrics prefill_transport, decode_transport;
    bool transport_phases_complete = false;
    ColdExpertMetrics cold_delta;
    double request_ms = 0, ttft_ms = -1, prompt_eval_ms = 0, token_eval_ms = 0;
    int32_t prompt_eval_tokens = 0, token_eval_tokens = 0, graph_reuses = 0;
};

struct BenchmarkReport {
    std::string mode, status, error, model, device, canonical_commit, architecture;
    std::string workload_identity, output_text;
    uint64_t cache_mib = 0;
    std::vector<int32_t> input_ids, output_ids;
    std::vector<Checkpoint> checkpoints;
    bool diagnostics = false, layer_complete = false;
    bool threadpool_resident = false, pinned_experts = false;
    uint32_t warmup_requests = 0, measure_requests = 1;
    std::vector<RequestReport> requests;
    ExpertStagingMetrics expert_transport;
    SourcePinMetrics source_pin;
    int32_t cold_threads = 0;
    bool cold_admit_free = true;
    bool cache_leaf_aliases = false;
    bool layer_fusion_seam = false;
    bool cache_prime = false, all_hit_lease = false;
    std::vector<RequestReport> priming_requests;
    ColdExpertMetrics cold;
    double load_wall_ms = 0, request_ms = 0, ttft_ms = -1;
    double prompt_eval_ms = 0, token_eval_ms = 0;
    int32_t prompt_eval_tokens = 0, token_eval_tokens = 0, graph_reuses = 0;
};

// Populate only from an authoritative backend snapshot. An unset metric is
// unavailable, never zero. This contract adds no production textual logging.
struct NativeMetrics {
    bool available = false;
    uint64_t h2d_bytes = 0, d2h_bytes = 0, expert_launches = 0;
    uint64_t host_continuations = 0, synchronizations = 0;
};

std::string json_string(const std::string& value);
void write_report(const std::filesystem::path& directory, const BenchmarkReport& report,
                  const NativeMetrics* metrics = nullptr);
bool compare_runs(const std::filesystem::path& reference, const std::filesystem::path& candidate,
                  const std::filesystem::path& output);

} // namespace strata::deepseek_v4
