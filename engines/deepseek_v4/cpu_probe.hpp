#pragma once

#include "checkpoint.hpp"

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {

struct CpuProbeOptions {
    std::string model;
    std::filesystem::path reference_dir, output_dir;
    int32_t n_threads = 4;
    uint32_t route_mask = 63;
    bool raw_experts_only = false;
    bool allow_device_reference = false;
};

struct CpuProbeResult {
    bool completed = false, expert_stages_exact = false;
    bool service_curve_available = false;
    bool threadpool_resident = false;
    // Older checkpoints may lack storage metadata; never infer it from bytes.
    bool reference_host_storage_recorded = false;
    std::string error, first_difference, router_first_difference, weighted_first_difference;
    std::string input_identity;
    std::vector<uint32_t> original_router_ranks;
    bool raw_experts_only = false;
    uint64_t contribution_output_bytes = 0;
    uint64_t compared_expert_stages = 0;
    // Logical distinct expert payload, not measured memory traffic or disk I/O.
    uint64_t effective_expert_payload_bytes = 0;
    int32_t n_threads = 0;
    double load_wall_ms = 0, compute_wall_ms = 0;
    double service_compute_wall_ms = -1;
    std::vector<double> service_compute_wall_ms_raw;
    std::vector<Checkpoint> checkpoints;
};

// Replays canonical layer-zero MoE from captured normalized activation, router
// logits and selected ids, without a llama_context, attention or expert cache.
// Completion does not imply equality: gate on expert_stages_exact separately.
bool run_cpu_probe(const CpuProbeOptions& options, CpuProbeResult& result) noexcept;

} // namespace strata::deepseek_v4
