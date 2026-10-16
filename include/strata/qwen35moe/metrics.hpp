#pragma once
#include "strata/qwen35moe/dense_cost.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::qwen35moe {
struct Metrics {
    uint64_t mtp_windows = 0, mtp_accepted = 0, verify_tokens = 0;
    uint64_t rollback_bytes = 0, kv_catchup_rows = 0;
    uint64_t expert_h2d = 0, visual_h2d = 0;
    uint64_t context_cache_d2d = 0;
    uint64_t expert_h2d_calls = 0;
    size_t gpu_min_free_bytes = size_t(-1);
    uint64_t cpu_expert_read = 0;
    uint64_t cpu_result_h2d = 0;
    double cpu_pack_ms = 0;
    double cpu_pool_ms = 0, cpu_gu_ms = 0, cpu_quant_ms = 0, cpu_down_ms = 0;
    uint64_t cpu_jobs = 0, cpu_paired_jobs = 0;
    double head_observe_ms = 0;
    uint64_t moe_graph_launches = 0, moe_host_submissions = 0;
    double route_wait_ms = 0;
    double diag_route_gpu_wait_ms = 0;
    double diag_route_copy_ms = 0;
    double diag_hit_gpu_wait_ms = 0;
    double diag_hit_copy_ms = 0;
    double diag_hit_sync_ms = 0;
    double diag_hit_classify_ms = 0;
    double diag_hit_table_ms = 0;
    double diag_hit_table_copy_ms = 0;
    double diag_miss_norm_copy_ms = 0;
    double diag_miss_norm_wait_ms = 0;
    double diag_token_wall_ms = 0;
    double diag_moe_wall_ms = 0;
    double diag_route_wall_ms = 0;
    double diag_classify_wall_ms = 0;
    double diag_hit_branch_wall_ms = 0;
    double diag_miss_branch_wall_ms = 0;
    double diag_cpu_prepare_ms = 0;
    double diag_result_h2d_ms = 0;
    double diag_join_submit_ms = 0;
    double diag_head_wall_ms = 0;
    double diag_head_copy_ms = 0;
    double diag_head_scan_ms = 0;
    double expert_enqueue_ms = 0;
    double head_wait_ms = 0;
    double pp_mtp_ms = 0, pp_embedding_submit_ms = 0;
    DenseCost pp_attention_detail, pp_gdn_detail;
    uint64_t pp_dense_diagnostic_events = 0;
    double pp_dense_event_ms = 0;
    double pp_attention_event_ms = 0, pp_gdn_event_ms = 0;
    double pp_route_event_ms = 0;
    double pp_address_ms = 0; // CPU address/residency submission, excludes GPU completion.
    double pp_copy_event_ms = 0;
    double pp_moe_event_ms = 0;
    double pp_gu_event_ms = 0, pp_down_event_ms = 0, pp_shared_event_ms = 0;
    uint64_t route_d2h = 0;
    uint64_t control_h2d = 0;
    uint64_t control_d2h = 0, cpu_norm_d2h = 0;
    uint64_t head_d2h = 0;
    uint64_t host_continuations = 0;
    uint64_t expert_launches = 0;
    uint64_t explicit_syncs = 0;
    uint64_t expert_cache_hits = 0;
    uint64_t expert_cache_misses = 0;
    uint64_t tg_gpu_hit_layers = 0;
    uint64_t tg_device_layers = 0;
    uint64_t tg_cpu_layers = 0;
};

} // namespace strata::qwen35moe
