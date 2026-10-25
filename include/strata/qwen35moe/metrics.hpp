#pragma once
#include "strata/qwen35moe/dense_cost.hpp"
#include "strata/qwen35moe/prefetch_cost.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::qwen35moe {
struct Metrics {
    PpPrefetchCost pp_prefetch;
    uint64_t mtp_windows = 0, mtp_accepted = 0, verify_tokens = 0;
    uint64_t rollback_bytes = 0, kv_catchup_rows = 0;
    uint64_t expert_h2d = 0, visual_h2d = 0;
    uint64_t context_cache_d2d = 0;
    uint64_t expert_h2d_calls = 0;
    uint64_t expert_h2d_logical_copies = 0, expert_h2d_2d_calls = 0;
    uint64_t expert_h2d_registered_bytes = 0, expert_h2d_pageable_bytes = 0;
    uint64_t pp_mapped_entries = 0, pp_mapped_unique = 0;
    // Unique G/U/D bytes and sum(weight_bytes*ceil(routes/tile_rows)); logical
    // demand estimates, never device-cache hits or measured PCIe transfer bytes.
    uint64_t pp_mapped_logical_weight_bytes = 0, pp_mapped_expected_tile_bytes = 0;
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
    uint64_t pp_gu_plan_calls = 0, pp_gu_plan_active_calls = 0, pp_gu_plan_invalid = 0;
    uint64_t pp_gu_plan_groups = 0, pp_gu_plan_max_rows = 0, pp_gu_plan_entries = 0;
    uint64_t pp_gu_plan_active_tiles = 0, pp_gu_plan_rectangular_tiles = 0;
    uint64_t pp_gu_plan_scheduled_tiles = 0, pp_gu_plan_tile_control_bytes = 0;
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
