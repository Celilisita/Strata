#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>
namespace strata::deepseek_v4 {
struct CacheCostSnapshot {
    uint64_t permanent_slots = 0, model_expert_keys = 0, resident_keys = 0;
    uint64_t weight_tensor_bytes = 0, backend_alloc_bytes = 0, invalid_directory_links = 0;
    uint64_t plans = 0, planned_fills = 0, planned_evictions = 0, repeated_fills = 0;
    uint64_t refills_within43_miss_plans = 0, incoming_score_not_greater = 0;
    uint64_t transient_slots = 0, transient_fills = 0, reused_transient_routes = 0, admission_bypasses = 0;
};
bool cache_cost_audit_requested();
// Bounded history of fill decisions, never a cache, slot map or replacement
// policy. Software decision events are labelled separately from DMA counters.
class CacheCostAudit {
public:
    explicit CacheCostAudit(std::size_t keys);
    void plan();
    void fill(int key, int evicted_key, uint32_t incoming_score, uint32_t outgoing_score);
    void bypass() { ++costs_.admission_bypasses; }
    void transport(size_t fills, size_t reused) { costs_.transient_fills += fills; costs_.reused_transient_routes += reused; }
    CacheCostSnapshot snapshot() const { return costs_; }
private:
    struct History { uint64_t last_fill = 0, last_eviction = 0; };
    std::vector<History> history_;
    CacheCostSnapshot costs_;
};
}
