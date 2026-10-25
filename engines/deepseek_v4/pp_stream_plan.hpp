#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

namespace strata::deepseek_v4 {
// Scratch windows carry no residency ownership. Route multiplicity and the
// original token/rank indices survive grouping; only execution order changes.
struct PpStreamSourceRun {
    int original_first=0,count=0,local_first=0;
};
struct PpStreamGroup {
    int first_expert = 0, experts = 0, first_row = 0, rows = 0, max_rows = 0;
    int cache_first_slot = -1;
    std::vector<int32_t> bounds;
    // Nonempty only for coalesced owned misses; local row j uses this real key.
    std::vector<int32_t> source_keys;
    std::vector<PpStreamSourceRun> source_runs;
};
struct PpStreamPlan {
    int tokens = 0, topk = 0, max_group_rows = 0;
    std::vector<int32_t> sorted_pairs, inverse;
    std::vector<PpStreamGroup> groups;
    uint64_t selected_experts = 0, payload_bytes = 0;
};
PpStreamPlan make_pp_stream_plan(const int32_t* ids, int tokens, int topk,
                               int experts, int window_experts, size_t triplet_bytes, const int32_t* cache_slots = nullptr, int prefix_boundary = 0);
}
