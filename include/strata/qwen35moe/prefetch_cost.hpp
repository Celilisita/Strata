#pragma once
#include <cstdint>
namespace strata::qwen35moe {
struct PpPrefetchCost {
    uint64_t layers=0, issued_unique=0, used_unique=0, unused_unique=0;
    uint64_t h2d_bytes=0, h2d_calls=0, used_bytes=0, unused_bytes=0;
    uint64_t logical_copies=0;
    uint64_t registered_bytes=0, pageable_bytes=0, warm_present_unique=0, warm_used_unique=0;
    uint64_t route_refs=0, demand_fallback_unique=0, joins=0, boundary_drains=0;
    uint64_t stage_layers=0, stage_fallbacks=0, stage_peak_slots=0, stage_evicted_keys=0;
    uint64_t stage_scatter_launches=0, stage_bytes=0;
    uint64_t gu_copy_layers=0,gu_copy_bytes=0,gu_copy_calls=0,gu_copy_waves=0,gu_copy_waits=0;
    double gu_copy_ms=0;
    double enqueue_ms=0, copy_ms=0;
};
}
