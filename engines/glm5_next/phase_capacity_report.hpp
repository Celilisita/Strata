#pragma once
#include "phase_growth.hpp"
#include <cstdio>
namespace strata::glm5_next {
inline void report_phase_capacity(const PhaseCapacity& c) {
    if(!c.requested)return;
    std::fprintf(stderr,"strata_glm_cache_capacity requested=%llu admitted_target=%llu weight_tensor_bytes=%llu actual_vmm_mapped=%llu slots=%llu admission_headroom=%llu cuda_free_snapshot=%llu dxgi_local_headroom_snapshot=%llu retired_pool_bytes=%llu attempts=%u rechecks=%u refused=%u reason=%u driver_error=%d pp_floor_limited=%d owner_gap=%s\n",
        (unsigned long long)c.requested,(unsigned long long)c.admitted,(unsigned long long)c.weight_bytes,
        (unsigned long long)c.allocated,(unsigned long long)c.slots,(unsigned long long)c.headroom,
        (unsigned long long)c.cuda_free,(unsigned long long)c.local_free,(unsigned long long)c.retired_pool,
        c.attempts,c.rechecks,c.refused,c.reason,c.driver_error,int(c.pp_floor_limited),c.unknown_cuda_dxgi_gap?"UNKNOWN_CUDA_DXGI_GAP":"NOT_OBSERVED");
}
}
