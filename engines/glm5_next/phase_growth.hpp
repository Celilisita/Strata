#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
namespace strata::glm5_next {
enum PhaseGrowthReason : uint32_t {
    growth_committed=0,growth_no_slots=1,growth_allocation_refused=2,
    growth_headroom_refused=3,growth_invalid_owner=4,growth_rollback_failed=5,
    growth_sync_failed=6,growth_driver_failed=7,growth_no_reported_room=8,
    growth_retry_limit=9,growth_requested_reached=10,growth_pp_floor=11
};
struct PhaseMapAudit {
    uint64_t allocated_before=0,allocated_after=0,requested_mapping=0;
    uint32_t reason=growth_no_slots,bank=0;int driver_error=0;
    bool rollback_verified=false;
};
using PhaseGrowBatch=int(*)(ggml_backend_t,ggml_backend_buffer_t*,const size_t*,size_t,bool(*)(void*),void*,PhaseMapAudit*);
struct PhaseCapacity {
    uint64_t requested=0,admitted=0,weight_bytes=0,allocated=0,slots=0,headroom=0;
    uint64_t cuda_free=0,local_free=0,retired_pool=0;
    bool pp_floor_limited=false;
    uint32_t attempts=0,rechecks=0,refused=0,reason=growth_no_slots;
    int driver_error=0;
    // CUDA free and DXGI local headroom are different reports. The gap has no
    // proven owner ledger here; it is never added to free or promised capacity.
    bool unknown_cuda_dxgi_gap=false;
};
}
