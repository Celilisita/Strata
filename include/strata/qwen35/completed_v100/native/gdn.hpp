#pragma once
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native {
struct GdnShape { int key_dim, value_dim, query_heads, value_heads, tokens; };
struct GdnInputs {
    const float *q, *k, *v, *log_decay, *beta, *state;
};
// q/k are normalized, contiguous [K,Hq,T]; v is [V,Hv,T]. State is [K,V,Hv].
// A caller-owned stream orders inputs and outputs. No allocation or join here.
// State output may alias state input: each CTA owns disjoint value rows.
void gdn_recurrence(GdnInputs, GdnShape, float *output, float *final_state, cudaStream_t);
// Row grouping reuses Q/K/decay across1,2,4or8value rows per warp.
void gdn_recurrence_rows(GdnInputs,GdnShape,float *,float *,int value_rows,cudaStream_t);
}
