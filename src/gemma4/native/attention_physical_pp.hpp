#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include "attention_physical_pp_geometry.hpp"

namespace strata::gemma4::native {
// Caller owns the buffers until all work on stream completes. The lease is
// valid only after store_kv on that stream, before the next raw K/V projection.
// owner identifies that invocation's Model/fixture; nothing is retained here.
struct PhysicalPpLease {
    const void* owner;
    cudaStream_t stream;
    void* dead_k;
    size_t dead_k_bytes;
    void* dead_v;
    size_t dead_v_bytes;
};
// Independent diagnostic operator, no Model dispatch or policy selection.
// Cache keys[0,end) are resident; device position must equal the host past.
// Actual device position still controls every causal mask. GPU P is read-only.
void physical_global_pp_attention(const float*,float*,CacheView,const int* position,
    int past,int heads,int tokens,AttentionWorkspace,PhysicalPpLease,cudaStream_t);
}
