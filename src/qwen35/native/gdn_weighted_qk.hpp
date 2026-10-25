#pragma once
#include <kreg.hpp>
namespace strata::qwen35::native::gdn_weighted_qk_unit {
struct KernelResources {int registers,blocks,binary_version;size_t shared_bytes,local_bytes;};
struct Resources {KernelResources prepare,scan;int kernel_count;};
// Two Half planes [value_head,256 blocks,32 tokens,128 features].
// This is the future F32 gated-output allocation, dead until scan completes.
struct Weighted {__half *q,*k;};
void configure();
Resources resources();
size_t workspace_bytes(GdnShape);
size_t weighted_bytes(GdnShape);
Weighted weighted(void*,GdnShape);
// Two submissions. No persistent allocation, copy, synchronization, or state reset.
// Input/final state may alias exactly; weighted storage cannot alias live views.
void run(GdnInputs,GdnShape,float *out,float *state,void *work,size_t,
         void *dead_gated,size_t,cudaStream_t);
}
