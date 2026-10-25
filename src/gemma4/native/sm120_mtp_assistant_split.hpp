#pragma once
#include "sm120_mtp_assistant.hpp"
#include <cstddef>

namespace strata::gemma4::native::mtp_assistant {
// Exclusive, stream-ordered borrow between Q projection completion and output
// projection start. The owner and stream outlive every enqueue/graph replay.
// The whole allocation is checked against all read/write argument spans.
struct SplitAttentionLease {
    const void* owner=nullptr;
    cudaStream_t stream=nullptr;
    void* data=nullptr;
    size_t bytes=0;
};
struct SplitKernelResources {int threads,num_regs,local_bytes,static_shared_bytes;};
struct SplitAttentionResources {SplitKernelResources segment,merge;};
// Cold diagnostics only; neither helper reads GPU Bounds or allocates memory.
size_t split_cross_attention_bytes(CacheView,bool sliding,int context);
SplitAttentionResources split_cross_attention_resources(int dimension);
// Default-OFF standalone candidate: FP32, direct Q8,128 keys/4 warps per
// segment. No GPU cache writes, allocations, synchronization or host reads.
void split_cross_attention(const float* query,float* output,CacheView,
    const Bounds*,bool sliding,int context,unsigned* invalid,
    SplitAttentionLease,cudaStream_t);
}
