#pragma once
#include "strata/gemma4/native/attention_native.hpp"

namespace strata::gemma4::native {
// Borrowed until all enqueued consumers finish; nothing is retained by the API.
struct GlobalT2Lease {const void* owner;cudaStream_t stream;void* data;size_t bytes;};
struct GlobalT2Resources {
    int threads,max_threads,num_regs;
    size_t local_bytes,shared_bytes;
};
size_t global_t2_partial_bytes(CacheView);
void shared_global_t2_attention(const float*,float*,CacheView,const int* position,
    int heads,int tokens,GlobalT2Lease,cudaStream_t);
// Cold only, never queried inside an attention invocation.
GlobalT2Resources shared_global_t2_resources();
}
