#pragma once
#include "strata/qwen35/native/attention_online.hpp"

namespace strata::qwen35::native {
PrefillOnlineResources prefill_gather_resources(int query_rows);
// Caller configures the selected geometry before capture; staging is borrowed.
void configure_prefill_gather(int query_rows);
size_t prefill_gather_shared_bytes(int query_rows);
void prefill_attention_gather(const float *,const void *,const void *,float *,void *,size_t,
                            PrefillAttentionShape,int query_rows,cudaStream_t);
}
