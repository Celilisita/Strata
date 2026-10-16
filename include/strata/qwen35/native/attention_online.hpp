#pragma once
#include "strata/qwen35/native/attention_prefill.hpp"

namespace strata::qwen35::native {
struct PrefillOnlineResources { int registers,blocks_per_SM,max_threads;size_t local_bytes,shared_bytes; };
PrefillOnlineResources prefill_online_resources(int query_rows);
// Caller configures the selected geometry before capture; staging is borrowed.
void configure_prefill_online(int query_rows);
size_t prefill_online_shared_bytes(int query_rows);
void prefill_attention_online(const float *,const void *,const void *,float *,void *,size_t,
                            PrefillAttentionShape,int query_rows,cudaStream_t);
}
