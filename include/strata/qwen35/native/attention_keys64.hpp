#pragma once
#include "strata/qwen35/native/attention_online.hpp"

namespace strata::qwen35::native {
PrefillOnlineResources prefill_keys64_resources(int query_rows);
// Caller configures the selected geometry before capture; staging is borrowed.
void configure_prefill_keys64(int query_rows);
size_t prefill_keys64_shared_bytes(int query_rows);
void prefill_attention_keys64(const float *,const void *,const void *,float *,void *,size_t,
                            PrefillAttentionShape,int query_rows,cudaStream_t);
}
