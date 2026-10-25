#pragma once
#include "strata/qwen35/completed_v100/native/attention_online.hpp"

namespace strata::completed_v100::qwen35::native {
PrefillOnlineResources prefill_reuse_resources(int query_rows);
// Caller configures the selected geometry before capture; staging is borrowed.
void configure_prefill_reuse(int query_rows);
size_t prefill_reuse_shared_bytes(int query_rows);
void prefill_attention_reuse(const float *,const void *,const void *,float *,void *,size_t,
                            PrefillAttentionShape,int query_rows,cudaStream_t);
}
