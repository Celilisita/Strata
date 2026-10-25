#pragma once
#include "strata/qwen35/completed_v100/native/attention_prefill.hpp"

namespace strata::completed_v100::qwen35::native {
struct PrefillWideResources { int registers,blocks_per_SM,max_threads;size_t local_bytes,shared_bytes; };
PrefillWideResources prefill_wide_resources(int query_rows);
// Caller configures the selected geometry before capture; staging is borrowed.
void configure_prefill_wide(int query_rows);
size_t prefill_wide_shared_bytes(int query_rows);
void prefill_attention_wide(const float *,const void *,const void *,float *,void *,size_t,
                            PrefillAttentionShape,int query_rows,cudaStream_t);
}
