#pragma once
#include "strata/qwen35/completed_v100/native/attention_keys64.hpp"
namespace strata::completed_v100::qwen35::native {
void prefill_short32_configure();
PrefillOnlineResources prefill_short32_resources(int);
void prefill_short32_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,int,cudaStream_t);
void prefill_attention_tuned(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,int,cudaStream_t);
}
