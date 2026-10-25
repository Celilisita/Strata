#pragma once
#include "strata/qwen35/native/attention_online.hpp"
namespace strata::qwen35::native {
void configure_pv_reuse_attention();
PrefillOnlineResources pv_reuse_attention_resources();
void pv_reuse_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,cudaStream_t);
}
