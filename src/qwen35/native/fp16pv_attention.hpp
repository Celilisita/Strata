#pragma once
#include "strata/qwen35/native/attention_online.hpp"
namespace strata::qwen35::native {
void configure_fp16pv_attention();
PrefillOnlineResources fp16pv_attention_resources();
void fp16pv_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,cudaStream_t);
}
