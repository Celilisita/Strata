#pragma once
#include "strata/qwen35/native/attention_online.hpp"
namespace strata::qwen35::native {
void configure_rescale_skip_attention();
PrefillOnlineResources rescale_skip_attention_resources();
void rescale_skip_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,cudaStream_t);
}
