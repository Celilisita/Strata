#pragma once
#include "strata/qwen35/native/attention_online.hpp"
namespace strata::qwen35::native {
void configure_row_inverse_attention();
PrefillOnlineResources row_inverse_attention_resources();
void row_inverse_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,cudaStream_t);
}
