#pragma once
#include <strata/qwen35/native/attention_online.hpp>
namespace strata::qwen35::native {
void configure_regp_attention();
PrefillOnlineResources regp_attention_resources();
int regp_binary_version();
void regp_attention(const float *,const void *,const void *,float *,void *,size_t,
                    PrefillAttentionShape,cudaStream_t);
}
