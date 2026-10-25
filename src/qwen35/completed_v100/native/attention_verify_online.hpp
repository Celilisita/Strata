#pragma once
#include "strata/qwen35/completed_v100/native/attention_verify.hpp"
namespace strata::completed_v100::qwen35::native {
void verify_online_configure();
void verify_online_attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,int,int,cudaStream_t);
void verify_attention_online_q8(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,int,cudaStream_t);
}
