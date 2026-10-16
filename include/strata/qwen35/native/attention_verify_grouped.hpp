#pragma once
#include "strata/qwen35/native/attention_verify.hpp"
namespace strata::qwen35::native {
struct VerifyGroupedResources {int registers,blocks_per_SM;size_t shared_bytes,local_bytes;};
VerifyGroupedResources verify_grouped_resources();
void verify_attention_grouped_q8(const float *,const void *,const void *,float *,void *,size_t,
                                 PrefillAttentionShape,int,cudaStream_t);
}
