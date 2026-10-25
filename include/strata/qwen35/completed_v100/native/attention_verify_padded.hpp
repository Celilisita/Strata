#pragma once
#include "strata/qwen35/completed_v100/native/attention_verify.hpp"
namespace strata::completed_v100::qwen35::native {
struct VerifyPaddedResources {int registers,blocks_per_SM;size_t shared_bytes,local_bytes;};
VerifyPaddedResources verify_padded_resources();
void verify_attention_padded_q8(const float *,const void *,const void *,float *,void *,size_t,
                                PrefillAttentionShape,int,cudaStream_t);
}
