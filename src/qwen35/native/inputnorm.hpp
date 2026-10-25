#pragma once
#include <fp4_compensated.hpp>
namespace strata::qwen35::native {
// Task-only producer. Caller keeps the original RMS/codec path outside this exact shape.
inline bool inputnorm_eligible(int features,int tokens){return features==5120&&tokens==8192;}
PackedFp4Resources inputnorm_resources();
void inputnorm_fused(const float *input,const float *gamma,Nvfp4Input encoded,void *optional_fp16,
                     int tokens,float epsilon,float global,cudaStream_t stream);
}
