#pragma once
#include <cuda_runtime.h>
namespace strata::qwen35::native {
struct ConvGdnInput {
    const float *mixed,*history,*filter,*alpha,*beta_raw,*bias,*decay;
};
struct ConvGdnOutput {float *q,*k,*v,*g,*beta,*history;};
void conv_gdn_fused(ConvGdnInput,ConvGdnOutput,int query_heads,int value_heads,int tokens,float epsilon,cudaStream_t);
}
