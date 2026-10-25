#pragma once
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native {
// PP32-only joint Q4/Q6 descriptors; q6 may be null for gated Out preparation.
void rms_norm_quantized32(const float *,const float *,float *,const float *,
                         int features,int heads,float epsilon,void *q4,void *q6,cudaStream_t);
// Contiguous feature-major rows. Optional gate applies SiLU after weighted RMS.
void rms_norm(const float *input, const float *weight, float *output,
              int features, int rows, float epsilon, cudaStream_t, const float *gate = nullptr,
              int input_row_stride = 0);
// History is [3,channels], key-contiguous in each three-element channel row.
// updated_history must be distinct from history; input/output are [channels,T].
void causal_conv4(const float *input, const float *history, const float *weights,
                  float *raw, float *activated, float *updated_history,
                  int channels, int tokens, cudaStream_t);
// Split normalized Q/K and contiguous V from [2*D*Hq+D*Hv,T] convolution output.
void gdn_split(const float *mixed, float *q, float *k, float *v,
               int dim, int query_heads, int value_heads, int tokens, float epsilon, cudaStream_t);
void gdn_parameters(const float *alpha, const float *beta_raw, const float *bias, const float *a,
                    float *decay, float *beta, int heads, int tokens, cudaStream_t);
void add(const float *first, const float *second, float *out, size_t count, cudaStream_t);
void add_rms_norm(const float *first,const float *second,const float *weight,float *residual,float *normalized,
                  int features,int rows,float epsilon,cudaStream_t);
void gdn_prepare(const float *mixed,const float *alpha,const float *beta_raw,const float *bias,const float *a,
                 float *q,float *k,float *v,float *decay,float *beta,int dim,int query_heads,int value_heads,
                 int tokens,float epsilon,cudaStream_t);

}
