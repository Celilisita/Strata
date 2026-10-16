#pragma once
#include "strata/qwen35/native/ffn.hpp"
#include "strata/qwen35/native/token_mask.hpp"
namespace strata::qwen35::native {
struct OutputHeadBatch {
    float *normalized,*logits;
    void *quantized,*greedy;
    size_t greedy_bytes;
    int *ids;
    unsigned *flags;
};
// Same model normalization and owned Q8 activation/Q6 weight mathematics.
// All rows share the weight fetch and greedy kernels; no host response here.
void output_head_batch(Matrix,const float *norm,const float *hidden,int rows,
                       float epsilon,OutputHeadBatch,cudaStream_t,TokenMask={});
}
