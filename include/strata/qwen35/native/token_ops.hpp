#pragma once
#include "strata/qwen35/native/quant.hpp"
#include "strata/qwen35/native/token_mask.hpp"
#include <cuda_runtime.h>

namespace strata::qwen35::native {
void gather_embedding(Matrix,const int *device_ids,float *,int tokens,cudaStream_t);
void set_positions(int *device_positions,int first,int tokens,cudaStream_t);
struct GreedyPartial { float value; int token; unsigned invalid; };
size_t greedy_workspace(int vocabulary);
// Token and finite flag stay on device. Host reads only at an output boundary.
void greedy_pick(const float *logits,int vocabulary,void *workspace,size_t bytes,
                 int *device_token,unsigned *device_invalid,cudaStream_t,TokenMask={});
void greedy_pick_batch(const float *logits,int vocabulary,int rows,void *workspace,size_t bytes,
                       int *device_tokens,unsigned *device_invalid,cudaStream_t,TokenMask={});
}
