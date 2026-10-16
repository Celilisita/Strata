#pragma once
#include <cuda_runtime.h>
#include "strata/qwen35/native/decode_graph_patch.hpp"
namespace strata::qwen35::native {
// Signed Q8_0, half delta +32 codes, token/head/block contiguous.
void q8_cache_write(const float *,void *,int,int,int,int,int,cudaStream_t);
void q8_cache_write_position(const float *,void *,int,int,const int *,cudaStream_t);
DecodeNode decode_q8_cache_node(void *);
}
