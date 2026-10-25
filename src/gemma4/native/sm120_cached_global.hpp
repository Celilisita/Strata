#pragma once
#include "pp_tc_attention.hpp"
#include <cuda_fp16.h>

namespace strata::gemma4::native {
bool cached_global_requested();
std::size_t cached_global_workspace_bytes(int maximum, int capacity);
void cached_global_attention(const float* q, float* out, CacheView cache,
    const int* position, int past, int tokens, int maximum,
    AttentionWorkspace workspace, cudaStream_t stream);

namespace cached_global_detail {
struct Buffers {
    half* key;
    half* value;
    unsigned* invalid;
    int keys;
};
Buffers bind(AttentionWorkspace workspace, int maximum, int capacity);
int staged_key_bound(int upper);
void stage(Buffers buffers, CacheView cache, const int* position,
    int tokens, int upper, cudaStream_t stream);
void propagate(Buffers buffers, float* out, int tokens, cudaStream_t stream);
}
}
