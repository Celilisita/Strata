#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstddef>
namespace strata::gemma4::native {
// Private experimental dtype: F16 QK inputs/F32 dot, F16 PV accumulator.
// No ggml context, pool, graph owner, scheduler or model object is created.
bool global_flash_requested();
int global_flash_key_bound(int active_upper);
std::size_t global_flash_workspace_bytes(int tokens,int key_upper);
void prepare_global_flash(cudaStream_t); // cold, before capture; no forward
void global_flash_attention(const float* q,float* out,CacheView cache,
    const int* position,int past_upper,int tokens,int maximum_tokens,
    AttentionWorkspace workspace,cudaStream_t stream);
namespace global_flash_detail {
struct Buffers {void* key;void* value;void* mask;unsigned* invalid;int keys;};
Buffers bind(AttentionWorkspace,int tokens,int key_upper);
}
}
