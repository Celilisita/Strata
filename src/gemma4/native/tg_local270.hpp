#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"

namespace strata::gemma4::native {
// Private T1 leaf: local D256/Q32/KV16, true physical ring, window1024.
// The existing attention owner lends its arena and ordered stream. The caller
// gates the default-off policy and passes the full request context, not ring size.
size_t tg_local270_workspace_bytes(CacheView cache, int heads);
void tg_local270_attention(const float* query, float* output, CacheView cache,
    const int* device_position, int heads, int tokens, int context,
    AttentionWorkspace workspace, cudaStream_t stream);
}
