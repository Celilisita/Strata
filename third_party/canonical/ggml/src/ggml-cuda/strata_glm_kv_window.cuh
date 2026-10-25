#pragma once
#include "ggml-backend.h"
#include <cstdint>

// Host history is authoritative. Tags identify the complete logical row held
// by each bounded device slot; an absent row is read from mapped host history.
struct StrataGlmKvWindow {
    unsigned char* host;
    unsigned char* device;
    int64_t row_bytes;
    int64_t resident_tokens;
    int64_t* tags;
    int64_t logical_bytes;
    ggml_type type;
    int64_t row_elements;
};
bool strata_glm_kv_window_view(const ggml_tensor*, StrataGlmKvWindow&);
// Called after a canonical CPY write on that same CUDA stream.
void strata_glm_kv_window_written(const ggml_tensor*, void* stream);
void strata_glm_kv_attention_update(int device, const StrataGlmKvWindow&, bool remove);

#ifdef __CUDACC__
static __device__ __forceinline__ const unsigned char* strata_glm_kv_read_ptr(
        const StrataGlmKvWindow& v, int64_t offset) {
    const int64_t row = offset/v.row_bytes;
    const int64_t slot = row%v.resident_tokens;
    return v.tags[slot] == row ? v.device+slot*v.row_bytes+offset%v.row_bytes : v.host+offset;
}
#endif
