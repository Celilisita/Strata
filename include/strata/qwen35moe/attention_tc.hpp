#pragma once
#include <cstdint>
#include <cstddef>
namespace strata::qwen35moe {
// Measured Volta PP only: F32 Q, persistent F16 KV, 16 query/2 KV heads,
// D=256, causal positions. Caller owns cache; capacity must include a64-row tail.
// Padding writes only after first+count, never any active KV row.
inline bool attention_tc_fits(int first, int count, int capacity) {
    return first >= 0 && count >= 1 && count <= 8192 && capacity >= 1 && capacity <= 196608 &&
        first <= capacity - count && ((first + count + 63) / 64) * 64 <= capacity;
}
void attention_tc_prefill(const float* q, uint16_t* k, uint16_t* v,
    int first, int count, int capacity, float* out, void* stream, bool compact = false, bool scratch_reuse = false);
struct AttentionTcResources { int registers; std::size_t shared_bytes, local_bytes; };
AttentionTcResources attention_tc_resources(bool compact, bool scratch_reuse = false);
}
