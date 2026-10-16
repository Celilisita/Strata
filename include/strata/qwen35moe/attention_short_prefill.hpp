#pragma once
#include <cstdint>
namespace strata::qwen35moe {
// Canonical short-vector attention over independent causal rows. Caller owns
// disjoint Q/output [count,4096], F16 KV [capacity,512], status and ordered stream.
void attention_short_prefill(const float* q, const uint16_t* k, const uint16_t* v,
    int first, int count, int64_t capacity, float* output, int32_t* status, void* stream);
}
