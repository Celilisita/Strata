#pragma once
#include "strata/qwen35moe/attention_tc.hpp"
namespace strata::qwen35moe {
// Independent operator qualification only; identical accepted Q/F16 KV geometry.
void attention_register_prefill(const float* q, uint16_t* k, uint16_t* v,
    int first, int count, int capacity, float* out, void* stream);
AttentionTcResources attention_register_resources();
int attention_register_active_blocks();
}
