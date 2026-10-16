#pragma once

#include <cstdint>

namespace strata::qwen35moe {

// Shared byte layout for the two-stage GPU selector and its 12-byte readback.
struct HeadPick {
    int32_t id;
    int32_t first_bad;
    float logit;
};

void select_head_greedy(const float* logits, int n_vocab, HeadPick* partials,
                        HeadPick* result, void* stream, const int32_t* failure = nullptr);

} // namespace strata::qwen35moe
