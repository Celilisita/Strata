#pragma once
namespace strata::qwen35moe {
// F32 router with canonical lane/K/reduction order, four independent tokens.
void route_project_tiled(const float* rows, const float* input, float* logits,
    int width, int outputs, int tokens, void* stream);
}
