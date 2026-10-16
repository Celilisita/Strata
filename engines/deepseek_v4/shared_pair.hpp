#pragma once
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
inline constexpr uint64_t shared_pair_tag = 0x5354525348504131ull;
struct SharedPairParams { int32_t n_in, n_out, tokens; float clamp; };
static_assert(sizeof(SharedPairParams) == 16);
inline bool is_shared_pair(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MAP_CUSTOM3) return false;
    uint64_t tag = 0;
    std::memcpy(&tag, reinterpret_cast<const char *>(tensor->op_params) + 16, sizeof(tag));
    return tag == shared_pair_tag;
}
inline SharedPairParams shared_pair_params(const ggml_tensor * tensor) {
    SharedPairParams params{};
    std::memcpy(&params, reinterpret_cast<const char *>(tensor->op_params) + 24, sizeof(params));
    return params;
}
inline ggml_tensor * build_shared_pair(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * gate, ggml_tensor * up, float clamp) {
    if (!ctx || !x || !gate || !up || x->type != GGML_TYPE_F32 ||
        gate->type != GGML_TYPE_Q8_0 || up->type != GGML_TYPE_Q8_0 ||
        x->ne[0] != 4096 || x->ne[1] < 1 || x->ne[1] > 2 ||
        gate->ne[0] != 4096 || up->ne[0] != 4096 || gate->ne[1] != 2048 || up->ne[1] != 2048 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(gate) || !ggml_is_contiguous(up) || clamp < 1e-6f)
        throw std::runtime_error("Strata shared pair tensor contract differs");
    auto * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 2048, x->ne[1]);
    output->op = GGML_OP_MAP_CUSTOM3;
    output->src[0] = x; output->src[1] = gate; output->src[2] = up;
    const int tasks = 1;
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 8, &tasks, sizeof(tasks));
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 16, &shared_pair_tag, sizeof(shared_pair_tag));
    const SharedPairParams params{4096, 2048, int32_t(x->ne[1]), clamp};
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 24, &params, sizeof(params));
    ggml_set_name(output, "strata_shared_gate_up_act");
    return output;
}
}
struct ggml_backend_cuda_context;
void strata_shared_pair_cuda(ggml_backend_cuda_context &, ggml_tensor *);
