#pragma once
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
inline constexpr uint64_t qkv_pair_tag = 0x535452514B565031ull;
struct QkvPairParams { int32_t n_in, q_rows, kv_rows, tokens; };
static_assert(sizeof(QkvPairParams) == 16);
inline bool is_qkv_pair(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MAP_CUSTOM3) return false;
    uint64_t tag = 0;
    std::memcpy(&tag, reinterpret_cast<const char *>(tensor->op_params) + 16, sizeof(tag));
    return tag == qkv_pair_tag;
}
inline QkvPairParams qkv_pair_params(const ggml_tensor * tensor) {
    QkvPairParams params{};
    std::memcpy(&params, reinterpret_cast<const char *>(tensor->op_params) + 24, sizeof(params));
    return params;
}
inline ggml_tensor * build_qkv_pair(ggml_context * ctx, ggml_tensor * x,
        ggml_tensor * q, ggml_tensor * kv) {
    if (!ctx || !x || !q || !kv || x->type != GGML_TYPE_F32 ||
        q->type != GGML_TYPE_Q8_0 || kv->type != GGML_TYPE_Q8_0 ||
        x->ne[0] != 4096 || x->ne[1] < 1 || x->ne[1] > 2 ||
        q->ne[0] != 4096 || kv->ne[0] != 4096 || q->ne[1] != 1024 || kv->ne[1] != 512 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(q) || !ggml_is_contiguous(kv))
        throw std::runtime_error("Strata Q/KV pair tensor contract differs");
    auto * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, 1536, x->ne[1]);
    output->op = GGML_OP_MAP_CUSTOM3;
    output->src[0] = x; output->src[1] = q; output->src[2] = kv;
    const int tasks = 1;
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 8, &tasks, sizeof(tasks));
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 16, &qkv_pair_tag, sizeof(qkv_pair_tag));
    const QkvPairParams params{4096, 1024, 512, int32_t(x->ne[1])};
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 24, &params, sizeof(params));
    ggml_set_name(output, "strata_q_kv_pair");
    return output;
}
}
struct ggml_backend_cuda_context;
void strata_qkv_pair_cuda(ggml_backend_cuda_context &, ggml_tensor *);
