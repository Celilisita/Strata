#pragma once
#include "ggml.h"
#include <cstdint>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
inline constexpr uint64_t overlap_compress_tag = 0x535452434F4D5031ull;
struct OverlapCompressParams { int32_t ratio, dim, blocks, rows; float eps; };
static_assert(sizeof(OverlapCompressParams) == 20);
inline bool is_overlap_compress(const ggml_tensor * tensor) {
    if (!tensor || tensor->op != GGML_OP_MAP_CUSTOM3) return false;
    uint64_t tag = 0;
    std::memcpy(&tag, reinterpret_cast<const char *>(tensor->op_params) + 16, sizeof(tag));
    return tag == overlap_compress_tag;
}
inline OverlapCompressParams overlap_compress_params(const ggml_tensor * tensor) {
    OverlapCompressParams params{};
    std::memcpy(&params, reinterpret_cast<const char *>(tensor->op_params) + 24, sizeof(params));
    return params;
}
inline ggml_tensor * build_overlap_compress(ggml_context * ctx, ggml_tensor * kv,
        ggml_tensor * score, ggml_tensor * indices, ggml_tensor * norm,
        int ratio, int blocks, float eps) {
    if (!ctx || !kv || !score || !indices || !norm || ratio != 4 || blocks < 1 || blocks > 8 ||
        kv->type != GGML_TYPE_F32 || score->type != GGML_TYPE_F32 ||
        indices->type != GGML_TYPE_I32 || norm->type != GGML_TYPE_F32 ||
        kv->ne[0] != score->ne[0] || kv->ne[1] != score->ne[1] ||
        kv->ne[0] != 2 * norm->ne[0] || norm->ne[0] < 1 || norm->ne[0] > 512 ||
        indices->ne[0] != 2 * ratio * blocks || !ggml_is_contiguous(kv) ||
        !ggml_is_contiguous(score) || !ggml_is_contiguous(indices) ||
        !ggml_is_contiguous(norm) || eps <= 0)
        throw std::runtime_error("Strata overlap compressor tensor contract differs");
    auto * output = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, norm->ne[0], blocks);
    output->op = GGML_OP_MAP_CUSTOM3;
    output->src[0] = kv; output->src[1] = score; output->src[2] = indices; output->src[3] = norm;
    const int n_tasks = 1;
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 8, &n_tasks, sizeof(n_tasks));
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 16, &overlap_compress_tag, sizeof(overlap_compress_tag));
    const OverlapCompressParams params{ratio, int(norm->ne[0]), blocks, int(kv->ne[1]), eps};
    std::memcpy(reinterpret_cast<char *>(output->op_params) + 24, &params, sizeof(params));
    ggml_set_name(output, "strata_overlap_compress");
    return output;
}
}
struct ggml_backend_cuda_context;
void strata_overlap_compress_cuda(ggml_backend_cuda_context &, ggml_tensor *);
