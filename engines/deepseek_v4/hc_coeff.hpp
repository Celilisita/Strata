#pragma once
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
inline constexpr int hc_coeff_tag_index = GGML_MAX_OP_PARAMS / sizeof(int32_t) - 1;
inline constexpr int32_t hc_coeff_tag = 0x48434331;
inline bool is_hc_coeff(const ggml_tensor* t) {
    return t && t->op == GGML_OP_SCALE && t->op_params[hc_coeff_tag_index] == hc_coeff_tag;
}
// Packed role dimension retains each original [4, tokens] coefficient layout.
// Only the isolated benchmark uses this operation until its real-data gate wins.
inline ggml_tensor* build_hc_coeff(ggml_context* context, ggml_tensor* mixes,
                                  ggml_tensor* scale, ggml_tensor* base, float eps) {
    if (!context || !mixes || !scale || !base || mixes->type != GGML_TYPE_F32 ||
        scale->type != GGML_TYPE_F32 || base->type != GGML_TYPE_F32 ||
        mixes->ne[0] != 24 || mixes->ne[1] < 1 || mixes->ne[1] > 8 ||
        mixes->ne[2] != 1 || mixes->ne[3] != 1 ||
        ggml_nelements(scale) != 3 || ggml_nelements(base) != 24 ||
        !ggml_is_contiguous(mixes) || !ggml_is_contiguous(scale) || !ggml_is_contiguous(base) ||
        !std::isfinite(eps) || eps <= 0)
        throw std::runtime_error("mHC coefficient actual contract differs");
    auto* result = ggml_new_tensor_3d(context, GGML_TYPE_F32, 4, mixes->ne[1], 2);
    result->op = GGML_OP_SCALE;
    result->src[0] = mixes; result->src[1] = scale; result->src[2] = base;
    std::memcpy(result->op_params, &eps, sizeof(eps));
    result->op_params[hc_coeff_tag_index] = hc_coeff_tag;
    ggml_set_name(result, "strata_hc_coefficients");
    return result;
}
inline ggml_tensor* hc_coeff_role(ggml_context* context, ggml_tensor* packed, int role) {
    if (!is_hc_coeff(packed) || role < 0 || role > 1) throw std::runtime_error("mHC coefficient role differs");
    return ggml_view_2d(context, packed, 4, packed->ne[1], packed->nb[1], role * packed->nb[2]);
}
}
struct ggml_backend_cuda_context;
bool strata_hc_coeff_cuda(ggml_backend_cuda_context&, ggml_tensor*);
