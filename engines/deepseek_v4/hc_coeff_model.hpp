#pragma once
#include "hc_coeff.hpp"
#include "ggml-backend.h"
#include <cstdlib>

namespace strata::deepseek_v4 {
inline ggml_tensor* try_hc_coeff(ggml_context* context, ggml_tensor* mixes,
                                ggml_tensor* scale, ggml_tensor* base, float eps) {
    const char* flag = std::getenv("STRATA_DSV4_HC_COEFF_FUSION");
    if (!flag || std::strcmp(flag, "1") || !mixes ||
        (mixes->ne[1] != 1 && mixes->ne[1] != 2 && mixes->ne[1] != 5)) return nullptr;
    // Only actual selected CUDA weight placement can create the tagged operation.
    if (!scale || !base || !scale->buffer || !base->buffer ||
        ggml_backend_buffer_is_host(scale->buffer) || ggml_backend_buffer_is_host(base->buffer)) return nullptr;
    const auto device = ggml_backend_buft_get_device(ggml_backend_buffer_get_type(scale->buffer));
    if (!device || std::strcmp(ggml_backend_reg_name(ggml_backend_dev_backend_reg(device)), "CUDA")) return nullptr;
    return build_hc_coeff(context, mixes, scale, base, eps);
}
}
