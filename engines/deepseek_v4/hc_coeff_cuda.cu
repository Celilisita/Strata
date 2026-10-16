#include "hc_coeff.hpp"
#include "ggml-cuda/common.cuh"

namespace {
__global__ void hc_coeff_kernel(const float* mixes, const float* scale, const float* base,
                                float* result, int tokens, float eps) {
    ggml_cuda_pdl_lc();
    const int index = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (index >= 8 * tokens) return;
    ggml_cuda_pdl_sync();
    const int role = index / (4 * tokens);
    const int token = (index / 4) % tokens;
    const int lane = index % 4;
    // Original MUL then ADD are separate kernels; disallow contraction across
    // those stores. Sigmoid and final SCALE use the canonical CUDA expressions.
    const float product = __fmul_rn(mixes[24 * token + 4 * role + lane], scale[role]);
    const float affine = __fadd_rn(product, base[4 * role + lane]);
    const float sigmoid = 1.0f / (1.0f + expf(-affine));
    result[index] = role ? 2.0f * sigmoid + 0.0f : sigmoid + eps;
}
}
bool strata_hc_coeff_cuda(ggml_backend_cuda_context& context, ggml_tensor* dst) {
    using namespace strata::deepseek_v4;
    if (!is_hc_coeff(dst)) return false;
    const auto* mixes = dst->src[0];
    const auto* scale = dst->src[1];
    const auto* base = dst->src[2];
    GGML_ASSERT(mixes && scale && base && mixes->type == GGML_TYPE_F32 &&
                scale->type == GGML_TYPE_F32 && base->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mixes->ne[0] == 24 && mixes->ne[1] >= 1 && mixes->ne[1] <= 8 &&
                ggml_nelements(mixes) == 24 * mixes->ne[1] && ggml_nelements(scale) == 3 &&
                ggml_nelements(base) == 24 && dst->ne[0] == 4 && dst->ne[1] == mixes->ne[1] && dst->ne[2] == 2);
    GGML_ASSERT(ggml_is_contiguous(mixes) && ggml_is_contiguous(scale) && ggml_is_contiguous(base) && ggml_is_contiguous(dst));
    float eps; std::memcpy(&eps, dst->op_params, sizeof(eps));
    const int elements = int(ggml_nelements(dst));
    const auto launch = ggml_cuda_kernel_launch_params((elements + 31) / 32, 32, 0, context.stream());
    ggml_cuda_kernel_launch(hc_coeff_kernel, launch, static_cast<const float*>(mixes->data),
        static_cast<const float*>(scale->data), static_cast<const float*>(base->data),
        static_cast<float*>(dst->data), int(mixes->ne[1]), eps);
    return true;
}
