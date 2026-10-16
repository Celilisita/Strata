#include "overlap_compress.hpp"
#include "ggml-cuda/common.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
__global__ void overlap_compress_kernel(const float * kv, const float * score,
        const int32_t * indices, const float * norm, float * output,
        int dim, int rows, int ratio, float eps) {
    ggml_cuda_pdl_lc();
    __shared__ float squared[512];
    const int d = threadIdx.x, block = blockIdx.x;
    float value = 0.f;
    ggml_cuda_pdl_sync();
    if (d < dim) {
        const int base = block * ratio;
        float maximum = -INFINITY;
        for (int i = 0; i < 2 * ratio; ++i) {
            const int row = indices[(i < ratio ? 0 : gridDim.x * ratio) + base + (i % ratio)];
            if (row >= 0 && row < rows) {
                const float s = score[row * (2 * dim) + d + (i < ratio ? 0 : dim)];
                maximum = fmaxf(maximum, s);
            }
        }
        float numerator = 0.f, denominator = 0.f;
        if (isfinite(maximum)) for (int i = 0; i < 2 * ratio; ++i) {
            const int row = indices[(i < ratio ? 0 : gridDim.x * ratio) + base + (i % ratio)];
            if (row >= 0 && row < rows) {
                const int offset = row * (2 * dim) + d + (i < ratio ? 0 : dim);
                const float weight = expf(score[offset] - maximum);
                numerator = fmaf(kv[offset], weight, numerator);
                denominator += weight;
            }
        }
        value = denominator > 0.f ? numerator / denominator : 0.f;
    }
    squared[d] = d < dim ? value * value : 0.f;
    __syncthreads();
    for (int stride = 256; stride; stride >>= 1) {
        if (d < stride) squared[d] += squared[d + stride];
        __syncthreads();
    }
    if (d < dim) output[block * dim + d] = value * rsqrtf(squared[0] / float(dim) + eps) * norm[d];
}
}
void strata_overlap_compress_cuda(ggml_backend_cuda_context & context, ggml_tensor * tensor) {
    using namespace strata::deepseek_v4;
    if (!is_overlap_compress(tensor)) throw std::runtime_error("Strata compressor dispatch tag differs");
    const auto p = overlap_compress_params(tensor);
    if (p.ratio != 4 || p.dim < 1 || p.dim > 512 || p.blocks < 1 || p.blocks > 8 || p.rows < 1 ||
        tensor->ne[0] != p.dim || tensor->ne[1] != p.blocks ||
        tensor->src[0]->type != GGML_TYPE_F32 || tensor->src[1]->type != GGML_TYPE_F32 ||
        tensor->src[2]->type != GGML_TYPE_I32 || tensor->src[3]->type != GGML_TYPE_F32)
        throw std::runtime_error("Strata compressor CUDA geometry differs");
    overlap_compress_kernel<<<p.blocks, 512, 0, context.stream()>>>(
        static_cast<const float *>(tensor->src[0]->data),
        static_cast<const float *>(tensor->src[1]->data),
        static_cast<const int32_t *>(tensor->src[2]->data),
        static_cast<const float *>(tensor->src[3]->data),
        static_cast<float *>(tensor->data), p.dim, p.rows, p.ratio, p.eps);
    CUDA_CHECK(cudaGetLastError());
}
