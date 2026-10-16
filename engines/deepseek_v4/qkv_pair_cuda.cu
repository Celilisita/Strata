#include "qkv_pair.hpp"
#include "q8_pair_common.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/quantize.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
using namespace strata::deepseek_v4::cuda_q8;
__global__ void qkv_pair_kernel(const Q81 * x, const Q80 * q, const Q80 * kv,
        float * output, int tokens) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x, block = threadIdx.x;
    const Q80 weight = row < 1024 ? q[row * 128 + block] : kv[(row - 1024) * 128 + block];
    __shared__ float sums[4];
    ggml_cuda_pdl_sync();
    for (int token = 0; token < tokens; ++token) {
        const Q81 activation = x[token * 128 + block];
        int dot = 0;
        for (int i = 0; i < 32; i += 4)
            dot = __dp4a(pack4(weight.qs + i), pack4(activation.qs + i), dot);
        float value = float(dot) * __half2float(__low2half(activation.ds)) * __half2float(weight.d);
        value = warp_sum(value);
        if ((block & 31) == 0) sums[block >> 5] = value;
        __syncthreads();
        if (block < 32) {
            value = warp_sum(block < 4 ? sums[block] : 0.f);
            if (block == 0) output[token * 1536 + row] = value;
        }
        __syncthreads();
    }
}
}
void strata_qkv_pair_cuda(ggml_backend_cuda_context & context, ggml_tensor * tensor) {
    using namespace strata::deepseek_v4;
    if (!is_qkv_pair(tensor)) throw std::runtime_error("Strata Q/KV pair dispatch tag differs");
    const auto p = qkv_pair_params(tensor);
    if (p.n_in != 4096 || p.q_rows != 1024 || p.kv_rows != 512 || p.tokens < 1 || p.tokens > 2 ||
        tensor->src[0]->type != GGML_TYPE_F32 || tensor->src[1]->type != GGML_TYPE_Q8_0 ||
        tensor->src[2]->type != GGML_TYPE_Q8_0)
        throw std::runtime_error("Strata Q/KV pair CUDA geometry differs");
    ggml_cuda_pool_alloc<Q81> quantized(context.pool(), size_t(128 * p.tokens));
    quantize_row_q8_1_cuda(static_cast<const float *>(tensor->src[0]->data), nullptr, quantized.ptr,
        GGML_TYPE_F32, 4096, 4096, 4096 * p.tokens, 4096 * p.tokens,
        4096, p.tokens, 1, 1, context.stream());
    qkv_pair_kernel<<<1536, 128, 0, context.stream()>>>(quantized.ptr,
        static_cast<const Q80 *>(tensor->src[1]->data), static_cast<const Q80 *>(tensor->src[2]->data),
        static_cast<float *>(tensor->data), p.tokens);
    CUDA_CHECK(cudaGetLastError());
}
