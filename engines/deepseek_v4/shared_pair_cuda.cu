#include "shared_pair.hpp"
#include "q8_pair_common.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/quantize.cuh"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
using namespace strata::deepseek_v4::cuda_q8;
__global__ void shared_pair_kernel(const Q81 * x, const Q80 * gate, const Q80 * up,
        float * output, int tokens, float clamp) {
    ggml_cuda_pdl_lc();
    const int row = blockIdx.x, block = threadIdx.x;
    const Q80 g = gate[row * 128 + block], u = up[row * 128 + block];
    __shared__ float gate_sums[4], up_sums[4];
    ggml_cuda_pdl_sync();
    for (int token = 0; token < tokens; ++token) {
        const Q81 a = x[token * 128 + block];
        int gdot = 0, udot = 0;
        for (int i = 0; i < 32; i += 4) {
            const int activation = pack4(a.qs + i);
            gdot = __dp4a(pack4(g.qs + i), activation, gdot);
            udot = __dp4a(pack4(u.qs + i), activation, udot);
        }
        const float scale = __half2float(__low2half(a.ds));
        float gv = float(gdot) * scale * __half2float(g.d);
        float uv = float(udot) * scale * __half2float(u.d);
        gv = warp_sum(gv); uv = warp_sum(uv);
        if ((block & 31) == 0) { gate_sums[block >> 5] = gv; up_sums[block >> 5] = uv; }
        __syncthreads();
        if (block < 32) {
            gv = warp_sum(block < 4 ? gate_sums[block] : 0.f);
            uv = warp_sum(block < 4 ? up_sums[block] : 0.f);
            if (block == 0) {
                gv = fminf(gv, clamp);
                uv = fmaxf(fminf(uv, clamp), -clamp);
                output[token * 2048 + row] = gv / (1.f + expf(-gv)) * uv;
            }
        }
        __syncthreads();
    }
}
}
void strata_shared_pair_cuda(ggml_backend_cuda_context & context, ggml_tensor * tensor) {
    using namespace strata::deepseek_v4;
    if (!is_shared_pair(tensor)) throw std::runtime_error("Strata shared pair dispatch tag differs");
    const auto p = shared_pair_params(tensor);
    if (p.n_in != 4096 || p.n_out != 2048 || p.tokens < 1 || p.tokens > 2 ||
        tensor->src[0]->type != GGML_TYPE_F32 || tensor->src[1]->type != GGML_TYPE_Q8_0 ||
        tensor->src[2]->type != GGML_TYPE_Q8_0)
        throw std::runtime_error("Strata shared pair CUDA geometry differs");
    ggml_cuda_pool_alloc<Q81> quantized(context.pool(), size_t(128 * p.tokens));
    quantize_row_q8_1_cuda(static_cast<const float *>(tensor->src[0]->data), nullptr, quantized.ptr,
        GGML_TYPE_F32, 4096, 4096, 4096 * p.tokens, 4096 * p.tokens,
        4096, p.tokens, 1, 1, context.stream());
    shared_pair_kernel<<<2048, 128, 0, context.stream()>>>(quantized.ptr,
        static_cast<const Q80 *>(tensor->src[1]->data), static_cast<const Q80 *>(tensor->src[2]->data),
        static_cast<float *>(tensor->data), p.tokens, p.clamp);
    CUDA_CHECK(cudaGetLastError());
}
