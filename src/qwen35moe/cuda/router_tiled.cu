#include "strata/qwen35moe/router_tiled.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
namespace strata::qwen35moe {
namespace {
__global__ void project_four(const float* __restrict__ rows, const float* __restrict__ x,
                            float* __restrict__ logits, int width, int outputs, int tokens) {
    const int row = int(blockIdx.x), first = 4 * int(blockIdx.y), lane = int(threadIdx.x);
    float sums[4]{};
    for (int i = lane; i < width; i += 128) {
        const float weight = rows[size_t(row) * width + i];
#pragma unroll
        for (int j = 0; j < 4; ++j)
            if (first + j < tokens) sums[j] = fmaf(weight, x[size_t(first + j) * width + i], sums[j]);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            sums[j] += __shfl_xor_sync(0xffffffffu, sums[j], mask);
    __shared__ float partial[4][4];
    if ((lane & 31) == 0) {
#pragma unroll
        for (int j = 0; j < 4; ++j) partial[j][lane / 32] = sums[j];
    }
    __syncthreads();
    if (lane < 32) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            float sum = lane < 4 ? partial[j][lane] : 0.0f;
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, mask);
            if (lane == 0 && first + j < tokens) logits[size_t(first + j) * outputs + row] = sum;
        }
    }
}
}
void route_project_tiled(const float* rows, const float* input, float* logits,
                        int width, int outputs, int tokens, void* stream) {
    if (!rows || !input || !logits || !stream || width <= 0 || width % 128 ||
        outputs <= 0 || outputs > 65535 || tokens < 4 || tokens > 8192)
        throw std::invalid_argument("invalid canonical tiled F32 router");
    project_four<<<dim3(unsigned(outputs), unsigned((tokens + 3) / 4)), 128, 0,
        static_cast<cudaStream_t>(stream)>>>(rows, input, logits, width, outputs, tokens);
    const auto e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
} // namespace strata::qwen35moe
