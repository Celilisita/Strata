#include "strata/qwen35moe/cuda_kernels.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
// Same per-lane four-term sum and XOR reduction as canonical CUDA L2 norm.
// Four independent heads share a block; no shared scratch or block barrier.
__global__ void warp_norm(float* values, float epsilon_squared, int rows, int stride) {
    const int lane = threadIdx.x & 31;
    const int head = int(blockIdx.x) * 4 + int(threadIdx.x) / 32;
    if (head >= rows) return;
    float* row = values + size_t(blockIdx.y) * stride + head * 128;
    float v[4]; float square = 0.0f;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        v[r] = row[lane + r * 32];
        square += v[r] * v[r];
    }
#pragma unroll
    for (int mask = 16; mask; mask >>= 1)
        square += __shfl_xor_sync(0xffffffffu, square, mask);
    const float scale = rsqrtf(fmaxf(square, epsilon_squared));
#pragma unroll
    for (int r = 0; r < 4; ++r) row[lane + r * 32] = v[r] * scale;
}
__global__ void norm(float * values, float epsilon_squared, int stride) {
    const int col = threadIdx.x;
    float * row = values + blockIdx.y * stride + blockIdx.x * 128;
    const float value = row[col];
    float square = value * value;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1)
        square += __shfl_xor_sync(0xffffffffu, square, mask);
    __shared__ float sums[4];
    if ((col & 31) == 0) sums[col / 32] = square;
    __syncthreads();
    if (col < 32) {
        square = col < 4 ? sums[col] : 0.0f;
#pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            square += __shfl_xor_sync(0xffffffffu, square, mask);
        if (col == 0) sums[0] = rsqrtf(fmaxf(square, epsilon_squared));
    }
    __syncthreads();
    row[col] = value * sums[0];
}
}

void gdn_l2_norm(float * values, int rows, int width, float epsilon, void * stream, int tokens, int stride, bool canonical_warp) {
    if (!values || !stream || rows <= 0 || width != 128 || tokens < 1 || tokens > 8192 ||
        (tokens > 1 && stride < rows * width) ||
        !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("invalid Qwen3.5 GDN L2 parameters");
    if (canonical_warp)
        warp_norm<<<dim3(unsigned((rows + 3) / 4), unsigned(tokens)), 128, 0,
            static_cast<cudaStream_t>(stream)>>>(values, epsilon * epsilon, rows, stride);
    else
        norm<<<dim3(unsigned(rows), unsigned(tokens)), 128, 0, static_cast<cudaStream_t>(stream)>>>(values, epsilon * epsilon, stride);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

} // namespace strata::qwen35moe
