#include "strata/qwen35moe/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {

__global__ void post(const float * __restrict__ output, const float * __restrict__ z,
                     const float * __restrict__ gamma, float * __restrict__ destination,
                     float epsilon) {
    const int col = threadIdx.x;
    const int offset = blockIdx.x * 128 + col;
    const float value = output[offset];
    float square = value * value;
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) square += __shfl_xor_sync(0xffffffffu, square, mask);
    __shared__ float warp_sums[4];
    if ((col & 31) == 0) warp_sums[col / 32] = square;
    __syncthreads();
    if (col < 32) {
        square = col < 4 ? warp_sums[col] : 0.0f;
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) square += __shfl_xor_sync(0xffffffffu, square, mask);
        if (col == 0) warp_sums[0] = rsqrtf(square / 128.0f + epsilon);
    }
    __syncthreads();
    const float weighted = (value * warp_sums[0]) * gamma[col];
    const float gate = z[offset];
    destination[offset] = weighted * (gate / (1.0f + expf(-gate)));
}

} // namespace

void gdn_post_silu(const float * output, const float * z, const float * gamma,
                   float * destination, int heads, float epsilon, void * stream) {
    if (!output || !z || !gamma || !destination || !stream || heads <= 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0f)
        throw std::invalid_argument("invalid qwen35moe GDN post parameters");
    post<<<heads, 128, 0, static_cast<cudaStream_t>(stream)>>>(output, z, gamma, destination, epsilon);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace strata::qwen35moe
