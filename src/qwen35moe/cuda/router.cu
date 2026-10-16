// Adapted from ggml-cuda's topk-moe implementation (MIT), with the
// Qwen3.5/3.6 256-expert, top-8 geometry. Copyright ggml authors.
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/core/device_miss.cuh"

#include <cuda_runtime.h>
#include <cfloat>
#include <cstddef>
#include <cmath>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value += __shfl_xor_sync(0xffffffffu, value, mask);
    return value;
}

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
    for (int mask = 16; mask; mask >>= 1) value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, mask));
    return value;
}

__global__ void project(const float * __restrict__ rows, const float * __restrict__ x,
                        float * __restrict__ logits, int width, int outputs) {
    x += size_t(blockIdx.y) * width; logits += size_t(blockIdx.y) * outputs;
    const int row = blockIdx.x;
    const int lane = threadIdx.x;
    float sum = 0.0f;
    for (int i = lane; i < width; i += 128)
        sum = fmaf(rows[size_t(row) * width + i], x[i], sum);
    for (int mask = 16; mask; mask >>= 1)
        sum += __shfl_xor_sync(0xffffffffu, sum, mask);
    __shared__ float partial[4];
    if ((lane & 31) == 0) partial[lane / 32] = sum;
    __syncthreads();
    if (lane < 32) {
        sum = lane < 4 ? partial[lane] : 0.0f;
        for (int mask = 16; mask; mask >>= 1)
            sum += __shfl_xor_sync(0xffffffffu, sum, mask);
        if (lane == 0) logits[row] = sum;
    }
}

template<bool Continue, bool Pair = false, bool Batch = false>
__global__ void route(const float * __restrict__ logits, int32_t * __restrict__ ids,
                      float * __restrict__ weights, const core::NativeMissArgs* continuation) {
    int32_t* all_ids = ids;
    const int token = Pair ? int(threadIdx.x) / 32 : (Batch ? int(blockIdx.x) : 0);
    const int lane = threadIdx.x & 31;
    logits += token * 257; ids += token * 8; weights += token * 8;
    float values[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) values[i] = logits[lane + 32 * i];
    float maximum = -FLT_MAX;
#pragma unroll
    for (int i = 0; i < 8; ++i) maximum = fmaxf(maximum, values[i]);
    maximum = warp_max(maximum);
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values[i] = expf(values[i] - maximum);
        sum += values[i];
    }
    const float inverse_sum = 1.0f / warp_sum(sum);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        values[i] *= inverse_sum;
        if (__isnanf(values[i])) values[i] = -FLT_MAX;
    }
    float selected = 0.0f, selected_sum = 0.0f;
    for (int rank = 0; rank < 8; ++rank) {
        float best = values[0];
        int expert = lane;
#pragma unroll
        for (int i = 1; i < 8; ++i) {
            if (values[i] > best) { best = values[i]; expert = lane + 32 * i; }
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) {
            const float other = __shfl_xor_sync(0xffffffffu, best, mask);
            const int other_id = __shfl_xor_sync(0xffffffffu, expert, mask);
            if (other > best || (other == best && other_id < expert)) {
                best = other;
                expert = other_id;
            }
        }
        if ((expert & 31) == lane) {
            values[expert / 32] = -FLT_MAX;
            ids[rank] = expert;
            selected_sum += best;
        }
        if (rank == lane) selected = best;
    }
    const float inverse_selected_sum = 1.0f / fmaxf(warp_sum(selected_sum), 6.103515625e-5f);
    if (lane < 8) weights[lane] = selected * inverse_selected_sum;
    if (Continue) {
        if (Pair) __syncthreads(); else __syncwarp();
        if (threadIdx.x < 32) core::publish_native_miss(all_ids, *continuation);
    }
}

void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace

void route_top8(const float * logits, int32_t * ids, float * weights, void * stream) {
    if (!logits || !ids || !weights || !stream)
        throw std::invalid_argument("qwen35moe route requires nonnull device spans and stream");
    route<false><<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, nullptr);
    check(cudaGetLastError());
}

void route_top8_batch(const float* logits, int32_t* ids, float* weights, int tokens, void* stream) {
    if (!logits || !ids || !weights || !stream || tokens < 1 || tokens > 8192)
        throw std::invalid_argument("invalid batched Qwen35 route");
    route<false, false, true><<<tokens, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, ids, weights, nullptr);
    check(cudaGetLastError());
}

void route_top8_continue(const float* logits, int32_t* ids, float* weights,
                          const core::NativeMissArgs* args, void* stream, int tokens) {
    if (!logits || !ids || !weights || !args || !stream || tokens < 1 || tokens > 2)
        throw std::invalid_argument("invalid Qwen35 continuation route");
    if (tokens == 2)
        route<true, true><<<1, 64, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, args);
    else route<true><<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(logits, ids, weights, args);
    check(cudaGetLastError());
}

void route_project(const float * rows, const float * x, float * logits,
                   int width, int outputs, void * stream, int tokens) {
    if (!rows || !x || !logits || !stream || width <= 0 || width % 128 ||
        outputs <= 0 || outputs > 65535 || tokens < 1 || tokens > 8192)
        throw std::invalid_argument("invalid Qwen3.5 F32 router projection");
    project<<<dim3(outputs, tokens), 128, 0, static_cast<cudaStream_t>(stream)>>>(rows, x, logits, width, outputs);
    check(cudaGetLastError());
}

} // namespace strata::qwen35moe
