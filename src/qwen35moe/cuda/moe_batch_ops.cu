#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/core/device_miss.cuh"

#include <cuda_runtime.h>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
__global__ void gate_shared_kernel(float* __restrict__ shared,
                                   const float* __restrict__ logits, int count) {
    const int i = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    if (i >= count * 2048) return;
    const int token = i / 2048;
    const float gate = logits[token * 257 + 256];
    shared[i] *= 1.0f / (1.0f + expf(-gate));
}

template<bool Continue>
__global__ void combine_kernel(float* __restrict__ parts,
                               const float* __restrict__ weights,
                               const float* __restrict__ shared,
                               const float* __restrict__ residual,
                               float* __restrict__ output, int count, const core::NativeMissArgs* continuation) {
    if (Continue) {
        __shared__ int ready;
        if (threadIdx.x == 0) ready = core::await_native_miss(*continuation);
        __syncthreads();
        if (!ready) return;
        const auto mask = continuation->frame->mask;
        if (mask) {
            for (int rank = 0; rank < continuation->ranks; ++rank) {
                if (!(mask & (1u << rank))) continue;
                for (int col = threadIdx.x; col < continuation->n_embd; col += blockDim.x) {
                    const size_t offset = size_t(rank) * continuation->n_embd + col;
                    parts[offset] = continuation->result_host_alias[offset];
                }
            }
            __syncthreads();
        }
    }
    const int first = int(blockIdx.x) * blockDim.x + int(threadIdx.x);
    const int end = Continue ? count * 2048 : first + 1;
    const int step = Continue ? int(blockDim.x) : 1;
    for (int i = first; i < end && i < count * 2048; i += step) {
        const int token = i / 2048, col = i % 2048;
        const float* expert = parts + size_t(token) * 8 * 2048;
        const float* weight = weights + token * 8;
        float sum = expert[col] * weight[0];
        for (int rank = 1; rank < 8; ++rank)
            sum += expert[size_t(rank) * 2048 + col] * weight[rank];
        sum += shared[i];
        output[i] = sum + residual[i];
    }
}
}

void gate_shared_batch(float* shared, const float* logits, int count, void* stream) {
    if (!shared || !logits || !stream || count < 1 || count > 8192)
        throw std::invalid_argument("invalid Qwen35 batched shared gate");
    gate_shared_kernel<<<(count * 2048 + 255) / 256, 256, 0,
                         static_cast<cudaStream_t>(stream)>>>(shared, logits, count);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

void moe_combine_batch(const float* parts, const float* weights, const float* shared,
                       const float* residual, float* output, int count, void* stream) {
    if (!parts || !weights || !shared || !residual || !output || !stream ||
        count < 1 || count > 8192)
        throw std::invalid_argument("invalid Qwen35 batched expert combine");
    combine_kernel<false><<<(count * 2048 + 255) / 256, 256, 0,
                     static_cast<cudaStream_t>(stream)>>>(const_cast<float*>(parts), weights, shared,
                                                          residual, output, count, nullptr);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void moe_combine_continue(float* parts, const float* weights, const float* shared,
                           const float* residual, float* output,
                           const core::NativeMissArgs* args, void* stream, int tokens) {
    if (!parts || !weights || !shared || !residual || !output || !args || !stream || tokens < 1 || tokens > 2)
        throw std::invalid_argument("invalid Qwen35 continuation combine");
    combine_kernel<true><<<1, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parts, weights, shared, residual, output, tokens, args);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::qwen35moe
