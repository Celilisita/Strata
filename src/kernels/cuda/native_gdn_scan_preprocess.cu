// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh,unary.cu,unary.cuh,ssm-conv.cu,scale.cu}.
// Compile with --use_fast_math, as the pinned backend does.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {
constexpr int S = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
    return value;
}
__device__ __forceinline__ float norm_sum(float value, float* sums) {
    const int lane = threadIdx.x % 32;
    value = warp_sum(value);
    if (lane == 0) sums[threadIdx.x / 32] = value;
    __syncthreads();
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(value);
}
__device__ __forceinline__ float sigmoid(float value) { return 1.0f / (1.0f + expf(-value)); }

template<bool Scan, bool Prefix = false>
__global__ void conv_silu(float* __restrict__ history, const float* __restrict__ input,
                           const float* __restrict__ weights, float* __restrict__ raw_output,
                           float* __restrict__ silu_output, int channels, int tokens, float* __restrict__ prefix) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], 0.0f};
    const int count = Scan ? tokens : 1;
    for (int t = 0; t < count; ++t) {
        values[3] = input[size_t(t) * channels + c];
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
        sum = __fadd_rn(sum, 0.0f);
        if (!Scan) raw_output[c] = sum;
        silu_output[size_t(t) * channels + c] = sum / (1.0f + expf(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) values[tap] = values[tap + 1];
        if (Prefix && t + 1 < count) {
#pragma unroll
            for (int tap = 0; tap < 3; ++tap)
                prefix[size_t(t) * channels * 3 + c * 3 + tap] = values[tap];
        }
    }
#pragma unroll
    for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap];
}

template<bool Scan>
__global__ void gate_softplus(const float* __restrict__ alpha, const float* __restrict__ dt,
                               const float* __restrict__ ssm_a, float* __restrict__ gate, int count, int heads) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int h = Scan ? i % heads : i;
    const float value = __fadd_rn(alpha[i], dt[h]);
    const float softplus = value > 20.0f ? value : logf(1.0f + expf(value));
    gate[i] = softplus * ssm_a[h];
}

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 1048576)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,1048576]");
}
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}

void native_gdn_conv_scan(float* history, const float* input, const float* weights,
                           float* output, int channels, int tokens, void* stream, float* prefix) {
    count_and_stream(channels, stream);
    if (tokens < 1 || tokens > 8192) throw std::invalid_argument("invalid convolution token count");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {output, size_t(tokens) * bytes}};
    const Span inputs[] = {{input, size_t(tokens) * bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    disjoint(writable[0], writable[1]);
    for (auto dst : writable) for (auto src : inputs) disjoint(dst, src);
    if (prefix) {
        if (tokens < 2) throw std::invalid_argument("history checkpoint requires multiple tokens");
        const Span checkpoint{prefix, size_t(tokens - 1) * 3 * bytes}; valid(checkpoint);
        for (auto span : writable) disjoint(checkpoint, span);
        for (auto span : inputs) disjoint(checkpoint, span);
        conv_silu<true, true><<<unsigned((channels + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
            history, input, weights, nullptr, output, channels, tokens, prefix);
    } else conv_silu<true><<<unsigned((channels + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        history, input, weights, nullptr, output, channels, tokens, nullptr);
    check_launch();
}
void native_gdn_gate_scan(const float* alpha, const float* dt, const float* ssm_a,
                           float* gate, int heads, int tokens, void* stream) {
    count_and_stream(heads, stream);
    if (tokens < 1 || tokens > 8192 || int64_t(heads) * tokens > 1048576)
        throw std::invalid_argument("invalid GDN gate token count");
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span writable{gate, size_t(tokens) * bytes};
    valid(writable);
    for (auto src : {Span{alpha, size_t(tokens) * bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(src); disjoint(writable, src);
    }
    gate_softplus<true><<<unsigned((heads * tokens + 255) / 256), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        alpha, dt, ssm_a, gate, heads * tokens, heads);
    check_launch();
}
} // namespace strata::kernels
