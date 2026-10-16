// Arithmetic adapted from gated_delta_net.cu at pinned llama.cpp
// 3cf03257f219afbe7334045ff7c6a06ac68c627d. Only state addressing differs:
// each warp owns one Strata (head,column) and retains its four rows in registers.
// Compile with --use_fast_math to match the pinned CUDA implementation.
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
#include "strata/kernels/native_gdn.hpp"
#include <cuda_runtime.h>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        value += __shfl_xor_sync(0xffffffff, value, offset, 32);
    return value;
}

#include "native_gdn_step.cuh"

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}

static void launch_gdn(bool scan, int tokens, int input_stride, float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, void* stream, float* prefix = nullptr, int columns = 1) {
    if ((columns != 1 && columns != 2) || (columns != 1 && (!scan || prefix)) || tokens < 1 || tokens > 8192 || input_stride < (2 * shape.h_k + shape.h_v) * S ||
        !stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 ||
        shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = (size_t(tokens - 1) * input_stride + size_t(S) * shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(tokens) * S * shape.h_v * sizeof(float);
    const size_t v_bytes = (size_t(tokens - 1) * input_stride + size_t(S) * shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(tokens) * shape.h_v * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, v_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i) {
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    }
    if (prefix) {
        const size_t checkpoint_bytes = size_t(tokens - 1) * state_bytes;
        if (!scan || tokens < 2 || !valid_span(prefix, checkpoint_bytes) ||
            overlap(prefix, checkpoint_bytes, state, state_bytes) ||
            overlap(prefix, checkpoint_bytes, output, output_bytes))
            throw std::invalid_argument("GDN checkpoint requires disjoint multi-token scan");
        for (int i = 0; i < 5; ++i)
            if (overlap(prefix, checkpoint_bytes, inputs[i], bytes[i]))
                throw std::invalid_argument("GDN checkpoint aliases input");
    }
    const float scale = 1.0f / sqrtf(float(S));
    const dim3 grid(unsigned(shape.h_v), 1, S / (4 * columns)), threads(32, 4);
    if (prefix)
        step<true, true><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, prefix);
    else if (scan && columns == 2)
        step<true, false, 2><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else if (scan)
        step<true><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, tokens, input_stride, nullptr);
    else
        step<false><<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
            state, q, k, v, gate, beta, output, int(shape.h_k), int(shape.h_v), scale, 1, input_stride, nullptr);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
void native_gdn_scan(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shape, int tokens, int input_stride, void* stream, float* prefix, int columns) {
    launch_gdn(true, tokens, input_stride, state, q, k, v, gate, beta, output, shape, stream, prefix, columns);
}
} // namespace strata::kernels
