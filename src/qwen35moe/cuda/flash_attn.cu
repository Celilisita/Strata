// Qwen3.5 16-head short decode attention, specialized from llama.cpp's
// F16/F16 FlashAttention arithmetic for the TITAN V Volta path.
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/qwen35moe/attention_short_prefill.hpp"
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::qwen35moe {
namespace {
template<int Width> __device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
    for (int offset = Width / 2; offset; offset >>= 1)
        x += __shfl_xor_sync(0xffffffffu, x, offset, Width);
    return x;
}
__device__ __forceinline__ float warp_max(float x) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffffu, x, offset, 32));
    return x;
}

// D=256,ncols=1,F16/F16; 128 threads, nthreads_KQ=nthreads_V=8,
// four values (float2) per load, four V columns per iteration. Padded length256
// gives ntiles_KV=ceil(256/D)=1, so the pinned launcher selects grid.y=1.
template<bool Batch>
__launch_bounds__(128, 1)
__global__ void attend(const float* __restrict__ q, const half* __restrict__ k,
                       const half* __restrict__ v, const int32_t* __restrict__ step,
                       int max_context, int padded_length, int head_group, float scale,
                       float* __restrict__ out, int32_t* __restrict__ status,
                       const half* __restrict__ mask, int first) {
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int head = blockIdx.x, kv = head / head_group;
    const int row = Batch ? int(blockIdx.y) : 0;
    q += size_t(row) * 4096;
    out += size_t(row) * 4096;
    const int width = Batch ? first + row + 1 : step[kernels::kStepWidth];
    const int nkv = Batch ? width : step[kernels::kStepNKv];
    const bool valid = width >= 1 && width <= max_context && nkv == width &&
                       (Batch || (step[kernels::kStepPos] == width - 1 &&
                                  step[kernels::kStepNBid] == width / 4));
    if (head == 0 && row == 0 && tid == 0)
        *status = valid ? kernels::kNativeFlashAttnSuccess : kernels::kNativeFlashAttnUnsupportedStep;
    if (!valid) {
        out[head * 256 + tid] = __int_as_float(0x7fc00000);
        out[head * 256 + tid + 128] = __int_as_float(0x7fc00000);
        return;
    }
    float2 qreg[16];
    float2 vkq[16] = {};
    __shared__ float tile[4 * 4 * 256];
    __shared__ float max_shared[32], sum_shared[32];
    float maximum = -FLT_MAX / 2.0f, sum = 0.0f;
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 32) {
        const int i = i0 + (lane % 8) * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int d = 2 * (i + j);
            float qx = q[head * 256 + d] * scale;
            float qy = q[head * 256 + d + 1] * scale;
            qx = __half2float(__float2half_rn(qx));
            qy = __half2float(__float2half_rn(qy));
            qreg[i0 / 8 + j] = make_float2(qx, qy);
        }
    }
    for (int base = 0; base < padded_length; base += 128) {
        float score = 0.0f, next_max = maximum;
#pragma unroll
        for (int row = 0; row < 8; ++row) {
            const int cell = base + warp * 32 + (lane & ~7) + row;
            float dot = 0.0f;
#pragma unroll
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * (i + j);
                    const float a = cell < width ? __half2float(k[(cell * 2 + kv) * 256 + d]) : 0.0f;
                    const float b = cell < width ? __half2float(k[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                    const half ax = __hmul(__float2half_rn(a), __float2half_rn(qreg[i0 / 8 + j].x));
                    const half by = __hmul(__float2half_rn(b), __float2half_rn(qreg[i0 / 8 + j].y));
                    dot += __half2float(ax) + __half2float(by);
                }
            }
            dot = warp_sum<8>(dot);
            dot += cell < width ? (mask ? __half2float(mask[cell]) : 0.0f) : __int_as_float(0xff800000u);
            next_max = fmaxf(next_max, dot + (3.0f * 0.6931f));
            if (lane % 8 == row) score = dot;
        }
#pragma unroll
        for (int offset = 8; offset < 32; offset <<= 1)
            next_max = fmaxf(next_max, __shfl_xor_sync(0xffffffffu, next_max, offset, 32));
        const float rescale = expf(maximum - next_max);
        maximum = next_max;
        score = expf(score - maximum);
        sum = sum * rescale + score;
        // Volta's tile path stores softmax weights and V accumulators in F16,
        // while its normalization denominator stays F32.
        tile[tid] = __half2float(__float2half_rn(score));
        __syncwarp();
#pragma unroll
        for (int k0 = 0; k0 < 32; k0 += 4) {
            const int local = warp * 32 + k0 + lane / 8, cell = base + local;
            const float weight = tile[local];
#pragma unroll
            for (int i0 = 0; i0 < 128; i0 += 32) {
                const int i = i0 + (lane % 8) * 4;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int d = 2 * (i + j);
                    const float a = cell < width ? __half2float(v[(cell * 2 + kv) * 256 + d]) : 0.0f;
                    const float b = cell < width ? __half2float(v[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                    const half weight_h = __float2half_rn(weight);
                    const half scale_h = __float2half_rn(rescale);
                    const half old_x = k0 == 0 ? __hmul(__float2half_rn(vkq[i0 / 8 + j].x), scale_h)
                                                 : __float2half_rn(vkq[i0 / 8 + j].x);
                    const half old_y = k0 == 0 ? __hmul(__float2half_rn(vkq[i0 / 8 + j].y), scale_h)
                                                 : __float2half_rn(vkq[i0 / 8 + j].y);
                    vkq[i0 / 8 + j].x = __half2float(__hadd(old_x,
                        __hmul(__float2half_rn(a), weight_h)));
                    vkq[i0 / 8 + j].y = __half2float(__hadd(old_y,
                        __hmul(__float2half_rn(b), weight_h)));
                }
            }
        }
    }
    if (warp == 0) { max_shared[lane] = -FLT_MAX / 2.0f; sum_shared[lane] = 0.0f; }
    __syncthreads();
    if (lane == 0) max_shared[warp] = maximum;
    __syncthreads();
    const float global_max = warp_max(max_shared[lane]);
    const float rescale = expf(maximum - global_max);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const half scale_h = __float2half_rn(rescale);
        vkq[i].x = __half2float(__hmul(__float2half_rn(vkq[i].x), scale_h));
        vkq[i].y = __half2float(__hmul(__float2half_rn(vkq[i].y), scale_h));
    }
#pragma unroll
    for (int i0 = 0; i0 < 128; i0 += 32) {
        const int start = warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            tile[start + 2 * j] = vkq[i0 / 8 + j].x;
            tile[start + 2 * j + 1] = vkq[i0 / 8 + j].y;
        }
    }
    sum *= rescale;
    sum = warp_sum<32>(sum);
    if (lane == 0) sum_shared[warp] = sum;
    __syncthreads();
    sum = warp_sum<32>(sum_shared[lane]);
#pragma unroll
    for (int i0 = 0; i0 < 256; i0 += 128) {
        float result = 0.0f;
#pragma unroll
        for (int w = 0; w < 4; ++w) {
#pragma unroll
            for (int group = 0; group < 4; ++group)
                result += tile[w * 4 * 256 + group * 256 + i0 + tid];
        }
        out[head * 256 + i0 + tid] = result / sum;
    }
}
struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}
} // namespace

void flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v,
                           const int32_t* step, int64_t capacity, int max_context, int width,
                           float* output, int32_t* status, const uint16_t* mask,
                           void* stream) {
    if (!stream || capacity < max_context ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 ||
        max_context < 1 || max_context > 1024 || width < 1 || width > max_context)
        throw std::invalid_argument("Qwen3.5 FlashAttention requires bounded capacity and width");
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 16 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2},
                          {step, kernels::kStepCount * 4, 4}, {output, 16 * 256 * 4, 4},
                          {status, 4, 4}, {mask, size_t(width) * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    attend<false><<<16, dim3(32, 4), 0, static_cast<cudaStream_t>(stream)>>>(q,
        reinterpret_cast<const half*>(k), reinterpret_cast<const half*>(v), step,
        max_context, (std::max)(256, ((width + 127) / 128) * 128), 8, 0.0625f,
        output, status, reinterpret_cast<const half*>(mask), 0);
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("Qwen3.5 FlashAttention launch: ") + cudaGetErrorString(result));
}
// Schedule independent short causal rows together. The device formula above
// is shared with decode; only row addressing and step construction differ.
void attention_short_prefill(const float* q, const uint16_t* k, const uint16_t* v,
                             int first, int count, int64_t capacity, float* output,
                             int32_t* status, void* stream) {
    if (!stream || first < 0 || first >= 256 || count < 1 || count > 256 - first ||
        capacity < first + count || uint64_t(capacity) > SIZE_MAX / 1024)
        throw std::invalid_argument("short attention prefill requires causal positions below256");
    const size_t kv_bytes = size_t(capacity) * 1024;
    const Span spans[] = {{q, size_t(count) * 4096 * 4, 4}, {k, kv_bytes, 2},
        {v, kv_bytes, 2}, {output, size_t(count) * 4096 * 4, 4}, {status, 4, 4}};
    validate_spans(spans, 5);
    attend<true><<<dim3(16, unsigned(count)), dim3(32, 4), 0,
        static_cast<cudaStream_t>(stream)>>>(q, reinterpret_cast<const half*>(k),
        reinterpret_cast<const half*>(v), nullptr, 256, 256, 8, 0.0625f,
        output, status, nullptr, first);
    const auto result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string("short attention prefill launch: ") + cudaGetErrorString(result));
}
} // namespace strata::qwen35moe
