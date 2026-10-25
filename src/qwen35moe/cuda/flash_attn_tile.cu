// Volta F16 tile arithmetic specialized from ggml's MIT-licensed fattn-tile.
// Copyright (c) 2023-2026 The ggml authors. See flash_attn.cu for the license.
#include "strata/qwen35moe/attention_tile.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "tg_attention_join_config.cuh"
#include "attention_pairheads4.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
constexpr int kHeads = 16, kGroup = 8, kDim = 256, kRows = 64;
constexpr int kPartial = kDim + 2;

__device__ float warp_maximum(float v) {
    for (int d = 16; d; d >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, d));
    return v;
}
__device__ float warp_total(float v) {
    for (int d = 16; d; d >>= 1) v += __shfl_xor_sync(0xffffffffu, v, d);
    return v;
}

// A block shares each K/V tile across eight Q heads. Each split owns exactly
// 64 cells, so its half accumulators do not grow with context length. The join
// rescales and reduces split outputs in F32, as the canonical tile launcher does.
template<bool Batch, int Queries = 1, bool Padded = false, bool Packed = false, bool FullShared = false>
__launch_bounds__(256, Packed ? 4 : 2)
__global__ void tile(const float* q, const half2* k, const half2* v,
                     const int32_t* step, int capacity, int width, int splits,
                     const half* mask, float* partial, int32_t* status, int first_position, int query_count) {
    const int lane = threadIdx.x, warp = threadIdx.y, tid = warp * 32 + lane;
    const int kv = blockIdx.x, head = kv * kGroup + warp, split = blockIdx.y;
    bool valid;
    int widths[Queries];
    int active = Queries;
    if constexpr (Batch) {
        const int query = blockIdx.z * Queries;
        if constexpr (Queries > 1) active = min(Queries, query_count - query);
        width = first_position + query + active;
        q += size_t(query) * kHeads * (Packed ? 2 * kDim : kDim);
        partial += size_t(query) * kHeads * splits * kPartial;
#pragma unroll
        for (int g = 0; g < Queries; ++g) widths[g] = first_position + query + g + 1;
        valid = active > 0 && width > 0 && width <= capacity;
    } else {
        widths[0] = width;
        valid = step[kernels::kStepWidth] == width && width <= capacity &&
            step[kernels::kStepNKv] == width && step[kernels::kStepPos] == width - 1 &&
            step[kernels::kStepNBid] == width / 4;
    }
    if (kv == 0 && split == 0 && tid == 0)
        *status = valid ? kernels::kNativeFlashAttnSuccess : kernels::kNativeFlashAttnUnsupportedStep;
    if (!valid) return;
    constexpr int cache_stride = Padded ? 33 : 36;
    __shared__ half2 qt[Queries * kGroup * 128];
    __shared__ half2 cache[kRows * cache_stride];
    __shared__ half probability[Queries * kGroup * kRows];
    for (int i = tid; i < Queries * kGroup * 128; i += 256) {
        const int g = i / (kGroup * 128);
        const int d = i % 128, h = kv * kGroup + (i / 128) % kGroup;
        if constexpr (Packed) {
            // Copy F16 bits directly: avoid materializing temporary half2 objects.
            reinterpret_cast<uint32_t*>(qt)[i] = g < active ? reinterpret_cast<const uint32_t*>(q)[
                (size_t(g) * kHeads + h) * (2 * kDim) + d] : 0u;
        } else {
            const half2 value = g < active ? __floats2half2_rn(q[size_t(g) * kHeads * kDim + h * kDim + 2 * d] * 0.0625f,
                q[size_t(g) * kHeads * kDim + h * kDim + 2 * d + 1] * 0.0625f) : __float2half2_rn(0);
            qt[i] = __hmul2(value, __float2half2_rn(0.25f));
        }
    }
    __syncthreads();
    const int base = split * kRows;
    float score[Queries][2] = {};
    // The K dimension is accumulated in the same pair order as fattn-tile;
    // keep half products and float pair sums rather than changing to F32 GEMM.
    for (int chunk = 0; chunk < 128; chunk += 32) {
        for (int i = tid; i < kRows * 32; i += 256) {
            const int row = i / 32, d = i % 32;
            cache[row * cache_stride + d] = base + row < width ?
                k[(base + row) * 256 + kv * 128 + chunk + d] : __float2half2_rn(0);
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < Queries; ++g) {
#pragma unroll
            for (int row = 0; row < 2; ++row) {
#pragma unroll
                for (int d = 0; d < 32; ++d) {
                    const float2 p = __half22float2(__hmul2(cache[(lane + row * 32) * cache_stride + d],
                        qt[(g * kGroup + warp) * 128 + chunk + d]));
                    score[g][row] += p.x + p.y;
                }
            }
        }
        __syncthreads();
    }
    float maximum[Queries], denominator[Queries] = {};
#pragma unroll
    for (int g = 0; g < Queries; ++g) {
        maximum[g] = -FLT_MAX / 2;
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            const int cell = base + lane + row * 32;
            score[g][row] = cell < widths[g] ? score[g][row] * 4.0f +
                (mask ? __half2float(mask[cell]) : 0.0f) : __int_as_float(0xff800000u);
            maximum[g] = fmaxf(maximum[g], score[g][row] + 3.0f * 0.6931f);
        }
        maximum[g] = warp_maximum(maximum[g]);
#pragma unroll
        for (int row = 0; row < 2; ++row) {
            const float weight = expf(score[g][row] - maximum[g]);
            denominator[g] += weight;
            probability[(g * kGroup + warp) * kRows + lane + row * 32] = __float2half_rn(weight);
        }
        denominator[g] = warp_total(denominator[g]);
    }
    half2 accumulator[Queries][4] = {};
    __syncthreads();
    // Reuse the K staging memory for 16 V cells at a time.
    for (int row0 = 0; row0 < kRows; row0 += 16) {
        for (int i = tid; i < 16 * 128; i += 256) {
            const int cell = base + row0 + i / 128;
            cache[i] = cell < width ? v[cell * 256 + kv * 128 + i % 128] : __float2half2_rn(0);
        }
        __syncthreads();
#pragma unroll
        for (int g = 0; g < Queries; ++g) {
#pragma unroll
            for (int row = 0; row < 16; ++row) {
                const half2 weight = __half2half2(probability[(g * kGroup + warp) * kRows + row0 + row]);
#pragma unroll
                for (int d = 0; d < 4; ++d) {
                    const int pair = lane * 2 + (d & 1) + (d / 2) * 64;
                    half2 value = cache[row * 128 + pair];
                    if constexpr (Queries > 1) {
                        if (base + row0 + row >= widths[g]) value = __float2half2_rn(0);
                    }
                    accumulator[g][d] = __hadd2(accumulator[g][d], __hmul2(value, weight));
                }
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int g = 0; g < Queries; ++g) {
        if (g >= active) continue;
        float* out = partial + (size_t(g) * kHeads * splits + head * splits + split) * kPartial;
#pragma unroll
        for (int d = 0; d < 4; ++d) {
            const int pair = lane * 2 + (d & 1) + (d / 2) * 64;
            const float2 value = __half22float2(accumulator[g][d]);
            out[2 * pair] = value.x; out[2 * pair + 1] = value.y;
        }
        if (lane == 0) { out[kDim] = maximum[g]; out[kDim + 1] = denominator[g]; }
    }
}

template<bool Batch, bool Shared = false>
__global__ void join(const float* partial, int splits, const int32_t* status, float* out) {
    const int head = blockIdx.x, d = threadIdx.x;
    if constexpr (Batch) {
        partial += size_t(blockIdx.y) * kHeads * splits * kPartial;
        out += size_t(blockIdx.y) * kHeads * kDim;
    }
    if (*status != kernels::kNativeFlashAttnSuccess) {
        out[head * kDim + d] = __int_as_float(0x7fc00000); return;
    }
    if constexpr (Shared) {
        __shared__ float common[2];
        __shared__ float scales[(kAttentionMaxContext + kRows - 1) / kRows];
        if (d == 0) {
            float maximum = -FLT_MAX / 2;
            for (int s = 0; s < splits; ++s)
                maximum = fmaxf(maximum, partial[(head * splits + s) * kPartial + kDim]);
            common[0] = maximum;
        }
        __syncthreads();
        for (int s = d; s < splits; s += kDim)
            scales[s] = expf(partial[(head * splits + s) * kPartial + kDim] - common[0]);
        __syncthreads();
        if (d == 0) {
            float denominator = 0;
            for (int s = 0; s < splits; ++s)
                denominator += partial[(head * splits + s) * kPartial + kDim + 1] * scales[s];
            common[1] = denominator;
        }
        __syncthreads();
        float numerator = 0;
        for (int s = 0; s < splits; ++s)
            numerator += partial[(head * splits + s) * kPartial + d] * scales[s];
        out[head * kDim + d] = numerator / common[1];
    } else {
        float maximum = -FLT_MAX / 2;
        for (int s = 0; s < splits; ++s)
            maximum = fmaxf(maximum, partial[(head * splits + s) * kPartial + kDim]);
        float numerator = 0, denominator = 0;
        for (int s = 0; s < splits; ++s) {
            const float* p = partial + (head * splits + s) * kPartial;
            const float scale = expf(p[kDim] - maximum);
            numerator += p[d] * scale; denominator += p[kDim + 1] * scale;
        }
        out[head * kDim + d] = numerator / denominator;
    }
}
}

size_t flash_attn_tile_scratch_bytes(int max_context) {
    if (max_context < 1 || max_context > kAttentionMaxContext)
        throw std::invalid_argument("invalid Qwen35 tile context");
    return size_t(kHeads) * ((max_context + kRows - 1) / kRows) * kPartial * sizeof(float);
}

bool flash_attn_tile_step_provider(const float* q, const uint16_t* k, const uint16_t* v,
                          const int32_t* step, int capacity, int width, float* output,
                          int32_t* status, const uint16_t* mask, float* scratch,
                          size_t scratch_bytes, void* stream, bool request_pairheads4) {
    if (!q || !k || !v || !step || !output || !status || !scratch || !stream ||
        width < 1 || width > capacity || capacity > kAttentionMaxContext ||
        scratch_bytes < flash_attn_tile_scratch_bytes(width))
        throw std::invalid_argument("invalid Qwen35 tile attention buffers");
    const bool shared_join = tg_attention_shared_join_enabled() && width >= 1024;
    const int splits = (width + kRows - 1) / kRows;
    const bool selected = request_pairheads4 && try_launch_attention_pairheads4(
        q, k, v, step, capacity, width, splits, mask, scratch, status, stream);
    if (!selected) {
    tile<false><<<dim3(2, splits), dim3(32, 8), 0, static_cast<cudaStream_t>(stream)>>>(
        q, reinterpret_cast<const half2*>(k), reinterpret_cast<const half2*>(v),
        step, capacity, width, splits, reinterpret_cast<const half*>(mask), scratch, status, -1, 1);
    }
    auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    if (shared_join)
        join<false, true><<<kHeads, kDim, 0, static_cast<cudaStream_t>(stream)>>>(scratch, splits, status, output);
    else
        join<false><<<kHeads, kDim, 0, static_cast<cudaStream_t>(stream)>>>(scratch, splits, status, output);
    error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return selected;
}
void flash_attn_tile_step(const float* q, const uint16_t* k, const uint16_t* v,
    const int32_t* step, int capacity, int width, float* output, int32_t* status,
    const uint16_t* mask, float* scratch, size_t scratch_bytes, void* stream) {
    (void)flash_attn_tile_step_provider(q, k, v, step, capacity, width, output,
        status, mask, scratch, scratch_bytes, stream, false);
}
AttentionPairHeads4Resources flash_attn_tile_native_resources() {
    // Probe-only observation; it does not impose new production eligibility.
    static thread_local const AttentionPairHeads4Resources cached=[] {
        AttentionPairHeads4Resources r{};r.threads=256;
        auto check=[](cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));};
        check(cudaGetDevice(&r.device));cudaDeviceProp p{};check(cudaGetDeviceProperties(&p,r.device));
        r.major=p.major;r.minor=p.minor;r.multiprocessors=p.multiProcessorCount;r.shared_limit=p.sharedMemPerBlock;
        cudaFuncAttributes a{};check(cudaFuncGetAttributes(&a,tile<false>));
        r.binary_version=a.binaryVersion;r.registers=a.numRegs;r.max_threads=a.maxThreadsPerBlock;
        r.static_shared=a.sharedSizeBytes;r.local_bytes=a.localSizeBytes;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&r.active_blocks,tile<false>,256,0));
        return r;
    }();
    return cached;
}
namespace {
__global__ void positions(int32_t* out, int rows, int heads, int first) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) out[row] = first + row / heads;
}
}
void attention_prefill_positions(int32_t* out, int count, int heads, int first, void* stream) {
    if (!out || !stream || count < 1 || count > 8192 || (heads != 2 && heads != 16) || first < 0)
        throw std::invalid_argument("invalid attention position workspace");
    const int rows = count * heads;
    positions<<<(rows + 255) / 256, 256, 0, static_cast<cudaStream_t>(stream)>>>(out, rows, heads, first);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
#include "attention_shared_config.cuh"
#include "attention_query_pack.cuh"

void flash_attn_tile_prefill(const float* q, const uint16_t* k, const uint16_t* v,
    int first, int count, int capacity, float* out, int32_t* status,
    float* scratch, size_t scratch_bytes, void* stream, int query_batch, int query_group, bool shared_join, bool padded_cache, const float* packed_q, bool full_shared) {
    if ((full_shared && (packed_q || padded_cache || shared_join || query_group != 2)) ||
        !q || !k || !v || !out || !status || !scratch || !stream || first < 256 ||
        count < 1 || count > 8192 || first > capacity || count > capacity - first ||
        (packed_q && (query_group != 2 || padded_cache || shared_join)) ||
        (padded_cache && query_group != 2) || (query_group != 1 && query_group != 2 && query_group != 4 && !(packed_q && query_group == 2)) || query_batch < 2 || query_batch > 64 || scratch_bytes < 2 * flash_attn_tile_scratch_bytes(first + count))
        throw std::invalid_argument("invalid bounded attention prefill workspace");
    // Bound query concurrency by the caller's existing scratch. Larger batches
    // retain each query's tile arithmetic and ordered split join exactly.
    if (full_shared) configure_attention_shared<false>();
    const size_t one_query = flash_attn_tile_scratch_bytes(first + count);
    const int batch = int((std::min)(size_t(query_batch), scratch_bytes / one_query));
    for (int offset = 0; offset < count; offset += batch) {
        const int queries = (std::min)(count - offset, batch);
        const int width = first + offset + queries;
        const int splits = (width + kRows - 1) / kRows;
        const auto st = static_cast<cudaStream_t>(stream);
        const auto* query = packed_q ? packed_q + size_t(offset) * kHeads * 2 * kDim
                                     : q + size_t(offset) * kHeads * kDim;
        const auto* keys = reinterpret_cast<const half2*>(k);
        const auto* values = reinterpret_cast<const half2*>(v);
        if (full_shared)
            tile<true, 2, false, false, true><<<dim3(2, splits, (queries + 1) / 2), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        else if (packed_q)
            tile<true, 2, false, true><<<dim3(2, splits, (queries + 1) / 2), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        else if (padded_cache)
            tile<true, 2, true><<<dim3(2, splits, (queries + 1) / 2), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        else if (query_group == 4)
            tile<true, 4><<<dim3(2, splits, (queries + 3) / 4), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        else if (query_group == 2)
            tile<true, 2><<<dim3(2, splits, (queries + 1) / 2), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        else
            tile<true><<<dim3(2, splits, queries), dim3(32, 8), 0, st>>>(
                query, keys, values, nullptr, capacity, width, splits, nullptr, scratch, status, first + offset, queries);
        auto error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
        if (shared_join)
            join<true, true><<<dim3(kHeads, queries), kDim, 0, st>>>(
                scratch, splits, status, out + size_t(offset) * kHeads * kDim);
        else
            join<true><<<dim3(kHeads, queries), kDim, 0, st>>>(
                scratch, splits, status, out + size_t(offset) * kHeads * kDim);
        error = cudaGetLastError();
        if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    }
}
} // namespace strata::qwen35moe
