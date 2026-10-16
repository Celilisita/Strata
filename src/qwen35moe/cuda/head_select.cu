#include "strata/qwen35moe/head_pick.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
constexpr int kTile = 4096;
constexpr int kThreads = 256;

__device__ __forceinline__ uint32_t ordered_key(uint32_t bits) {
    // F32 numerical order on finite values, with both signed zeros sharing one key.
    if ((bits & 0x7fffffffu) == 0) bits = 0;
    return (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
}

__device__ __forceinline__ void choose(uint32_t candidate_key, float candidate, int id,
                                        uint32_t& best_key, float& best, int& best_id) {
    if (candidate_key > best_key ||
        (candidate_key == best_key && id < best_id)) {
        best_key = candidate_key;
        best = candidate;
        best_id = id;
    }
}

__device__ __forceinline__ void reduce_warp(uint32_t& best_key, float& best,
                                            int& id, int& bad) {
    for (int step = 16; step; step >>= 1) {
        const uint32_t other_key = __shfl_down_sync(0xffffffffu, best_key, step);
        const float other = __shfl_down_sync(0xffffffffu, best, step);
        const int other_id = __shfl_down_sync(0xffffffffu, id, step);
        const int other_bad = __shfl_down_sync(0xffffffffu, bad, step);
        choose(other_key, other, other_id, best_key, best, id);
        bad = min(bad, other_bad);
    }
}

__global__ void select_tiles(const float* logits, int n_vocab, HeadPick* partials) {
    const int lane = threadIdx.x;
    const int first = int(blockIdx.x) * kTile;
    uint32_t best_key = 0;
    float best = __int_as_float(0xff800000u);
    int best_id = n_vocab;
    int bad = INT_MAX;
    for (int i = lane; i < kTile && first + i < n_vocab; i += kThreads) {
        const int id = first + i;
        const float value = logits[id];
        // Integer bits preserve this check under -use_fast_math, including NaN and both infinities.
        const uint32_t bits = __float_as_uint(value);
        if ((bits & 0x7f800000u) == 0x7f800000u) bad = min(bad, id);
        else choose(ordered_key(bits), value, id, best_key, best, best_id);
    }
    reduce_warp(best_key, best, best_id, bad);
    __shared__ uint32_t warp_key[8];
    __shared__ float warp_best[8];
    __shared__ int warp_id[8], warp_bad[8];
    if ((lane & 31) == 0) {
        const int warp = lane >> 5;
        warp_key[warp] = best_key;
        warp_best[warp] = best;
        warp_id[warp] = best_id;
        warp_bad[warp] = bad;
    }
    __syncthreads();
    if (lane < 32) {
        best_key = lane < 8 ? warp_key[lane] : 0;
        best = lane < 8 ? warp_best[lane] : __int_as_float(0xff800000u);
        best_id = lane < 8 ? warp_id[lane] : n_vocab;
        bad = lane < 8 ? warp_bad[lane] : INT_MAX;
        reduce_warp(best_key, best, best_id, bad);
        if (lane == 0) partials[blockIdx.x] = {best_id, bad, best};
    }
}

__global__ void select_final(const HeadPick* partials, int count, int n_vocab,
                             HeadPick* result, const int32_t* failure) {
    const int lane = threadIdx.x;
    uint32_t best_key = 0;
    float best = __int_as_float(0xff800000u);
    int best_id = n_vocab;
    int bad = INT_MAX;
    for (int i = lane; i < count; i += kThreads) {
        const auto part = partials[i];
        if (part.id < n_vocab)
            choose(ordered_key(__float_as_uint(part.logit)), part.logit, part.id,
                   best_key, best, best_id);
        bad = min(bad, part.first_bad);
    }
    reduce_warp(best_key, best, best_id, bad);
    __shared__ uint32_t warp_key[8];
    __shared__ float warp_best[8];
    __shared__ int warp_id[8], warp_bad[8];
    if ((lane & 31) == 0) {
        const int warp = lane >> 5;
        warp_key[warp] = best_key;
        warp_best[warp] = best;
        warp_id[warp] = best_id;
        warp_bad[warp] = bad;
    }
    __syncthreads();
    if (lane < 32) {
        best_key = lane < 8 ? warp_key[lane] : 0;
        best = lane < 8 ? warp_best[lane] : __int_as_float(0xff800000u);
        best_id = lane < 8 ? warp_id[lane] : n_vocab;
        bad = lane < 8 ? warp_bad[lane] : INT_MAX;
        reduce_warp(best_key, best, best_id, bad);
        if (lane == 0) {
            const int code = failure ? *failure : 0;
            *result = {code ? -1 : best_id, code ? -2 - code : (bad == INT_MAX ? -1 : bad), best};
        }
    }
}
}

void select_head_greedy(const float* logits, int n_vocab, HeadPick* partials,
                        HeadPick* result, void* stream, const int32_t* failure) {
    if (!logits || n_vocab < 1 || !partials || !result || !stream)
        throw std::invalid_argument("invalid Qwen35 GPU head selector");
    const int blocks = (n_vocab + kTile - 1) / kTile;
    select_tiles<<<blocks, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        logits, n_vocab, partials);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    select_final<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        partials, blocks, n_vocab, result, failure);
    status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

} // namespace strata::qwen35moe
