#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "codec.cuh"
#include <mma.h>

namespace strata::completed_v100::qwen35::native {
namespace {
namespace mma = nvcuda::wmma;
// A CTA owns R output features and C tokens. Both projections consume the same
// input tile; only quantized weight values, not global FP16 weights, are read.
template<int R, int C, int K, bool Pair>
__global__ void project(Matrix a, Matrix b, const float *input, float *first,
                        float *second, int tokens) {
    constexpr int Warps = 8;
    constexpr int Fragments = R * C / (256 * Warps);
    // Avoid power-of-two row strides when warps load matrix fragments. All
    // fragment origins remain 32-byte aligned. Staging dies before epilogue.
    constexpr int Stride = K + 16;
    constexpr int WeightElements = (Pair ? 2 : 1) * R * Stride;
    constexpr int StagingBytes = (WeightElements + C * Stride) * int(sizeof(half));
    constexpr int ResultBytes = (Pair ? 2 : 1) * R * C * int(sizeof(float));
    __shared__ __align__(32) unsigned char storage[StagingBytes > ResultBytes ? StagingBytes : ResultBytes];
    half *weights = reinterpret_cast<half *>(storage);
    half *inputs = weights + WeightElements;
    float *result = reinterpret_cast<float *>(storage);
    const int warp = threadIdx.x / 32;
    const int feature = int(blockIdx.x) * R;
    const int token = int(blockIdx.y) * C;
    mma::fragment<mma::accumulator, 16, 16, 16, float> accum[Fragments];
    mma::fragment<mma::accumulator, 16, 16, 16, float> other[Fragments];
    for (int f = 0; f < Fragments; ++f) {
        mma::fill_fragment(accum[f], 0.0f);
        if constexpr (Pair) mma::fill_fragment(other[f], 0.0f);
    }
    for (int base = 0; base < a.columns; base += K) {
        for (int j = threadIdx.x; j < R * K; j += Warps * 32) {
            const int row = feature + j / K;
            const int column = base + j % K;
            const int offset = (j / K) * Stride + j % K;
            weights[offset] = row < a.rows ? warp_matrix_element(a, row, column) : __float2half(0);
            if constexpr (Pair)
                weights[R * Stride + offset] = row < b.rows ? warp_matrix_element(b, row, column) : __float2half(0);
        }
        for (int j = threadIdx.x; j < C * K; j += Warps * 32) {
            const int column = token + j / K;
            inputs[(j / K) * Stride + j % K] = column < tokens ? __float2half_rn(input[size_t(column) * a.columns + base + j % K])
                                        : __float2half(0);
        }
        __syncthreads();
        for (int f = 0; f < Fragments; ++f) {
            const int tile = warp + f * Warps;
            const int r = (tile % (R / 16)) * 16;
            const int c = (tile / (R / 16)) * 16;
            for (int k = 0; k < K; k += 16) {
                mma::fragment<mma::matrix_a, 16, 16, 16, half, mma::row_major> wa;
                mma::fragment<mma::matrix_b, 16, 16, 16, half, mma::col_major> x;
                mma::load_matrix_sync(wa, weights + r * Stride + k, Stride);
                mma::load_matrix_sync(x, inputs + c * Stride + k, Stride);
                mma::mma_sync(accum[f], wa, x, accum[f]);
                if constexpr (Pair) {
                    mma::load_matrix_sync(wa, weights + R * Stride + r * Stride + k, Stride);
                    mma::mma_sync(other[f], wa, x, other[f]);
                }
            }
        }
        __syncthreads();
    }
    for (int f = 0; f < Fragments; ++f) {
        const int tile = warp + f * Warps;
        const int r = (tile % (R / 16)) * 16;
        const int c = (tile / (R / 16)) * 16;
        mma::store_matrix_sync(result + c * R + r, accum[f], R, mma::mem_col_major);
        if constexpr (Pair)
            mma::store_matrix_sync(result + R * C + c * R + r, other[f], R, mma::mem_col_major);
    }
    __syncthreads();
    for (int j = threadIdx.x; j < R * C; j += Warps * 32) {
        const int row = feature + j % R, column = token + j / R;
        if (row < a.rows && column < tokens) {
            const size_t out = size_t(column) * a.rows + row;
            first[out] = result[j];
            if constexpr (Pair) second[out] = result[R * C + j];
        }
    }
}
template<bool Pair>
void launch(Matrix a, Matrix b, const float *input, float *first, float *second,
            int tokens, Tile tile, cudaStream_t stream) {
    const dim3 grid((a.rows + tile.rows - 1) / tile.rows, (tokens + tile.tokens - 1) / tile.tokens);
    if (tile.rows == 64 && tile.tokens == 32)
        project<64, 32, 32, Pair><<<grid, 256, 0, stream>>>(a, b, input, first, second, tokens);
    else if (tile.rows == 64 && tile.tokens == 64 && tile.reduction == 64)
        project<64, 64, 64, Pair><<<grid, 256, 0, stream>>>(a, b, input, first, second, tokens);
    else throw std::invalid_argument("uncompiled native Tensor Core tile");
    check_cuda(cudaGetLastError(), "native quantized tiled projection");
}
}
void tiled_pair(Matrix a, Matrix b, const float *x, float *g, float *u,
                int tokens, Tile tile, cudaStream_t stream) {
    launch<true>(a, b, x, g, u, tokens, tile, stream);
}
void tiled_linear(Matrix a, const float *x, float *out, int tokens, Tile tile, cudaStream_t stream) {
    launch<false>(a, a, x, out, nullptr, tokens, tile, stream);
}
} // namespace strata::completed_v100::qwen35::native
