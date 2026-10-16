// Q6_K MMQ with the canonical native accumulator. GGUF blocks and the lossless
// packed MMQ input use register matrix tiles; the Qwen graph owns no new math.
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include "mmq.cuh"
#include <stdexcept>

namespace strata::prefill::mmq {
namespace {
constexpr int WARP = 32, WARPS = 4;
using Q6KBlock = block_q6_K;
using Q81Block = block_q8_1;
__device__ __forceinline__ int load_int_b2(const void* source, int word) {
    const auto* h = static_cast<const uint16_t*>(source) + 2*word;
    return int(uint32_t(h[0]) | (uint32_t(h[1]) << 16));
}
// Shared canonical scale, integer dot, and two-term F32 arithmetic.
#include "../kernels/cuda/native_k_dot.cuh"
#include "../kernels/cuda/native_q6_traits.cuh"

constexpr int Rows = 2, Columns = 8;
template<bool Tail>
__launch_bounds__(WARPS*WARP, Tail ? 1 : 8)
__global__ void canonical_q6_mmq(const Q6KBlock* __restrict__ weights,
    const block_q8_1_mmq* __restrict__ input, float* __restrict__ output,
    int width, int tokens, int stride, int token_start) {
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int row0 = int(blockIdx.x)*Rows;
    const int token0 = token_start+int(blockIdx.y)*Columns;
    const int blocks = width/256;
    float sum[Columns][Rows] = {};
    // Four canonical K chains and the retained V174 eight-token/two-row
    // register tile. The independent coalesced pack eliminates its measured tax.
    for (int kb = warp; kb < blocks; kb += WARPS) {
        Q6KTraits::W w[Rows];
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
            const int source_row = row0+row;
            w[row] = Q6KTraits::load(weights+size_t(source_row)*blocks+kb,lane);
        }
#pragma unroll
        for (int col = 0; col < Columns; ++col) {
            if (Tail && token0+col >= tokens) continue;
            Q6KTraits::Input activation{};
            if (!Tail || token0+col < tokens) {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const int q8 = w[0].bq8_offset+2*i;
                    const auto& packed = input[size_t(kb*2+q8/4)*tokens+token0+col];
                    activation.u[i] = reinterpret_cast<const int*>(packed.qs)[(q8%4)*8+lane%8];
                    activation.d8[i] = packed.d4[q8%4];
                }
            }
#pragma unroll
            for (int row = 0; row < Rows; ++row)
                sum[col][row] += Q6KTraits::apply_preloaded(w[row],activation);
        }
    }
    __shared__ float partial[WARPS-1][Columns][Rows][WARP];
    if (warp) {
#pragma unroll
        for (int col = 0; col < Columns; ++col)
#pragma unroll
            for (int row = 0; row < Rows; ++row)
                partial[warp-1][col][row][lane] = sum[col][row];
    }
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int col = 0; col < Columns; ++col) {
        if (Tail && token0+col >= tokens) continue;
#pragma unroll
        for (int row = 0; row < Rows; ++row) {
#pragma unroll
            for (int chain = 0; chain < WARPS-1; ++chain)
                sum[col][row] += partial[chain][col][row][lane];
#pragma unroll
            for (int offset = WARP/2; offset > 0; offset >>= 1)
                sum[col][row] += __shfl_xor_sync(0xffffffff,sum[col][row],offset,WARP);
            if (!lane)
                output[size_t(token0+col)*stride+row0+row] = sum[col][row];
        }
    }
}
}
int run_q6_canonical(const Product& p, void* stream) {
    if (!stream || !p.w || !p.xq || !p.dst || p.type != GGML_TYPE_Q6_K ||
        p.n != 1 || p.ids || p.bounds || p.total_rows < 1 || p.total_rows > 8192 ||
        p.max_rows != p.total_rows || p.w_cols < 1024 || p.w_cols > 16384 || p.w_cols%1024 ||
        p.w_rows < Rows || p.w_rows > 8192 || p.w_rows%Rows || p.ld_dst < p.w_rows || p.ld_dst > 8192 ||
        p.expert_bytes < size_t(p.w_rows)*(p.w_cols/256)*sizeof(Q6KBlock))
        throw std::invalid_argument("canonical Q6 MMQ requires a bounded dense packed product");
    static_assert(sizeof(Q6KBlock) == 210 && sizeof(Q81Block) == 36);
    const int full = int(p.total_rows)/Columns;
    const int tail = int(p.total_rows)%Columns;
    const auto stream_cuda = static_cast<cudaStream_t>(stream);
    const auto* weights = static_cast<const Q6KBlock*>(p.w);
    const auto* input = static_cast<const block_q8_1_mmq*>(p.xq);
    if (full)
        canonical_q6_mmq<false><<<dim3(unsigned(p.w_rows/Rows),unsigned(full)),dim3(WARP,WARPS),0,stream_cuda>>>(
            weights,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.ld_dst),0);
    if (tail)
        canonical_q6_mmq<true><<<dim3(unsigned(p.w_rows/Rows),1),dim3(WARP,WARPS),0,stream_cuda>>>(
            weights,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.ld_dst),full*Columns);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return int(full > 0)+int(tail > 0);
}
} // namespace strata::prefill::mmq
