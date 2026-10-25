// Q6_K MMQ with the canonical native accumulator. GGUF blocks and the lossless
// packed MMQ input use register matrix tiles; the Qwen graph owns no new math.
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include "mmq.cuh"
#include "strata/prefill/native_grouped_mmq.hpp"
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#ifdef STRATA_BLACKWELL_MMQ
#include "strata/prefill/native_blackwell_mmq.hpp"
#endif

namespace strata::prefill::mmq {
namespace {
constexpr int WARP = 32, WARPS = 4, QK = 256, Q8K = 32, QI = 32, VDR = 2;
using Q4KBlock = block_q4_K;
using Q6KBlock = block_q6_K;
using Q81Block = block_q8_1;
__device__ __forceinline__ int load_int_b2(const void* source, int word) {
    const auto* h = static_cast<const uint16_t*>(source) + 2*word;
    return int(uint32_t(h[0]) | (uint32_t(h[1]) << 16));
}
// Shared canonical scale, integer dot, and two-term F32 arithmetic.
#include "../kernels/cuda/native_k_dot.cuh"
#include "../kernels/cuda/native_q6_traits.cuh"
#include "../kernels/cuda/native_q4_traits.cuh"
#include "native_canonical_mmq.cuh"
#include "native_warp_matrix_mmq.cuh"

constexpr int Rows = 2, Columns = 8;
template<bool Tail, int TileRows>
__launch_bounds__(WARPS*WARP, Tail ? 1 : TileRows == 2 ? 8 : TileRows == 4 ? 4 : 2)
__global__ void canonical_q6_mmq(const Q6KBlock* __restrict__ weights,
    const block_q8_1_mmq* __restrict__ input, float* __restrict__ output,
    int width, int tokens, int stride, int token_start) {
    const int lane = int(threadIdx.x), warp = int(threadIdx.y);
    const int row0 = int(blockIdx.x)*TileRows;
    const int token0 = token_start+int(blockIdx.y)*Columns;
    const int blocks = width/256;
    float sum[Columns][TileRows] = {};
    // Four canonical K chains and the retained V174 eight-token/two-row
    // register tile. The independent coalesced pack eliminates its measured tax.
    for (int kb = warp; kb < blocks; kb += WARPS) {
        Q6KTraits::W w[TileRows];
#pragma unroll
        for (int row = 0; row < TileRows; ++row) {
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
            for (int row = 0; row < TileRows; ++row)
                sum[col][row] += Q6KTraits::apply_preloaded(w[row],activation);
        }
    }
    __shared__ float partial[WARPS-1][Columns][TileRows][WARP];
    if (warp) {
#pragma unroll
        for (int col = 0; col < Columns; ++col)
#pragma unroll
            for (int row = 0; row < TileRows; ++row)
                partial[warp-1][col][row][lane] = sum[col][row];
    }
    __syncthreads();
    if (warp) return;
#pragma unroll
    for (int col = 0; col < Columns; ++col) {
        if (Tail && token0+col >= tokens) continue;
#pragma unroll
        for (int row = 0; row < TileRows; ++row) {
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
int canonical_dense_rows() {
    const char* value=std::getenv("STRATA_NATIVE_DENSE_ROWS");
    if(!value || !std::strcmp(value,"2")) return 2;
    if(!std::strcmp(value,"4")) return 4;
    if(!std::strcmp(value,"8")) return 8;
    throw std::invalid_argument("canonical dense rows require2,4or8");
}
template<int TileRows> int launch_q6_dense(const Product& p,void* stream) {
    const int full=int(p.total_rows)/Columns,tail=int(p.total_rows)%Columns;
    const auto st=static_cast<cudaStream_t>(stream);
    const auto* weights=static_cast<const Q6KBlock*>(p.w);
    const auto* input=static_cast<const block_q8_1_mmq*>(p.xq);
    if(full) canonical_q6_mmq<false,TileRows><<<dim3(unsigned(p.w_rows/TileRows),unsigned(full)),dim3(WARP,WARPS),0,st>>>(
        weights,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.ld_dst),0);
    if(tail) canonical_q6_mmq<true,TileRows><<<dim3(unsigned(p.w_rows/TileRows),1),dim3(WARP,WARPS),0,st>>>(
        weights,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.ld_dst),full*Columns);
    const auto error=cudaGetLastError();
    if(error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return int(full>0)+int(tail>0);
}
int run_q6_canonical(const Product& p, void* stream) {
    if (!stream || !p.w || !p.xq || !p.dst || p.type != GGML_TYPE_Q6_K ||
        p.n != 1 || p.ids || p.bounds || p.total_rows < 1 || p.total_rows > 8192 ||
        p.max_rows != p.total_rows || p.w_cols < 1024 || p.w_cols > 16384 || p.w_cols%1024 ||
        p.w_rows < Rows || p.w_rows > 8192 || p.w_rows%Rows || p.ld_dst < p.w_rows || p.ld_dst > 8192 ||
        p.expert_bytes < size_t(p.w_rows)*(p.w_cols/256)*sizeof(Q6KBlock))
        throw std::invalid_argument("canonical Q6 MMQ requires a bounded dense packed product");
    static_assert(sizeof(Q6KBlock) == 210 && sizeof(Q81Block) == 36);
    const int rows=canonical_dense_rows();
    if(p.w_rows%rows) throw std::invalid_argument("dense row tile does not divide output width");
    if(rows==8) return launch_q6_dense<8>(p,stream);
    if(rows==4) return launch_q6_dense<4>(p,stream);
    return launch_q6_dense<2>(p,stream);
}
int run_q4_canonical(const Product& p, void* stream) {
    if (!stream || !p.w || !p.xq || !p.dst || p.type != GGML_TYPE_Q4_K || p.n != 1 ||
        p.ids || p.bounds || p.total_rows < 1 || p.total_rows > 8192 ||
        p.w_cols < 512 || p.w_cols%512 || p.w_rows < 2 || p.w_rows%2 || p.ld_dst != p.w_rows)
        throw std::invalid_argument("invalid bounded canonical Q4 matrix");
    const int rows=canonical_dense_rows();
    if(p.w_rows%rows) throw std::invalid_argument("dense row tile does not divide output width");
    const dim3 grid(unsigned(p.w_rows/rows),unsigned((p.total_rows+7)/8));
    const auto st=static_cast<cudaStream_t>(stream);
    const auto* weights=static_cast<const Q4KBlock*>(p.w);
    const auto* input=static_cast<const block_q8_1_mmq*>(p.xq);
    if(rows==8) canonical_matrix_mmq<PackedQ4,false,false,8><<<grid,128,0,st>>>(weights,nullptr,nullptr,nullptr,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.w_rows),int(p.ld_dst),1);
    else if(rows==4) canonical_matrix_mmq<PackedQ4,false,false,4><<<grid,128,0,st>>>(weights,nullptr,nullptr,nullptr,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.w_rows),int(p.ld_dst),1);
    else canonical_matrix_mmq<PackedQ4,false,false,2><<<grid,128,0,st>>>(weights,nullptr,nullptr,nullptr,input,p.dst,int(p.w_cols),int(p.total_rows),int(p.w_rows),int(p.ld_dst),1);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    return 1;
}
template<class F> void grouped_canonical(const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* ids, const void* input, float* output, int rows, int max_rows,
    int width, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count) {
    const auto st = static_cast<cudaStream_t>(stream);
    const dim3 grid(unsigned(outputs/4),unsigned(tiles ? tile_count*8 : ((max_rows+63)/64)*8),
                    unsigned(tiles ? parts : groups*parts));
    if (parts == 2) {
        if (tiles) canonical_matrix_mmq<F,true,true,4><<<grid,128,0,st>>>(nullptr,matrices,tiles,ids,
            static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,outputs,parts);
        else canonical_matrix_mmq<F,true,false,4><<<grid,128,0,st>>>(nullptr,matrices,nullptr,ids,
            static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,outputs,parts);
        return;
    }
    if (tiles) canonical_warp_matrix_mmq<F,true><<<grid,128,0,st>>>(matrices,tiles,ids,
        static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,parts);
    else canonical_warp_matrix_mmq<F,false><<<grid,128,0,st>>>(matrices,nullptr,ids,
        static_cast<const block_q8_1_mmq*>(input),output,width,rows,outputs,parts);
}
void native_grouped_canonical(int type, const NativeMmqMatrix* matrices, int groups, int parts,
    const int32_t* ids, const void* input, float* output, int rows, int max_rows,
    int width, int outputs, void* stream, const NativeMmqTile* tiles, int tile_count, const NativeMmqEncodedInput* encoded) {
    #ifdef STRATA_BLACKWELL_MMQ
    if (const int backend = blackwell_mmq_backend(type,parts)) {
        if (!encoded) throw std::invalid_argument("Blackwell MMQ requires an encoded input view");
        run_blackwell_grouped(type,matrices,groups,parts,ids,input,output,rows,max_rows,width,outputs,stream,tiles,tile_count,backend,*encoded);
        return;
    }
    #endif
    if (type != GGML_TYPE_Q4_K && type != GGML_TYPE_Q6_K)
        throw std::invalid_argument("No canonical grouped fallback for this quant format");
    if (type == GGML_TYPE_Q4_K) grouped_canonical<PackedQ4>(matrices,groups,parts,ids,input,output,
        rows,max_rows,width,outputs,stream,tiles,tile_count);
    else grouped_canonical<PackedQ6>(matrices,groups,parts,ids,input,output,
        rows,max_rows,width,outputs,stream,tiles,tile_count);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
} // namespace strata::prefill::mmq
