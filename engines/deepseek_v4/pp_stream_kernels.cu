#include "pp_stream_cuda.cuh"
#include "mmq.cuh"
#include "pp_ragged_mmq.cuh"
#include "unary.cuh"
#include <algorithm>
#include <climits>

namespace strata::deepseek_v4 {
template<ggml_type type, ggml_prec precision>
static void product_typed(ggml_backend_cuda_context& ctx, const void* w, const void* xq,
                       const int32_t* bounds, const int32_t* identity, float* dst,
                       int experts, int rows, int quant_rows, int max_rows, int cols, int out_cols,
                       size_t expert_bytes, bool fp4, cudaStream_t stream, int destination_rows,
                       const int32_t* ragged_prefix,const int32_t* host_bounds, const int32_t* host_tile_counts) {
    GGML_ASSERT(expert_bytes%ggml_type_size(type)==0);
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const size_t sm = ggml_cuda_info().devices[ctx.device].smpbo;
    const auto prec = precision;
    (void)fp4;
    int j_best = 0, tiles_best = INT_MAX;
    for (int j = 8; j <= 128; j += 8) {
        const auto c = ggml_cuda_mmq_get_config(type, j, false, cc, prec);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, cc) > sm) continue;
        int tiles = (max_rows + c.J - 1) / c.J;
        if(ragged_prefix&&host_bounds) {
            if(host_tile_counts && c.J>=8 && c.J<=128 && c.J%8==0)
                tiles=host_tile_counts[(c.J/8-1)*(experts+1)+experts];
            else {
                tiles=0;for(int expert=0;expert<experts;++expert)
                    tiles+=(host_bounds[expert+1]-host_bounds[expert]+c.J-1)/c.J;
            }
        }
        if (tiles < tiles_best) { j_best = j; tiles_best = tiles; }
    }
    GGML_ASSERT(j_best > 0);
    const mmq_args a = {static_cast<const char*>(w), type,
        static_cast<const int*>(xq), identity, bounds, dst, nullptr,
        cols, out_cols, rows, cols / ggml_blck_size(type), quant_rows, destination_rows ? destination_rows : out_cols,
        experts, experts, int64_t(expert_bytes / ggml_type_size(type)), 0, 0,
        1, 1, 0, 0, 0, max_rows, j_best};
    const int rectangle_tiles=experts*((max_rows+j_best-1)/j_best);
    // Balanced geometry measured3.5% slower with the compact wrapper; only
    // select it when it deletes a substantial amount of empty tile work.
    if(ragged_prefix&&host_bounds&&5*tiles_best<=4*rectangle_tiles) {
        const auto* prefix=ragged_prefix+(j_best/8-1)*(experts+1);
        pp_ragged_dispatch<type,precision>(a,prefix,tiles_best,cc,stream);
    } else mul_mat_q_case<type,precision>(ctx, a, stream);
    CUDA_CHECK(cudaGetLastError());
}
void pp_stream_product(ggml_backend_cuda_context& ctx, const void* w, const void* xq,
                       const int32_t* bounds, const int32_t* identity, float* dst,
                       int experts, int rows, int quant_rows, int max_rows, int cols, int out_cols,
                       size_t expert_bytes, bool fp4, cudaStream_t stream, int destination_rows,
                       const int32_t* ragged_prefix, const int32_t* host_bounds, ggml_type type, const int32_t* host_tile_counts) {
#define STRATA_PP_PRODUCT(t,p) product_typed<t,p>(ctx,w,xq,bounds,identity,dst,experts,rows,quant_rows,max_rows,cols,out_cols,expert_bytes,fp4,stream,destination_rows,ragged_prefix,host_bounds,host_tile_counts)
    switch(type) {
        case GGML_TYPE_MXFP4:
            if(fp4) STRATA_PP_PRODUCT(GGML_TYPE_MXFP4,GGML_PREC_Q4);
            else STRATA_PP_PRODUCT(GGML_TYPE_MXFP4,GGML_PREC_Q8);
            break;
        case GGML_TYPE_IQ3_S: STRATA_PP_PRODUCT(GGML_TYPE_IQ3_S,GGML_PREC_Q8); break;
        case GGML_TYPE_IQ4_XS: STRATA_PP_PRODUCT(GGML_TYPE_IQ4_XS,GGML_PREC_Q8); break;
        case GGML_TYPE_Q6_K: STRATA_PP_PRODUCT(GGML_TYPE_Q6_K,GGML_PREC_Q8); break;
        default: GGML_ABORT("PP weight type has no canonical MMQ dispatch");
    }
#undef STRATA_PP_PRODUCT
}
__global__ void pp_swiglu(const float* gu, float* h, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int row = i / 2048, col = i % 2048;
    const float gate = gu[row * 4096 + col], up = gu[row * 4096 + 2048 + col];
    h[i] = limit > 1e-6f ? ggml_cuda_op_swiglu_clamp_single(gate, up, limit) :
                           ggml_cuda_op_silu_single(gate) * up;
}
__global__ void pp_scatter(const float* down, const int32_t* pairs, const float* weights,
                           float* dst, int n, int topk) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int pair = pairs[i / 4096], col = i % 4096;
    // A token may have several experts in this group. Preserve every rank's
    // contribution; the bounded float reduction is checked at the operator seam.
    atomicAdd(dst + (pair / topk) * 4096 + col, down[i] * weights[pair]);
}
__global__ void pp_iota(int32_t* ids, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ids[i] = i;
}
void pp_stream_swiglu(const float* gu, float* h, int rows, float clamp, cudaStream_t stream) {
    pp_swiglu<<<(rows * 2048 + 255) / 256, 256, 0, stream>>>(gu, h, rows * 2048, clamp);
    CUDA_CHECK(cudaGetLastError());
}
void pp_stream_scatter(const float* down, const int32_t* pairs, const float* weights,
                       float* dst, int rows, cudaStream_t stream, int topk) {
    pp_scatter<<<(rows * 4096 + 255) / 256, 256, 0, stream>>>(down, pairs, weights, dst, rows * 4096, topk);
    CUDA_CHECK(cudaGetLastError());
}
void pp_stream_iota(int32_t* ids, int n, cudaStream_t stream) {
    pp_iota<<<(n + 255) / 256, 256, 0, stream>>>(ids, n);
    CUDA_CHECK(cudaGetLastError());
}
}
