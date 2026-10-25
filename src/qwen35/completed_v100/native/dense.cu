#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "strata/qwen35/completed_v100/native/projection.hpp"
#include <cuda_fp16.h>
#include <mma.h>
#include "math.cuh"

namespace strata::completed_v100::qwen35::native {
namespace {
namespace mma = nvcuda::wmma;
#include "dense_native.cuh"
#include "dense_balanced_pipeline.cuh"
#include "dense_joint_shared.cuh"
#include "dense_joint.cuh"
#include "dense_folded.cuh"
#include "dense_folded_compact.cuh"
#include "dense_folded_gray.cuh"
#include "dense_folded_widewarp.cuh"
#include "dense_folded_ring_shared.cuh"
#include "dense_folded_ring.cuh"
#include "dense_token256.cuh"
#include "dense_warp64.cuh"
#include "dense_operand_lookahead.cuh"
#include "dense_cache_policy.cuh"
// Staging and compute have separate lifetimes: quantized weights decode once,
// while each CTA consumes coalesced half vectors and reuses them across tokens.
template<bool Pair>
__global__ void dense(const half *weights, const half *input, float *first,
                      float *second, int rows, int columns, int tokens, int splits, int partition_columns) {
    constexpr int R = 64, C = 64, K = 64, Stride = K + 16, Warps = 4;
    constexpr int Fragments = R * C / (256 * Warps);
    __shared__ __align__(32) half shared[(R + C) * Stride];
    const int warp = threadIdx.x / 32;
    const int feature = int(blockIdx.x) * R, token = int(blockIdx.y) * C;
    const size_t plane = Pair ? size_t(blockIdx.z) * rows * columns : 0;
    half *wa = shared, *xb = shared + R * Stride;
    mma::fragment<mma::accumulator, 16, 16, 16, float> accum[Fragments];
    for (int f = 0; f < Fragments; ++f) mma::fill_fragment(accum[f], 0.0f);
    const int begin = Pair ? 0 : int(blockIdx.z) * partition_columns;
    const int limit = begin + (Pair ? columns : partition_columns);
    const int end = limit < columns ? limit : columns;
    for (int base = begin; base < end; base += K) {
        for (int v = threadIdx.x; v < R * K / 8; v += Warps * 32) {
            const int row = v / (K / 8), k = (v % (K / 8)) * 8;
            uint4 value{};
            if (feature + row < rows && base + k < end)
                value = *reinterpret_cast<const uint4 *>(weights + plane + size_t(feature + row) * columns + base + k);
            *reinterpret_cast<uint4 *>(wa + row * Stride + k) = value;
        }
        for (int v = threadIdx.x; v < C * K / 8; v += Warps * 32) {
            const int col = v / (K / 8), k = (v % (K / 8)) * 8;
            uint4 value{};
            if (token + col < tokens && base + k < end)
                value = *reinterpret_cast<const uint4 *>(input + size_t(token + col) * columns + base + k);
            *reinterpret_cast<uint4 *>(xb + col * Stride + k) = value;
        }
        __syncthreads();
        for (int k = 0; k < K; k += 16) {
            for (int f = 0; f < Fragments; ++f) {
                const int tile = warp + f * Warps;
                const int r = (tile % (R / 16)) * 16, c = (tile / (R / 16)) * 16;
                mma::fragment<mma::matrix_a, 16, 16, 16, half, mma::row_major> a;
                mma::fragment<mma::matrix_b, 16, 16, 16, half, mma::col_major> b;
                mma::load_matrix_sync(a, wa + r * Stride + k, Stride);
                mma::load_matrix_sync(b, xb + c * Stride + k, Stride);
                mma::mma_sync(accum[f], a, b, accum[f]);
            }
        }
        __syncthreads();
    }
    // The same storage holds one warp's result at a time; each warp owns a
    // disjoint 16x16 tile. No shared operand survives the final barrier.
    float *result = reinterpret_cast<float *>(shared);
    float *out = Pair && blockIdx.z ? second : first;
    if constexpr (!Pair) out += size_t(blockIdx.z) * tokens * rows;
    for (int f = 0; f < Fragments; ++f) {
        mma::store_matrix_sync(result + warp * 256, accum[f], 16, mma::mem_col_major);
        __syncwarp();
        const int tile = warp + f * Warps;
        const int r = (tile % (R / 16)) * 16, c = (tile / (R / 16)) * 16;
        for (int j = threadIdx.x % 32; j < 256; j += 32) {
            const int row = feature + r + j % 16, col = token + c + j / 16;
            if (row < rows && col < tokens) out[size_t(col) * rows + row] = result[warp * 256 + j];
        }
        __syncwarp();
    }
}
template<bool Pair>
void launch(const void *weights, const void *input, float *first, float *second,
            int rows, int columns, int tokens, cudaStream_t stream) {
    if (columns % 64) throw std::invalid_argument("native dense reduction must be divisible by 64");
    const dim3 grid((rows + 63) / 64, (tokens + 63) / 64, Pair ? 2 : 1);
    dense<Pair><<<grid, 128, 0, stream>>>(static_cast<const half *>(weights), static_cast<const half *>(input),
                                       first, second, rows, columns, tokens, 1, columns);
    check_cuda(cudaGetLastError(), "native decoded Tensor Core projection");
}
__global__ void finish_split(const float *partial, float *output, size_t count, int splits) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float value = partial[i];
    for (int s = 1; s < splits; ++s) value = __fadd_rn(value, partial[size_t(s) * count + i]);
    output[i] = value;
}
template<bool Pair,bool Balanced=false,bool TokenFirst=false>
void launch_native(const void *weights,const void *input,float *first,float *second,
                   int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(columns%64||splits<1||columns%(splits*64))throw std::invalid_argument("invalid owned register dense geometry");
    const int row_tiles=(rows+63)/64,token_tiles=(tokens+127)/128;
    const dim3 grid(TokenFirst?token_tiles:row_tiles,TokenFirst?row_tiles:token_tiles,Pair?2:splits);
    dense_native<Pair,Balanced,TokenFirst><<<grid,256,0,stream>>>(static_cast<const half *>(weights),static_cast<const half *>(input),
                                           first,second,rows,columns,tokens,columns/splits);
    check_cuda(cudaGetLastError(),"owned register Tensor Core dense projection");
}
}
void dense_pair(const void *weights, const void *input, float *gate, float *up,
                int rows, int columns, int tokens, cudaStream_t stream) {
    launch<true>(weights, input, gate, up, rows, columns, tokens, stream);
}
void dense_linear(const void *weights, const void *input, float *output,
                  int rows, int columns, int tokens, cudaStream_t stream) {
    launch<false>(weights, input, output, nullptr, rows, columns, tokens, stream);
}
void dense_split(const void *weights, const void *input, float *output, float *partial,
                 int rows, int columns, int tokens, int splits, cudaStream_t stream) {
    if (!partial || splits < 2 || columns % (splits * 64))
        throw std::invalid_argument("invalid native dense reduction partition");
    const dim3 grid((rows + 63) / 64, (tokens + 63) / 64, splits);
    dense<false><<<grid, 128, 0, stream>>>(static_cast<const half *>(weights), static_cast<const half *>(input),
                                        partial, nullptr, rows, columns, tokens, splits, columns / splits);
    check_cuda(cudaGetLastError(), "native partitioned Tensor Core projection");
    const size_t count = size_t(rows) * tokens;
    finish_split<<<unsigned((count + 255) / 256), 256, 0, stream>>>(partial, output, count, splits);
    check_cuda(cudaGetLastError(), "native ordered partition reduction");
}
void dense_wide_pair(const void *weights,const void *input,float *gate,float *up,
                     int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_wide_linear(const void *weights,const void *input,float *output,
                      int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<false>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_wide_split(const void *weights,const void *input,float *output,float *partial,
                     int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid owned register dense partial owner");
    launch_native<false>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"owned register dense partition reduction");
}
void configure_dense_wide(){
    check_cuda(cudaFuncSetAttribute(dense_native<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"register paired shared policy");
    check_cuda(cudaFuncSetAttribute(dense_native<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"register linear shared policy");
}
void dense_balanced_pair(const void *weights,const void *input,float *gate,float *up,
                        int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<true,true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_balanced_linear(const void *weights,const void *input,float *output,
                          int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<false,true>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_balanced_split(const void *weights,const void *input,float *output,float *partial,
                         int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid balanced dense partial owner");
    launch_native<false,true>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"balanced dense partition reduction");
}
void configure_dense_balanced(){
    check_cuda(cudaFuncSetAttribute(dense_native<true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"balanced pair shared policy");
    check_cuda(cudaFuncSetAttribute(dense_native<false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"balanced linear shared policy");
}
void dense_raster_pair(const void *weights,const void *input,float *gate,float *up,
                      int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<true,true,true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_raster_linear(const void *weights,const void *input,float *output,
                        int rows,int columns,int tokens,cudaStream_t stream){
    launch_native<false,true,true>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_raster_split(const void *weights,const void *input,float *output,float *partial,
                       int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid raster dense partial owner");
    launch_native<false,true,true>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"raster dense partition reduction");
}
void configure_dense_raster(){
    check_cuda(cudaFuncSetAttribute(dense_native<true,true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"raster pair shared policy");
    check_cuda(cudaFuncSetAttribute(dense_native<false,true,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"raster linear shared policy");
}
template<bool Pair,bool Swizzle=false,bool FullK=false,bool Gray=false>
void launch_joint(const void *weights,const void *input,float *first,float *second,
                  int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(columns%64||splits<1||columns%(splits*64))throw std::invalid_argument("invalid joint dense geometry");
    const dim3 grid((tokens+127)/128,(rows+(Pair?63:127))/(Pair?64:128),Pair?1:splits);
    dense_joint<Pair,Swizzle,FullK,Gray><<<grid,256,0,stream>>>(static_cast<const half *>(weights),static_cast<const half *>(input),
                                          first,second,rows,columns,tokens,columns/splits);
    check_cuda(cudaGetLastError(),"joint input reuse dense projection");
}
void dense_joint_pair(const void *weights,const void *input,float *gate,float *up,
                     int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_joint_linear(const void *weights,const void *input,float *output,
                       int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<false>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_joint_split(const void *weights,const void *input,float *output,float *partial,
                      int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid joint dense partial owner");
    launch_joint<false>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"joint dense partition reduction");
}
void configure_dense_joint(){
    check_cuda(cudaFuncSetAttribute(dense_joint<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"joint pair shared policy");
    check_cuda(cudaFuncSetAttribute(dense_joint<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"joint linear shared policy");
}
void configure_dense_operand_lookahead(){
    check_cuda(cudaFuncSetAttribute(dense_operand_lookahead,cudaFuncAttributePreferredSharedMemoryCarveout,100),"operand lookahead shared policy");
    int active=0;check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_operand_lookahead,256,0),"operand lookahead residency");
    if(active<2)throw std::invalid_argument("operand lookahead requires two resident256-thread CTAs");
}
void dense_operand_lookahead_pair(const void *weights,const void *input,float *gate,float *up,
                                int rows,int columns,int tokens,cudaStream_t stream){
    if(columns%64||tokens<128)throw std::invalid_argument("operand lookahead requires full K64 and wide token span");
    dense_operand_lookahead<<<dim3((tokens+127)/128,(rows+63)/64),256,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,rows,columns,tokens,columns);
    check_cuda(cudaGetLastError(),"shared operand lookahead Gate/Up");
}
void configure_dense_warp64(){
    check_cuda(cudaFuncSetAttribute(dense_warp64,cudaFuncAttributePreferredSharedMemoryCarveout,100),"warp64 shared policy");
    int active=0;check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active,dense_warp64,512,0),"warp64 actual occupancy budget");
    if(active<2)throw std::invalid_argument("warp64 requires two resident CTAs without spilling");
}
void dense_warp64_pair(const void *weights,const void *input,float *gate,float *up,
                      int rows,int columns,int tokens,cudaStream_t stream){
    if(columns%64||tokens<128)throw std::invalid_argument("warp64 requires full K64 and wide token span");
    if(rows%64||tokens%128){dense_joint_pair(weights,input,gate,up,rows,columns,tokens,stream);return;}
    dense_warp64<<<dim3((tokens+127)/128,(rows+63)/64),512,0,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,rows,columns,tokens,columns);
    check_cuda(cudaGetLastError(),"warp64 operand reuse Gate/Up");
}
void configure_dense_token256(){
    int device=0,limit=0;check_cuda(cudaGetDevice(&device),"token256 execution device");
    check_cuda(cudaDeviceGetAttribute(&limit,cudaDevAttrMaxSharedMemoryPerBlockOptin,device),"token256 opt-in capacity");
    cudaFuncAttributes attributes{};
    check_cuda(cudaFuncGetAttributes(&attributes,dense_token256),"token256 compiled shared budget");
    constexpr int bytes=(128+256)*72*2;
    if(size_t(limit)<size_t(bytes)+attributes.sharedSizeBytes)
        throw std::invalid_argument("token256 dynamic plus static shared exceeds opt-in capacity");
    check_cuda(cudaFuncSetAttribute(dense_token256,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes),"token256 dynamic shared budget");
    check_cuda(cudaFuncSetAttribute(dense_token256,cudaFuncAttributePreferredSharedMemoryCarveout,100),"token256 shared policy");
}
void dense_token256_pair(const void *weights,const void *input,float *gate,float *up,
                         int rows,int columns,int tokens,cudaStream_t stream){
    if(columns%64||tokens<256)throw std::invalid_argument("token256 requires full K64 and wide token span");
    dense_token256<<<dim3((tokens+255)/256,(rows+63)/64),512,(128+256)*72*2,stream>>>(
        static_cast<const half *>(weights),static_cast<const half *>(input),gate,up,rows,columns,tokens,columns);
    check_cuda(cudaGetLastError(),"token256 shared weight Gate/Up");
}
void dense_full_k_pair(const void *weights,const void *input,float *gate,float *up,
                       int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<true,false,true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_full_k_linear(const void *weights,const void *input,float *output,
                         int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<false,false,true>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_full_k_split(const void *weights,const void *input,float *output,float *partial,
                        int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid full K dense partial owner");
    launch_joint<false,false,true>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"full K dense partition reduction");
}
void configure_dense_full_k(){
    check_cuda(cudaFuncSetAttribute(dense_joint<true,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"full K pair shared policy");
    check_cuda(cudaFuncSetAttribute(dense_joint<false,false,true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"full K linear shared policy");
}
#include "dense_bank.cuh"
#include "dense_folded_launch.cuh"
#include "dense_folded_compact_launch.cuh"
#include "dense_folded_gray_launch.cuh"
#include "dense_folded_widewarp_launch.cuh"
#include "dense_folded_ring_launch.cuh"
void dense_shared_pair(const void *weights,const void *input,float *gate,float *up,
                      int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<true,true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_shared_linear(const void *weights,const void *input,float *output,
                        int rows,int columns,int tokens,cudaStream_t stream){
    launch_joint<false,true>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_shared_split(const void *weights,const void *input,float *output,float *partial,
                       int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid shared dense partial owner");
    launch_joint<false,true>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"shared dense partition reduction");
}
void configure_dense_shared(){
    configure_dense_cache(dense_joint<true,true>);
    configure_dense_cache(dense_joint<false,true>);
}
template<bool Pair>
void launch_pipeline(const void *weights,const void *input,float *first,float *second,
                     int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(columns%64||splits<1||columns%(splits*64))throw std::invalid_argument("invalid pipeline dense geometry");
    const dim3 grid((rows+63)/64,(tokens+127)/128,Pair?2:splits);
    dense_balanced_pipeline<Pair><<<grid,256,0,stream>>>(static_cast<const half *>(weights),static_cast<const half *>(input),
                                                       first,second,rows,columns,tokens,columns/splits);
    check_cuda(cudaGetLastError(),"owned pipeline dense projection");
}
void dense_pipeline_pair(const void *weights,const void *input,float *gate,float *up,
                         int rows,int columns,int tokens,cudaStream_t stream){
    launch_pipeline<true>(weights,input,gate,up,rows,columns,tokens,1,stream);
}
void dense_pipeline_linear(const void *weights,const void *input,float *output,
                           int rows,int columns,int tokens,cudaStream_t stream){
    launch_pipeline<false>(weights,input,output,nullptr,rows,columns,tokens,1,stream);
}
void dense_pipeline_split(const void *weights,const void *input,float *output,float *partial,
                          int rows,int columns,int tokens,int splits,cudaStream_t stream){
    if(!partial||splits<2)throw std::invalid_argument("invalid pipeline partial owner");
    launch_pipeline<false>(weights,input,partial,nullptr,rows,columns,tokens,splits,stream);
    const size_t count=size_t(rows)*tokens;
    finish_split<<<unsigned((count+255)/256),256,0,stream>>>(partial,output,count,splits);
    check_cuda(cudaGetLastError(),"pipeline dense partition reduction");
}
void configure_dense_pipeline(){
    check_cuda(cudaFuncSetAttribute(dense_balanced_pipeline<true>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"pipeline pair shared policy");
    check_cuda(cudaFuncSetAttribute(dense_balanced_pipeline<false>,cudaFuncAttributePreferredSharedMemoryCarveout,100),"pipeline linear shared policy");
}
void dense_partitioned(const void *weights, const void *input, float *output, float *partial,
                       int rows, int columns, int tokens, int splits, int granularity, cudaStream_t stream) {
    if (!weights || !input || !output || !partial || rows < 1 || columns < 64 || columns % 64 || tokens < 1 ||
        splits < 2 || splits > 32 || (granularity != 32 && granularity != 64 && granularity != 128))
        throw std::invalid_argument("invalid native bounded reduction partition");
    const int span = ((columns + splits * granularity - 1) / (splits * granularity)) * granularity;
    const dim3 grid((rows + 63) / 64, (tokens + 63) / 64, splits);
    dense<false><<<grid, 128, 0, stream>>>(static_cast<const half *>(weights), static_cast<const half *>(input),
                                        partial, nullptr, rows, columns, tokens, splits, span);
    check_cuda(cudaGetLastError(), "native bounded partitioned Tensor Core projection");
    const size_t count = size_t(rows) * tokens;
    finish_split<<<unsigned((count + 255) / 256), 256, 0, stream>>>(partial, output, count, splits);
    check_cuda(cudaGetLastError(), "native bounded partition reduction");
}
} // namespace strata::completed_v100::qwen35::native
