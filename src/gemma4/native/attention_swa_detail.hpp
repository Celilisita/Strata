#pragma once
#include "attention_swa_ops.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::gemma4::native::swa403::detail403 {
inline void blas(cublasStatus_t code){
    if(code!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma403 SWA cuBLAS");
}
// One body for the production Empty observer and an independent cold unit.
// Observer hooks see actual RN16 Q/K/V/P_h/P_l and actual GEMM outputs before
// their storage is reused. A cold oracle must consume/copy them on this stream.
template<class Observer>
void execute(const float* query,float* output,CacheView cache,const int* position,
    int past,int heads,int tokens,Lease lease,Binding live,Observer& observer){
    validate(query,output,cache,position,past,heads,tokens,lease,live);
    const auto w=lease.workspace;const auto stream=live.stream;
    const auto tile=bind(w.data,w.bytes);
    blas(cublasSetStream(w.handle,stream));
    blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));
    blas(cublasSetMathMode(w.handle,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float one=1.f,zero=0.f,correction=1.f/4096.f;
    constexpr long long kv_stride=KeyTile*Dim;
    constexpr long long query_stride=HeadsPerKv*QueryTile*Dim;
    constexpr long long score_stride=HeadsPerKv*QueryTile*KeyTile;
    try {
        for(int first=0;first<tokens;first+=QueryTile){
            native::detail::initialize_tile(tile.f32,Heads,Dim,stream);
            pack_query(tile,query,position,past,first,stream);
            observer.query(tile,first);
            const int begin=(std::max)(0,past+first-Window+1);
            const int last=past+first+QueryTile-1;
            for(int key=begin;key<=last;key+=KeyTile){
                const int keys=(std::min)(KeyTile,last-key+1);
                pack_kv(tile,cache,key,keys,stream);
                // The proven195 D128 bound is retained: two ordered FP32
                // score accumulations, while each KV head serves both Q heads.
                for(int d=0;d<Dim;d+=128)
                    blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,
                        keys,HeadsPerKv*QueryTile,128,&one,tile.key+d,CUDA_R_16F,Dim,kv_stride,
                        tile.query+d,CUDA_R_16F,Dim,query_stride,d?&one:&zero,
                        tile.f32.scores,CUDA_R_32F,KeyTile,score_stride,KvHeads,
                        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                observer.qk(tile,first,key,keys);
                native::detail::normalize_tile(tile.f32,cache,position,first,key,keys,QueryTile,stream);
                pack_probability(tile,keys,stream);
                observer.probability(tile,first,key,keys);
                const half* planes[]={tile.probability,tile.residual};
                for(int plane=0;plane<2;++plane)
                    blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                        Dim,HeadsPerKv*QueryTile,keys,plane?&correction:&one,
                        tile.value,CUDA_R_16F,Dim,kv_stride,planes[plane],CUDA_R_16F,KeyTile,score_stride,
                        plane?&one:&zero,tile.f32.partial,CUDA_R_32F,Dim,query_stride,KvHeads,
                        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                observer.pv(tile,first,key,keys);
                native::detail::accumulate_tile(tile.f32,Heads,QueryTile,Dim,stream);
            }
            native::detail::finish_tile(tile.f32,output,Heads,QueryTile,Dim,first,stream);
            observer.finish(tile,output,first);
        }
    } catch(...) {
        (void)cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH);throw;
    }
    blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    primitive::check_cuda(cudaGetLastError(),"Gemma403 SWA complete attention");
}
}
