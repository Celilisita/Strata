#include "attention_half_ops.hpp"
#include <algorithm>

namespace strata::gemma4::native {
namespace {
void check_blas(cublasStatus_t status){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma FP16 PP cuBLAS attention");
}
void validate_shape(int heads,int dim){
    if(heads!=32||(dim!=256&&dim!=512))
        throw std::invalid_argument("Gemma FP16 attention tile shape");
}
}

namespace detail {
size_t half_attention_tile_bytes(int heads,int dim){
    validate_shape(heads,dim);
    const size_t kv=size_t(heads)*KeyTile*dim,out=size_t(heads)*QueryTile*dim;
    const size_t scores=size_t(heads)*KeyTile*QueryTile,rows=size_t(heads)*QueryTile;
    return (2*kv+2*out+scores+4*rows)*sizeof(float)+(2*kv+out+2*scores)*sizeof(half);
}

HalfAttentionTile bind_half_attention_tile(void* data,size_t bytes,int heads,int dim){
    if(!data||half_attention_tile_bytes(heads,dim)>bytes)
        throw std::invalid_argument("Gemma FP16 tiled attention arena too small");
    const size_t kv=size_t(heads)*KeyTile*dim,out=size_t(heads)*QueryTile*dim;
    const size_t scores=size_t(heads)*KeyTile*QueryTile,rows=size_t(heads)*QueryTile;
    auto* at=static_cast<float*>(data);
    auto take_float=[&](size_t n){auto* p=at;at+=n;return p;};
    AttentionTile f32{take_float(kv),take_float(kv),take_float(scores),
        take_float(out),take_float(out),take_float(rows),take_float(rows),
        take_float(rows),take_float(rows)};
    auto* encoded=reinterpret_cast<half*>(at);
    auto take_half=[&](size_t n){auto* p=encoded;encoded+=n;return p;};
    return {f32,take_half(out),take_half(kv),take_half(kv),take_half(scores),take_half(scores)};
}
}

void half_blas_attention(const float* q,float* output,CacheView c,const int* position,
    int past,int heads,int tokens,AttentionWorkspace w,cudaStream_t stream){
    if(!q||!output||q==output||!position||!c.key||!c.value||heads!=32||
       c.heads<1||heads%c.heads||c.capacity<tokens||tokens<1||tokens>8192||past<0||
       c.window<0||(c.dim!=256&&c.dim!=512)||!w.data||!w.handle||!w.blas_data)
        throw std::invalid_argument("Gemma FP16 tiled PP attention contract");
    const int required=c.window?(std::min)(past+tokens,c.window+tokens-1):past+tokens;
    if(required>c.capacity)throw std::invalid_argument("Gemma FP16 PP history no longer retained");
    using namespace detail;
    HalfAttentionTile tile=bind_half_attention_tile(w.data,w.bytes,heads,c.dim);
    check_blas(cublasSetStream(w.handle,stream));
    check_blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    check_blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));
    check_blas(cublasSetMathMode(w.handle,static_cast<cublasMath_t>(
        CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float one=1.f,zero=0.f,probability_correction=1.f/4096.f;
    const long long kv_stride=KeyTile*c.dim,score_stride=KeyTile*QueryTile;
    const long long out_stride=QueryTile*c.dim;
    for(int query_first=0;query_first<tokens;query_first+=QueryTile){
        const int queries=(std::min)(QueryTile,tokens-query_first);
        const int first=c.window?(std::max)(0,past+query_first-c.window+1):0;
        const int last=past+query_first+queries-1;
        initialize_tile(tile.f32,heads,c.dim,stream);
        pack_half_query(tile,q,query_first,queries,heads,c.dim,stream);
        for(int key_first=first;key_first<=last;key_first+=KeyTile){
            const int keys=(std::min)(KeyTile,last-key_first+1);
            unpack_tile(tile.f32,c,key_first,keys,heads,stream);
            pack_half_kv(tile,keys,heads,c.dim,stream);
            // Bound the Tensor Core dot accumulation before the FP16
            // probability quantizer. Preserve the original matrix strides.
            for(int dot_first=0;dot_first<c.dim;dot_first+=128){
                const int dot_count=(std::min)(128,c.dim-dot_first);
                check_blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,
                    keys,queries,dot_count,&one,tile.k+dot_first,CUDA_R_16F,c.dim,kv_stride,
                    tile.q+dot_first,CUDA_R_16F,c.dim,out_stride,dot_first?&one:&zero,
                    tile.f32.scores,CUDA_R_32F,KeyTile,score_stride,heads,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            }
            normalize_tile(tile.f32,c,position,query_first,key_first,keys,queries,stream);
            pack_half_probabilities(tile,keys,queries,heads,stream);
            check_blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                c.dim,queries,keys,&one,tile.v,CUDA_R_16F,c.dim,kv_stride,
                tile.probabilities,CUDA_R_16F,KeyTile,score_stride,&zero,
                tile.f32.partial,CUDA_R_32F,c.dim,out_stride,heads,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            check_blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,
                c.dim,queries,keys,&probability_correction,tile.v,CUDA_R_16F,c.dim,kv_stride,
                tile.probability_residual,CUDA_R_16F,KeyTile,score_stride,&one,
                tile.f32.partial,CUDA_R_32F,c.dim,out_stride,heads,
                CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            accumulate_tile(tile.f32,heads,queries,c.dim,stream);
        }
        finish_tile(tile.f32,output,heads,queries,c.dim,query_first,stream);
    }
    check_blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    primitive::check_cuda(cudaGetLastError(),"Gemma bounded FP16 cuBLAS PP attention tiles");
}
}
