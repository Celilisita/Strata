#include "attention_blas_ops.hpp"
#include <algorithm>
namespace strata::gemma4::native {
namespace {
void check_blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma PP cuBLAS attention");}
}
void blas_attention(const float* q,float* output,CacheView c,const int* position,int past,int heads,int tokens,AttentionWorkspace w,cudaStream_t stream){
    if(!q||!output||q==output||!position||!c.key||!c.value||heads!=32||c.heads<1||heads%c.heads||c.capacity<tokens||tokens<1||tokens>8192||past<0||
        c.window<0||(c.dim!=256&&c.dim!=512)||!w.data||!w.handle||!w.blas_data)throw std::invalid_argument("Gemma tiled PP attention contract");
    const int required=c.window?(std::min)(past+tokens,c.window+tokens-1):past+tokens;
    if(required>c.capacity)throw std::invalid_argument("Gemma PP history no longer retained");
    using namespace detail;const size_t kv=size_t(heads)*KeyTile*c.dim,out=size_t(heads)*QueryTile*c.dim,scores=size_t(heads)*KeyTile*QueryTile,rows=size_t(heads)*QueryTile;
    const size_t elements=2*kv+2*out+scores+4*rows;
    if(elements*sizeof(float)>w.bytes)throw std::invalid_argument("Gemma tiled attention arena too small");
    auto* at=static_cast<float*>(w.data);auto take=[&](size_t n){auto* p=at;at+=n;return p;};
    AttentionTile b{take(kv),take(kv),take(scores),take(out),take(out),take(rows),take(rows),take(rows),take(rows)};
    check_blas(cublasSetStream(w.handle,stream));check_blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    check_blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));check_blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    const float one=1.f,zero=0.f;const long long kv_stride=KeyTile*c.dim,score_stride=KeyTile*QueryTile,out_stride=QueryTile*c.dim;
    for(int query_first=0;query_first<tokens;query_first+=QueryTile){
        const int queries=(std::min)(QueryTile,tokens-query_first),first=c.window?(std::max)(0,past+query_first-c.window+1):0,last=past+query_first+queries-1;
        initialize_tile(b,heads,c.dim,stream);
        for(int key_first=first;key_first<=last;key_first+=KeyTile){
            const int keys=(std::min)(KeyTile,last-key_first+1);unpack_tile(b,c,key_first,keys,heads,stream);
            check_blas(cublasSgemmStridedBatched(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,keys,queries,c.dim,&one,
                b.k,c.dim,kv_stride,q+size_t(query_first)*heads*c.dim,heads*c.dim,c.dim,&zero,b.scores,KeyTile,score_stride,heads));
            normalize_tile(b,c,position,query_first,key_first,keys,queries,stream);
            check_blas(cublasSgemmStridedBatched(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,c.dim,queries,keys,&one,
                b.v,c.dim,kv_stride,b.scores,KeyTile,score_stride,&zero,b.partial,c.dim,out_stride,heads));
            accumulate_tile(b,heads,queries,c.dim,stream);
        }
        finish_tile(b,output,heads,queries,c.dim,query_first,stream);
    }
    primitive::check_cuda(cudaGetLastError(),"Gemma bounded cuBLAS PP attention tiles");
}
}
