#include "whole_history.hpp"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
std::size_t aligned(std::size_t n){return(n+255)&~std::size_t(255);}
int key_bound(int upper){if(upper<1||upper>262144)throw std::invalid_argument("wholehistory actual extent");return(upper+255)&~255;}
void valid_tokens(int tokens){if(tokens<4||tokens>2048)throw std::invalid_argument("wholehistory PP batch4..2048");}
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("wholehistory GemmBatchedEx");}
struct Span{const void* data;std::size_t bytes;};
void disjoint(const Span* all,int count){for(int i=0;i<count;++i){const auto x=reinterpret_cast<std::uintptr_t>(all[i].data);
    if(!x||!all[i].bytes||all[i].bytes>std::numeric_limits<std::uintptr_t>::max()-x)throw std::invalid_argument("wholehistory span");
    for(int j=0;j<i;++j){const auto y=reinterpret_cast<std::uintptr_t>(all[j].data);
        if(x<y+all[j].bytes&&y<x+all[i].bytes)throw std::invalid_argument("wholehistory input/output/arena alias");}}}
}
bool whole_history_requested(){static const bool enabled=[](){const char* v=std::getenv("STRATA_GEMMA_PP_WHOLE_HISTORY");
    if(!v||std::strcmp(v,"0")==0)return false;if(std::strcmp(v,"1")==0)return true;
    throw std::invalid_argument("STRATA_GEMMA_PP_WHOLE_HISTORY requires0or1");}();return enabled;}
std::size_t whole_history_workspace_bytes(int tokens,int upper){valid_tokens(tokens);using namespace whole_history_detail;const int keys=key_bound(upper);
    return 256+2*aligned(std::size_t(keys)*KvHeads*Dim*2)+aligned(std::size_t(tokens)*Heads*Dim*2)+
        aligned(std::size_t(Heads)*Query*keys*4)+aligned(std::size_t(Heads)*Query*keys*2)+
        aligned(std::size_t(Heads)*Query*Dim*4)+aligned(std::size_t(Heads)*Query*4)+aligned(6*Heads*sizeof(void*));}
namespace whole_history_detail {
Buffers bind(AttentionWorkspace work,int tokens,int upper){const auto bytes=whole_history_workspace_bytes(tokens,upper);
    if(!work.data||reinterpret_cast<std::uintptr_t>(work.data)%256||work.bytes<bytes)throw std::invalid_argument("wholehistory owned arena");
    Buffers b{};b.keys=key_bound(upper);auto* p=static_cast<unsigned char*>(work.data);b.invalid=reinterpret_cast<unsigned*>(p);p+=256;
    const auto take=[&](std::size_t n){void* result=p;p+=aligned(n);return result;};
    b.key=take(std::size_t(b.keys)*KvHeads*Dim*2);b.value=take(std::size_t(b.keys)*KvHeads*Dim*2);
    b.query=take(std::size_t(tokens)*Heads*Dim*2);b.score=static_cast<float*>(take(std::size_t(Heads)*Query*b.keys*4));
    b.probability=take(std::size_t(Heads)*Query*b.keys*2);b.partial=static_cast<float*>(take(std::size_t(Heads)*Query*Dim*4));
    b.den=static_cast<float*>(take(std::size_t(Heads)*Query*4));b.pointers=static_cast<void**>(take(6*Heads*sizeof(void*)));return b;
}
}
void whole_history_attention(const float* q,float* out,CacheView cache,const int* position,int past,
        int tokens,int maximum,AttentionWorkspace work,cudaStream_t stream){
    using namespace whole_history_detail;
    if(!stream||!q||!out||!position||cache.dim!=Dim||cache.heads!=KvHeads||cache.window||tokens<4||tokens>maximum||maximum>2048||
       past<0||cache.capacity<tokens||past>cache.capacity-tokens||!work.handle||!work.blas_data||!work.blas_bytes)
        throw std::invalid_argument("wholehistory global PP geometry/owner");
    const auto qbytes=std::size_t(tokens)*Heads*Dim*4,kvbytes=std::size_t(cache.capacity)*KvHeads*(Dim/32)*34;
    const Span spans[]={{q,qbytes},{out,qbytes},{cache.key,kvbytes},{cache.value,kvbytes},{position,4},{work.data,work.bytes},{work.blas_data,work.blas_bytes}};
    disjoint(spans,7);const auto b=bind(work,tokens,past+tokens);
    blas(cublasSetStream(work.handle,stream));blas(cublasSetWorkspace(work.handle,work.blas_data,work.blas_bytes));
    blas(cublasSetPointerMode(work.handle,CUBLAS_POINTER_MODE_HOST));
    blas(cublasSetMathMode(work.handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float one=1,zero=0;
    try{stage(b,q,cache,position,tokens,stream);
        for(int first=0;first<tokens;first+=Query){const int queries=std::min(Query,tokens-first);pointers(b,first,tokens,stream);
            blas(cublasGemmBatchedEx(work.handle,CUBLAS_OP_T,CUBLAS_OP_N,b.keys,queries,Dim,&one,
                reinterpret_cast<const void* const*>(b.pointers),CUDA_R_16F,Dim,
                reinterpret_cast<const void* const*>(b.pointers+Heads),CUDA_R_16F,Dim,&zero,
                reinterpret_cast<void* const*>(b.pointers+2*Heads),CUDA_R_32F,b.keys,Heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            normalize(b,position,first,queries,tokens,cache.capacity,stream);
            blas(cublasGemmBatchedEx(work.handle,CUBLAS_OP_N,CUBLAS_OP_N,Dim,queries,b.keys,&one,
                reinterpret_cast<const void* const*>(b.pointers+3*Heads),CUDA_R_16F,Dim,
                reinterpret_cast<const void* const*>(b.pointers+4*Heads),CUDA_R_16F,b.keys,&zero,
                reinterpret_cast<void* const*>(b.pointers+5*Heads),CUDA_R_32F,Dim,Heads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
            finish(b,out,first,queries,stream);
        }propagate(b,out,tokens,stream);
    }catch(...){(void)cublasSetMathMode(work.handle,CUBLAS_DEFAULT_MATH);throw;}
    blas(cublasSetMathMode(work.handle,CUBLAS_DEFAULT_MATH));
}
}
