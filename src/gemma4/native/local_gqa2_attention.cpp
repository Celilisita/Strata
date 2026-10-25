#include "local_gqa2_ops.hpp"
#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native::local_gqa2_candidate {
namespace {
void blas(cublasStatus_t s){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Gemma PP TC GemmBatchedEx");}
struct Span{const void* p;std::size_t n;};
void spans(const Span* all,int count){
    for(int i=0;i<count;++i){const auto x=reinterpret_cast<std::uintptr_t>(all[i].p);
        if(!x||!all[i].n||x>std::numeric_limits<std::uintptr_t>::max()-all[i].n)
            throw std::invalid_argument("Gemma PP TC live span overflow");
        for(int j=0;j<i;++j){const auto y=reinterpret_cast<std::uintptr_t>(all[j].p);
            if(x<y+all[j].n&&y<x+all[i].n)throw std::invalid_argument("Gemma PP TC live spans alias");}
    }
}
}
void pp_tc_attention(const float* q,float* out,CacheView c,const int* position,int past,int tokens,
                     int maximum,AttentionWorkspace w,cudaStream_t stream){
    const bool local=c.dim==256&&c.heads==16&&c.window==1024;
    const bool global=c.dim==512&&c.heads==4&&c.window==0;
    if(!stream||!position||(!local&&!global)||past<0||tokens<4||tokens>maximum||maximum>8192||
       past>262144-tokens||!w.handle||!w.blas_data||!w.blas_bytes)
        throw std::invalid_argument("Gemma PP TC exact geometry/full history");
    const int required=local?std::min(past+tokens,1024+tokens-1):past+tokens;
    if(c.capacity<required||c.capacity>262144)throw std::invalid_argument("Gemma PP TC retained KV extent");
    const std::size_t qbytes=std::size_t(tokens)*32*c.dim*4,kvbytes=std::size_t(c.capacity)*c.heads*(c.dim/32)*34;
    const Span live[]={{q,qbytes},{out,qbytes},{c.key,kvbytes},{c.value,kvbytes},{position,4},{w.data,w.bytes},{w.blas_data,w.blas_bytes}};
    spans(live,7);using namespace pp_tc_detail;const auto b=pp_tc_detail::bind(w.data,w.bytes,maximum);
    blas(cublasSetStream(w.handle,stream));blas(cublasSetWorkspace(w.handle,w.blas_data,w.blas_bytes));
    blas(cublasSetPointerMode(w.handle,CUBLAS_POINTER_MODE_HOST));
    blas(cublasSetMathMode(w.handle,static_cast<cublasMath_t>(CUBLAS_TENSOR_OP_MATH|CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
    const float one=1,zero=0;
    const bool grouped=LocalGroupedGqa2&&local&&(tokens==2048||tokens==1328);
    try{
        pp_tc_detail::prepare(b,q,out,tokens,c.dim,grouped,stream);
        const int first=local?std::max(0,past-1023):0,last=past+tokens-1;
        for(int key=first;key<=last;key+=Keys){
            const int keys=std::min(Keys,last-key+1);pp_tc_detail::decode(b,c,key,keys,stream);
            for(int query=0;query<tokens;query+=Query){
                const int queries=std::min(Query,tokens-query);
                const int columns=grouped?queries*2:queries,batches=grouped?16:Heads;
                if(key>past+query+queries-1||(local&&key+keys-1<past+query-1023))continue;
                pp_tc_detail::pointers(b,c,tokens,query,grouped,stream);
                blas(cublasGemmBatchedEx(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,keys,columns,c.dim,&one,
                    b.kt,CUDA_R_16F,c.dim,b.qt,CUDA_R_16F,c.dim,&zero,b.st,CUDA_R_32F,Keys,batches,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                pp_tc_detail::normalize(b,c,position,query,key,keys,queries,grouped,stream);
                blas(cublasGemmBatchedEx(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,c.dim,columns,keys,&one,
                    b.vt,CUDA_R_16F,c.dim,b.pt,CUDA_R_16F,Keys,&zero,b.ot,CUDA_R_32F,c.dim,batches,
                    CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
                pp_tc_detail::accumulate(b,out,query,queries,c.dim,grouped,stream);
            }
        }
        pp_tc_detail::finish(b,out,tokens,c.dim,stream);
    }catch(...){(void)cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH);throw;}
    blas(cublasSetMathMode(w.handle,CUBLAS_DEFAULT_MATH));
    primitive::check_cuda(cudaGetLastError(),"Gemma PP TC key-outer result");
}
}
