#include "attention_gqa_t2_ops.hpp"
#include <stdexcept>
namespace strata::gemma4::native::gqa371 {
namespace {
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("371 cuBLAS failure");}
class HandleState {
    cublasHandle_t handle_;cublasMath_t math_;cublasPointerMode_t pointer_;bool active_=false;
public:
    HandleState(AttentionWorkspace w,cudaStream_t stream):handle_(w.handle){
        cudaStream_t actual=nullptr;blas(cublasGetStream(handle_,&actual));
        if(actual!=stream)throw std::invalid_argument("371 handle belongs to another stream");
        blas(cublasGetMathMode(handle_,&math_));blas(cublasGetPointerMode(handle_,&pointer_));active_=true;
        try{
            blas(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST));
            blas(cublasSetMathMode(handle_,static_cast<cublasMath_t>(CUBLAS_DEFAULT_MATH|
                CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)));
            // Never call SetStream: it would reset the explicitly supplied workspace.
            // The owner contract binds this SAME dedicated32MiB buffer to the handle.
            blas(cublasSetWorkspace(handle_,w.blas_data,w.blas_bytes));
        }catch(...){(void)cublasSetMathMode(handle_,math_);(void)cublasSetPointerMode(handle_,pointer_);active_=false;throw;}
    }
    HandleState(const HandleState&)=delete;HandleState& operator=(const HandleState&)=delete;
    ~HandleState(){if(active_){(void)cublasSetMathMode(handle_,math_);(void)cublasSetPointerMode(handle_,pointer_);}}
    void restore(){const auto a=cublasSetMathMode(handle_,math_),b=cublasSetPointerMode(handle_,pointer_);
        blas(a);blas(b);active_=false;}
};
}
void attention(const float* query,float* output,CacheView cache,const int* position,int heads,int tokens,
    AttentionWorkspace w,Lease lease,Binding live,const unsigned* invalid){
    const auto l=layout(cache.capacity);detail::validate(query,output,cache,position,heads,tokens,w,lease,live,invalid,l);
    auto* base=static_cast<unsigned char*>(w.data);
    const auto h=[&](std::size_t at){return reinterpret_cast<half*>(base+at);};
    const auto f=[&](std::size_t at){return reinterpret_cast<float*>(base+at);};
    detail::Tile tile{static_cast<half*>(lease.dead_k),static_cast<half*>(lease.dead_v),h(l.query),h(l.probability),
        f(l.score),f(l.partial),f(l.split_partial),f(l.maximum),f(l.denominator),reinterpret_cast<unsigned*>(base+l.query_bad),
        reinterpret_cast<int*>(base+l.first_bad_key),cache.capacity,l.parts};
    HandleState handle(w,live.stream);const float one=1.f,zero=0.f;
    detail::initialize(tile,live.stream);
    detail::stage_kv(tile,cache,position,invalid,live.stream);
    detail::pack_query(tile,query,position,invalid,live.stream);
    // K_h is [4][C columns][512 rows]. Q_h is [4][16 columns][512 rows].
    // Four positive, disjoint strides; QK=(K_h)^T Q_h, F32 C x16 scores.
    blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_T,CUBLAS_OP_N,
        cache.capacity,Rows,Dim,&one,tile.key,CUDA_R_16F,Dim,l.kv_stride,
        tile.query,CUDA_R_16F,Dim,l.query_stride,&zero,tile.score,CUDA_R_32F,cache.capacity,l.score_stride,
        KvHeads,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    detail::softmax(tile,position,invalid,live.stream);
    // P_h=RN16(exp(score-max)); its unrounded F32 denominator remains separate.
    // Each PV batch reduces one PvKeys tile; positive strides flatten head/part.
    // F32 partials are disjoint. finish merges them; no beta1 RMW/low-plane products.
    blas(cublasGemmStridedBatchedEx(w.handle,CUBLAS_OP_N,CUBLAS_OP_N,
        Dim,Rows,PvKeys,&one,tile.value,CUDA_R_16F,Dim,l.pv_kv_stride,
        tile.probability,CUDA_R_16F,PvKeys,l.pv_probability_stride,&zero,tile.split_partial,CUDA_R_32F,Dim,l.pv_partial_stride,
        KvHeads*l.parts,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    detail::finish(tile,output,position,invalid,live.stream);
    primitive::check_cuda(cudaGetLastError(),"371 physical GQA T2 enqueue");handle.restore();
}
}
