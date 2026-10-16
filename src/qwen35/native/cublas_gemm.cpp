#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cublas_v2.h>
#include <stdexcept>
#include <string>
#include <climits>

namespace strata::qwen35::native {
namespace {
void checked(cublasStatus_t status,const char *operation){
    if(status!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation)+": cuBLAS status "+std::to_string(int(status)));
}
class CublasGemm final:public DenseGemm {
    cublasHandle_t handle_=nullptr;
    void *workspace_=nullptr;
    size_t bytes_=0;
    cudaStream_t stream_=nullptr;
    int device_=0;
    unsigned long long calls_=0;
    bool paired_=false;
public:
    CublasGemm(cudaStream_t stream,size_t bytes,bool paired=false):bytes_(bytes),stream_(stream),paired_(paired){
        if(bytes<16*1024)throw std::invalid_argument("cuBLAS workspace too small");
        check_cuda(cudaGetDevice(&device_),"cuBLAS execution device");
        try{
            check_cuda(cudaMalloc(&workspace_,bytes_),"cuBLAS owned workspace");
            checked(cublasCreate(&handle_),"cuBLAS handle");
            checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST),"cuBLAS scalar mode");
            checked(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH),"cuBLAS compute policy");
            checked(cublasSetStream(handle_,stream_),"cuBLAS stream");
            checked(cublasSetWorkspace(handle_,workspace_,bytes_),"cuBLAS workspace binding");
        }catch(...){if(handle_)cublasDestroy(handle_);cudaFree(workspace_);throw;}
    }
    ~CublasGemm()override{
        // Drain the borrowed stream while caller operand arenas are still alive.
        cudaStreamSynchronize(stream_);
        if(handle_)cublasDestroy(handle_);
        cudaFree(workspace_);
    }
    void project(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t stream)override{
        if(!w||!x||!y||r<1||k<1||t<1||stream!=stream_)
            throw std::invalid_argument("cuBLAS borrowed stream/operand geometry mismatch");
        // Stream and workspace are bound once; SetStream would reset workspace.
        const float alpha=1.0f,beta=0.0f;
        checked(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,r,t,k,
            &alpha,w,CUDA_R_16F,k,x,CUDA_R_16F,k,&beta,y,CUDA_R_32F,r,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"cuBLAS Dense GEMM");
        ++calls_;
    }
    size_t workspace_bytes()const override{return bytes_;}
    unsigned long long calls()const override{return calls_;}
    GemmWorkspace borrowed_workspace()const override{return {workspace_,bytes_,stream_};}
    bool project_pair(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t stream)override{
        if(!paired_)return false;
        if(r<1||r>INT_MAX/2)throw std::invalid_argument("concatenated GEMM row extent invalid");
        project(w,x,y,2*r,k,t,stream);return true;
    }
    void select_pair(bool enabled)override{paired_=enabled;}
    void batched_fp16(const void *a,const void *b,float *c,GemmBatch p,cudaStream_t stream)override{
        if(!a||!b||!c||stream!=stream_||p.m<1||p.n<1||p.k<1||p.batches<1)
            throw std::invalid_argument("cuBLAS batched operands/stream mismatch");
        checked(cublasGemmStridedBatchedEx(handle_,p.transpose_a?CUBLAS_OP_T:CUBLAS_OP_N,
            p.transpose_b?CUBLAS_OP_T:CUBLAS_OP_N,p.m,p.n,p.k,&p.alpha,
            a,CUDA_R_16F,p.lda,p.stride_a,b,CUDA_R_16F,p.ldb,p.stride_b,
            &p.beta,c,CUDA_R_32F,p.ldc,p.stride_c,p.batches,CUBLAS_COMPUTE_32F,
            CUBLAS_GEMM_DEFAULT_TENSOR_OP),"cuBLAS batched attention GEMM");
        ++calls_;
    }
};
}
std::unique_ptr<DenseGemm> make_cublas_gemm(cudaStream_t stream,size_t bytes){
    return std::make_unique<CublasGemm>(stream,bytes);
}
std::unique_ptr<DenseGemm> make_cublas_pair_gemm(cudaStream_t stream,size_t bytes){
    return std::make_unique<CublasGemm>(stream,bytes,true);
}
}
