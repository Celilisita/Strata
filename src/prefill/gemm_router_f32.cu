#include "strata/prefill/gemm.hpp"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace strata::prefill {
namespace {
void check_router_blas(cublasStatus_t status) {
    if(status!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error("router F32 cuBLAS status "+std::to_string(int(status)));
}
}
bool Gemm::router_f32(const float* X,const float* W,float* Y,int tokens) {
    if(tokens<1024 || tokens>8192) return false;
    if(!X || !W || !Y || !handle_ || !stream_ || !workspace_)
        throw std::invalid_argument("uninitialized router F32 GEMM buffers/owner");
    const void* pointers[3]={X,W,Y};
    const size_t sizes[3]={size_t(tokens)*2048*4,size_t(257)*2048*4,size_t(tokens)*257*4};
    for(int i=0;i<3;++i) {
        const auto a=reinterpret_cast<uintptr_t>(pointers[i]);
        if((a&3) || sizes[i]>UINTPTR_MAX-a)
            throw std::invalid_argument("router F32 GEMM alignment/span");
        for(int j=0;j<i;++j) {
            const auto b=reinterpret_cast<uintptr_t>(pointers[j]);
            if(a<b+sizes[j] && b<a+sizes[i]) throw std::invalid_argument("router F32 GEMM alias");
        }
    }
    const auto handle=static_cast<cublasHandle_t>(handle_);
    cudaStream_t actual_stream=nullptr;cublasMath_t math{};cublasPointerMode_t pointer_mode{};
    check_router_blas(cublasGetStream(handle,&actual_stream));
    check_router_blas(cublasGetMathMode(handle,&math));
    check_router_blas(cublasGetPointerMode(handle,&pointer_mode));
    if(actual_stream!=static_cast<cudaStream_t>(stream_) || math!=CUBLAS_DEFAULT_MATH ||
        pointer_mode!=CUBLAS_POINTER_MODE_HOST) return false;
    const float alpha=1.0f,beta=0.0f;
    // Column-major view: W[K,257]^T * X[K,T] -> Y[257,T],
    // exactly the caller's original token-major Y[T,257] storage.
    check_router_blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,257,tokens,2048,
        &alpha,W,2048,X,2048,&beta,Y,257));
    return true;
}
}
