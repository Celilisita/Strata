#include "strata/prefill/gemm.hpp"
#include <cublas_v2.h>
#include <stdexcept>
namespace strata::prefill {
void Gemm::f16(const uint16_t* X, const uint16_t* W, float* Y, int64_t T, int64_t N,
               int64_t K, int64_t ldy, float beta, float alpha) {
    if (alpha == 1) { f16(X, W, Y, T, N, K, ldy, beta); return; }
    if (T <= 0 || N <= 0) return;
    if (ldy <= 0) ldy = N;
    auto status = cublasGemmEx(static_cast<cublasHandle_t>(handle_), CUBLAS_OP_T, CUBLAS_OP_N,
        int(N), int(T), int(K), &alpha, W, CUDA_R_16F, int(K), X, CUDA_R_16F, int(K), &beta,
        Y, CUDA_R_32F, int(ldy), CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    if (status != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("scaled FP16 GEMM failed");
}
}
