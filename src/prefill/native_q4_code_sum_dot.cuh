// Pinned ggml Q4_K MMQ integer products + shared canonical native K finalize.
// DS upper 16 bits carry an exact signed code sum, never an F16 value.
namespace strata_native_k_math {
#include "../kernels/cuda/native_k_dot.cuh"
}

template<int J, bool fallback>
static __device__ __forceinline__ void strata_q4_code_sum_dot(
    const int* __restrict__ x, const int* __restrict__ y,
    float* __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps = ggml_cuda_mmq_get_nthreads(GGML_TYPE_Q4_K, J, fallback) / warp_size;
    constexpr int I = ggml_cuda_mmq_get_I(GGML_TYPE_Q4_K, J, fallback);
    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    const int* x_qs = x;
    const half2* x_dm = reinterpret_cast<const half2*>(x_qs) + txs.qs;
    const int* x_sc = reinterpret_cast<const int*>(x_dm) + txs.dm;
    const int* y_qs = y + 4;
    const half2* y_ds = reinterpret_cast<const half2*>(y);
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K * VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;
#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;
#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const auto* sc = reinterpret_cast<const uint8_t*>(&x_sc[i * (MMQ_TILE_NE_K / 8) + i / 8 + k0 / 32]) + 2 * (k01 / 16);
                const int* v = &x_qs[i * (MMQ_TILE_NE_K + 1) + k0 / 2];
                const int* u = &y_qs[j * MMQ_TILE_Y_K + k01];
                const half2* ds = &y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1];
                int dots[2]{}, code_sums[2]; float d8[2];
#pragma unroll
                for (int part = 0; part < 2; ++part) {
#pragma unroll
                    for (int k = 0; k < QI8_1; ++k)
                        dots[part] = ggml_cuda_dp4a((v[k] >> (4 * part)) & 0x0f0f0f0f, u[part * QI8_1 + k], dots[part]);
                    code_sums[part] = reinterpret_cast<const int16_t*>(ds)[2 * part + 1];
                    d8[part] = __low2float(ds[part]);
                }
                sum[j0 / nwarps * I / warp_size + i0 / warp_size] +=
                    strata_native_k_math::q4_q8_finalize(dots, code_sums, sc, sc + 8, x_dm[i], d8);
            }
        }
    }
}

#define STRATA_CODE_SUM_DOT(J, F) \
    template<> __device__ constexpr ggml_cuda_mmq_vec_dot_t \
    strata_code_sum_get_vec_dot<GGML_TYPE_Q4_K, J, F>() { return strata_q4_code_sum_dot<J, F>; }
#define STRATA_CODE_SUM_J(J) STRATA_CODE_SUM_DOT(J, false) STRATA_CODE_SUM_DOT(J, true)
STRATA_CODE_SUM_J(8) STRATA_CODE_SUM_J(16) STRATA_CODE_SUM_J(24) STRATA_CODE_SUM_J(32)
STRATA_CODE_SUM_J(40) STRATA_CODE_SUM_J(48) STRATA_CODE_SUM_J(56) STRATA_CODE_SUM_J(64)
STRATA_CODE_SUM_J(72) STRATA_CODE_SUM_J(80) STRATA_CODE_SUM_J(88) STRATA_CODE_SUM_J(96)
STRATA_CODE_SUM_J(104) STRATA_CODE_SUM_J(112) STRATA_CODE_SUM_J(120) STRATA_CODE_SUM_J(128)
#undef STRATA_CODE_SUM_J
#undef STRATA_CODE_SUM_DOT
