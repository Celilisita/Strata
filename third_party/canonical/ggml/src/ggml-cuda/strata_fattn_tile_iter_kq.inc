template <int warp_size, int nwarps, int ncols1, int ncols2, int DKQ, int nbatch_fa, int nbatch_K,
    bool use_logit_softcap, bool oob_check, typename T_vec_dot, typename T_KV>
static __device__ __forceinline__ void flash_attn_tile_iter_KQ(
        T_vec_dot   * const Q_tmp,
        const T_KV * const __restrict__ K_h2,
        T_vec_dot   * const KV_tmp,
        const int stride_K2,
        const int k_VKQ_0,
        const int k_VKQ_sup,
        const int k_KQ_0,
        float * KQ_acc,const int32_t* selected) {
    constexpr int cpy_nb = ggml_cuda_get_max_cpy_bytes();
    constexpr int cpy_ne = cpy_nb / 4;

    constexpr int ncols = ncols1*ncols2;
    constexpr int cpw   = ncols > nwarps ? ncols/nwarps : 1; // Q columns per warp
    constexpr int np    = nwarps > ncols ? nwarps/ncols : 1; // number of parallel warps per Q column

#ifdef STRATA_GLM53_F32_FLASH_TILE
    if constexpr(sizeof(T_KV)==sizeof(block_q8_0)) {
        flash_attn_tile_load_tile<warp_size,nwarps,nbatch_fa,nbatch_K,cpy_ne,oob_check>
            (K_h2+(selected?0:int64_t(k_VKQ_0)*stride_K2)+k_KQ_0/QK8_0,KV_tmp,stride_K2,k_VKQ_sup,selected?selected+k_VKQ_0:nullptr);
    } else
#endif
    if constexpr(sizeof(T_KV)==sizeof(float2)
#ifdef STRATA_GLM53_F32_FLASH_TILE
                 || sizeof(T_KV)==sizeof(half2)
#endif
    ) {
        flash_attn_tile_load_tile<warp_size,nwarps,nbatch_fa,nbatch_K,cpy_ne,oob_check>
            (K_h2+(selected?0:int64_t(k_VKQ_0)*stride_K2)+k_KQ_0/2,KV_tmp,stride_K2,k_VKQ_sup,selected?selected+k_VKQ_0:nullptr);
    } else {
    flash_attn_tile_load_tile<warp_size, nwarps, nbatch_fa, nbatch_K, cpy_ne, oob_check>
        (K_h2 + int64_t(k_VKQ_0)*stride_K2 + k_KQ_0/2, KV_tmp, stride_K2, k_VKQ_sup);
    }
    __syncthreads();

#ifdef FAST_FP16_AVAILABLE
    static_assert((nbatch_K/2) % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K/2; k_KQ_1 += cpy_ne) {
        __align__(16) half2 K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) half2 Q_k[cpw][cpy_ne];
#else
    static_assert(nbatch_K % cpy_ne == 0, "bad nbatch_K");
#pragma unroll
    for (int k_KQ_1 = 0; k_KQ_1 < nbatch_K; k_KQ_1 += cpy_ne) {
        __align__(16) float K_k[nbatch_fa/(np*warp_size)][cpy_ne];
        __align__(16) float Q_k[cpw][cpy_ne];
#endif // FAST_FP16_AVAILABLE

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
            const int i_KQ = i_KQ_0 + (threadIdx.y % np)*warp_size + threadIdx.x;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K/2 + cpy_ne) + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&K_k[i_KQ_0/(np*warp_size)], &KV_tmp[i_KQ*(nbatch_K   + cpy_ne) + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }
#pragma unroll
        for (int jc0 = 0; jc0 < cpw; ++jc0) {
            const int jc = jc0 + (threadIdx.y / np)*cpw;

#ifdef FAST_FP16_AVAILABLE
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc*(DKQ/2) + k_KQ_0/2 + k_KQ_1]);
#else
            ggml_cuda_memcpy_1<cpy_nb>(&Q_k[jc0], &Q_tmp[jc* DKQ    + k_KQ_0   + k_KQ_1]);
#endif // FAST_FP16_AVAILABLE
        }

#pragma unroll
        for (int i_KQ_0 = 0; i_KQ_0 < nbatch_fa; i_KQ_0 += np*warp_size) {
#pragma unroll
            for (int jc0 = 0; jc0 < cpw; ++jc0) {
#pragma unroll
                for (int k = 0; k < cpy_ne; ++k) {
                    ggml_cuda_mad(KQ_acc[i_KQ_0/(np*warp_size)*cpw + jc0], K_k[i_KQ_0/(np*warp_size)][k], Q_k[jc0][k]);
                }
            }
        }
    }

    if (k_KQ_0 + nbatch_K < DKQ) {
        __syncthreads(); // Sync not needed on last iteration.
    }
}

