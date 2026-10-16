__device__ __forceinline__ float q4_q8_finalize(const int* dots, const int* sums,
    const uint8_t* sc, const uint8_t* m, const half2& dm4, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        sumf_d += d8[i] * (dots[i] * sc[i]);
        sumf_m += d8[i] * (sums[i] * m[i]);
    }
    const float2 dm4f = __half22float2(dm4);
    return dm4f.x * sumf_d - dm4f.y * sumf_m;
}

__device__ __forceinline__ float q6_q8_finalize(const int* dots,
    const int8_t* scales, float d, const float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i)
        sumf += d8[i] * (dots[i] * scales[4 * i]);
    return d * sumf;
}

// Exact pinned vec_dot_q4_K_q8_1_impl_vmmq expression and integer dot order.
