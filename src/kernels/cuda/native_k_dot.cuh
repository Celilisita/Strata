// Canonical pinned Q4_K/Q6_K scale/min expressions. Integer-dot providers
// (DP4A and FP16 WMMA) share this arithmetic and its operation order.
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
__device__ __forceinline__ float q4_q8_dot_impl(
    const int* __restrict__ v, const int* __restrict__ u,
    const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m, const half2& dm4,
    const float* __restrict__ d8) {
    int dots[2], sums[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = __dp4a(v1i, u[2 * i + 1], __dp4a(v0i, u[2 * i], 0));
        const int dot2 = __dp4a(0x01010101, u[2 * i + 1], __dp4a(0x01010101, u[2 * i], 0));
        dots[i] = dot1;
        sums[i] = dot2;
    }
    return q4_q8_finalize(dots, sums, sc, m, dm4, d8);
}

// Exact pinned vec_dot_q6_K_q8_1: keep signed per-16-element scales,
// signed-byte subtraction, DP4A order, and the float accumulation sequence.
__device__ __forceinline__ float q6_q8_dot_impl(int vl, int vh, const int* __restrict__ u,
                                              const int8_t* __restrict__ scales,
                                              float d, const float* __restrict__ d8) {
    int dots[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = __vsubss4(vil | vih, 0x20202020);
        dots[i] = __dp4a(vi, u[i], 0);
    }
    return q6_q8_finalize(dots, scales, d, d8);
}

