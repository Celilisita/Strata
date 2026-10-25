// CUDA12.8 headers use this PTX operand order: low element is the first input.
// All radix components are exactly representable; conversion adds no rounding.
template<bool Fp4>
__device__ __forceinline__ float blackwell_component(int x, int digit, int digits) {
    if constexpr (Fp4) {
        // Balanced radix8 low digits stay in [-4,3]; the last digit stays in [-4,4], exactly representable in E2M1.
        // Q4/Q6 weights need two digits; signed Q8 inputs need three.
#pragma unroll
        for (int i = 0; i < digit; ++i) x = (x+4) >> 3;
        return float(digit == digits-1 ? x : ((x+4)&7)-4);
    } else {
        // One-digit weights are exact signed E4M3 (Q4:0..15, Q5:-16..15).
        // Q8 activations/Q6 retain their existing two radix16 digits.
        return digits==1 ? float(x) : float(radix16_digit(x,digit)*(digit ? 16 : 1));
    }
}
template<bool Fp4>
__device__ __forceinline__ unsigned blackwell_convert_pair(float a, float b) {
    uint16_t packed;
    if constexpr (Fp4) {
        asm("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t,%2,%1; mov.b16 %0,{t,0}; }"
            : "=h"(packed) : "f"(a),"f"(b));
    } else {
        asm("cvt.rn.satfinite.e4m3x2.f32 %0,%2,%1;"
            : "=h"(packed) : "f"(a),"f"(b));
    }
    return packed;
}
template<class F, bool Fp4, bool Weight, bool Int8 = false>
__device__ __forceinline__ unsigned blackwell_code_word(const typename F::W& w,
    const typename F::Input& x, int word, int digit) {
    constexpr int E = Fp4 ? 8 : 4;
    constexpr int Digits = Fp4 ? (Weight ? F::WeightDigits4 : 3) : 2;
    unsigned result = 0;
#pragma unroll
    for (int pair = 0; pair < E/2; ++pair) {
        const int k0 = word*E+2*pair, k1 = k0+1;
        int a, b;
        if constexpr (Weight) {
            a = F::weight(w,k0/F::K,k0%F::K); b = F::weight(w,k1/F::K,k1%F::K);
        } else {
            a = F::code(x,k0/F::K,k0%F::K); b = F::code(x,k1/F::K,k1%F::K);
        }
        if constexpr (Int8) {
            result |= (unsigned(a)&255u) << (16*pair);
            result |= (unsigned(b)&255u) << (16*pair+8);
        } else result |= blackwell_convert_pair<Fp4>(blackwell_component<Fp4>(a,digit,Digits),
            blackwell_component<Fp4>(b,digit,Digits)) << ((Fp4 ? 8 : 16)*pair);
    }
    return result;
}
