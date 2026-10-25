// Exact integer digits, not quantization of dequantized model values.
struct BwQ4 : PackedQ4 {
    static constexpr int K = 8, WeightDigits8 = 1, WeightDigits4 = 2;
    __device__ static int weight(const W& w, int p, int k) {
        return (unsigned(w.v[k/4]) >> (4*p+8*(k%4))) & 15;
    }
    __device__ static int code(const Input& x, int p, int k) {
        return int(int8_t(unsigned(x.u[2*p+k/4]) >> (8*(k%4))));
    }
    template<class Dot>
    __device__ static float finish(const W& w, const Input& x, const Dot* dots, const int* sums) {
        const auto* sc = reinterpret_cast<const uint8_t*>(w.aux);
        return q4_q8_finalize(dots,sums,sc,sc+2,w.dm,x.d8);
    }
};
struct BwQ6 : PackedQ6 {
    static constexpr int K = 4, WeightDigits8 = 2, WeightDigits4 = 2;
    __device__ static int weight(const W& w, int p, int k) {
        const unsigned low = (unsigned(w.vl) >> (4*p+8*k)) & 15;
        const unsigned high = ((unsigned(w.vh) >> (4*p+8*k)) & 3) << 4;
        return int(low | high)-32;
    }
    __device__ static int code(const Input& x, int p, int k) {
        return int(int8_t(unsigned(x.u[p]) >> (8*k)));
    }
    template<class Dot>
    __device__ static float finish(const W& w, const Input& x, const Dot* dots, const int*) {
        return q6_q8_finalize(dots,w.scales,w.d,x.d8);
    }
};
__device__ __forceinline__ int radix16_digit(int x, int digit) {
    return digit ? x >> 4 : x & 15;
}
template<bool Fp4>
__device__ __forceinline__ void blackwell_mma(const unsigned* a, const unsigned* b, float* d, int ad, int bd) {
    if constexpr (Fp4) {
        // Exact radix powers are carried by hardware UE8M0 block scales.
        const unsigned sa = 0x01010101u*unsigned(127+3*ad);
        const unsigned sb = 0x01010101u*unsigned(127+3*bd);
        asm("mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
    } else {
        asm("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
            : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
            : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
    }
}
