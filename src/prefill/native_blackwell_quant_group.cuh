// Lossless integer digit representation of one GGUF quantization group.
// Only F32 reduction order changes; scales, minima and activation codes do not.
template<class Block>
__device__ uint2 affine_group_meta(const Block& w, int g) {
    const auto* sc=w.scales;
    const unsigned s=g<4 ? sc[g]&63 : (sc[g+4]&15)|((sc[g-4]>>6)<<4);
    const unsigned m=g<4 ? sc[g+4]&63 : (sc[g+4]>>4)|((sc[g]>>6)<<4);
    return make_uint2(*reinterpret_cast<const unsigned*>(&w.dm),s|(m<<8));
}
struct GroupQ4 {
    using Block = Q4KBlock;
    static constexpr int K = 32, AD8 = 1, AD4 = 2;
    __device__ static int code(const Block& w, int g, int k) {
        return (w.qs[(g/2)*32+k] >> (4*(g%2))) & 15;
    }
    __device__ static uint2 meta(const Block& w, int g) {
        return affine_group_meta(w,g);
    }
    __device__ static float finish(uint2 w, float d8, int sum, float dot) {
        const float d=__half2float(__ushort_as_half(w.x & 65535));
        const float m=__half2float(__ushort_as_half(w.x >> 16));
        const float a=__fmul_rn(d,__fmul_rn(d8,__fmul_rn(dot,float(w.y & 255))));
        const float b=__fmul_rn(m,__fmul_rn(d8,float(sum*int(w.y >> 8))));
        return __fsub_rn(a,b);
    }
};
// Centering keeps all Q5 codes exactly representable by a single E4M3 digit.
// Restore the original integer dot before the unchanged affine scale/min math.
struct GroupQ5 {
    using Block = Q5KBlock;
    static constexpr int K = 32, AD8 = 1, AD4 = 2;
    __device__ static int code(const Block& w, int g, int k) {
        const int low=(w.qs[(g/2)*32+k] >> (4*(g%2))) & 15;
        return (low | (((w.qh[k] >> g) & 1) << 4))-16;
    }
    __device__ static uint2 meta(const Block& w, int g) {
        return affine_group_meta(w,g);
    }
    __device__ static float finish(uint2 w, float d8, int sum, float dot) {
        const float original=__fadd_rn(dot,__fmul_rn(16.0f,float(sum)));
        return GroupQ4::finish(w,d8,sum,original);
    }
};
struct GroupQ6 {
    using Block = Q6KBlock;
    static constexpr int K = 16, AD8 = 2, AD4 = 2;
    __device__ static int code(const Block& w, int g, int k) {
        const int pos=g*16+k, half=pos/128, slot=(pos%128)/32, lane=pos%32;
        const int lo=(w.ql[half*64+(slot%2)*32+lane] >> (4*(slot/2))) & 15;
        const int hi=(w.qh[half*32+lane] >> (2*slot)) & 3;
        return (lo|(hi<<4))-32;
    }
    __device__ static uint2 meta(const Block& w, int g) {
        return make_uint2(unsigned(*reinterpret_cast<const uint16_t*>(&w.d)) |
            (unsigned(uint8_t(w.scales[g]))<<16),0);
    }
    __device__ static float finish(uint2 w, float d8, int, float dot) {
        const float d=__half2float(__ushort_as_half(w.x & 65535));
        return __fmul_rn(d,__fmul_rn(d8,__fmul_rn(dot,float(int(int8_t(w.x >> 16))))));
    }
};
struct GroupInputMeta { float d8; int sum; };
template<bool Fp4, bool F16=false, class Code>
__device__ unsigned group_code_word(Code code, int word, int digit, int digits) {
    constexpr int E=F16 ? 2 : Fp4 ? 8 : 4;
    unsigned result=0;
#pragma unroll
    for(int p=0;p<E/2;++p) {
        if constexpr(F16) result |= unsigned(__half_as_ushort(__float2half_rn(float(code(word*E+2*p))))) |
            (unsigned(__half_as_ushort(__float2half_rn(float(code(word*E+2*p+1)))))<<16);
        else result |= blackwell_convert_pair<Fp4>(
        blackwell_component<Fp4>(code(word*E+2*p),digit,digits),
        blackwell_component<Fp4>(code(word*E+2*p+1),digit,digits)) << ((Fp4 ? 8 : 16)*p);
    }
    return result;
}
