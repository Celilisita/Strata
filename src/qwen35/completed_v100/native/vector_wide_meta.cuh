#pragma once
#include "vector_fragment.cuh"
namespace strata::completed_v100::qwen35::native::detail {
template<Quant Type, bool PackedMetadata,bool Tiled=false>
__device__ __forceinline__ float accumulate_wide_fragment(const unsigned char *p, const ActivationBlock *x, float previous,const unsigned char *tile_codes) {
    constexpr int Workers=Type==Quant::Q4K?16:32;
    const int local=threadIdx.x%Workers;
    const int pair=Type==Quant::Q4K?local/4:local/8;
    const int position=(local%(Type==Quant::Q4K?4:8))*4;
    const int group = Type == Quant::Q4K ? pair * 2 : (pair / 2) * 4 + pair % 2;
    constexpr int Distance = Type == Quant::Q4K ? 1 : 2;
    int product[2] = {}, activation_sum[2] = {};
    // Q4 fragments pair four values at position and position+16. Keep
    // this numeric grouping separate from contiguous encoded-plane loads.
    for (int i=0;i<(Type==Quant::Q4K?32:4);i+=(Type==Quant::Q4K?16:4)) {
        const auto codes = packed_pair<Type,Tiled>(p, group, position + i,tile_codes);
        const int a0 = *reinterpret_cast<const int *>(x[group].codes + position + i);
        const int a1 = *reinterpret_cast<const int *>(x[group + Distance].codes + position + i);
        product[0] = __dp4a(codes.first, a0, product[0]);
        product[1] = __dp4a(codes.second, a1, product[1]);
        if constexpr (Type == Quant::Q4K) {
            activation_sum[0] = __dp4a(0x01010101, a0, activation_sum[0]);
            activation_sum[1] = __dp4a(0x01010101, a1, activation_sum[1]);
        }
    }
    float scaled = 0, correction = 0;
    unsigned metadata = 0;
    if constexpr (Type == Quant::Q4K && PackedMetadata) {
        const int byte=pair*3,aligned=byte&~3;
        const unsigned low=*reinterpret_cast<const unsigned *>(p+4+aligned);
        const unsigned high=*reinterpret_cast<const unsigned *>(p+8+aligned);
        metadata=__funnelshift_r(low,high,(byte&3)*8);
    }
    for (int part = 0; part < 2; ++part) {
        const int g = group + part * Distance;
        const float delta = __half2float(x[g].delta);
        if constexpr (Type == Quant::Q4K) {
            const int q = g & 3;
            int scale, minimum;
            if constexpr (PackedMetadata) {
                const unsigned value = metadata >> (part * 12);
                scale = value & 63; minimum = (value >> 6) & 63;
            } else {
                scale = g < 4 ? p[4 + q] & 63 : (p[12 + q] & 15) | ((p[4 + q] >> 6) << 4);
                minimum = g < 4 ? p[8 + q] & 63 : (p[12 + q] >> 4) | ((p[8 + q] >> 6) << 4);
            }
            scaled = __fmaf_rn(delta, float(scale * product[part]), scaled);
            correction = __fmaf_rn(delta, float(minimum * activation_sum[part]), correction);
        } else {
            const int scale = int(reinterpret_cast<const signed char *>(p + 192)[g * 2 + position / 16]);
            scaled = __fmaf_rn(delta, float(scale * product[part]), scaled);
        }
    }
    if constexpr (Type == Quant::Q4K) {
        const float d=__half2float(*reinterpret_cast<const half *>(p));
        const float minimum=__half2float(*reinterpret_cast<const half *>(p+2));
        const float term=__fmaf_rn(d,scaled,-__fmul_rn(minimum,correction));
        return __fadd_rn(previous, term);
    } else return __fmaf_rn(__half2float(*reinterpret_cast<const half *>(p + 208)), scaled, previous);
}
}
