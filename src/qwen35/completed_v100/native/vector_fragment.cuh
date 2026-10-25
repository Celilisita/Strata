#pragma once
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <cstdint>
namespace strata::completed_v100::qwen35::native::detail {
struct alignas(4) ActivationBlock {
    half delta;
    unsigned short padding;
    signed char codes[32];
};
static_assert(sizeof(ActivationBlock) == VectorActivationBytes, "owned activation extent");
static_assert(offsetof(ActivationBlock, codes) == 4, "aligned activation words");
__device__ __forceinline__ uint32_t bytes4(const unsigned char *p) {
    // Q6 blocks/offsets have two-byte alignment, even when not four-byte aligned.
    const auto *words = reinterpret_cast<const unsigned short *>(p);
    return uint32_t(words[0]) | (uint32_t(words[1]) << 16);
}
struct Codes { int first, second; };
template<Quant Type,bool Tiled=false>
__device__ __forceinline__ Codes packed_pair(const unsigned char *p, int group, int position,const unsigned char *tile_codes) {
    if constexpr (Type == Quant::Q4K) {
        const auto *codes=Tiled?tile_codes+(group/2)*32+position:p+16+(group/2)*32+position;
        const uint32_t word = *reinterpret_cast<const uint32_t *>(codes);
        return {int(word & 0x0f0f0f0fu), int((word >> 4) & 0x0f0f0f0fu)};
    } else {
        const int section = group / 4, plane = group % 4;
        const uint32_t high = bytes4(p + 128 + section * 32 + position);
        const uint32_t low = bytes4(p + section * 64 + plane * 32 + position);
        const uint32_t code0 = (low & 0x0f0f0f0fu) | (((high >> (plane * 2)) & 0x03030303u) << 4);
        const uint32_t code1 = ((low >> 4) & 0x0f0f0f0fu) | (((high >> (plane * 2 + 4)) & 0x03030303u) << 4);
        return {int((code0 + 0x60606060u) ^ 0x80808080u), int((code1 + 0x60606060u) ^ 0x80808080u)};
    }
}
// Two logical groups share their encoded plane; decode it once. Each lane owns
// sixteen values, so eight superblocks are in flight in one four-warp CTA.
template<Quant Type, bool PackedMetadata,bool Tiled=false>
__device__ __forceinline__ float accumulate_fragment(const unsigned char *p, const ActivationBlock *x, float previous,const unsigned char *tile_codes) {
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
        const int offset = 4 + pair * 3;
        metadata = unsigned(p[offset]) | (unsigned(p[offset + 1]) << 8) | (unsigned(p[offset + 2]) << 16);
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
