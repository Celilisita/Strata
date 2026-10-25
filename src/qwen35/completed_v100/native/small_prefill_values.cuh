#pragma once
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"
#include <cuda_fp16.h>

namespace strata::completed_v100::qwen35::native::detail {
struct alignas(4) TokenBlock {
    float delta;
    half rounded_delta, original_sum;
    signed char codes[32];
};
static_assert(sizeof(TokenBlock)==SmallPrefillActivationBytes,"owned PP block extent");
template<bool Aligned=false>
__device__ inline unsigned word(const unsigned char *p){
    if constexpr(Aligned)return *reinterpret_cast<const unsigned *>(p);
    const auto *v=reinterpret_cast<const unsigned short *>(p);
    return unsigned(v[0])|(unsigned(v[1])<<16);
}
template<Quant Type,bool Aligned=false>
__device__ int weight_codes(const unsigned char *p,int group,int position){
    if constexpr(Type==Quant::Q4K)
        return int((word<Aligned>(p+16+(group/2)*32+position)>>((group%2)*4))&0x0f0f0f0fu);
    else{
        const int section=group/4,plane=group%4;
        const unsigned low=word<Aligned>(p+section*64+(plane%2)*32+position);
        const unsigned high=word<Aligned>(p+128+section*32+position);
        const unsigned codes=((low>>(plane>=2?4:0))&0x0f0f0f0fu)|(((high>>(plane*2))&0x03030303u)<<4);
        return int((codes+0x60606060u)^0x80808080u);
    }
}
__device__ inline unsigned metadata(const unsigned char *p,int group,bool prepared){
    if(prepared){
        const int offset=4+(group/2)*3;
        return (unsigned(p[offset])|(unsigned(p[offset+1])<<8)|(unsigned(p[offset+2])<<16))>>((group%2)*12);
    }
    const int q=group&3;
    const unsigned scale=group<4?p[4+q]&63:(p[12+q]&15)|((p[4+q]>>6)<<4);
    const unsigned minimum=group<4?p[8+q]&63:(p[12+q]>>4)|((p[8+q]>>6)<<4);
    return scale|(minimum<<6);
}
}
