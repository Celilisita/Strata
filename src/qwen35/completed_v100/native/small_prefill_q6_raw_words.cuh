#pragma once
namespace strata::completed_v100::qwen35::native::detail {
template<bool Aligned>
__device__ __forceinline__ unsigned raw_q6_word(const unsigned char *p){
    if constexpr(Aligned)return __ldg(reinterpret_cast<const unsigned *>(p));
    else{
        // The caller supplies a2-byte aligned address. Previous/next two
        // bytes are inside the owned GGUF allocation, never outside a tensor.
        const auto *words=reinterpret_cast<const unsigned *>(p-2);
        const unsigned lo=__ldg(words),hi=__ldg(words+1);
        return __funnelshift_r(lo,hi,16);
    }
}
template<bool Aligned>
__device__ __forceinline__ unsigned raw_q6_codes(const unsigned char *p,int group,int position){
    const int section=group/4,plane=group%4;
    const unsigned low=raw_q6_word<Aligned>(p+section*64+(plane%2)*32+position);
    const unsigned high=raw_q6_word<Aligned>(p+128+section*32+position);
    const unsigned codes=((low>>(plane>=2?4:0))&0x0f0f0f0fu)|(((high>>(plane*2))&0x03030303u)<<4);
    return (codes+0x60606060u)^0x80808080u;
}
}
