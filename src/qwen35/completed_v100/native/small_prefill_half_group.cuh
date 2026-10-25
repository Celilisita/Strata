#pragma once
#include <cuda_fp16.h>

namespace strata::completed_v100::qwen35::native::detail {
__device__ inline unsigned rounded_half_bits(float value,float reciprocal){
    const float rounded=roundf(__fmul_rn(value,reciprocal));
    const unsigned bits=__half_as_ushort(__float2half_rn(rounded));
    return (bits&0x7fffu)?bits:0u;
}
// One8-lane subgroup owns32 values. Used by standalone and fused producers.
template<class Block,bool Affine=false>
__device__ inline void encode_half_group(float4 v,Block *output,int group,int token,int token_tiles,int lane){
    float maximum=fmaxf(fmaxf(fabsf(v.x),fabsf(v.y)),fmaxf(fabsf(v.z),fabsf(v.w)));
    float sum=0;
    if constexpr(Affine)sum=__fadd_rn(__fadd_rn(__fadd_rn(v.x,v.y),v.z),v.w);
    for(int offset=4;offset;offset/=2){
        maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffffu,maximum,offset));
        if constexpr(Affine)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffffu,sum,offset));
    }
    const float reciprocal=maximum?__fdividef(127.0f,maximum):0;
    const float delta=maximum?__fdividef(1.0f,reciprocal):0;
    const unsigned x=rounded_half_bits(v.x,reciprocal),y=rounded_half_bits(v.y,reciprocal);
    const unsigned z=rounded_half_bits(v.z,reciprocal),w=rounded_half_bits(v.w,reciprocal);
    auto &block=output[(group/8)*token_tiles+token/32];
    const int word=(group%8)*16+lane*2;
    *reinterpret_cast<uint2 *>(block.codes[token%32]+word)=make_uint2(x|(y<<16),z|(w<<16));
    if(!lane){
        if constexpr(Affine)block.ds[group%8][token%32]=unsigned(__half_as_ushort(__float2half_rn(delta)))|
            (unsigned(__half_as_ushort(__float2half_rn(sum)))<<16);
        else block.delta[group%8][token%32]=delta;
    }
}
}
