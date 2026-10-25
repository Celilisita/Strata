#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
namespace strata::moe_nvfp4 {
enum class Quant : uint32_t {Q4K=12,Q5K=13,Q6K=14};
enum class BlockLayout {Gguf};
// Original GGUF K-blocks only. Explicit operations retain source F32 rounding;
// no device repack, F16 weight cast, shared scale, or MoE route semantics here.
__device__ inline float block_element(const unsigned char* p,Quant type,int j,BlockLayout) {
    if(type==Quant::Q4K||type==Quant::Q5K){
        const int group=j/32,part=group&3;
        const int scale=group<4?p[4+part]&63:(p[12+part]&15)|((p[4+part]>>6)<<4);
        const int minimum=group<4?p[8+part]&63:(p[12+part]>>4)|((p[8+part]>>6)<<4);
        const int offset=type==Quant::Q5K?48:16;
        int code=(p[offset+(j/64)*32+j%32]>>((group&1)*4))&15;
        if(type==Quant::Q5K)code|=((p[16+j%32]>>group)&1)<<4;
        const float a=__fmul_rn(__half2float(*reinterpret_cast<const half*>(p)),float(scale));
        const float b=__fmul_rn(__half2float(*reinterpret_cast<const half*>(p+2)),float(minimum));
        return __fsub_rn(__fmul_rn(a,float(code)),b);
    }
    const int section=j/128,group=(j%128)/32,lane=j%32;
    const int low=(p[section*64+(group&1)*32+lane]>>((group/2)*4))&15;
    const int high=(p[128+section*32+lane]>>(group*2))&3;
    const int code=(low|(high<<4))-32;
    const int scale=reinterpret_cast<const signed char*>(p+192)[j/16];
    return __fmul_rn(__fmul_rn(__half2float(*reinterpret_cast<const half*>(p+208)),float(scale)),float(code));
}
} // namespace strata::moe_nvfp4
