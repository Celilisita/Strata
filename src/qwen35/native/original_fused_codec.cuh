#pragma once
#include "nvfp4_codec.cuh"
#include <cuda_fp16.h>

namespace strata::qwen35::native::original_fused {
// Logical K permutes each original group16 to permit four half2 outputs from
// one uint32. Activations use the same permutation; original weight bytes stay.
__host__ __device__ __forceinline__ int nvfp4_k(int k) {
    const int j=k&15;
    return (k&~15)|((j&1)<<1)|((j&2)<<2)|((j&4)>>2)|((j&8)>>1);
}
__device__ __forceinline__ void nvfp4_eight_values(unsigned scale_code,unsigned word,half* output) {
    // UE4M3 widened into FP16 /256; 0x7f is the canonical masked scale.
    const half scale=__ushort_as_half(static_cast<unsigned short>(scale_code==127?0:scale_code<<7));
    const half2 scaled=__hmul2(__halves2half2(scale,scale),__float2half2_rn(16384.f));
    // E2M1 bit widening: FP4/2^14, then *256 gives FP4/64. The scale above
    // contributes UE4M3*64. Products exactly recover original FP4*UE4M3.
    // Bit-widening reference: vllm/csrc/quantization/marlin/dequant.h.
    #pragma unroll
    for(int pair=0;pair<4;++pair) {
        const unsigned q=word<<(12-4*pair);
        const unsigned bits=(q&0x80008000u)|((q&0x70007000u)>>3);
        const half2 tiny=__halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),
            __ushort_as_half(static_cast<unsigned short>(bits>>16)));
        reinterpret_cast<half2*>(output)[pair]=__hmul2(__hmul2(tiny,__float2half2_rn(256.f)),scaled);
    }
}
__device__ __forceinline__ void nvfp4_eight(const unsigned char* raw,int columns,
    int row,int column,half* output) {
    const auto* block=raw+size_t(row)*(columns/64)*36+size_t(column/64)*36;
    const int group=(column>>4)&3;
    nvfp4_eight_values(__ldg(block+group),
        __ldg(reinterpret_cast<const unsigned*>(block+4+group*8+(column&8)/2)),output);
}
// Lossless cold layout: eight rows x32K stores32 words contiguously, then
// a separate scale plane stores16 bytes for the same tile. No FP conversion.
__device__ __forceinline__ void nvfp4_tiled_eight(const unsigned char* raw,int columns,
    int rows,int row,int column,half* output) {
    const size_t tile=size_t(row/8)*(columns/32)+column/32;
    const unsigned word=__ldg(reinterpret_cast<const unsigned*>(raw)+tile*32+(row&7)*4+((column&31)/8));
    const unsigned scale=__ldg(raw+size_t(rows)*columns/2+tile*16+(row&7)*2+((column&31)/16));
    nvfp4_eight_values(scale,word,output);
}
// Adjacent logical columns contain W_high and 4096*(W-W_high). An int8 times
// FP16 scale has an exactly FP16-representable residual after FP16 rounding.
__device__ __forceinline__ void q8_eight(const unsigned char* raw,int columns,
    int logical_row,int column,half* output) {
    const int row=logical_row/2;const bool residual=logical_row&1;
    const auto* block=raw+(size_t(row)*(columns/32)+column/32)*34;
    const half scale=__ldg(reinterpret_cast<const half*>(block));
    const half2 scales=__halves2half2(scale,scale);
    const auto* pairs=reinterpret_cast<const unsigned short*>(block+2+(column&31));
    #pragma unroll
    for(int pair=0;pair<4;++pair) {
        const unsigned p=__ldg(pairs+pair)^0x8080u;
        const unsigned bits=__byte_perm(p,0x64646464u,0x5140);
        const half2 biased=__halves2half2(__ushort_as_half(static_cast<unsigned short>(bits)),
            __ushort_as_half(static_cast<unsigned short>(bits>>16)));
        const half2 q=__hsub2(biased,__float2half2_rn(1152.f));
        const half2 high=__hmul2(q,scales);
        reinterpret_cast<half2*>(output)[pair]=residual?
            __hmul2(__hfma2(q,scales,__hneg2(high)),__float2half2_rn(4096.f)):high;
    }
}
}
