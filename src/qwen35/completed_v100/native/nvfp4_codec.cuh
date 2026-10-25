#pragma once
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>

namespace strata::completed_v100::qwen35::native::nvfp4 {
// Original GGUF type40: 64 values, four UE4M3 scales, 32 E2M1 code bytes.
// Global and optional input scales remain independent tensor metadata; callers
// must apply them at the matrix boundary. Neither is silently absorbed here.
constexpr unsigned BlockValues=64, BlockBytes=36;

__device__ __forceinline__ float block_scale(unsigned char code) {
    if(code==0||code==0x7f)return 0.f;
    const unsigned exponent=(code>>3)&15, mantissa=code&7;
    const float magnitude=exponent?__uint_as_float(((exponent+120)<<23)|(mantissa<<20)):
                                   float(mantissa)*(1.f/512.f);
    return magnitude*.5f;
}

__device__ __forceinline__ int doubled_value(unsigned nibble) {
    const unsigned magnitude=nibble&7;
    const int value=magnitude<4?int(magnitude):int(2*magnitude-4);
    // Magnitudes0..7 are 0,1,2,3,4,6,8,12 (twice E2M1).
    const int corrected=magnitude==7?12:value;
    return (nibble&8)?-corrected:corrected;
}

__device__ __forceinline__ float value(const unsigned char* block,unsigned within) {
    const unsigned group=within/16, j=within%16;
    const unsigned packed=block[4+group*8+(j&7)];
    return float(doubled_value((packed>>((j/8)*4))&15))*block_scale(block[group]);
}

__device__ __forceinline__ float matrix_value(const void* payload,int columns,int row,int column) {
    const size_t index=size_t(row)*columns+column;
    const auto* block=static_cast<const unsigned char*>(payload)+(index/BlockValues)*BlockBytes;
    return value(block,unsigned(index%BlockValues));
}
}
