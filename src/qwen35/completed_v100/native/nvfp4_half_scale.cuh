#pragma once
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native::nvfp4 {
// UE4M3/2 equals its FP16 bit widening times128. Mask only the canonical
// 0x7f code; preserve the original treatment of all256 possible input bytes.
__device__ __forceinline__ float half_scale(unsigned char code) {
    const unsigned short bits=static_cast<unsigned short>(code==127?0:(unsigned(code)&127u)<<7);
    return __half2float(__ushort_as_half(bits))*128.f;
}
}
