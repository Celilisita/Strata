#pragma once
#include <cuda_fp16.h>

namespace strata::gemma4::native::detail245 {
// Only raw GGUF Q4_K blocks: base4 alignment and144-byte stride are checked
// by the caller. This is202's aligned decode, with unchanged nibble values.
__device__ inline int codes(const unsigned char* raw, int group, int position) {
    const unsigned packed = *reinterpret_cast<const unsigned*>(raw + 16 + (group / 2) * 32 + position);
    return int((packed >> ((group % 2) * 4)) & 0x0f0f0f0fu);
}
__device__ inline unsigned metadata(const unsigned char* raw, int group) {
    const unsigned shift = unsigned(group & 3) * 8;
    const unsigned a = (*reinterpret_cast<const unsigned*>(raw + 4) >> shift) & 255u;
    const unsigned b = (*reinterpret_cast<const unsigned*>(raw + 8) >> shift) & 255u;
    if (group < 4) return (a & 63u) | ((b & 63u) << 6);
    const unsigned high = (*reinterpret_cast<const unsigned*>(raw + 12) >> shift) & 255u;
    return ((high & 15u) | ((a >> 6) << 4)) |
        (((high >> 4) | ((b >> 6) << 4)) << 6);
}
}
