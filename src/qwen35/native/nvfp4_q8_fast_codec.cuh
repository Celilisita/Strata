#pragma once
#include <cuda_runtime.h>
namespace strata::qwen35::native::q8fast {
__device__ __forceinline__ int unpack_fast(unsigned word,bool high) {
    const unsigned bits=high?word>>16:word;
    const unsigned magnitude=__byte_perm(0x03020100,0x0c080604,bits&0x7777);
    unsigned sign;
    // CUDA __byte_perm exposes byte selection, not raw PRMT sign replication.
    asm("prmt.b32 %0, %1, %2, 0xd9c8;" : "=r"(sign) : "r"(bits<<4), "r"(bits));
    return int(__vsub4(magnitude^sign,sign));
}
}
