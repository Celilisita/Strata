#pragma once
#include <cuda_runtime.h>

namespace strata::qwen35::native::detail {
// Produce a Q6 row's210 bytes and14 padding bytes. The caller can use a
// larger shared pitch; those extra bytes are not read. Raw bases are16-byte aligned. A two-byte lookbehind is inside the same inventory
// because the first block's address is aligned; the final partial word is bounded.
__device__ inline void copy_raw_q6_row(const unsigned char *source,unsigned *destination,
                                     int lane,bool valid){
    const bool shifted=(reinterpret_cast<uintptr_t>(source)&2u)!=0;
    const auto *aligned=reinterpret_cast<const unsigned *>(source-(shifted?2:0));
    unsigned one=0,two=0;
    if(valid){
        one=__ldg(aligned+lane);
        if(lane<20 || (shifted && lane==20))two=__ldg(aligned+32+lane);
        else if(lane==20)two=__ldg(reinterpret_cast<const unsigned short *>(source)+104);
    }
    if(shifted){
        // Warp neighbors already hold the next aligned word; do not issue a
        // second payload load for every two-byte-offset word.
        const unsigned next=__shfl_down_sync(0xffffffffu,one,1);
        const unsigned bridge=__shfl_sync(0xffffffffu,two,0);
        const unsigned tail=__shfl_down_sync(0xffffffffu,two,1);
        one=__funnelshift_r(one,lane==31?bridge:next,16);
        two=lane<20?__funnelshift_r(two,tail,16):lane==20?two>>16:0;
    }
    destination[lane]=one;
    if(lane<24)destination[32+lane]=two;
}
}
