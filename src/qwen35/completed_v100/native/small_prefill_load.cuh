#pragma once
#include <cuda_runtime.h>

namespace strata::completed_v100::qwen35::native::detail {
// All workers of a participating warp call this, including masked rows.
// A210B block may begin2B off alignment; obtain the next aligned word from
// the adjacent worker, using one extra global load at each warp boundary.
template<int Bytes>
__device__ inline unsigned load_raw_word(const unsigned char *p,int word,bool active){
    constexpr int Words=(Bytes+3)/4;
    const unsigned offset=unsigned(reinterpret_cast<unsigned long long>(p)&2);
    const auto *aligned=reinterpret_cast<const unsigned *>(p-offset);
    unsigned low=0;
    if(active){
        if constexpr(Bytes%4==2){
            if(word==Words-1&&!offset)low=*reinterpret_cast<const unsigned short *>(p+Bytes-2);
            else low=aligned[word];
        }else low=aligned[word];
    }
    if constexpr(Bytes%4==2){
        unsigned next=__shfl_down_sync(0xffffffff,low,1);
        if(active&&offset&&word<Words-1){
            if(threadIdx.x%32==31)next=aligned[word+1];
            return (low>>16)|(next<<16);
        }
        if(word==Words-1)return (offset?low>>16:low)&65535u;
    }
    return low;
}
}
