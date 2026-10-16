#pragma once
#include <cuda_fp16.h>

namespace strata::qwen35::native {
namespace {
// NVIDIA PTX m8n8k4 F16 register mapping: each quad-pair computes one 8x8 tile.
// A is feature x key (row major); B is key x query (column major).
struct VoltaValueAccumulator {
    unsigned c[4][4]{};
    __device__ void clear(){
        #pragma unroll
        for(int a=0;a<4;++a){
            #pragma unroll
            for(int b=0;b<4;++b)c[a][b]=0;
        }
    }
    __device__ static unsigned pack(__half lo,__half hi){
        return unsigned(__half_as_ushort(lo)) | (unsigned(__half_as_ushort(hi))<<16);
    }
    __device__ void rescale(const float *scale,int warp){
        const int query_base=(warp/2)*8;
        #pragma unroll
        for(int slot=0;slot<4;++slot){
            #pragma unroll
            for(int pair=0;pair<4;++pair){
                const __half lo=__ushort_as_half(c[slot][pair]&65535);
                const __half hi=__ushort_as_half(c[slot][pair]>>16);
                c[slot][pair]=pack(__hmul(lo,__float2half_rn(scale[query_base+pair*2])),
                                  __hmul(hi,__float2half_rn(scale[query_base+pair*2+1])));
            }
        }
    }
    __device__ void accumulate(const __half *v,const __half *p,int warp,int lane,int part){
#if __CUDA_ARCH__ == 700
        const int query=(warp/2)*8+lane%4+(lane>=16?4:0);
        const int feature_base=(warp%2)*128+((lane/4)%4)*8+lane%4+(lane>=16?4:0);
        #pragma unroll
        for(int key=0;key<32;key+=4){
            if((key/8)%2!=part)continue;
            const unsigned b0=pack(p[query*32+key],p[query*32+key+1]);
            const unsigned b1=pack(p[query*32+key+2],p[query*32+key+3]);
            #pragma unroll
            for(int slot=0;slot<4;++slot){
                const int f=feature_base+slot*32;
                const unsigned a0=pack(v[key*256+f],v[(key+1)*256+f]);
                const unsigned a1=pack(v[(key+2)*256+f],v[(key+3)*256+f]);
                asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 "
                    "{%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%0,%1,%2,%3};"
                    : "+r"(c[slot][0]),"+r"(c[slot][1]),"+r"(c[slot][2]),"+r"(c[slot][3])
                    : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
            }
        }
#endif
    }
    __device__ void store(__half *out,int warp,int lane){
        const int query_base=(warp/2)*8;
        const int feature_base=(warp%2)*128+((lane/4)%4)*8+lane%4+(lane>=16?4:0);
        #pragma unroll
        for(int slot=0;slot<4;++slot){
            const int f=feature_base+slot*32;
            #pragma unroll
            for(int pair=0;pair<4;++pair){
                out[(query_base+pair*2)*256+f]=__ushort_as_half(c[slot][pair]&65535);
                out[(query_base+pair*2+1)*256+f]=__ushort_as_half(c[slot][pair]>>16);
            }
        }
    }
};
}
}
