#pragma once
#include "small_prefill_mma.cuh"

namespace strata::qwen35::native::detail::scaled {
constexpr int Rows=64,Tokens=32,Threads=256,RowTiles=Rows/8;
constexpr int SharedBytes=(Rows+Tokens)*HalfOperandPitch*4+8*Tokens*4+
                          Rows*8*4+RowTiles*8*8*4+RowTiles*8*4+Rows*4;

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
__device__ inline unsigned shared_half(unsigned base,int column){
    unsigned value;
    const unsigned address=base+unsigned(column)*2;
    asm volatile("ld.shared.u16 %0, [%1];" : "=r"(value) : "r"(address) : "memory");
    return value;
}
// Four independent8x8 matrices use their own8-row K union. Paired row
// quadrants run the same padded K4 count; absent union slots have zero A.
// Add integer residuals directly into the high dot: exact below2^24, with
// no separate correction accumulator or per-output floating addition.
__device__ inline void residual_dot(const unsigned *bitmaps,const unsigned *indices,
                                    const unsigned *counts,const unsigned *input,
                                    int group,int local_group,int warp,int lane,float (&sum)[8]){
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=(warp/2)*16+(matrix/2)*8+local;
    const int token=(warp%2)*16+(matrix%2)*8+local,tile=row/8;
    const unsigned mask=bitmaps[row*8+group];
    const unsigned *list=indices+(tile*8+group)*8;
    const int steps=(max(int(counts[tile*8+group]),int(counts[(tile^1)*8+group]))+3)/4;
    const unsigned source=unsigned(__cvta_generic_to_shared(input+token*HalfOperandPitch+local_group*16));
    for(int step=0;step<steps;++step){
        const unsigned packed=list[step];
        const unsigned k0=packed&255,k1=(packed>>8)&255,k2=(packed>>16)&255,k3=packed>>24;
        const unsigned c0=(k0&128)?0:((mask>>(k0&31))&1)*0x3c00u;
        const unsigned c1=(k1&128)?0:((mask>>(k1&31))&1)*0x3c00u;
        const unsigned c2=(k2&128)?0:((mask>>(k2&31))&1)*0x3c00u;
        const unsigned c3=(k3&128)?0:((mask>>(k3&31))&1)*0x3c00u;
        const unsigned a0=c0|(c1<<16),a1=c2|(c3<<16);
        const unsigned b0=__byte_perm(shared_half(source,k0&31),shared_half(source,k1&31),0x5410);
        const unsigned b1=__byte_perm(shared_half(source,k2&31),shared_half(source,k3&31),0x5410);
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(sum[0]),"+f"(sum[1]),"+f"(sum[2]),"+f"(sum[3]),
                       "+f"(sum[4]),"+f"(sum[5]),"+f"(sum[6]),"+f"(sum[7])
                     : "r"(a0),"r"(a1),"r"(b0),"r"(b1));
    }
}
// Descending original K positions, four byte indices per word. Padding0x80
// maps to legal input column0 but suppresses the residual coefficient.
__device__ inline void build_lists(const unsigned *bitmaps,unsigned *indices,unsigned *counts,int tid){
    if(tid>=RowTiles*8)return;
    const int tile=tid/8,group=tid%8;
    unsigned remaining=0;
    #pragma unroll
    for(int row=0;row<8;++row)remaining|=bitmaps[(tile*8+row)*8+group];
    counts[tid]=__popc(remaining);
    for(int step=0;step<8;++step){
        unsigned packed=0x80808080u;
        #pragma unroll
        for(int item=0;item<4;++item){
            if(remaining){
                const unsigned column=31-__clz(remaining);
                remaining&=~(1u<<column);
                packed=(packed&~(255u<<(item*8)))|(column<<(item*8));
            }
        }
        indices[tid*8+step]=packed;
    }
}
#endif
}
