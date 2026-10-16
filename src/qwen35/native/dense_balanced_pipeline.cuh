#pragma once
#include "dense_swizzle_stage.cuh"
// Retain the measured 32x32 warp reuse; prefetch next K32 operands separately.
template<bool Pair>
__global__ __launch_bounds__(256,3)
void dense_balanced_pipeline(const half *__restrict__ weights,const half *__restrict__ input,
                             float *__restrict__ first,float *__restrict__ second,
                             int rows,int columns,int tokens,int span){
    constexpr int R=64,C=128,K=32,Stride=32,Stage=(R+C)*Stride;
    __shared__ __align__(32) half shared[2*Stage];
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int feature=int(blockIdx.x)*R,token=int(blockIdx.y)*C;
    const size_t plane=Pair?size_t(blockIdx.z)*rows*columns:0;
    const half *w=weights+plane;
    const int begin=Pair?0:int(blockIdx.z)*span;
    const int end=min(columns,begin+(Pair?columns:span));
    float accum[4][8]{};
#if __CUDA_ARCH__ >= 800
    dense_swizzle_async(w,input,shared,feature,token,rows,columns,tokens,begin,end);
    dense_wait_async();
#else
    dense_swizzle_commit(dense_fetch(w,input,feature,token,rows,columns,tokens,begin,end),shared);
#endif
    __syncthreads();
    int stage=0;
    for(int base=begin;base<end;base+=K){
        const bool more=base+K<end;
#if __CUDA_ARCH__ >= 800
        if(more)dense_swizzle_async(w,input,shared+(stage^1)*Stage,feature,token,rows,columns,tokens,base+K,end);
#else
        DenseFetch pending{};
        if(more)pending=dense_fetch(w,input,feature,token,rows,columns,tokens,base+K,end);
#endif
        half *wa=shared+stage*Stage,*xb=wa+R*Stride;
#if __CUDA_ARCH__ >= 800
        const int a_row=(warp%2)*32+(lane%8)+((lane/8)%2)*8;
        #pragma unroll 1
        for(int k=0;k<K;k+=16){
            uint4 a[2];uint2 b[2][2];
            #pragma unroll
            for(int m=0;m<2;++m)a[m]=dense_matrix4(wa+dense_swizzle_index(a_row+m*16,k+(lane/16)*8));
            #pragma unroll
            for(int n=0;n<2;++n){
                #pragma unroll
                for(int h=0;h<2;++h){
                    const int col=(warp/2)*32+n*16+h*8+lane%8;
                    b[n][h]=dense_matrix2(xb+dense_swizzle_index(col,k+((lane%16)/8)*8));
                }
            }
            #pragma unroll
            for(int m=0;m<2;++m){
                #pragma unroll
                for(int n=0;n<2;++n){
                    #pragma unroll
                    for(int h=0;h<2;++h){
                        float *c=accum[m*2+n]+h*4;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                                     : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                                     : "r"(a[m].x),"r"(a[m].y),"r"(a[m].z),"r"(a[m].w),"r"(b[n][h].x),"r"(b[n][h].y));
                    }
                }
            }
        }
        if(more)dense_wait_async();
        // Publishes async operands and releases every reader of the old stage.
        __syncthreads();
#else
        const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
        const int row=(warp%2)*32+(matrix/2)*8+local;
        const int col=(warp/2)*32+(matrix%2)*8+local;
        const unsigned a_base=unsigned(__cvta_generic_to_shared(wa))+row*Stride*2;
        const unsigned b_base=unsigned(__cvta_generic_to_shared(xb))+col*Stride*2;
        const int a_xor=dense_swizzle(row)*2,b_xor=dense_swizzle(col)*2;
        #pragma unroll 1
        for(int k=0;k<K;k+=8){
            const uint4 a0=dense_swizzle_load4(a_base+((k*2)^a_xor));
            const uint4 a1=dense_swizzle_load4(a_base+16*Stride*2+((k*2)^a_xor));
            #pragma unroll
            for(int n=0;n<2;++n){
                const uint4 b=dense_swizzle_load4(b_base+n*16*Stride*2+((k*2)^b_xor));
                dense_volta_step<false>(a0,b,accum[n]);dense_volta_step<false>(a1,b,accum[2+n]);
                dense_volta_step<true>(a0,b,accum[n]);dense_volta_step<true>(a1,b,accum[2+n]);
            }
        }
        // The previous iteration's barrier released this inactive stage.
        if(more)dense_swizzle_commit(pending,shared+(stage^1)*Stage);
        __syncthreads();
#endif
        stage^=1;
    }
    float *out=Pair&&blockIdx.z?second:first;
    if constexpr(!Pair)out+=size_t(blockIdx.z)*tokens*rows;
    #pragma unroll
    for(int f=0;f<4;++f){
        #pragma unroll
        for(int i=0;i<8;++i){
#if __CUDA_ARCH__ >= 800
            const int r=feature+(warp%2)*32+(f/2)*16+lane/4+((i%4)>=2?8:0);
            const int c=token+(warp/2)*32+(f%2)*16+(i/4)*8+2*(lane%4)+(i&1);
#else
            const int matrix=(lane%16)/4;
            const int r=feature+(warp%2)*32+(f/2)*16+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
            const int c=token+(warp/2)*32+(f%2)*16+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
#endif
            if(r<rows&&c<tokens)out[size_t(c)*rows+r]=accum[f][i];
        }
    }
}
