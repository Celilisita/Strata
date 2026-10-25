#pragma once
#include "dense_operands.cuh"
// Included within the owned dense kernel namespace. Each warp's register
// ownership follows the PTX ISA; no WMMA fragment layout or result staging.
template<bool Pair,bool Balanced=false,bool TokenFirst=false>
__global__ __launch_bounds__(256,3)
void dense_native(const half *__restrict__ weights,const half *__restrict__ input,
                            float *__restrict__ first,float *__restrict__ second,
                            int rows,int columns,int tokens,int span){
    constexpr int R=64,C=128,K=64,Stride=72,Workers=256,Fragments=4;
    __shared__ __align__(32) half shared[(R+C)*Stride];
    half *wa=shared,*xb=shared+R*Stride;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int feature=int(TokenFirst?blockIdx.y:blockIdx.x)*R;
    const int token=int(TokenFirst?blockIdx.x:blockIdx.y)*C;
    const size_t plane=Pair?size_t(blockIdx.z)*rows*columns:0;
    const int begin=Pair?0:int(blockIdx.z)*span;
    const int end=min(columns,begin+(Pair?columns:span));
    float accum[Fragments][8]{};
    for(int base=begin;base<end;base+=K){
        for(int v=tid;v<R*K/8;v+=Workers){
            const int row=v/(K/8),k=(v%(K/8))*8;
            uint4 value{};
            if(feature+row<rows&&base+k<end)
                value=*reinterpret_cast<const uint4 *>(weights+plane+size_t(feature+row)*columns+base+k);
            *reinterpret_cast<uint4 *>(wa+row*Stride+k)=value;
        }
        for(int v=tid;v<C*K/8;v+=Workers){
            const int col=v/(K/8),k=(v%(K/8))*8;
            uint4 value{};
            if(token+col<tokens&&base+k<end)
                value=*reinterpret_cast<const uint4 *>(input+size_t(token+col)*columns+base+k);
            *reinterpret_cast<uint4 *>(xb+col*Stride+k)=value;
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
        // SM80+ m16n8k16. One A operand feeds both 8-token halves of each tile.
        const int a_row=(Balanced?(warp%2)*32:(warp%4)*16)+(lane%8)+((lane/8)%2)*8;
        #pragma unroll 1
        for(int k=0;k<K;k+=16){
            if constexpr(Balanced){
                uint4 a[2];uint2 b[2][2];
                #pragma unroll
                for(int m=0;m<2;++m)a[m]=dense_matrix4(wa+(a_row+m*16)*Stride+k+(lane/16)*8);
                #pragma unroll
                for(int n=0;n<2;++n){
                    #pragma unroll
                    for(int h=0;h<2;++h){
                        const int col=(warp/2)*32+n*16+h*8+lane%8;
                        b[n][h]=dense_matrix2(xb+col*Stride+k+((lane%16)/8)*8);
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
            }else{
            const uint4 a=dense_matrix4(wa+a_row*Stride+k+(lane/16)*8);
            #pragma unroll
            for(int f=0;f<Fragments;++f){
                #pragma unroll
                for(int n=0;n<2;++n){
                    const int col=(warp/4)*16+f*32+n*8+lane%8;
                    const uint2 b=dense_matrix2(xb+col*Stride+k+((lane%16)/8)*8);
                    float *c=accum[f]+n*4;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                                 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                                 : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b.x),"r"(b.y));
                }
            }
            }
        }
#else
        // SM70 m8n8k4: four quad-pairs together cover a 16x16 output tile.
        const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
        const int row=(Balanced?(warp%2)*32:(warp%4)*16)+(matrix/2)*8+local;
        #pragma unroll 1
        for(int k=0;k<K;k+=8){
            if constexpr(Balanced){
                const uint4 a0=dense_shared4(wa+row*Stride+k);
                const uint4 a1=dense_shared4(wa+(row+16)*Stride+k);
                const int col=(warp/2)*32+(matrix%2)*8+local;
                const uint4 b0=dense_shared4(xb+col*Stride+k);
                const uint4 b1=dense_shared4(xb+(col+16)*Stride+k);
                dense_volta_step<false>(a0,b0,accum[0]);dense_volta_step<false>(a0,b1,accum[1]);
                dense_volta_step<false>(a1,b0,accum[2]);dense_volta_step<false>(a1,b1,accum[3]);
                dense_volta_step<true>(a0,b0,accum[0]);dense_volta_step<true>(a0,b1,accum[1]);
                dense_volta_step<true>(a1,b0,accum[2]);dense_volta_step<true>(a1,b1,accum[3]);
            }else{
            const uint4 a=dense_shared4(wa+row*Stride+k);
            uint4 b[Fragments];
            #pragma unroll
            for(int f=0;f<Fragments;++f){
                const int col=(warp/4)*16+f*32+(matrix%2)*8+local;
                b[f]=dense_shared4(xb+col*Stride+k);
            }
            #pragma unroll
            for(int f=0;f<Fragments;++f)dense_volta_step<false>(a,b[f],accum[f]);
            #pragma unroll
            for(int f=0;f<Fragments;++f)dense_volta_step<true>(a,b[f],accum[f]);
            }
        }
#endif
        __syncthreads();
    }
    float *out=Pair&&blockIdx.z?second:first;
    if constexpr(!Pair)out+=size_t(blockIdx.z)*tokens*rows;
    #pragma unroll
    for(int f=0;f<Fragments;++f){
        #pragma unroll
        for(int i=0;i<8;++i){
#if __CUDA_ARCH__ >= 800
            const int r=feature+(Balanced?(warp%2)*32+(f/2)*16:(warp%4)*16)+lane/4+((i%4)>=2?8:0);
            const int c=token+(Balanced?(warp/2)*32+(f%2)*16:(warp/4)*16+f*32)+(i/4)*8+2*(lane%4)+(i&1);
#else
            const int matrix=(lane%16)/4;
            const int r=feature+(Balanced?(warp%2)*32+(f/2)*16:(warp%4)*16)+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
            const int c=token+(Balanced?(warp/2)*32+(f%2)*16:(warp/4)*16+f*32)+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
#endif
            if(r<rows&&c<tokens)out[size_t(c)*rows+r]=accum[f][i];
        }
    }
}
