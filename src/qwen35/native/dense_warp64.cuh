#pragma once
// Pair joins 64 Gate and 64 Up rows; Down joins two neighboring 64-row blocks.
// Sixteen32x32 warp tiles,32 FP32 accumulators/thread; one published K32 stage.
__global__ __launch_bounds__(512,2)
void dense_warp64(const half *__restrict__ weights,const half *__restrict__ input,
                 float *__restrict__ first,float *__restrict__ second,
                 int rows,int columns,int tokens,int span){
    constexpr bool Pair=true,Swizzle=false,FullK=false;
    constexpr int R=128,C=128,K=32,Stride=40,Workers=512;
    __shared__ __align__(32) half shared[(R+C)*Stride];
    half *wa=shared,*xb=shared+R*Stride;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int feature=int(blockIdx.y)*(Pair?64:R),token=int(blockIdx.x)*C;
    const int begin=Pair?0:int(blockIdx.z)*span,end=min(columns,begin+(Pair?columns:span));
    float accum[4][8]{};
    for(int base=begin;base<end;base+=K){
        for(int v=tid;v<R*K/8;v+=Workers){
            const int row=v/(K/8),k=(v%(K/8))*8;
            const int source_row=feature+(Pair?row%64:row);
            const size_t plane=Pair&&row>=64?size_t(rows)*columns:0;
            uint4 value{};
            value=*reinterpret_cast<const uint4 *>(weights+plane+size_t(source_row)*columns+base+k);
            *reinterpret_cast<uint4 *>(wa+row*Stride+k)=value;
        }
        for(int v=tid;v<C*K/8;v+=Workers){
            const int col=v/(K/8),k=(v%(K/8))*8;
            uint4 value{};
            value=*reinterpret_cast<const uint4 *>(input+size_t(token+col)*columns+base+k);
            *reinterpret_cast<uint4 *>(xb+col*Stride+k)=value;
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
        const int a_row=(warp%4)*32+(lane%8)+((lane/8)%2)*8;
        #pragma unroll 1
        for(int k=0;k<K;k+=16){
            uint4 a[2];
            #pragma unroll
            for(int m=0;m<2;++m)a[m]=dense_matrix4(wa+a_row*Stride+k+(lane/16)*8+m*16*Stride);
            #pragma unroll
            for(int n=0;n<2;++n){
                #pragma unroll
                for(int h=0;h<2;++h){
                    const int col=(warp/4)*32+n*16+h*8+lane%8;
                    const uint2 b=dense_matrix2(xb+col*Stride+k+((lane%16)/8)*8);
                    #pragma unroll
                    for(int m=0;m<2;++m){
                        float *c=accum[m*2+n]+h*4;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                     "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                                     : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                                     : "r"(a[m].x),"r"(a[m].y),"r"(a[m].z),"r"(a[m].w),"r"(b.x),"r"(b.y));
                    }
                }
            }
        }
#else
        const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
        const int row=(warp%4)*32+(matrix/2)*8+local;
        #pragma unroll 1
        for(int k=0;k<K;k+=8){
            uint4 a[2];
            #pragma unroll
            for(int m=0;m<2;++m)a[m]=dense_shared4(wa+row*Stride+k+m*16*Stride);
            #pragma unroll
            for(int group=0;group<1;++group){
                uint4 b[2];
                #pragma unroll
                for(int n=0;n<2;++n){
                    const int col=(warp/4)*32+(group*2+n)*16+(matrix%2)*8+local;
                    b[n]=dense_shared4(xb+col*Stride+k);
                }
                // Four independent accumulator fragments separate lower/upper.
                #pragma unroll
                for(int n=0;n<2;++n)for(int m=0;m<2;++m)
                    dense_volta_step<false>(a[m],b[n],accum[m*2+group*2+n]);
                #pragma unroll
                for(int n=0;n<2;++n)for(int m=0;m<2;++m)
                    dense_volta_step<true>(a[m],b[n],accum[m*2+group*2+n]);
            }
        }
#endif
        __syncthreads();
    }
    float *out=Pair&&(warp%4>=2)?second:first;
    if constexpr(!Pair)out+=size_t(blockIdx.z)*tokens*rows;
    #pragma unroll
    for(int f=0;f<4;++f){
        #pragma unroll
        for(int i=0;i<8;++i){
#if __CUDA_ARCH__ >= 800
            const int local_row=(warp%4)*32+(f/2)*16+lane/4+((i%4)>=2?8:0);
            const int c=token+(warp/4)*32+(f%2)*16+(i/4)*8+2*(lane%4)+(i&1);
#else
            const int matrix=(lane%16)/4;
            const int local_row=(warp%4)*32+(f/2)*16+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
            const int c=token+(warp/4)*32+(f%2)*16+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
#endif
            const int r=feature+(Pair?local_row%64:local_row);
            out[size_t(c)*rows+r]=accum[f][i];
        }
    }
}
