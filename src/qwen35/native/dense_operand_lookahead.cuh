#pragma once
// Pair joins 64 Gate and 64 Up rows; Down joins two neighboring 64-row blocks.
// Eight 64x32 warp tiles; next B is loaded before independent current A MMAs.
__global__ __launch_bounds__(256,2)
void dense_operand_lookahead(const half *__restrict__ weights,const half *__restrict__ input,
                 float *__restrict__ first,float *__restrict__ second,
                 int rows,int columns,int tokens,int span){
    constexpr bool Pair=true,Swizzle=false,FullK=false;
    constexpr int R=128,C=128,K=64,Stride=Swizzle?64:72,Workers=256;
    __shared__ __align__(32) half shared[(R+C)*Stride];
    half *wa=shared,*xb=shared+R*Stride;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int feature=int(blockIdx.y)*(Pair?64:R),token=int(blockIdx.x)*C;
    const int begin=Pair?0:int(blockIdx.z)*span,end=min(columns,begin+(Pair?columns:span));
    float accum[8][8]{};
    for(int base=begin;base<end;base+=K){
        for(int v=tid;v<R*K/8;v+=Workers){
            const int row=v/(K/8),k=(v%(K/8))*8;
            const int source_row=feature+(Pair?row%64:row);
            const size_t plane=Pair&&row>=64?size_t(rows)*columns:0;
            uint4 value{};
            if(source_row<rows&&(FullK||base+k<end))
                value=*reinterpret_cast<const uint4 *>(weights+plane+size_t(source_row)*columns+base+k);
            *reinterpret_cast<uint4 *>(wa+joint_offset<Swizzle>(row,k))=value;
        }
        for(int v=tid;v<C*K/8;v+=Workers){
            const int col=v/(K/8),k=(v%(K/8))*8;
            uint4 value{};
            if(token+col<tokens&&(FullK||base+k<end))
                value=*reinterpret_cast<const uint4 *>(input+size_t(token+col)*columns+base+k);
            *reinterpret_cast<uint4 *>(xb+joint_offset<Swizzle>(col,k))=value;
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
        const int a_row=(warp%2)*64+(lane%8)+((lane/8)%2)*8;
        uint2 b[2][2];
        #pragma unroll
        for(int n=0;n<2;++n)for(int h=0;h<2;++h){
            const int col=(warp/2)*32+n*16+h*8+lane%8;
            b[n][h]=dense_matrix2(xb+joint_offset<Swizzle>(col,((lane%16)/8)*8));
        }
        #pragma unroll
        for(int k=0;k<K;k+=16){
            uint2 pending[2][2];
            if(k+16<K){
                #pragma unroll
                for(int n=0;n<2;++n)for(int h=0;h<2;++h){
                    const int col=(warp/2)*32+n*16+h*8+lane%8;
                    pending[n][h]=dense_matrix2(xb+joint_offset<Swizzle>(col,k+16+((lane%16)/8)*8));
                }
            }
            #pragma unroll
            for(int m=0;m<4;++m){
                const uint4 a=dense_matrix4(wa+joint_offset<Swizzle>(a_row,k+(lane/16)*8)+m*16*Stride);
                #pragma unroll
                for(int n=0;n<2;++n)for(int h=0;h<2;++h){
                    float *c=accum[m*2+n]+h*4;
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                                 "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
                                 : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
                                 : "r"(a.x),"r"(a.y),"r"(a.z),"r"(a.w),"r"(b[n][h].x),"r"(b[n][h].y));
                }
            }
            if(k+16<K){
                #pragma unroll
                for(int n=0;n<2;++n)for(int h=0;h<2;++h)b[n][h]=pending[n][h];
            }
        }
#else
        const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
        const int row=(warp%2)*64+(matrix/2)*8+local;
        uint4 b[2];
        #pragma unroll
        for(int n=0;n<2;++n){
            const int col=(warp/2)*32+n*16+(matrix%2)*8+local;
            b[n]=dense_shared4(xb+joint_offset<Swizzle>(col,0));
        }
        #pragma unroll
        for(int k=0;k<K;k+=8){
            uint4 pending[2];
            if(k+8<K){
                #pragma unroll
                for(int n=0;n<2;++n){
                    const int col=(warp/2)*32+n*16+(matrix%2)*8+local;
                    pending[n]=dense_shared4(xb+joint_offset<Swizzle>(col,k+8));
                }
            }
            #pragma unroll
            for(int m=0;m<4;++m){
                const uint4 a=dense_shared4(wa+joint_offset<Swizzle>(row,k)+m*16*Stride);
                #pragma unroll
                for(int n=0;n<2;++n){
                    dense_volta_step<false>(a,b[n],accum[m*2+n]);
                    dense_volta_step<true>(a,b[n],accum[m*2+n]);
                }
            }
            if(k+8<K){
                #pragma unroll
                for(int n=0;n<2;++n)b[n]=pending[n];
            }
        }
#endif
        __syncthreads();
    }
    float *out=Pair&&(warp%2)?second:first;
    if constexpr(!Pair)out+=size_t(blockIdx.z)*tokens*rows;
    #pragma unroll
    for(int f=0;f<8;++f){
        #pragma unroll
        for(int i=0;i<8;++i){
#if __CUDA_ARCH__ >= 800
            const int local_row=(warp%2)*64+(f/2)*16+lane/4+((i%4)>=2?8:0);
            const int c=token+(warp/2)*32+(f%2)*16+(i/4)*8+2*(lane%4)+(i&1);
#else
            const int matrix=(lane%16)/4;
            const int local_row=(warp%2)*64+(f/2)*16+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
            const int c=token+(warp/2)*32+(f%2)*16+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
#endif
            const int r=feature+(Pair?local_row%64:local_row);
            if(r<rows&&c<tokens)out[size_t(c)*rows+r]=accum[f][i];
        }
    }
}
