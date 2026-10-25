#pragma once
// Each warp folds16 Gate and16 Up rows; nonlinear epilogue follows the full K reduction.
// Eight 32x32 warp tiles consume one shared 128-token input tile per CTA.
template<bool Diagnostic>
__global__ __launch_bounds__(256,Diagnostic?1:3)
void dense_folded_compact(const half *__restrict__ weights,const half *__restrict__ input,
                 float *__restrict__ first,float *__restrict__ second,float *__restrict__ activation,
                 int rows,int columns,int tokens,int span){
    constexpr bool Pair=true,Swizzle=false,FullK=false,Gray=false;
    constexpr int R=64,C=128,K=64,Stride=(Swizzle||Gray)?64:72,Workers=256;
    __shared__ __align__(32) half shared[(R+C)*Stride];
    half *wa=shared,*xb=shared+R*Stride;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int feature=int(blockIdx.y)*(Pair?32:R),token=int(blockIdx.x)*C;
    const int begin=Pair?0:int(blockIdx.z)*span,end=min(columns,begin+(Pair?columns:span));
    float accum[4][8]{};
    for(int base=begin;base<end;base+=K){
        for(int v=tid;v<R*K/8;v+=Workers){
            const int row=v/(K/8),k=(v%(K/8))*8;
            const int source_row=feature+row%16+(row/32)*16;
            const size_t plane=(row%32)>=16?size_t(rows)*columns:0;
            uint4 value{};
            if(source_row<rows&&(FullK||base+k<end))
                value=*reinterpret_cast<const uint4 *>(weights+plane+size_t(source_row)*columns+base+k);
            *reinterpret_cast<uint4 *>(wa+joint_offset<Swizzle,Gray>(row,k))=value;
        }
        for(int v=tid;v<C*K/8;v+=Workers){
            const int col=v/(K/8),k=(v%(K/8))*8;
            uint4 value{};
            if(token+col<tokens&&(FullK||base+k<end))
                value=*reinterpret_cast<const uint4 *>(input+size_t(token+col)*columns+base+k);
            *reinterpret_cast<uint4 *>(xb+joint_offset<Swizzle,Gray>(col,k))=value;
        }
        __syncthreads();
#if __CUDA_ARCH__ >= 800
        const int a_row=(warp%2)*32+(lane%8)+((lane/8)%2)*8;
        #pragma unroll 1
        for(int k=0;k<K;k+=16){
            uint4 a[2];
            #pragma unroll
            for(int m=0;m<2;++m)a[m]=dense_matrix4(wa+joint_offset<Swizzle,Gray>(a_row,k+(lane/16)*8)+m*16*Stride);
            #pragma unroll
            for(int n=0;n<2;++n){
                #pragma unroll
                for(int h=0;h<2;++h){
                    const int col=(warp/2)*32+n*16+h*8+lane%8;
                    const uint2 b=dense_matrix2(xb+joint_offset<Swizzle,Gray>(col,k+((lane%16)/8)*8));
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
        const int row=(warp%2)*32+(matrix/2)*8+local;
        #pragma unroll 1
        for(int k=0;k<K;k+=8){
            uint4 a[2];
            #pragma unroll
            for(int m=0;m<2;++m)a[m]=dense_shared4(wa+joint_offset<Swizzle,Gray>(row,k)+m*16*Stride);
            #pragma unroll
            for(int n=0;n<2;++n){
                const int col=(warp/2)*32+n*16+(matrix%2)*8+local;
                const uint4 b=dense_shared4(xb+joint_offset<Swizzle,Gray>(col,k));
                #pragma unroll
                for(int m=0;m<2;++m)dense_volta_step<false>(a[m],b,accum[m*2+n]);
                #pragma unroll
                for(int m=0;m<2;++m)dense_volta_step<true>(a[m],b,accum[m*2+n]);
            }
        }
#endif
        __syncthreads();
    }
    #pragma unroll
    for(int f=0;f<2;++f){
        #pragma unroll
        for(int i=0;i<8;++i){
#if __CUDA_ARCH__ >= 800
            const int local_row=(f/2)*16+lane/4+((i%4)>=2?8:0);
            const int c=token+(warp/2)*32+(f%2)*16+(i/4)*8+2*(lane%4)+(i&1);
            constexpr int Neighbor=4;
#else
            const int matrix=(lane%16)/4;
            const int local_row=(f/2)*16+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
            const int c=token+(warp/2)*32+(f%2)*16+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
            constexpr int Neighbor=1;
#endif
            const int r=feature+(warp%2)*16+local_row;
            const size_t at=size_t(c)*rows+r;
            const float g=accum[f][i],u=accum[f+2][i],value=swiglu_value(g,u);
            if constexpr(Diagnostic){first[at]=g;second[at]=u;activation[at]=value;}
            else{
                const unsigned bits=__half_as_ushort(__float2half_rn(value));
                const unsigned other=__shfl_xor_sync(0xffffffff,bits,Neighbor);
                if((lane&Neighbor)==0)
                    *reinterpret_cast<unsigned *>(reinterpret_cast<half *>(activation)+at)=bits|(other<<16);
            }
        }
    }
}
