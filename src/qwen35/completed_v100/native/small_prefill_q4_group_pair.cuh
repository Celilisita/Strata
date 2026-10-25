#pragma once
// Four row warps reuse decoded Q4 operands across two token quadrants.
// Keep550 block scale/minimum math and the same lossless compact payload.
struct GroupPairShared {
    unsigned short coefficient[2][64][10];
    unsigned weight_ds[2][64];
    unsigned char scale[2][64][8];
    union { unsigned input[32][detail::CompactHalfOperandPitch];float activation[32][64]; };
    unsigned ds[8][32];
};
static_assert(sizeof(GroupPairShared)==13568,"compact group pair shared extent");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ __forceinline__ void group_pair_positive(const GroupPairShared &s,const unsigned char *gate,const unsigned char *up,
                                         size_t payload,int chunk,int warp,int lane,
                                         const float (&delta)[2][2],float (&result)[2][2][8]){
    const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
    const int row=warp*16+(matrix/2)*8+local;
    #pragma unroll
    for(int m=0;m<2;++m){
        float scaled[2][8]{};
        #pragma unroll 1
        for(int pair=0;pair<2;++pair){
            const auto *p=(m?up:gate)+payload+size_t(chunk*2+pair)*64*32+row*32;
            const uint4 raw0=__ldg(reinterpret_cast<const uint4 *>(p)),raw1=__ldg(reinterpret_cast<const uint4 *>(p+16));
            #pragma unroll
            for(int parity=0;parity<2;++parity){
                const int g=pair*2+parity,group=chunk*4+g;
                float dot[2][8]{};
                #pragma unroll
                for(int part=0;part<4;++part){
                    const uint4 a=register_q4_operand(raw0,raw1,part,parity*4);
                    #pragma unroll
                    for(int n=0;n<2;++n){
                        const int token=n*16+(matrix%2)*8+local;
                        const uint4 b=detail::input_operand4<true>(&s.input[0][0],token,g*16+part*4);
                        mma_step(a.x,a.y,b.x,b.y,dot[n]);mma_step(a.z,a.w,b.z,b.w,dot[n]);
                    }
                }
                float scale[2];
                #pragma unroll
                for(int r=0;r<2;++r)scale[r]=float(s.scale[m][detail::result_row(warp*2,lane,r*2)][group]);
                #pragma unroll
                for(int n=0;n<2;++n){
                    float activation_delta[4];
                    #pragma unroll
                    for(int t=0;t<4;++t)activation_delta[t]=__half2float(__ushort_as_half(static_cast<unsigned short>(s.ds[group][detail::result_token(warp*2+n,lane,(t&1)+(t/2)*4)])));
                    #pragma unroll
                    for(int i=0;i<8;++i){const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                        scaled[n][i]=__fmaf_rn(activation_delta[t],__fmul_rn(scale[r],dot[n][i]),scaled[n][i]);
                    }
                }
            }
        }
        #pragma unroll
        for(int n=0;n<2;++n){
            #pragma unroll
            for(int i=0;i<8;++i)result[m][n][i]=__fmaf_rn(delta[m][(i/2)%2],scaled[n][i],result[m][n][i]);
        }
    }
}
__device__ __forceinline__ void group_pair_minimum(const GroupPairShared &s,int warp,int lane,const float (&minimum)[2][2],float (&result)[2][2][8]){
    const int matrix=(lane%16)/4,local=lane%4+(lane>=16?4:0);
    const int row=warp*16+(matrix/2)*8+local;
    #pragma unroll
    for(int m=0;m<2;++m){
        const auto *a=reinterpret_cast<const unsigned *>(s.coefficient[m][row]);
        float bias[2][8]{};
        #pragma unroll
        for(int n=0;n<2;++n){
            const int token=n*16+(matrix%2)*8+local;
            #pragma unroll
            for(int step=0;step<2;++step){
                const int g=step*4;
                const unsigned bx=(s.ds[g][token]>>16)|(s.ds[g+1][token]&0xffff0000u);
                const unsigned by=(s.ds[g+2][token]>>16)|(s.ds[g+3][token]&0xffff0000u);
                mma_step(a[step*2],a[step*2+1],bx,by,bias[n]);
            }
            #pragma unroll
            for(int i=0;i<8;++i)result[m][n][i]=__fmaf_rn(-minimum[m][(i/2)%2],bias[n][i],result[m][n][i]);
        }
    }
}
#endif
__global__ __launch_bounds__(128,4)
void paired_q4_group_register(Q4GroupCodes gate,Q4GroupCodes up,const detail::HalfAffineInputBlock *input,
                              float *gout,float *uout,float *activation,bool materialize,detail::HalfInputBlock *quantized){
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    __shared__ __align__(16) GroupPairShared s;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,tile=blockIdx.x,first=tile*64,blocks=gate.columns/256;
    const auto *gbase=static_cast<const unsigned char *>(gate.data),*ubase=static_cast<const unsigned char *>(up.data);
    float result[2][2][8]{};
    for(int block=0;block<blocks;++block){
        if(tid<128){
            const int m=tid/64,row=tid%64;
            const auto *p=(m?ubase:gbase)+(size_t(tile)*blocks+block)*64*16+row*16;
            const uint4 raw=__ldg(reinterpret_cast<const uint4 *>(p));s.weight_ds[m][row]=raw.x;
            #pragma unroll
            for(int group=0;group<8;++group){
                const int word=(group*12)>>5,shift=(group*12)&31;
                const unsigned low=word==0?raw.y:word==1?raw.z:raw.w,high=word==0?raw.z:word==1?raw.w:0;
                const unsigned meta=__funnelshift_r(low,high,shift)&4095u;
                s.scale[m][row][group]=static_cast<unsigned char>(meta&63);
                s.coefficient[m][row][group]=__half_as_ushort(__float2half_rn(float(meta>>6)));
            }
        }
        for(int i=tid;i<256;i+=128)s.ds[i/32][i%32]=input[block].ds[i/32][i%32];
        const size_t payload=size_t(gate.rows)*blocks*16+(size_t(tile)*blocks+block)*64*128;
        float delta[2][2],minimum[2][2];
        for(int chunk=0;chunk<2;++chunk){
            for(int token=warp;token<32;token+=4)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            __syncthreads();
            if(!chunk){
                #pragma unroll
                for(int m=0;m<2;++m){
                    #pragma unroll
                    for(int r=0;r<2;++r){const unsigned ds=s.weight_ds[m][detail::result_row(warp*2,lane,r*2)];
                        delta[m][r]=__half2float(__ushort_as_half(static_cast<unsigned short>(ds)));
                        minimum[m][r]=__half2float(__ushort_as_half(static_cast<unsigned short>(ds>>16)));
                    }
                }
            }
            group_pair_positive(s,gbase,ubase,payload,chunk,warp,lane,delta,result);
            if(!chunk)__syncthreads();
        }
        group_pair_minimum(s,warp,lane,minimum,result);__syncthreads();
    }
    #pragma unroll
    for(int n=0;n<2;++n){
        #pragma unroll
        for(int i=0;i<8;++i){
            const int row=first+detail::result_row(warp*2,lane,i),token=detail::result_token(warp*2+n,lane,i);
            const size_t at=size_t(token)*gate.rows+row;
            if(materialize){gout[at]=result[0][n][i];uout[at]=result[1][n][i];}
            const float value=swiglu_value(result[0][n][i],result[1][n][i]);if(materialize||!quantized)activation[at]=value;
            if(quantized)s.activation[token][row-first]=value;
        }
    }
    if(quantized){
        __syncthreads();
        for(int task=tid/8;task<64;task+=16){const int token=task/2,group=task%2;
            const float4 v=reinterpret_cast<const float4 *>(s.activation[token])[group*8+tid%8];
            detail::encode_half_group(v,quantized,first/32+group,token,1,tid%8);
        }
    }
    #endif
}
