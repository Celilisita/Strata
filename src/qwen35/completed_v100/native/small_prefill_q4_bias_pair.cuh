#pragma once
// Keep unscaled exact Q4 codes; only the minimum correction uses Tensor Cores.
struct PairBiasShared : Shared<2> { unsigned char scale[2][64][8]; };
static_assert(sizeof(PairBiasShared)==48392,"two bias blocks fit96KiB shared");
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ inline void pair_positive_chunk(const PairBiasShared &s,int chunk,int warp,int lane,
                                          const float (&delta)[2][2],float (&result)[2][8]){
    float scaled[2][8]{};
    #pragma unroll 1
    for(int local=0;local<4;++local){
        float dot[2][8]{};
        grouped_dot16(static_cast<const Shared<2>&>(s),local,0,warp,lane,dot);
        grouped_dot16(static_cast<const Shared<2>&>(s),local,1,warp,lane,dot);
        const int group=chunk*4+local;
        float activation_delta[4],scale[2][2];
        #pragma unroll
        for(int t=0;t<4;++t){
            const int token=detail::result_token(warp,lane,(t&1)+(t/2)*4);
            activation_delta[t]=__half2float(__ushort_as_half(static_cast<unsigned short>(s.ds[group][token])));
        }
        #pragma unroll
        for(int m=0;m<2;++m){
            #pragma unroll
            for(int r=0;r<2;++r)scale[m][r]=float(s.scale[m][detail::result_row(warp,lane,r*2)][group]);
        }
        #pragma unroll
        for(int i=0;i<8;++i){
            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
            #pragma unroll
            for(int m=0;m<2;++m)
                scaled[m][i]=__fmaf_rn(activation_delta[t],__fmul_rn(scale[m][r],dot[m][i]),scaled[m][i]);
        }
    }
    #pragma unroll
    for(int m=0;m<2;++m){
        #pragma unroll
        for(int i=0;i<8;++i)result[m][i]=__fmaf_rn(delta[m][(i/2)%2],scaled[m][i],result[m][i]);
    }
}
__device__ inline void pair_minimum_block(const PairBiasShared &s,int warp,int lane,
                                         const float (&minimum)[2][2],float (&result)[2][8]){
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=(warp/2)*16+(matrix/2)*8+local,token=(warp%2)*16+(matrix%2)*8+local;
    float bias[2][8]{};
    #pragma unroll
    for(int step=0;step<2;++step){
        const int g=step*4;
        const unsigned bx=(s.ds[g][token]>>16)|(s.ds[g+1][token]&0xffff0000u);
        const unsigned by=(s.ds[g+2][token]>>16)|(s.ds[g+3][token]&0xffff0000u);
        #pragma unroll
        for(int m=0;m<2;++m){
            const auto *a=reinterpret_cast<const unsigned *>(s.coefficient[m][row]);
            mma_step(a[step*2],a[step*2+1],bx,by,bias[m]);
        }
    }
    #pragma unroll
    for(int m=0;m<2;++m){
        #pragma unroll
        for(int i=0;i<8;++i)result[m][i]=__fmaf_rn(-minimum[m][(i/2)%2],bias[m][i],result[m][i]);
    }
}
#endif
__global__ __launch_bounds__(256,2)
void paired_q4_tensor_bias(Matrix gate,Matrix up,const detail::HalfAffineInputBlock *input,
                           float *gout,float *uout,float *activation,bool materialize,
                           detail::HalfInputBlock *quantized){
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    __shared__ __align__(16) PairBiasShared s;
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32,first=int(blockIdx.x)*64,blocks=gate.columns/256;
    float result[2][8]{};
    for(int block=0;block<blocks;++block){
        if(tid<128){
            const int m=tid/64,row=tid%64;
            const auto *p=static_cast<const unsigned char *>(m?up.data:gate.data)+(size_t(first+row)*blocks+block)*144;
            const uint4 raw=__ldg(reinterpret_cast<const uint4 *>(p));
            s.weight_ds[m][row]=raw.x;
            #pragma unroll
            for(int group=0;group<8;++group){
                const int word=(group*12)>>5,shift=(group*12)&31;
                const unsigned low=word==0?raw.y:word==1?raw.z:raw.w;
                const unsigned high=word==0?raw.z:word==1?raw.w:0;
                const unsigned meta=__funnelshift_r(low,high,shift)&4095u;
                s.scale[m][row][group]=static_cast<unsigned char>(meta&63);
                s.coefficient[m][row][group]=__half_as_ushort(__float2half_rn(float(meta>>6)));
            }
        }
        s.ds[tid/32][tid%32]=input[block].ds[tid/32][tid%32];
        float delta[2][2],minimum[2][2];
        for(int chunk=0;chunk<2;++chunk){
            unsigned packed[2][8];
            #pragma unroll
            for(int m=0;m<2;++m){
                const auto *base=static_cast<const unsigned char *>(m?up.data:gate.data);
                #pragma unroll
                for(int r=0;r<8;++r){
                    const int row=warp+r*8,group=chunk*4+lane/8,position=(lane%8)*4;
                    const auto *p=base+(size_t(first+row)*blocks+block)*144;
                    packed[m][r]=(__ldg(reinterpret_cast<const unsigned *>(p+16+(group/2)*32+position))>>((group%2)*4))&0x0f0f0f0fu;
                }
            }
            #pragma unroll
            for(int m=0;m<2;++m){
                #pragma unroll
                for(int r=0;r<8;++r){
                    const int row=warp+r*8;const unsigned value=packed[m][r];
                    *reinterpret_cast<uint2 *>(s.weight[m][row]+lane*2)=make_uint2(detail::half_codes(value,false,0),detail::half_codes(value,false,2));
                }
            }
            for(int token=warp;token<32;token+=8)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            __syncthreads();
            if(!chunk){
                #pragma unroll
                for(int m=0;m<2;++m){
                    #pragma unroll
                    for(int r=0;r<2;++r){
                        const unsigned ds=s.weight_ds[m][detail::result_row(warp,lane,r*2)];
                        delta[m][r]=__half2float(__ushort_as_half(static_cast<unsigned short>(ds)));
                        minimum[m][r]=__half2float(__ushort_as_half(static_cast<unsigned short>(ds>>16)));
                    }
                }
            }
            pair_positive_chunk(s,chunk,warp,lane,delta,result);
            if(!chunk)__syncthreads();
        }
        // Read-only correction shares the final chunk's metadata lifetime.
        pair_minimum_block(s,warp,lane,minimum,result);
        __syncthreads();
    }
    #pragma unroll
    for(int i=0;i<8;++i){
        const int row=first+detail::result_row(warp,lane,i),token=detail::result_token(warp,lane,i);
        const size_t at=size_t(token)*gate.rows+row;
        if(materialize){gout[at]=result[0][i];uout[at]=result[1][i];}
        const float value=swiglu_value(result[0][i],result[1][i]);
        if(materialize||!quantized)activation[at]=value;
        if(quantized)s.activation[token][row-first]=value;
    }
    if(quantized){
        __syncthreads();
        for(int task=tid/8;task<64;task+=32){
            const int token=task/2,group=task%2;
            const float4 v=reinterpret_cast<const float4 *>(s.activation[token])[group*8+tid%8];
            detail::encode_half_group(v,quantized,first/32+group,token,1,tid%8);
        }
    }
    #endif
}
