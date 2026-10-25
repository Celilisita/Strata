#pragma once
// Included inside the kernel namespace after mma_step. Raw Q4 only.
struct RegisterQ4Shared {
    unsigned input[32][detail::CompactHalfOperandPitch];
    float2 ds[8][32];
    float2 affine[8][64];
    float2 weight_ds[64];
};
static_assert(sizeof(RegisterQ4Shared)==15104,"register Q4 shared extent");

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ < 800
__device__ __forceinline__ uint4 register_q4_operand(uint4 first,uint4 second,int part,int shift){
    uint2 raw=part==0?make_uint2(first.x,first.y):part==1?make_uint2(first.z,first.w):
              part==2?make_uint2(second.x,second.y):make_uint2(second.z,second.w);
    raw.x=(raw.x>>shift)&0x0f0f0f0fu;raw.y=(raw.y>>shift)&0x0f0f0f0fu;
    return make_uint4(detail::half_codes(raw.x,false,0),detail::half_codes(raw.x,false,2),
                      detail::half_codes(raw.y,false,0),detail::half_codes(raw.y,false,2));
}
#endif

template<int Partitions,bool Paired=false>
__global__ __launch_bounds__(128,4)
void register_q4(Matrix weight,const detail::HalfAffineInputBlock *input,float *output,Matrix second={}){
    #if __CUDA_ARCH__ >= 800
    return;
    #else
    static_assert(Partitions==0||Partitions==1||Partitions==2||Partitions==3||Partitions==5||Partitions==6);
    __shared__ __align__(16) RegisterQ4Shared s;
    if constexpr(Paired)if(blockIdx.y==1){weight=second;output+=size_t(weight.columns/256)*32*weight.rows;}
    const int tid=threadIdx.x,warp=tid/32,lane=tid%32;
    const int first=int(blockIdx.x)*64,blocks=weight.columns/256;
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=first+warp*16+(matrix/2)*8+local;
    const auto *base=static_cast<const unsigned char *>(weight.data);
    float result[2][8]{};
    int begin=0,end=0;
    if constexpr(Partitions==0){begin=blocks-1-int(blockIdx.z);end=begin+1;}
    else{begin=int(blockIdx.z)*blocks/Partitions;end=int(blockIdx.z+1)*blocks/Partitions;}
    if constexpr(Partitions==2||Partitions==3){
        // Validated SM70 service: split the existing owned row128 numerical
        // cells, while each CTA independently tiles64rows*32tokens with MMA.
        const int width=int(int64_t((weight.rows+127)/128)*blocks/80),start=(first/128)*blocks;
        int stop=start+blocks;
        for(int part=0;part<int(blockIdx.z)&&stop>start;++part)
            stop=max(start,((stop-1)/width)*width);
        begin=max(start,stop>start?((stop-1)/width)*width:start)-start;
        end=stop-start;
    }
    for(int block=begin;block<end;++block){
        if(tid<64){
            const auto *p=base+(size_t(first+tid)*blocks+block)*144;
            const uint4 raw=first+tid<weight.rows?__ldg(reinterpret_cast<const uint4 *>(p)):make_uint4(0,0,0,0);
            const float d=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x)));
            const float dm=__half2float(__ushort_as_half(static_cast<unsigned short>(raw.x>>16)));
            s.weight_ds[tid]=make_float2(d,dm);
            #pragma unroll
            for(int g=0;g<8;++g){
                const int word=(g*12)>>5,shift=(g*12)&31;
                const unsigned lo=word==0?raw.y:word==1?raw.z:raw.w,hi=word==0?raw.z:word==1?raw.w:0;
                const unsigned descriptor=__funnelshift_r(lo,hi,shift)&4095u;
                s.affine[g][tid]=make_float2(float(descriptor&63),float(descriptor>>6));
            }
        }
        #pragma unroll
        for(int i=tid;i<256;i+=128){
            const unsigned bits=input[block].ds[i/32][i%32];
            s.ds[i/32][i%32]=make_float2(__half2float(__ushort_as_half(static_cast<unsigned short>(bits))),
                                       __half2float(__ushort_as_half(static_cast<unsigned short>(bits>>16))));
        }
        for(int chunk=0;chunk<2;++chunk){
            for(int token=warp;token<32;token+=4)
                *reinterpret_cast<uint2 *>(s.input[token]+lane*2)=__ldg(reinterpret_cast<const uint2 *>(input[block].codes[token]+chunk*64)+lane);
            __syncthreads();
            float2 weight_ds[2];
            #pragma unroll
            for(int r=0;r<2;++r)weight_ds[r]=s.weight_ds[detail::result_row(warp*2,lane,r*2)];
            #pragma unroll
            for(int pair=0;pair<2;++pair){
                float scaled[2][8]{},correction[2][8]{};
                uint4 raw0=make_uint4(0,0,0,0),raw1=raw0;
                if(row<weight.rows){
                    const auto *p=base+(size_t(row)*blocks+block)*144+16+(chunk*2+pair)*32;
                    raw0=__ldg(reinterpret_cast<const uint4 *>(p));
                    raw1=__ldg(reinterpret_cast<const uint4 *>(p+16));
                }
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
                    float2 coefficient[2];
                    #pragma unroll
                    for(int r=0;r<2;++r)coefficient[r]=s.affine[group][detail::result_row(warp*2,lane,r*2)];
                    #pragma unroll
                    for(int n=0;n<2;++n){
                        float delta[4],sum[4];
                        #pragma unroll
                        for(int t=0;t<4;++t){
                            const int token=detail::result_token(warp*2+n,lane,(t&1)+(t/2)*4);
                            const float2 ds=s.ds[group][token];
                            delta[t]=ds.x;sum[t]=ds.y;
                        }
                        #pragma unroll
                        for(int i=0;i<8;++i){
                            const int r=(i/2)%2,t=(i&1)+(i/4)*2;
                            // Scale0..63 times the bounded integer dot is exact F32.
                            const float scaled_dot=__fmul_rn(coefficient[r].x,dot[n][i]);
                            scaled[n][i]=__fmaf_rn(delta[t],scaled_dot,scaled[n][i]);
                            correction[n][i]=__fmaf_rn(sum[t],coefficient[r].y,correction[n][i]);
                        }
                    }
                }
                #pragma unroll
                for(int n=0;n<2;++n){
                    #pragma unroll
                    for(int i=0;i<8;++i){const int r=(i/2)%2;
                        result[n][i]=__fadd_rn(result[n][i],__fmaf_rn(weight_ds[r].x,scaled[n][i],-__fmul_rn(weight_ds[r].y,correction[n][i])));
                    }
                }
            }
            // Protect input reuse and next-block metadata writes.
            __syncthreads();
        }
    }
    #pragma unroll
    for(int n=0;n<2;++n){
        #pragma unroll
        for(int i=0;i<8;++i){
            const int r=first+detail::result_row(warp*2+n,lane,i),t=detail::result_token(warp*2+n,lane,i);
            if(r<weight.rows)output[(size_t(blockIdx.z)*32+t)*weight.rows+r]=result[n][i];
        }
    }
    #endif
}
