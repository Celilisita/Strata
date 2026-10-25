#include "tiny32.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cuda_fp16.h>
#include <cstdint>

namespace strata::qwen35::native {
namespace {
struct Activation {__half delta;unsigned short reserved;signed char codes[32];};
static_assert(sizeof(Activation)==NvTinyActivationBytes,"NVFP4 tiny activation ABI");
__device__ unsigned signed_four(unsigned packed){
    // PRMT maps E2M1 magnitudes to their exact integer doubles, four at once.
    const unsigned magnitude=__byte_perm(0x03020100u,0x0c080604u,packed&0x7777u);
    const unsigned sign=__byte_perm(0u,0xffffffffu,(packed&0x8888u)>>1);
    return __vsub4(magnitude^sign,sign);
}
struct Fragment32 {unsigned codes[8];float scale[2];};
__device__ Fragment32 fragment32(Nvfp4View weight,int row,int group){
    const auto packed=reinterpret_cast<const uint4 *>(weight.values)[size_t(row)*(weight.columns/32)+group];
    const auto *scales=static_cast<const unsigned char *>(weight.scales);
    const unsigned short sf=*reinterpret_cast<const unsigned short *>(scales+nvfp4_detail::scale_index(row,group*2,weight.columns));
    const float global=__fmul_rn(.5f,weight.weight_scale);
    return {{signed_four(packed.x),signed_four(packed.x>>16),signed_four(packed.y),signed_four(packed.y>>16),
             signed_four(packed.z),signed_four(packed.z>>16),signed_four(packed.w),signed_four(packed.w>>16)},
        {__fmul_rn(global,nvfp4_detail::scale_float(sf&255)),__fmul_rn(global,nvfp4_detail::scale_float(sf>>8))}};
}
__device__ float product32(Fragment32 weight,const Activation &input,float accumulator){
    const float delta=__half2float(input.delta);
    const auto *codes=reinterpret_cast<const int *>(input.codes);
    #pragma unroll
    for(int block=0;block<2;++block){
        int dot=0;
        #pragma unroll
        for(int j=0;j<4;++j)dot=__dp4a(int(weight.codes[block*4+j]),codes[block*4+j],dot);
        accumulator=__fmaf_rn(float(dot),__fmul_rn(weight.scale[block],delta),accumulator);
    }
    return accumulator;
}
template<bool Pair,int Tokens> __global__ void multiply(Nvfp4View a,Nvfp4View b,const Activation *input,
                                        float *first,float *second,float *activation){
    __shared__ float partial[Tokens][Pair?2:1][4];
    const int row=int(blockIdx.x),tid=threadIdx.x,lane=tid%32,warp=tid/32;
    float one[Tokens]{},two[Tokens]{};
    for(int group=tid;group<a.columns/32;group+=128){
        const auto g=fragment32(a,row,group);
        Fragment32 u{};if constexpr(Pair)u=fragment32(b,row,group);
        #pragma unroll
        for(int t=0;t<Tokens;++t){
            const auto *x=input+size_t(t)*(a.columns/32);
            one[t]=product32(g,x[group],one[t]);
            if constexpr(Pair)two[t]=product32(u,x[group],two[t]);
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        for(int d=16;d;d>>=1){
            one[t]=__fadd_rn(one[t],__shfl_xor_sync(0xffffffff,one[t],d));
            if constexpr(Pair)two[t]=__fadd_rn(two[t],__shfl_xor_sync(0xffffffff,two[t],d));
        }
        if(!lane){partial[t][0][warp]=one[t];if constexpr(Pair)partial[t][1][warp]=two[t];}
    }
    __syncthreads();
    if(tid<32){
        #pragma unroll
        for(int t=0;t<Tokens;++t){
            float g=lane<4?partial[t][0][lane]:0,u=0;
            if constexpr(Pair)u=lane<4?partial[t][1][lane]:0;
            for(int d=16;d;d>>=1){
                g=__fadd_rn(g,__shfl_xor_sync(0xffffffff,g,d));
                if constexpr(Pair)u=__fadd_rn(u,__shfl_xor_sync(0xffffffff,u,d));
            }
            if(!lane){const size_t index=size_t(t)*a.rows+row;first[index]=g;
                if constexpr(Pair){second[index]=u;activation[index]=swiglu_value(g,u);}}
        }
    }
}
void validate(Nvfp4View w,const void *input,const float *output,int tokens){
    if(!w.values||!w.scales||!input||!output||w.rows<1||w.columns<256||w.columns%256||tokens<1||tokens>4)
        throw std::invalid_argument("NVFP4 tiny geometry/storage");
}
template<bool Pair> void launch(Nvfp4View a,Nvfp4View b,const void *input,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    const auto *x=static_cast<const Activation *>(input);
    if(tokens==1)multiply<Pair,1><<<a.rows,128,0,stream>>>(a,b,x,first,second,act);
    else if(tokens==2)multiply<Pair,2><<<a.rows,128,0,stream>>>(a,b,x,first,second,act);
    else if(tokens==3)multiply<Pair,3><<<a.rows,128,0,stream>>>(a,b,x,first,second,act);
    else multiply<Pair,4><<<a.rows,128,0,stream>>>(a,b,x,first,second,act);
    check_cuda(cudaGetLastError(),"NVFP4 direct tiny projection");
}
}
void nvfp4_tiny32_linear(Nvfp4View w,const void *x,float *out,int tokens,cudaStream_t stream){
    validate(w,x,out,tokens);launch<false>(w,w,x,out,nullptr,nullptr,tokens,stream);
}
void nvfp4_tiny32_pair(Nvfp4View g,Nvfp4View u,const void *x,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    validate(g,x,first,tokens);validate(u,x,second,tokens);
    if(g.rows!=u.rows||g.columns!=u.columns||!act)throw std::invalid_argument("NVFP4 tiny pair mismatch");
    launch<true>(g,u,x,first,second,act,tokens,stream);
}
}
