#include "warp_logical.hpp"
#include "tiny_nvfp4.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cuda_fp16.h>
namespace strata::qwen35::native {namespace {
struct Activation {__half delta;unsigned short reserved;signed char codes[32];};
static_assert(sizeof(Activation)==NvTinyActivationBytes,"existing Q8 tiny ABI");
struct Prepared {int codes[4];float delta;};
__device__ Prepared prepare(const Activation *input,int group){
    const auto &x=input[group/2];const auto *codes=reinterpret_cast<const int*>(x.codes+(group%2)*16);
    return {{codes[0],codes[1],codes[2],codes[3]},__half2float(x.delta)};
}
__device__ unsigned signed_four(unsigned packed){
    const unsigned magnitude=__byte_perm(0x03020100u,0x0c080604u,packed&0x7777u);
    const unsigned sign=__byte_perm(0u,0xffffffffu,(packed&0x8888u)>>1);
    return __vsub4(magnitude^sign,sign);
}
struct Fragment {unsigned codes[4];float scale;};
__device__ Fragment fragment(Nvfp4View weight,int row,int group){
    const auto pair=reinterpret_cast<const uint2*>(weight.values)[size_t(row)*(weight.columns/16)+group];
    const auto *scales=static_cast<const unsigned char*>(weight.scales);
    const float scale=nvfp4_detail::scale_float(scales[nvfp4_detail::scale_index(row,group,weight.columns)]);
    return {{signed_four(pair.x),signed_four(pair.x>>16),signed_four(pair.y),signed_four(pair.y>>16)},
        __fmul_rn(__fmul_rn(.5f,weight.weight_scale),scale)};
}
__device__ float product(Fragment weight,Prepared input,float accumulator){
    int dot=0;
    #pragma unroll
    for(int j=0;j<4;++j)dot=__dp4a(int(weight.codes[j]),input.codes[j],dot);
    return __fmaf_rn(float(dot),__fmul_rn(weight.scale,input.delta),accumulator);
}
template<bool Pair,int Tokens>__global__ void multiply(Nvfp4View a,Nvfp4View b,const Activation *input,
    float *first,float *second,float *activation){
    const int lane=threadIdx.x%32,row=int(blockIdx.x)*4+threadIdx.x/32;
    if(row>=a.rows)return;
    float one[Tokens][4]{},two[Tokens][4]{};
    for(int base=lane;base<a.columns/16;base+=128){
        #pragma unroll
        for(int stripe=0;stripe<4;++stripe){
            const int group=base+stripe*32;
            if(group<a.columns/16){
                const auto g=fragment(a,row,group);Fragment u{};
                if constexpr(Pair)u=fragment(b,row,group);
                #pragma unroll
                for(int t=0;t<Tokens;++t){
                    const auto x=prepare(input+size_t(t)*(a.columns/32),group);
                    one[t][stripe]=product(g,x,one[t][stripe]);
                    if constexpr(Pair)two[t][stripe]=product(u,x,two[t][stripe]);
                }
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        #pragma unroll
        for(int stripe=0;stripe<4;++stripe){
            for(int d=16;d;d>>=1){
                one[t][stripe]=__fadd_rn(one[t][stripe],__shfl_xor_sync(0xffffffff,one[t][stripe],d));
                if constexpr(Pair)two[t][stripe]=__fadd_rn(two[t][stripe],__shfl_xor_sync(0xffffffff,two[t][stripe],d));
            }
            // Match the three zero branches before the retained four-part final tree.
            #pragma unroll
            for(int z=0;z<3;++z){
                one[t][stripe]=__fadd_rn(one[t][stripe],0.f);
                if constexpr(Pair)two[t][stripe]=__fadd_rn(two[t][stripe],0.f);
            }
        }
        if(!lane){
            const float g=__fadd_rn(__fadd_rn(one[t][0],one[t][2]),__fadd_rn(one[t][1],one[t][3]));
            const size_t index=size_t(t)*a.rows+row;first[index]=g;
            if constexpr(Pair){
                const float u=__fadd_rn(__fadd_rn(two[t][0],two[t][2]),__fadd_rn(two[t][1],two[t][3]));
                second[index]=u;activation[index]=swiglu_value(g,u);
            }
        }
    }
}

void validate(Nvfp4View w,const void *input,const float *out,int tokens){
    if(!w.values||!w.scales||!input||!out||w.rows<1||w.columns<256||w.columns%256||tokens<1||tokens>3)
        throw std::invalid_argument("logical warp rows geometry");
}
template<bool Pair>void launch(Nvfp4View a,Nvfp4View b,const void *input,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    const auto *x=static_cast<const Activation*>(input);const int blocks=(a.rows+3)/4;
    if(tokens==1)multiply<Pair,1><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    else if(tokens==2)multiply<Pair,2><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    else multiply<Pair,3><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    check_cuda(cudaGetLastError(),"logical warp rows");
}
}
void nvfp4_warp_logical_linear(Nvfp4View w,const void *x,float *out,int tokens,cudaStream_t stream){
    validate(w,x,out,tokens);launch<false>(w,w,x,out,nullptr,nullptr,tokens,stream);
}
void nvfp4_warp_logical_pair(Nvfp4View g,Nvfp4View u,const void *x,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    validate(g,x,first,tokens);validate(u,x,second,tokens);
    if(g.rows!=u.rows||g.columns!=u.columns||!act)throw std::invalid_argument("logical warp pair mismatch");
    launch<true>(g,u,x,first,second,act,tokens,stream);
}
}
