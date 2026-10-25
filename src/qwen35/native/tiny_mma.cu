#include "tiny_mma.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cuda_fp16.h>
namespace strata::qwen35::native {namespace {
struct Activation {__half delta;unsigned short reserved;signed char codes[32];};
static_assert(sizeof(Activation)==36,"shared tinyQ8 ABI");
__device__ unsigned signed_four(unsigned packed){
    const unsigned magnitude=__byte_perm(0x03020100u,0x0c080604u,packed&0x7777u);
    const unsigned sign=__byte_perm(0u,0xffffffffu,(packed&0x8888u)>>1);
    return __vsub4(magnitude^sign,sign);
}
struct Fragment {unsigned a,b;float first,second;};
__device__ Fragment fragment(Nvfp4View w,int row,int group,int column){
    Fragment f{};const auto *data=static_cast<const unsigned short*>(w.values);
    const auto *scales=static_cast<const unsigned char*>(w.scales);
    const float global=__fmul_rn(.5f,w.weight_scale);
    if(row<w.rows){
        f.a=signed_four(data[size_t(row)*(w.columns/4)+group*4+column]);
        f.first=__fmul_rn(global,nvfp4_detail::scale_float(scales[nvfp4_detail::scale_index(row,group,w.columns)]));
    }
    if(row+8<w.rows){
        f.b=signed_four(data[size_t(row+8)*(w.columns/4)+group*4+column]);
        f.second=__fmul_rn(global,nvfp4_detail::scale_float(scales[nvfp4_detail::scale_index(row+8,group,w.columns)]));
    }
    return f;
}
__device__ void accumulate(Fragment a,unsigned b,const float *delta,float *sum){
    int d0,d1,d2,d3;
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%7,%7,%7};"
        :"=r"(d0),"=r"(d1),"=r"(d2),"=r"(d3):"r"(a.a),"r"(a.b),"r"(b),"r"(0));
    sum[0]=__fmaf_rn(float(d0),__fmul_rn(a.first,delta[0]),sum[0]);
    sum[1]=__fmaf_rn(float(d1),__fmul_rn(a.first,delta[1]),sum[1]);
    sum[2]=__fmaf_rn(float(d2),__fmul_rn(a.second,delta[0]),sum[2]);
    sum[3]=__fmaf_rn(float(d3),__fmul_rn(a.second,delta[1]),sum[3]);
}
template<bool Pair,int Tokens> __global__ void multiply(Nvfp4View a,Nvfp4View b,const Activation *input,
    float *first,float *second,float *activation){
    constexpr int Warps=4;
    __shared__ float partial[Pair?2:1][Warps][16][8];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=int(blockIdx.x)*16+lane/4,col=lane%4;
    float one[4]{},two[4]{};
    for(int group=warp;group<a.columns/16;group+=Warps){
        unsigned codes=0;const int token=lane/4;
        if(token<Tokens){
            const auto &x=input[size_t(token)*(a.columns/32)+group/2];
            codes=reinterpret_cast<const unsigned*>(x.codes+(group%2)*16)[col];
        }
        float delta[2]{};
        #pragma unroll
        for(int i=0;i<2;++i){const int t=col*2+i;
            if(t<Tokens)delta[i]=__half2float(input[size_t(t)*(a.columns/32)+group/2].delta);}
        accumulate(fragment(a,row,group,col),codes,delta,one);
        if constexpr(Pair)accumulate(fragment(b,row,group,col),codes,delta,two);
    }
    #pragma unroll
    for(int i=0;i<4;++i){
        const int r=lane/4+(i/2)*8,c=col*2+i%2;
        partial[0][warp][r][c]=one[i];if constexpr(Pair)partial[1][warp][r][c]=two[i];
    }
    __syncthreads();
    if(!warp){
        #pragma unroll
        for(int i=0;i<4;++i){
            const int r=lane/4+(i/2)*8,c=col*2+i%2,actual=int(blockIdx.x)*16+r;
            if(c<Tokens&&actual<a.rows){
                float g=0,u=0;
                #pragma unroll
                for(int w=0;w<Warps;++w){g=__fadd_rn(g,partial[0][w][r][c]);
                    if constexpr(Pair)u=__fadd_rn(u,partial[1][w][r][c]);}
                const size_t offset=size_t(c)*a.rows+actual;first[offset]=g;
                if constexpr(Pair){second[offset]=u;activation[offset]=swiglu_value(g,u);}
            }
        }
    }
}
void validate(Nvfp4View w,const void *x,float *out,int tokens){
    if(!w.values||!w.scales||!x||!out||w.rows<1||w.columns<256||w.columns%256||tokens<1||tokens>4)
        throw std::invalid_argument("INT8 TensorCore tiny NVFP4/Q8 geometry");
}
template<bool Pair>void launch(Nvfp4View a,Nvfp4View b,const void *input,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    const auto *x=static_cast<const Activation*>(input);const int blocks=(a.rows+15)/16;
    if(tokens==1)multiply<Pair,1><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    else if(tokens==2)multiply<Pair,2><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    else if(tokens==3)multiply<Pair,3><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    else multiply<Pair,4><<<blocks,128,0,stream>>>(a,b,x,first,second,act);
    check_cuda(cudaGetLastError(),"INT8 TensorCore tiny projection");
}
}
void nvfp4_tiny_mma_linear(Nvfp4View w,const void *x,float *out,int tokens,cudaStream_t stream){
    validate(w,x,out,tokens);launch<false>(w,w,x,out,nullptr,nullptr,tokens,stream);
}
void nvfp4_tiny_mma_pair(Nvfp4View g,Nvfp4View u,const void *x,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    validate(g,x,first,tokens);validate(u,x,second,tokens);
    if(g.rows!=u.rows||g.columns!=u.columns||!act)throw std::invalid_argument("INT8 TensorCore pair mismatch");
    launch<true>(g,u,x,first,second,act,tokens,stream);
}
}
