#include "q6_warp_logical.hpp"
#include "vector_fragment.cuh"
namespace strata::qwen35::native {namespace {
using detail::ActivationBlock;
template<int Tokens>__global__ void multiply(Matrix w,const ActivationBlock *input,float *out){
    const int lane=threadIdx.x%32,row=int(blockIdx.x)*4+threadIdx.x/32;
    if(row>=w.rows)return;
    float value[Tokens][4]{};
    #pragma unroll
    for(int stripe=0;stripe<4;++stripe){
        // The original four logical K stripes are retained by the same lane.
        for(int block=stripe;block<w.columns/256;block+=4){
            const auto *p=static_cast<const unsigned char*>(w.data)+(size_t(row)*(w.columns/256)+block)*210;
            #pragma unroll
            for(int t=0;t<Tokens;++t){
                const auto *x=input+size_t(t)*(w.columns/32)+block*8;
                value[t][stripe]=detail::accumulate_fragment<Quant::Q6K,false>(p,x,value[t][stripe],nullptr);
            }
        }
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        // Q6 combines stripes before the XOR tree; keep that exact F32 order.
        float sum=value[t][0];
        #pragma unroll
        for(int stripe=1;stripe<4;++stripe)sum=__fadd_rn(sum,value[t][stripe]);
        for(int d=16;d;d/=2)sum=__fadd_rn(sum,__shfl_xor_sync(0xffffffff,sum,d));
        if(!lane)out[size_t(t)*w.rows+row]=sum;
    }
}
}
void q6_warp_logical(Matrix w,const void *input,float *out,int tokens,cudaStream_t stream){
    matrix_bytes(w);
    if(w.quant!=Quant::Q6K||w.layout!=BlockLayout::Gguf||w.columns!=5120||w.rows!=248320||!input||!out||tokens<1||tokens>4)
        throw std::invalid_argument("logical Q6 head geometry");
    const auto *x=static_cast<const ActivationBlock*>(input);const int blocks=(w.rows+3)/4;
    if(tokens==1)multiply<1><<<blocks,128,0,stream>>>(w,x,out);
    else if(tokens==2)multiply<2><<<blocks,128,0,stream>>>(w,x,out);
    else if(tokens==3)multiply<3><<<blocks,128,0,stream>>>(w,x,out);
    else multiply<4><<<blocks,128,0,stream>>>(w,x,out);
    check_cuda(cudaGetLastError(),"logical Q6 head result");
}
}
