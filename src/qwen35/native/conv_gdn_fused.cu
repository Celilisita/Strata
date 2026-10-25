#include "conv_gdn_fused.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "math.cuh"
#include <cmath>
namespace strata::qwen35::native {
namespace {
__device__ float convolution(ConvGdnInput input,float *next,int channel,int token,int channels,int tokens){
    float value=0;
    #pragma unroll
    for(int j=0;j<4;++j){
        const int source=token+j-3;
        const float x=source<0?input.history[size_t(channel)*3+source+3]:input.mixed[size_t(source)*channels+channel];
        value=__fmaf_rn(x,input.filter[size_t(channel)*4+j],value);
    }
    if(token==tokens-1){
        #pragma unroll
        for(int j=0;j<3;++j){
            const int source=tokens-3+j;
            next[size_t(channel)*3+j]=source<0?input.history[size_t(channel)*3+source+3]:input.mixed[size_t(source)*channels+channel];
        }
    }
    return swiglu_value(value,1.f);
}
__device__ float sum(float x){
    for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__global__ void prepare(ConvGdnInput input,ConvGdnOutput output,int hq,int hv,int tokens,float epsilon){
    constexpr int Dim=128;
    const int row=int(blockIdx.x)*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=tokens*(hq+hv))return;
    const int head=row%(hq+hv),token=row/(hq+hv),channels=(2*hq+hv)*Dim;
    if(head<hq){
        float q[4],k[4],qs=0,ks=0;
        #pragma unroll
        for(int j=0;j<4;++j){
            const int c=head*Dim+lane+j*32;
            q[j]=convolution(input,output.history,c,token,channels,tokens);
            k[j]=convolution(input,output.history,c+hq*Dim,token,channels,tokens);
            qs=__fmaf_rn(q[j],q[j],qs);ks=__fmaf_rn(k[j],k[j],ks);
        }
        qs=rsqrtf(fmaxf(sum(qs),epsilon));ks=rsqrtf(fmaxf(sum(ks),epsilon));
        #pragma unroll
        for(int j=0;j<4;++j){
            const size_t at=(size_t(token)*hq+head)*Dim+lane+j*32;
            output.q[at]=__fmul_rn(q[j],qs);output.k[at]=__fmul_rn(k[j],ks);
        }
    }else{
        const int value_head=head-hq;
        #pragma unroll
        for(int j=0;j<4;++j){
            const int d=lane+j*32,c=(2*hq+value_head)*Dim+d;
            output.v[(size_t(token)*hv+value_head)*Dim+d]=convolution(input,output.history,c,token,channels,tokens);
        }
        if(!lane){
            const size_t i=size_t(token)*hv+value_head;
            const float x=__fadd_rn(input.alpha[i],input.bias[value_head]);
            const float softplus=x>20?x:__logf(__fadd_rn(1.f,__expf(x)));
            output.g[i]=__fmul_rn(softplus,input.decay[value_head]);
            output.beta[i]=__fdividef(1.f,__fadd_rn(1.f,__expf(-input.beta_raw[i])));
        }
    }
}
}
void conv_gdn_fused(ConvGdnInput i,ConvGdnOutput o,int hq,int hv,int tokens,float epsilon,cudaStream_t stream){
    if(!i.mixed||!i.history||!i.filter||!i.alpha||!i.beta_raw||!i.bias||!i.decay||
       !o.q||!o.k||!o.v||!o.g||!o.beta||!o.history||o.history==i.history||
       hq<1||hv<hq||hv>256||hv%hq||tokens<1||tokens>8192||!(epsilon>0)||!std::isfinite(epsilon))
        throw std::invalid_argument("fused convolution/GDN preparation geometry");
    prepare<<<(tokens*(hq+hv)+3)/4,128,0,stream>>>(i,o,hq,hv,tokens,epsilon);
    check_cuda(cudaGetLastError(),"fused convolution normalized GDN preparation");
}
}
