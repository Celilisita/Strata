#include "strata/gemma4/native/vector_grouped.hpp"
#include "vector3_layout.cuh"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
namespace strata::gemma4::native {
namespace {
using detail::ResidualBlock;
// Eight lanes own one row, one full32-value quantization group per lane.
// Integer dots precede the group scale/minimum, avoiding four repeated F32
// scale applications and repeated metadata per group in the parent layout.
template<primitive::Quant Type>
__device__ float term(const unsigned char* raw,const ResidualBlock& x,int group,float previous){
    int weight[8];
    #pragma unroll
    for(int w=0;w<8;++w)weight[w]=primitive::detail::weight_codes<Type>(raw,group,w*4);
    float product=0,correction=0;
    #pragma unroll
    for(int r=0;r<3;++r){
        int first=0,second=0;
        #pragma unroll
        for(int w=0;w<8;++w){const int code=reinterpret_cast<const int*>(x.code[r])[w];
            if(w<4)first=__dp4a(weight[w],code,first);else second=__dp4a(weight[w],code,second);}
        if constexpr(Type==primitive::Quant::Q4K){
            product=__fmaf_rn(x.d[r],float(first+second),product);correction=__fmaf_rn(x.d[r],float(x.sum[r]),correction);
        }else{
            const auto* scales=reinterpret_cast<const signed char*>(raw+192)+group*2;
            product=__fmaf_rn(x.d[r],float(int(scales[0])*first+int(scales[1])*second),product);
        }
    }
    if constexpr(Type==primitive::Quant::Q4K){
        const unsigned meta=primitive::detail::metadata(raw,group,false);
        const float scale=__half2float(*reinterpret_cast<const half*>(raw))*float(meta&63);
        const float minimum=__half2float(*reinterpret_cast<const half*>(raw+2))*float(meta>>6);
        return __fadd_rn(previous,__fmaf_rn(scale,product,-__fmul_rn(minimum,correction)));
    }else return __fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208)),product,previous);
}
__device__ float sum8(float v,unsigned mask){for(int d=4;d;d/=2)v=__fadd_rn(v,__shfl_xor_sync(mask,v,d));return v;}
template<primitive::Quant A,primitive::Quant B,int Tokens,bool Pair>
__global__ void multiply(primitive::Matrix first,primitive::Matrix second,const ResidualBlock* input,float* x,float* y,float* activation){
    const int row=blockIdx.x*16+threadIdx.x/8,group=threadIdx.x%8;if(row>=first.rows)return;
    const unsigned active=__activemask();constexpr int BytesA=A==primitive::Quant::Q4K?144:210,BytesB=B==primitive::Quant::Q4K?144:210;
    const int blocks=first.columns/256;float a[Tokens]{},b[Tokens]{};
    for(int k=0;k<blocks;++k){
        const auto* p=static_cast<const unsigned char*>(first.data)+(size_t(row)*blocks+k)*BytesA;
        const auto* q=Pair?static_cast<const unsigned char*>(second.data)+(size_t(row)*blocks+k)*BytesB:nullptr;
        #pragma unroll
        for(int t=0;t<Tokens;++t){const auto& code=input[size_t(t)*(first.columns/32)+k*8+group];
            a[t]=term<A>(p,code,group,a[t]);if constexpr(Pair)b[t]=term<B>(q,code,group,b[t]);}
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){const float v=sum8(a[t],active);float u=0;if constexpr(Pair)u=sum8(b[t],active);
        if(!group){const size_t at=size_t(t)*first.rows+row;x[at]=v;
            if constexpr(Pair){y[at]=u;const float z=0.7978845608028654f*(v+0.044715f*v*v*v);activation[at]=(0.5f*v*(1.f+tanhf(z)))*u;}}}
}
void validate(primitive::Matrix m){
    if(!m.data||m.rows<1||m.columns<256||m.columns%256||m.layout!=primitive::BlockLayout::Gguf||
        (m.quant!=primitive::Quant::Q4K&&m.quant!=primitive::Quant::Q6K))throw std::invalid_argument("Gemma grouped raw inventory");
}
template<primitive::Quant A,primitive::Quant B,bool Pair>
void launch(primitive::Matrix a,primitive::Matrix b,const void* in,float* x,float* y,float* act,int tokens,cudaStream_t stream){
    const auto* p=static_cast<const ResidualBlock*>(in);const int grid=(a.rows+15)/16;
    if(tokens==1)multiply<A,B,1,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
    else if(tokens==2)multiply<A,B,2,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
    else multiply<A,B,3,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
}
template<bool Pair>
void dispatch(primitive::Matrix a,primitive::Matrix b,const void* in,float* x,float* y,float* act,int tokens,cudaStream_t stream){
    using Q=primitive::Quant;
    if(a.quant==Q::Q4K){if(b.quant==Q::Q4K)launch<Q::Q4K,Q::Q4K,Pair>(a,b,in,x,y,act,tokens,stream);else launch<Q::Q4K,Q::Q6K,Pair>(a,b,in,x,y,act,tokens,stream);}
    else{if(b.quant==Q::Q4K)launch<Q::Q6K,Q::Q4K,Pair>(a,b,in,x,y,act,tokens,stream);else launch<Q::Q6K,Q::Q6K,Pair>(a,b,in,x,y,act,tokens,stream);}
    primitive::check_cuda(cudaGetLastError(),"Gemma grouped integer projection");
}
}
void linear_vector_grouped(primitive::Matrix m,const void* input,float* output,int tokens,cudaStream_t stream){
    validate(m);if(!input||!output||tokens<1||tokens>3)throw std::invalid_argument("Gemma grouped vector span");
    dispatch<false>(m,m,input,output,nullptr,nullptr,tokens,stream);
}
void pair_vector_grouped(primitive::Matrix a,primitive::Matrix b,const void* input,float* x,float* y,float* activation,int tokens,cudaStream_t stream){
    validate(a);validate(b);if(!input||!x||!y||!activation||x==y||x==activation||y==activation||tokens<1||tokens>3||a.rows!=b.rows||a.columns!=b.columns)
        throw std::invalid_argument("Gemma grouped pair geometry");
    dispatch<true>(a,b,input,x,y,activation,tokens,stream);
}
}
