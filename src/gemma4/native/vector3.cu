#include "strata/gemma4/native/vector3.hpp"
#include "small_prefill_values.cuh"
#include "vector3_layout.cuh"
#include <cuda_fp16.h>
namespace strata::gemma4::native {
namespace {
using detail::ResidualBlock;
__device__ float warp_sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void encode(const float* input,ResidualBlock* out,int columns){
    const int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32,t=blockIdx.y;if(group>=columns/32)return;
    float residual=input[size_t(t)*columns+group*32+lane];auto& block=out[size_t(t)*(columns/32)+group];
    #pragma unroll
    for(int r=0;r<3;++r){
        const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(residual));float a=fabsf(residual);
        for(int d=16;d;d/=2)a=fmaxf(a,__shfl_xor_sync(0xffffffff,a,d));
        const float scale=invalid?nanf(""):a/127.f;
        const int code=invalid||!(scale>0)?0:max(-127,min(127,__float2int_rn(residual/scale)));
        block.code[r][lane]=static_cast<signed char>(code);int total=code;
        for(int d=16;d;d/=2)total+=__shfl_xor_sync(0xffffffff,total,d);
        if(!lane){block.d[r]=scale;block.sum[r]=total;}
        residual=__fmaf_rn(-scale,float(code),residual);
    }
}
template<primitive::Quant Type>
__device__ float term(const unsigned char* raw,const ResidualBlock* x,int group,int word,float previous){
    using primitive::detail::weight_codes;using primitive::detail::metadata;
    const int a=weight_codes<Type>(raw,group,word*8),b=weight_codes<Type>(raw,group,word*8+4);
    float product=0,correction=0;
    #pragma unroll
    for(int r=0;r<3;++r){
        const int q0=*reinterpret_cast<const int*>(x[group].code[r]+word*8),q1=*reinterpret_cast<const int*>(x[group].code[r]+word*8+4);
        const int dot=__dp4a(a,q0,__dp4a(b,q1,0));product=__fmaf_rn(x[group].d[r],float(dot),product);
        if constexpr(Type==primitive::Quant::Q4K)if(!word)correction=__fmaf_rn(x[group].d[r],float(x[group].sum[r]),correction);
    }
    if constexpr(Type==primitive::Quant::Q4K){
        const unsigned m=metadata(raw,group,false);const float scale=__half2float(*reinterpret_cast<const half*>(raw))*float(m&63);
        previous=__fmaf_rn(scale,product,previous);
        if(!word)previous=__fmaf_rn(-__half2float(*reinterpret_cast<const half*>(raw+2))*float(m>>6),correction,previous);
    }else{
        const int scale=reinterpret_cast<const signed char*>(raw+192)[group*2+word/2];
        previous=__fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208))*float(scale),product,previous);
    }
    return previous;
}
template<primitive::Quant A,primitive::Quant B,int Tokens,bool Pair>
__global__ void multiply(primitive::Matrix first,primitive::Matrix second,const ResidualBlock* input,float* x,float* y,float* activation){
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;if(row>=first.rows)return;
    constexpr int BytesA=A==primitive::Quant::Q4K?144:210,BytesB=B==primitive::Quant::Q4K?144:210;
    const int blocks=first.columns/256,group=lane/4,word=lane%4;float a[Tokens]{},b[Tokens]{};
    for(int k=0;k<blocks;++k){
        const auto* p=static_cast<const unsigned char*>(first.data)+(size_t(row)*blocks+k)*BytesA;
        const auto* q=Pair?static_cast<const unsigned char*>(second.data)+(size_t(row)*blocks+k)*BytesB:nullptr;
        #pragma unroll
        for(int t=0;t<Tokens;++t){const auto* code=input+size_t(t)*(first.columns/32)+k*8;
            a[t]=term<A>(p,code,group,word,a[t]);if constexpr(Pair)b[t]=term<B>(q,code,group,word,b[t]);}
    }
    #pragma unroll
    for(int t=0;t<Tokens;++t){
        const float v=warp_sum(a[t]);float u=0;if constexpr(Pair)u=warp_sum(b[t]);
        if(!lane){const size_t at=size_t(t)*first.rows+row;x[at]=v;
            if constexpr(Pair){y[at]=u;const float z=0.7978845608028654f*(v+0.044715f*v*v*v);activation[at]=(0.5f*v*(1.f+tanhf(z)))*u;}}
    }
}
void validate(primitive::Matrix m){
    if(!m.data||m.rows<1||m.columns<256||m.columns%256||m.layout!=primitive::BlockLayout::Gguf||
        (m.quant!=primitive::Quant::Q4K&&m.quant!=primitive::Quant::Q6K))throw std::invalid_argument("Gemma residual vector raw inventory");
}
template<primitive::Quant A,primitive::Quant B,bool Pair>
void launch(primitive::Matrix a,primitive::Matrix b,const void* in,float* x,float* y,float* act,int tokens,cudaStream_t stream){
    const auto* p=static_cast<const ResidualBlock*>(in);const int grid=(a.rows+3)/4;
    if(tokens==1)multiply<A,B,1,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
    else if(tokens==2)multiply<A,B,2,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
    else multiply<A,B,3,Pair><<<grid,128,0,stream>>>(a,b,p,x,y,act);
}
template<bool Pair>
void dispatch(primitive::Matrix a,primitive::Matrix b,const void* in,float* x,float* y,float* act,int tokens,cudaStream_t stream){
    using Q=primitive::Quant;
    if(a.quant==Q::Q4K){if(b.quant==Q::Q4K)launch<Q::Q4K,Q::Q4K,Pair>(a,b,in,x,y,act,tokens,stream);else launch<Q::Q4K,Q::Q6K,Pair>(a,b,in,x,y,act,tokens,stream);}
    else{if(b.quant==Q::Q4K)launch<Q::Q6K,Q::Q4K,Pair>(a,b,in,x,y,act,tokens,stream);else launch<Q::Q6K,Q::Q6K,Pair>(a,b,in,x,y,act,tokens,stream);}
    primitive::check_cuda(cudaGetLastError(),"Gemma three-residual integer projection");
}
}
void prepare_vector3(const float* input,void* output,size_t capacity,int columns,int tokens,cudaStream_t stream){
    if(!input||!output||input==output||columns<256||columns%256||tokens<1||tokens>3||size_t(columns/32)*tokens*sizeof(ResidualBlock)>capacity)
        throw std::invalid_argument("Gemma three-residual activation arena");
    encode<<<dim3((columns/32+3)/4,tokens),128,0,stream>>>(input,static_cast<ResidualBlock*>(output),columns);
    primitive::check_cuda(cudaGetLastError(),"Gemma GPU F32-scale residual encoding");
}
void linear_vector3(primitive::Matrix m,const void* input,float* output,int tokens,cudaStream_t stream){
    validate(m);if(!input||!output||tokens<1||tokens>3)throw std::invalid_argument("Gemma residual projection span");
    dispatch<false>(m,m,input,output,nullptr,nullptr,tokens,stream);
}
void pair_vector3(primitive::Matrix a,primitive::Matrix b,const void* input,float* x,float* y,float* activation,int tokens,cudaStream_t stream){
    validate(a);validate(b);if(!input||!x||!y||!activation||x==y||x==activation||y==activation||tokens<1||tokens>3||a.rows!=b.rows||a.columns!=b.columns)
        throw std::invalid_argument("Gemma residual pair geometry");
    dispatch<true>(a,b,input,x,y,activation,tokens,stream);
}
}
