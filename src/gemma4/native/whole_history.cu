#include "whole_history.hpp"
#include <cuda_fp16.h>
#include <cmath>
namespace strata::gemma4::native::whole_history_detail {
namespace p=strata::qwen35::native;
namespace {
__device__ bool valid(const int* position,int tokens,int capacity,int keys,unsigned* invalid){const int first=*position;
    const bool ok=first>=0&&tokens<=capacity&&first<=capacity-tokens&&first<=keys-tokens;if(!ok)atomicOr(invalid,1u);return ok;}
__global__ void pack_kernel(Buffers b,const float* q,CacheView cache,const int* position,int tokens){
    const bool ok=valid(position,tokens,cache.capacity,b.keys,b.invalid);const int active=ok?*position+tokens:0;
    const auto kvcount=std::size_t(b.keys)*KvHeads*Dim;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<kvcount;i+=std::size_t(blockDim.x)*gridDim.x){
        const int d=int(i%Dim),row=int(i/Dim),key=row%b.keys,head=row/b.keys;half k=__float2half_rn(0),v=k;
        if(key<active){const auto at=((std::size_t(key)*KvHeads+head)*(Dim/32)+d/32)*34;
            const auto* kr=static_cast<const unsigned char*>(cache.key)+at;const auto* vr=static_cast<const unsigned char*>(cache.value)+at;
            const float a=__half2float(*reinterpret_cast<const half*>(kr))*float(reinterpret_cast<const signed char*>(kr+2)[d%32]);
            const float c=__half2float(*reinterpret_cast<const half*>(vr))*float(reinterpret_cast<const signed char*>(vr+2)[d%32]);
            k=__float2half_rn(a);v=__float2half_rn(c);if(!isfinite(a)||!isfinite(c)||!isfinite(__half2float(k))||!isfinite(__half2float(v)))atomicOr(b.invalid,2u);}
        static_cast<half*>(b.key)[i]=k;static_cast<half*>(b.value)[i]=v;
    }
    const auto qcount=std::size_t(tokens)*Heads*Dim;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<qcount;i+=std::size_t(blockDim.x)*gridDim.x){
        const int d=int(i%Dim),row=int(i/Dim),t=row%tokens,h=row/tokens;const float a=q[(std::size_t(t)*Heads+h)*Dim+d];
        const half x=__float2half_rn(a);if(!isfinite(a)||!isfinite(__half2float(x)))atomicOr(b.invalid,2u);static_cast<half*>(b.query)[i]=x;
    }
}
__global__ void pointer_kernel(Buffers b,int first,int tokens){const int h=threadIdx.x;if(h>=Heads)return;
    b.pointers[h]=static_cast<half*>(b.key)+std::size_t(h/8)*b.keys*Dim;
    b.pointers[Heads+h]=static_cast<half*>(b.query)+(std::size_t(h)*tokens+first)*Dim;
    b.pointers[2*Heads+h]=b.score+std::size_t(h)*Query*b.keys;
    b.pointers[3*Heads+h]=static_cast<half*>(b.value)+std::size_t(h/8)*b.keys*Dim;
    b.pointers[4*Heads+h]=static_cast<half*>(b.probability)+std::size_t(h)*Query*b.keys;
    b.pointers[5*Heads+h]=b.partial+std::size_t(h)*Query*Dim;
}
__global__ void normalize_kernel(Buffers b,const int* position,int first,int queries,int tokens,int capacity){
    const int h=blockIdx.x/queries,t=blockIdx.x%queries,lane=threadIdx.x;
    const bool ok=valid(position,tokens,capacity,b.keys,b.invalid);const int upper=ok?*position+first+t:-1;
    float* const scores=b.score+(std::size_t(h)*Query+t)*b.keys;
    half* const probability=static_cast<half*>(b.probability)+(std::size_t(h)*Query+t)*b.keys;
    __shared__ float reduction[256];float maximum=-INFINITY;
    for(int k=lane;k<b.keys;k+=256){float s=k<=upper?scores[k]:-INFINITY;
        if(k<=upper&&!isfinite(s)){atomicOr(b.invalid,4u);s=-INFINITY;}scores[k]=s;maximum=fmaxf(maximum,s);}
    reduction[lane]=maximum;__syncthreads();
    for(int stride=128;stride;stride/=2){if(lane<stride)reduction[lane]=fmaxf(reduction[lane],reduction[lane+stride]);__syncthreads();}
    maximum=reduction[0];__syncthreads();float den=0;
    for(int k=lane;k<b.keys;k+=256){const float e=isfinite(maximum)?expf(scores[k]-maximum):0.f;
        den+=e;probability[k]=__float2half_rn(e);}
    reduction[lane]=den;__syncthreads();
    for(int stride=128;stride;stride/=2){if(lane<stride)reduction[lane]+=reduction[lane+stride];__syncthreads();}
    if(lane==0){b.den[h*Query+t]=reduction[0];if(!(reduction[0]>0)||!isfinite(reduction[0]))atomicOr(b.invalid,8u);}
}
__global__ void finish_kernel(Buffers b,float* out,int first,int queries){const auto count=std::size_t(Heads)*queries*Dim;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x){
        const int d=int(i%Dim),row=int(i/Dim),t=row%queries,h=row/queries;
        const float x=b.partial[(std::size_t(h)*Query+t)*Dim+d]/b.den[h*Query+t];
        if(!isfinite(x))atomicOr(b.invalid,16u);out[(std::size_t(first+t)*Heads+h)*Dim+d]=x;
    }
}
__global__ void propagate_kernel(Buffers b,float* out,std::size_t count){if(!*b.invalid)return;
    for(std::size_t i=std::size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=std::size_t(blockDim.x)*gridDim.x)out[i]=nanf("");}
}
void stage(Buffers b,const float* q,CacheView cache,const int* position,int tokens,cudaStream_t stream){
    p::check_cuda(cudaMemsetAsync(b.invalid,0,4,stream),"wholehistory fresh invalid");pack_kernel<<<256,256,0,stream>>>(b,q,cache,position,tokens);
    p::check_cuda(cudaGetLastError(),"wholehistory once Q8KV/F16Q staging+padding");}
void pointers(Buffers b,int first,int tokens,cudaStream_t stream){pointer_kernel<<<1,32,0,stream>>>(b,first,tokens);p::check_cuda(cudaGetLastError(),"wholehistory32 pointer batch");}
void normalize(Buffers b,const int* position,int first,int queries,int tokens,int capacity,cudaStream_t stream){
    normalize_kernel<<<Heads*queries,256,0,stream>>>(b,position,first,queries,tokens,capacity);p::check_cuda(cudaGetLastError(),"wholehistory F32expden/P16 causal softmax");}
void finish(Buffers b,float* out,int first,int queries,cudaStream_t stream){finish_kernel<<<256,256,0,stream>>>(b,out,first,queries);p::check_cuda(cudaGetLastError(),"wholehistory F32 final divide/store");}
void propagate(Buffers b,float* out,int tokens,cudaStream_t stream){propagate_kernel<<<256,256,0,stream>>>(b,out,std::size_t(tokens)*Heads*Dim);p::check_cuda(cudaGetLastError(),"wholehistory complete invalid propagation");}
}
