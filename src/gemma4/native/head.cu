#include "strata/gemma4/native/decoder_ops.hpp"
#include <cmath>
#include <climits>

namespace strata::gemma4::native {
namespace {
struct Candidate {float value;int id;};
__device__ Candidate better(Candidate a,Candidate b){
    return b.value>a.value||(b.value==a.value&&b.id<a.id)?b:a;
}
__device__ Candidate reduce(Candidate x){
    for(int d=16;d;d/=2){Candidate y{__shfl_down_sync(0xffffffff,x.value,d),__shfl_down_sync(0xffffffff,x.id,d)};
        if((threadIdx.x%32)+d<32)x=better(x,y);}
    return x;
}
__global__ void partial_pick(float* logits,Candidate* partial,unsigned* invalid,int count,float cap){
    __shared__ Candidate warps[8];
    const int id=blockIdx.x*256+threadIdx.x;Candidate candidate{-INFINITY,INT_MAX};
    if(id<count){
        const float raw=logits[id];
        if(!isfinite(raw))atomicExch(invalid,1u);
        const float value=cap*tanhf(raw/cap);logits[id]=value;
        if(isfinite(raw))candidate={value,id};
    }
    candidate=reduce(candidate);if(!(threadIdx.x%32))warps[threadIdx.x/32]=candidate;__syncthreads();
    if(threadIdx.x<32){candidate=threadIdx.x<8?warps[threadIdx.x]:Candidate{-INFINITY,INT_MAX};
        candidate=reduce(candidate);if(!threadIdx.x)partial[blockIdx.x]=candidate;}
}
__global__ void final_pick(const Candidate* partial,int count,int* next,int* position,unsigned* invalid,int tokens){
    __shared__ Candidate warps[8];Candidate candidate{-INFINITY,INT_MAX};
    for(int i=threadIdx.x;i<count;i+=256)candidate=better(candidate,partial[i]);
    candidate=reduce(candidate);if(!(threadIdx.x%32))warps[threadIdx.x/32]=candidate;__syncthreads();
    if(threadIdx.x<32){candidate=threadIdx.x<8?warps[threadIdx.x]:Candidate{-INFINITY,INT_MAX};candidate=reduce(candidate);
        if(!threadIdx.x){
            if(candidate.id==INT_MAX||!isfinite(candidate.value))atomicExch(invalid,1u);
            *next=candidate.id;*position+=tokens;
        }
    }
}
__global__ void advance_position(int* position,int tokens,int capacity,unsigned* invalid){
    if(*position<0||*position>capacity-tokens){atomicExch(invalid,1u);return;}*position+=tokens;
}
}
void pick(float* logits,float* partial,int* next,int* position,unsigned* invalid,int vocabulary,int tokens,float cap,cudaStream_t stream){
    static_assert(sizeof(Candidate)==2*sizeof(float));
    if(!logits||!partial||!next||!position||!invalid||vocabulary<1||vocabulary>262144||tokens<1||tokens>8192||!std::isfinite(cap)||cap<=0)
        throw std::invalid_argument("Gemma capped greedy sampling contract");
    const int blocks=(vocabulary+255)/256;auto* candidates=reinterpret_cast<Candidate*>(partial);
    partial_pick<<<blocks,256,0,stream>>>(logits,candidates,invalid,vocabulary,cap);
    final_pick<<<1,256,0,stream>>>(candidates,blocks,next,position,invalid,tokens);
    primitive::check_cuda(cudaGetLastError(),"Gemma capped head and device position");
}
void advance(int* position,int tokens,int capacity,unsigned* invalid,cudaStream_t stream){
    if(!position||!invalid||tokens<1||tokens>capacity)throw std::invalid_argument("Gemma device position extent");
    advance_position<<<1,1,0,stream>>>(position,tokens,capacity,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma device position advance");
}
}
