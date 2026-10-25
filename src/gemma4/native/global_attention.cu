#include "strata/gemma4/native/attention_parallel_merge.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Partition=128,Warps=4;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
template<int Dim>
__global__ void segment(const float* query,float* partial,CacheView c,const int* position,int heads,int parts){
    __shared__ float vectors[Warps*Dim],m[Warps],z[Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y,t=blockIdx.z;
    const int qpos=*position+t,first=c.window?max(0,qpos-c.window+1):0;
    if(qpos<0||(!c.window&&qpos>=c.capacity)){
        float* invalid=partial+((size_t(t)*heads+head)*parts+part)*(Dim+2);
        for(int i=threadIdx.x;i<Dim+2;i+=128)invalid[i]=CUDART_NAN_F;return;
    }
    const int begin=first+part*Partition,end=min(qpos,begin+Partition-1),kh=head/(heads/c.heads);
    float q[Dim/32],a[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j)q[j]=query[(size_t(t)*heads+head)*Dim+lane+32*j];
    float top=-CUDART_INF_F,denominator=0;int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=end;key+=Warps){
        const size_t at=(size_t(slot)*c.heads+kh)*(Dim/32)*34;float score=0;
        #pragma unroll
        for(int j=0;j<Dim/32;++j)score=fmaf(q[j],read_q8(c.key,at,lane+32*j),score);
        score=sum(score);const float next=fmaxf(top,score),old=expf(top-next),current=expf(score-next);
        #pragma unroll
        for(int j=0;j<Dim/32;++j)a[j]=fmaf(current,read_q8(c.value,at,lane+32*j),old*a[j]);
        denominator=fmaf(denominator,old,current);top=next;slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    if(!lane){m[warp]=top;z[warp]=denominator;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j)vectors[warp*Dim+lane+32*j]=a[j];
    __syncthreads();
    const float all=fmaxf(fmaxf(m[0],m[1]),fmaxf(m[2],m[3]));float weights[Warps],total=0;
    #pragma unroll
    for(int w=0;w<Warps;++w){weights[w]=z[w]==0?0:expf(m[w]-all);total=fmaf(z[w],weights[w],total);}
    float* out=partial+((size_t(t)*heads+head)*parts+part)*(Dim+2);
    if(!threadIdx.x){out[0]=all;out[1]=total;}
    for(int i=threadIdx.x;i<Dim;i+=128){float value=0;
        #pragma unroll
        for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[w*Dim+i],value);
        out[2+i]=value;}
}
bool span(const void* p,size_t n,size_t a){
    const auto x=reinterpret_cast<std::uintptr_t>(p);return p&&n&&x%a==0&&n<=(std::numeric_limits<std::uintptr_t>::max)()-x;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<=y?y-x<an:x-y<bn;
}
}
namespace merge321 {
void split_attention_parallel_merge(const float* q,float* output,CacheView c,const int* position,
    int heads,int tokens,Lease lease,Binding live,Mode mode){
    if(heads!=Heads||c.heads!=4||c.dim!=Dim||c.window!=0||c.capacity<1||c.capacity>262144||tokens<1||tokens>3||tokens>c.capacity||
       (mode!=Mode::Parent&&mode!=Mode::Parallel))throw std::invalid_argument("321 global Q32/KV4/D512/T1..3 geometry");
    const auto l=layout((c.capacity+127)/128,tokens);const size_t needed=partial_bytes(l),io=output_bytes(l);
    const size_t kv=size_t(c.capacity)*4*544;const auto owned=lease.binding;const auto w=lease.partial;
    if(!live.owner||!live.epoch||!live.stream||owned.owner!=live.owner||owned.epoch!=live.epoch||owned.stream!=live.stream||
       !span(q,io,4)||!span(output,io,4)||!span(position,4,4)||!span(c.key,kv,2)||!span(c.value,kv,2)||
       !span(w.data,w.bytes,4)||w.bytes<needed)throw std::invalid_argument("321 global typed source/partial lease");
    const void* sources[]{q,c.key,c.value,position};const size_t bytes[]{io,kv,kv,4};
    if(overlap(output,io,w.data,w.bytes))throw std::invalid_argument("321 global output/partial alias");
    for(int i=0;i<4;++i)if(overlap(output,io,sources[i],bytes[i])||overlap(w.data,w.bytes,sources[i],bytes[i]))
        throw std::invalid_argument("321 global source/output/partial alias");
    if(mode==Mode::Parent){native::split_attention(q,output,c,position,heads,tokens,w,live.stream);return;}
    // This is frozen174's full static-grid segment, NOT266's unwritten-future variant.
    segment<Dim><<<dim3(Heads,l.parts,tokens),128,0,live.stream>>>(q,static_cast<float*>(w.data),c,position,heads,l.parts);
    merge_aos(output,l,{position,0,c.capacity,nullptr,nullptr,nullptr},lease,live);
    primitive::check_cuda(cudaGetLastError(),"321 original174 segment + parallel explicit-stride merge");
}
}
}
