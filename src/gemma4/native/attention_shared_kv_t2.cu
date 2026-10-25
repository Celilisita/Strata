#include "strata/gemma4/native/attention_shared_kv_t2.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <limits>
#include <stdexcept>

namespace strata::gemma4::native::shared336 {
namespace {
constexpr int Dim=512,Heads=32,Partition=128,Warps=4,Threads=128;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
__global__ void segment_pair(const float* query,float* partial,CacheView c,const int* position,int parts){
    __shared__ float vectors[2][Warps*Dim],m[2][Warps],z[2][Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y;
    const long long p0=static_cast<long long>(*position),p1=p0+1;
    const bool valid0=p0>=0&&p0<c.capacity,valid1=p1>=0&&p1<c.capacity;
    const int begin=part*Partition,kh=head/8;
    const int end0=valid0?min(int(p0),begin+Partition-1):-1;
    const int end1=valid1?min(int(p1),begin+Partition-1):-1;
    float q0[Dim/32],q1[Dim/32],a0[Dim/32]{},a1[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j){
        q0[j]=valid0?query[size_t(head)*Dim+lane+32*j]:0.f;
        q1[j]=valid1?query[(size_t(Heads)+head)*Dim+lane+32*j]:0.f;
    }
    float top0=-CUDART_INF_F,top1=-CUDART_INF_F,denominator0=0,denominator1=0;
    int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=max(end0,end1);key+=Warps){
        const bool active0=key<=end0,active1=key<=end1;
        const size_t at=(size_t(slot)*c.heads+kh)*(Dim/32)*34;
        float score0=0,score1=0;
        #pragma unroll
        for(int j=0;j<Dim/32;++j){const float value=read_q8(c.key,at,lane+32*j);
            if(active0)score0=fmaf(q0[j],value,score0);
            if(active1)score1=fmaf(q1[j],value,score1);
        }
        score0=sum(score0);score1=sum(score1);
        float old0=0,current0=0,old1=0,current1=0;
        if(active0){const float next=fmaxf(top0,score0);old0=expf(top0-next);current0=expf(score0-next);top0=next;}
        if(active1){const float next=fmaxf(top1,score1);old1=expf(top1-next);current1=expf(score1-next);top1=next;}
        #pragma unroll
        for(int j=0;j<Dim/32;++j){const float value=read_q8(c.value,at,lane+32*j);
            if(active0)a0[j]=fmaf(current0,value,old0*a0[j]);
            if(active1)a1[j]=fmaf(current1,value,old1*a1[j]);
        }
        if(active0)denominator0=fmaf(denominator0,old0,current0);
        if(active1)denominator1=fmaf(denominator1,old1,current1);
        slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    if(!lane){m[0][warp]=top0;z[0][warp]=denominator0;m[1][warp]=top1;z[1][warp]=denominator1;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j){vectors[0][warp*Dim+lane+32*j]=a0[j];vectors[1][warp*Dim+lane+32*j]=a1[j];}
    __syncthreads();
    #pragma unroll
    for(int t=0;t<2;++t){
        float* out=partial+((size_t(t)*Heads+head)*parts+part)*(Dim+2);
        if(t==0?!valid0:!valid1){for(int i=threadIdx.x;i<Dim+2;i+=Threads)out[i]=CUDART_NAN_F;continue;}
        const float all=fmaxf(fmaxf(m[t][0],m[t][1]),fmaxf(m[t][2],m[t][3]));
        float weights[Warps],total=0;
        #pragma unroll
        for(int w=0;w<Warps;++w){weights[w]=z[t][w]==0?0:expf(m[t][w]-all);total=fmaf(z[t][w],weights[w],total);}
        if(!threadIdx.x){out[0]=all;out[1]=total;}
        for(int i=threadIdx.x;i<Dim;i+=Threads){float value=0;
            #pragma unroll
            for(int w=0;w<Warps;++w)value=fmaf(weights[w],vectors[t][w*Dim+i],value);
            out[2+i]=value;
        }
    }
}
bool span(const void* p,size_t n,size_t a){
    const auto x=reinterpret_cast<std::uintptr_t>(p);return p&&n&&x%a==0&&n<=(std::numeric_limits<std::uintptr_t>::max)()-x;
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);return x<=y?y-x<an:x-y<bn;
}
}
void split_attention_shared_kv_t2(const float* q,float* output,CacheView c,const int* position,
    int heads,int tokens,merge321::Lease lease,merge321::Binding live){
    if(heads!=Heads||c.heads!=4||c.dim!=Dim||c.window!=0||c.capacity<2||c.capacity>32768||tokens!=2)
        throw std::invalid_argument("336 global Q32/KV4/D512/T2 capacity2..32768");
    const auto l=merge321::layout((c.capacity+127)/128,2);
    const size_t needed=merge321::partial_bytes(l),io=merge321::output_bytes(l),kv=size_t(c.capacity)*4*544;
    const auto owned=lease.binding;const auto w=lease.partial;
    if(!live.owner||!live.epoch||!live.stream||owned.owner!=live.owner||owned.epoch!=live.epoch||owned.stream!=live.stream||
       !span(q,io,4)||!span(output,io,4)||!span(position,4,4)||!span(c.key,kv,2)||!span(c.value,kv,2)||
       !span(w.data,w.bytes,4)||w.bytes<needed)throw std::invalid_argument("336 global typed source/partial lease");
    const void* sources[]{q,c.key,c.value,position};const size_t bytes[]{io,kv,kv,4};
    if(overlap(output,io,w.data,w.bytes))throw std::invalid_argument("336 global output/partial alias");
    for(int i=0;i<4;++i)if(overlap(output,io,sources[i],bytes[i])||overlap(w.data,w.bytes,sources[i],bytes[i]))
        throw std::invalid_argument("336 global source/output/partial alias");
    segment_pair<<<dim3(Heads,l.parts),Threads,0,live.stream>>>(q,static_cast<float*>(w.data),c,position,l.parts);
    merge321::merge_aos(output,l,{position,0,c.capacity,nullptr,nullptr,nullptr},lease,live);
    primitive::check_cuda(cudaGetLastError(),"336 paired T2 shared-KV segment + frozen321 merge");
}
}
