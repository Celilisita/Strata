#include "strata/gemma4/native/attention_global_aos_merge.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Dim=512,Heads=32,KvHeads=4,Threads=128,Tile=128,MaxParts=2048;
bool span(const void* p,size_t n,size_t alignment){
    const auto x=reinterpret_cast<std::uintptr_t>(p);
    return p&&n&&x%alignment==0&&n<=(std::numeric_limits<std::uintptr_t>::max)()-x;
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
constexpr int Partition=128,Warps=4;
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read_q8(const void* data,size_t at,int i){
    const auto* p=static_cast<const unsigned char*>(data)+at+(i/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[i%32]);
}
template<int Dim>
__global__ void segment(const float* query,float* partial,CacheView c,const int* position,int heads,int parts,int context){
    __shared__ float vectors[Warps*Dim],m[Warps],z[Warps];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,head=blockIdx.x,part=blockIdx.y,t=blockIdx.z;
    const int qpos=*position+t,first=c.window?max(0,qpos-c.window+1):0;
    if(qpos<0||qpos>=context||part*Partition>qpos)return; // No invalid/future KV or scratch access.
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
__global__ void merge(const float* partial,float* output,
    const int* position,int context,int max_parts){
    __shared__ float weights[MaxParts],reduction[Warps],vectors[Warps*Tile],top,denominator;
    __shared__ unsigned char present[MaxParts];
    __shared__ unsigned invalid[Warps],bad_stats;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,tile=blockIdx.y,p=*position;
    const size_t out=size_t(head)*Dim+tile*Tile+tid;
    if(p<0||p>=context){output[out]=CUDART_NAN_F;return;}
    const int parts=(p+Partition)/Partition;const size_t base=size_t(head)*max_parts;
    float value=-CUDART_INF_F;bool bad=false;
    for(int part=tid;part<parts;part+=Threads){
        const float z=partial[(base+part)*(Dim+2)+1],m=partial[(base+part)*(Dim+2)];
        bad|=!isfinite(z)||z<0||(z>0&&!isfinite(m));if(z>0)value=fmaxf(value,m);
    }
    const unsigned mask=__ballot_sync(0xffffffff,bad);value=maximum(value);
    if(!lane){invalid[warp]=mask;reduction[warp]=value;}__syncthreads();
    if(!warp){const float all=maximum(lane<Warps?reduction[lane]:-CUDART_INF_F);
        if(!lane){top=all;bad_stats=invalid[0]|invalid[1]|invalid[2]|invalid[3];}}
    __syncthreads();
    float total=0;
    for(int part=tid;part<parts;part+=Threads){
        const float z=partial[(base+part)*(Dim+2)+1];const bool active=z>0;
        const float weight=active?expf(partial[(base+part)*(Dim+2)]-top):0.f;
        present[part]=active;weights[part]=weight;total=fmaf(z,weight,total);
    }
    total=sum(total);if(!lane)reduction[warp]=total;__syncthreads();
    if(!warp){const float all=sum(lane<Warps?reduction[lane]:0.f);if(!lane)denominator=all;}
    __syncthreads();
    // Four part stripes, AoS metadata/numerators read directly; no SoA copy.
    // present (z>0), rather than weight!=0, retains active NaN propagation.
    float accum[Tile/32]{};
    for(int part=warp;part<parts;part+=Warps)if(present[part]){
        const float weight=weights[part];
        #pragma unroll
        for(int j=0;j<Tile/32;++j)
            accum[j]=fmaf(weight,partial[(base+part)*(Dim+2)+2+tile*Tile+lane+j*32],accum[j]);
    }
    #pragma unroll
    for(int j=0;j<Tile/32;++j)vectors[warp*Tile+lane+j*32]=accum[j];
    __syncthreads();value=0;
    #pragma unroll
    for(int w=0;w<Warps;++w)value=__fadd_rn(value,vectors[w*Tile+tid]);
    output[out]=bad_stats||!(denominator>0)||!isfinite(value)?CUDART_NAN_F:value/denominator;
}
}

void global_parallel_aos_merge_t1(const float* query,float* output,CacheView cache,const int* position,
    int heads,int tokens,AosMergeT1Lease lease,AosMergeT1Binding live){
    const auto layout=global_parallel_aos_merge_t1_layout(cache,heads,lease.max_position_exclusive);
    const auto owned=lease.binding;const auto w=lease.workspace;
    const size_t vector_bytes=size_t(Heads)*Dim*sizeof(float),kv_bytes=size_t(cache.capacity)*KvHeads*(Dim/32)*34;
    if(tokens!=1||!live.owner||!live.epoch||!live.stream||owned.owner!=live.owner||owned.epoch!=live.epoch||owned.stream!=live.stream||
       !span(query,vector_bytes,4)||!span(output,vector_bytes,4)||!span(position,4,4)||
       !span(cache.key,kv_bytes,2)||!span(cache.value,kv_bytes,2)||!span(w.data,w.bytes,256)||w.bytes<layout.bytes)
        throw std::invalid_argument("Gemma AoS merge T1 source/owner/epoch/stream/workspace");
    const void* sources[]{query,cache.key,cache.value,position};const size_t bytes[]{vector_bytes,kv_bytes,kv_bytes,sizeof(int)};
    if(overlap(output,vector_bytes,w.data,w.bytes))throw std::invalid_argument("Gemma AoS merge output/arena alias");
    for(int i=0;i<4;++i)if(overlap(output,vector_bytes,sources[i],bytes[i])||overlap(w.data,w.bytes,sources[i],bytes[i]))
        throw std::invalid_argument("Gemma AoS merge source/output/arena alias");
    auto* partial=static_cast<float*>(w.data);
    segment<Dim><<<dim3(Heads,layout.parts,1),Threads,0,live.stream>>>(query,partial,cache,position,heads,layout.parts,lease.max_position_exclusive);
    merge<<<dim3(Heads,Dim/Tile),Threads,0,live.stream>>>(partial,output,position,lease.max_position_exclusive,layout.parts);
    primitive::check_cuda(cudaGetLastError(),"Gemma original174 segment + parallel AoS merge T1");
}
}
