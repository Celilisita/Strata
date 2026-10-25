#include "strata/gemma4/native/attention_swa_warp_merge.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Dim=256,Heads=32,KvHeads=16,Parts=32,Tile=32;
constexpr size_t ScratchBytes=size_t(Heads)*Parts*(Dim+2)*sizeof(float);
static_assert(ScratchBytes==1056768);
bool span(const void* p,size_t bytes,size_t align){const auto x=reinterpret_cast<std::uintptr_t>(p);
    return p&&bytes&&x%align==0&&bytes<=(std::numeric_limits<std::uintptr_t>::max)()-x;}
bool overlap(const void* a,size_t na,const void* b,size_t nb){const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;}
constexpr int Partition=32,Warps=4;
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
    if(qpos<0||qpos>=context)return; // Uniform context guard; active174 math/ring unchanged.
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
__global__ void merge_warp(const float* partial,float* output,const int* position,int context){
    const int lane=threadIdx.x,head=blockIdx.x,tile=blockIdx.y,p=*position;
    const size_t out=size_t(head)*Dim+tile*Tile+lane;
    if(p<0||p>=context){output[out]=CUDART_NAN_F;return;}
    const int active=(min(p+1,1024)+31)/32;
    const float* row=partial+size_t(head)*Parts*(Dim+2);
    const float z=lane<active?row[lane*(Dim+2)+1]:0.f;
    const float m=lane<active?row[lane*(Dim+2)]:-CUDART_INF_F;
    const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(z)||z<0||(z>0&&!isfinite(m)));
    const float top=maximum(z>0?m:-CUDART_INF_F),weight=z>0?expf(m-top):0.f;
    const float denominator=sum(fmaf(z,weight,0.f));float value=0.f;
    #pragma unroll
    for(int part=0;part<Parts;++part){
        const float part_z=__shfl_sync(0xffffffff,z,part),part_weight=__shfl_sync(0xffffffff,weight,part);
        // Keep activeVNaN visible even if the wholepart probability underflows0.
        if(part_z>0)value=fmaf(part_weight,row[part*(Dim+2)+2+tile*Tile+lane],value);
    }
    output[out]=invalid||!(denominator>0)||!isfinite(value)?CUDART_NAN_F:value/denominator;
}
}
size_t swa_warp_merge_t1_bytes(CacheView c,int heads){
    if(heads!=Heads||c.heads!=KvHeads||c.dim!=Dim||c.window!=1024||c.capacity<1024||c.capacity>9216)
        throw std::invalid_argument("Gemma270 fixedSWA geometry");return ScratchBytes;
}
void swa_warp_merge_t1(const float* query,float* output,CacheView cache,const int* position,
    int heads,int tokens,SwaWarpMergeT1Lease lease,SwaWarpMergeT1Binding live){
    const auto binding=lease.binding;const auto stream=live.stream;
    const size_t bytes=swa_warp_merge_t1_bytes(cache,heads),vectors=size_t(Heads)*Dim*4;
    const size_t kv=size_t(cache.capacity)*KvHeads*(Dim/32)*34;
    if(tokens!=1||!live.owner||!live.epoch||!stream||binding.owner!=live.owner||binding.epoch!=live.epoch||binding.stream!=stream||
       lease.max_position_exclusive<cache.capacity||
       lease.max_position_exclusive>262144||!span(query,vectors,4)||!span(output,vectors,4)||!span(position,4,4)||
       !span(cache.key,kv,2)||!span(cache.value,kv,2)||!span(lease.data,lease.bytes,256)||lease.bytes<bytes)
        throw std::invalid_argument("Gemma270 liveSWA lease/source/context");
    if(overlap(output,vectors,lease.data,lease.bytes))throw std::invalid_argument("Gemma270 output/arena alias");
    const void* sources[]{query,cache.key,cache.value,position};const size_t sizes[]{vectors,kv,kv,sizeof(int)};
    for(int i=0;i<4;++i)if(overlap(output,vectors,sources[i],sizes[i])||overlap(lease.data,lease.bytes,sources[i],sizes[i]))
        throw std::invalid_argument("Gemma270 source/output/arena alias");
    auto* partial=static_cast<float*>(lease.data);
    segment<Dim><<<dim3(Heads,Parts,1),128,0,stream>>>(query,partial,cache,position,heads,Parts,lease.max_position_exclusive);
    merge_warp<<<dim3(Heads,Dim/Tile),32,0,stream>>>(partial,output,position,lease.max_position_exclusive);
    primitive::check_cuda(cudaGetLastError(),"Gemma270 original174 SWA + warpD32 merge");
}
}
