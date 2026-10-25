#include "strata/gemma4/native/attention_global_gqa_shared.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <limits>
namespace strata::gemma4::native {
namespace {
// The266 merge below is retained verbatim. Its four warps are independent of
// the new segment's eight Q-head warps and256threads.
constexpr int Dim=512,Heads=32,KvHeads=4,Threads=128,Tile=128,MaxParts=2048;
constexpr int Partition=128,Warps=4,GqaWarps=8,TileKeys=4,GqaThreads=256;
constexpr size_t SharedBytes=size_t(2)*TileKeys*Dim*sizeof(float);
static_assert(SharedBytes==16384);
bool span(const void* p,size_t n,size_t alignment){
    const auto x=reinterpret_cast<std::uintptr_t>(p);
    return p&&n&&x%alignment==0&&n<=(std::numeric_limits<std::uintptr_t>::max)()-x;
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
__device__ float sum(float x){for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffff,x,d);return x;}
__device__ float maximum(float x){for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read_q8(const void* data,size_t at,int d){
    const auto* raw=static_cast<const unsigned char*>(data)+at+(d/32)*34;
    return __half2float(*reinterpret_cast<const half*>(raw))*float(reinterpret_cast<const signed char*>(raw+2)[d%32]);
}
__global__ void segment_shared(const float* query,float* partial,CacheView c,
    const int* position,int parts,int context){
    __shared__ float kv[2*TileKeys*Dim];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,kh=blockIdx.x,part=blockIdx.y;
    const int p=*position,begin=part*Partition;
    // Uniform CTA return before any barrier/query/KV/scratch access. Merge
    // handles invalidP and reads only active parts, including after rewind.
    if(p<0||p>=context||begin>p)return;
    const int end=min(p,begin+Partition-1),head=kh*GqaWarps+warp;
    float q[Dim/32],a[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j)q[j]=query[size_t(head)*Dim+lane+32*j];
    float top=-CUDART_INF_F,denominator=0.f;
    for(int first=begin;first<=end;first+=TileKeys){
        const int valid=min(TileKeys,end-first+1);
        // Every physical K/V coordinate is decoded once by the CTA. No read
        // beyond the actual GPU position, even inside the final tile.
        for(int index=tid;index<2*TileKeys*Dim;index+=GqaThreads){
            const int which=index/(TileKeys*Dim),key=(index/Dim)%TileKeys,d=index%Dim;
            if(key<valid){
                const size_t at=(size_t(first+key)*KvHeads+kh)*(Dim/32)*34;
                kv[index]=read_q8(which?c.value:c.key,at,d);
            }
        }
        __syncthreads();
        for(int key=0;key<valid;++key){
            float score=0.f;
            #pragma unroll
            for(int j=0;j<Dim/32;++j)score=fmaf(q[j],kv[key*Dim+lane+32*j],score);
            score=sum(score);const float next=fmaxf(top,score);
            const float old=expf(top-next),current=expf(score-next);
            #pragma unroll
            for(int j=0;j<Dim/32;++j)
                a[j]=fmaf(current,kv[(TileKeys+key)*Dim+lane+32*j],old*a[j]);
            denominator=fmaf(denominator,old,current);top=next;
        }
        __syncthreads(); // Protect this tile until every Q-head warp is done.
    }
    float* out=partial+(size_t(head)*parts+part)*(Dim+2);
    if(!lane){out[0]=top;out[1]=denominator;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j)out[2+lane+32*j]=a[j];
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
void global_gqa_shared_t1(const float* query,float* output,CacheView cache,const int* position,
    int heads,int tokens,AosMergeT1Lease lease,AosMergeT1Binding live){
    const auto layout=global_gqa_shared_t1_layout(cache,heads,lease.max_position_exclusive);
    const auto binding=lease.binding;const auto w=lease.workspace;
    const size_t vectors=size_t(Heads)*Dim*sizeof(float),kv=size_t(cache.capacity)*KvHeads*(Dim/32)*34;
    if(tokens!=1||!live.owner||!live.epoch||!live.stream||binding.owner!=live.owner||binding.epoch!=live.epoch||binding.stream!=live.stream||
        !span(query,vectors,4)||!span(output,vectors,4)||!span(position,4,4)||
        !span(cache.key,kv,2)||!span(cache.value,kv,2)||!span(w.data,w.bytes,256)||w.bytes<layout.bytes)
        throw std::invalid_argument("Gemma280 globalGQA T1 source/owner/epoch/stream/workspace");
    const void* sources[]{query,cache.key,cache.value,position};const size_t bytes[]{vectors,kv,kv,sizeof(int)};
    if(overlap(output,vectors,w.data,w.bytes))throw std::invalid_argument("Gemma280 output/arena alias");
    for(int i=0;i<4;++i)if(overlap(output,vectors,sources[i],bytes[i])||overlap(w.data,w.bytes,sources[i],bytes[i]))
        throw std::invalid_argument("Gemma280 source/output/arena alias");
    auto* partial=static_cast<float*>(w.data);
    segment_shared<<<dim3(KvHeads,layout.parts),GqaThreads,0,live.stream>>>(
        query,partial,cache,position,layout.parts,lease.max_position_exclusive);
    merge<<<dim3(Heads,Dim/Tile),Threads,0,live.stream>>>(partial,output,position,lease.max_position_exclusive,layout.parts);
    primitive::check_cuda(cudaGetLastError(),"Gemma280 GQAsharedQ8 segment + unchanged266 AoSmerge");
}
}
