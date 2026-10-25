#include "sm120_mtp_assistant_split.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

namespace strata::gemma4::native::mtp_assistant {
namespace {
constexpr int Keys=128,Warps=4,MaximumParts=1024;
__device__ float sum(float x){
    for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__device__ float maximum(float x){
    for(int d=16;d;d/=2)x=fmaxf(x,__shfl_xor_sync(0xffffffff,x,d));return x;
}
__device__ bool valid(Bounds b,CacheView c,bool sliding,int context){
    const int origin=sliding?b.sliding_origin:b.global_origin;
    const int length=sliding?b.sliding_length:b.global_length;
    const int left=sliding?b.sliding_left:origin;
    return b.rope_position>=1&&b.rope_position<context&&b.kv_end==b.rope_position-1&&origin>=0&&
        length>=1&&length<=c.capacity&&int64_t(origin)+length-1==b.kv_end&&left>=origin&&left<=b.kv_end&&
        (sliding?b.kv_end-left+1<=1025:origin==0);
}
__device__ float read_q8(const void* data,size_t row,int element){
    const auto* block=static_cast<const unsigned char*>(data)+row+(element/32)*34;
    return __half2float(*reinterpret_cast<const half*>(block))*
        float(reinterpret_cast<const signed char*>(block+2)[element%32]);
}
template<int Dim>
__global__ void segment(const float* query,float* partial,CacheView c,const Bounds* bounds,
    bool sliding,int context,int parts,unsigned* invalid){
    __shared__ float vectors[Warps*Dim],tops[Warps],denominators[Warps];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x,part=blockIdx.y;
    float* destination=partial+(size_t(head)*parts+part)*(Dim+2);
    const Bounds b=*bounds;
    if(!valid(b,c,sliding,context)){
        if(!tid)atomicOr(invalid,2u);
        for(int j=tid;j<Dim+2;j+=128)destination[j]=NAN;return;
    }
    const int first=sliding?b.sliding_left:b.global_origin;
    const int begin=first+part*Keys,end=min(b.kv_end,begin+Keys-1);
    const int kvhead=head/(32/c.heads);
    float q[Dim/32],acc[Dim/32]{};
    #pragma unroll
    for(int j=0;j<Dim/32;++j){q[j]=query[size_t(head)*Dim+lane+j*32];
        if(!isfinite(q[j]))atomicOr(invalid,4u);}
    float top=-INFINITY,denominator=0;int slot=(begin+warp)%c.capacity;
    for(int key=begin+warp;key<=end;key+=Warps){
        const size_t row=(size_t(slot)*c.heads+kvhead)*(Dim/32)*34;float score=0;
        #pragma unroll
        for(int j=0;j<Dim/32;++j)score=__fmaf_rn(q[j],read_q8(c.key,row,lane+j*32),score);
        score=sum(score);const float next=fmaxf(top,score),a=__expf(top-next),p=__expf(score-next);
        #pragma unroll
        for(int j=0;j<Dim/32;++j)acc[j]=__fmaf_rn(p,read_q8(c.value,row,lane+j*32),a*acc[j]);
        denominator=__fmaf_rn(denominator,a,p);top=next;
        slot+=Warps;if(slot>=c.capacity)slot-=c.capacity;
    }
    if(!lane){tops[warp]=top;denominators[warp]=denominator;}
    #pragma unroll
    for(int j=0;j<Dim/32;++j)vectors[warp*Dim+lane+j*32]=acc[j];
    __syncthreads();
    const float all=fmaxf(fmaxf(tops[0],tops[1]),fmaxf(tops[2],tops[3]));
    float weights[Warps],total=0;
    #pragma unroll
    for(int w=0;w<Warps;++w){weights[w]=denominators[w]==0?0:__expf(tops[w]-all);
        total=__fmaf_rn(denominators[w],weights[w],total);}
    if(!tid){destination[0]=all;destination[1]=total;}
    for(int j=tid;j<Dim;j+=128){float value=0;
        #pragma unroll
        for(int w=0;w<Warps;++w)value=__fmaf_rn(weights[w],vectors[w*Dim+j],value);
        destination[j+2]=value;
    }
}
template<int Dim,int Threads>
__global__ void merge(const float* partial,float* output,CacheView c,const Bounds* bounds,
    bool sliding,int context,int parts,unsigned* invalid){
    __shared__ float weights[MaximumParts],reduction[8],block_top,block_denominator;
    __shared__ unsigned flags[8],block_errors;
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,head=blockIdx.x;const Bounds b=*bounds;
    if(!valid(b,c,sliding,context)){
        for(int j=tid;j<Dim;j+=Threads)output[size_t(head)*Dim+j]=NAN;return;
    }
    const float* source=partial+size_t(head)*parts*(Dim+2);
    float top=-INFINITY;bool bad=false;
    for(int i=tid;i<parts;i+=Threads){const float z=source[i*(Dim+2)+1],m=source[i*(Dim+2)];
        bad|=!isfinite(z)||(z>0&&!isfinite(m));if(z>0)top=fmaxf(top,m);}
    unsigned errors=__ballot_sync(0xffffffff,bad);top=maximum(top);
    if constexpr(Threads>32){
        if(!lane){reduction[warp]=top;flags[warp]=errors;}__syncthreads();
        if(!warp){float value=lane<Threads/32?reduction[lane]:-INFINITY;
            unsigned bits=lane<Threads/32?flags[lane]:0;
            value=maximum(value);for(int d=16;d;d/=2)bits|=__shfl_xor_sync(0xffffffff,bits,d);
            if(!lane){block_top=value;block_errors=bits;}}
        __syncthreads();top=block_top;errors=block_errors;
    }
    float denominator=0;
    for(int i=tid;i<parts;i+=Threads){const float z=source[i*(Dim+2)+1];
        const float w=z==0?0:__expf(source[i*(Dim+2)]-top);weights[i]=w;
        denominator=__fmaf_rn(z,w,denominator);}
    denominator=sum(denominator);
    if constexpr(Threads>32){
        if(!lane)reduction[warp]=denominator;__syncthreads();
        if(!warp){float value=lane<Threads/32?reduction[lane]:0;value=sum(value);
            if(!lane)block_denominator=value;}
        __syncthreads();denominator=block_denominator;
    }else __syncwarp();
    for(int j=tid;j<Dim;j+=Threads){float value=0;
        for(int i=0;i<parts;++i)if(weights[i]!=0)
            value=__fmaf_rn(weights[i],source[i*(Dim+2)+2+j],value);
        value=errors||!(denominator>0)?NAN:value/denominator;
        output[size_t(head)*Dim+j]=value;if(!isfinite(value))atomicOr(invalid,4u);
    }
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<=y?y-x<na:x-y<nb;
}
int partition_count(CacheView c,bool sliding,int context){
    if(context<2||context>262144||c.capacity<1||c.heads!=(sliding?16:4)||
        c.dim!=(sliding?256:512)||c.window!=(sliding?1024:0))
        throw std::invalid_argument("229 Assistant split attention geometry");
    const int count=(std::min)(c.capacity,sliding?(std::min)(1025,context-1):context-1);
    const int parts=(count+Keys-1)/Keys;
    if(parts>MaximumParts)throw std::invalid_argument("Assistant large global context requires history backend");
    return parts;
}
template<int Dim>SplitAttentionResources resources(){
    constexpr int Threads=Dim==256?32:256;
    cudaFuncAttributes first{},last{};
    primitive::check_cuda(cudaFuncGetAttributes(&first,segment<Dim>),"229 cold segment resources");
    primitive::check_cuda(cudaFuncGetAttributes(&last,merge<Dim,Threads>),"229 cold merge resources");
    return {{128,first.numRegs,int(first.localSizeBytes),int(first.sharedSizeBytes)},
        {Threads,last.numRegs,int(last.localSizeBytes),int(last.sharedSizeBytes)}};
}
}
size_t split_cross_attention_bytes(CacheView c,bool sliding,int context){
    return size_t(partition_count(c,sliding,context))*32*(c.dim+2)*sizeof(float);
}
SplitAttentionResources split_cross_attention_resources(int dimension){
    if(dimension==256)return resources<256>();if(dimension==512)return resources<512>();
    throw std::invalid_argument("229 Assistant split resource dimension");
}
void split_cross_attention(const float* query,float* output,CacheView c,const Bounds* bounds,
    bool sliding,int context,unsigned* invalid,SplitAttentionLease w,cudaStream_t stream){
    const int parts=partition_count(c,sliding,context);
    const size_t needed=size_t(parts)*32*(c.dim+2)*sizeof(float);
    if(!query||!output||!c.key||!c.value||!bounds||!invalid||!stream||!w.owner||!w.data||w.stream!=stream||
        w.bytes<needed||reinterpret_cast<uintptr_t>(w.data)%256||
        reinterpret_cast<uintptr_t>(query)%4||reinterpret_cast<uintptr_t>(output)%4||
        reinterpret_cast<uintptr_t>(bounds)%4||reinterpret_cast<uintptr_t>(invalid)%4||
        reinterpret_cast<uintptr_t>(c.key)%2||reinterpret_cast<uintptr_t>(c.value)%2)
        throw std::invalid_argument("229 Assistant split attention typed arena lease");
    const size_t kv=size_t(c.capacity)*c.heads*(c.dim/32)*34;
    const void* spans[]{query,output,c.key,c.value,bounds,invalid,w.data};
    const size_t bytes[]{size_t(32)*c.dim*4,size_t(32)*c.dim*4,kv,kv,sizeof(Bounds),sizeof(unsigned),w.bytes};
    for(int i=0;i<7;++i)for(int j=i+1;j<7;++j)if(overlap(spans[i],bytes[i],spans[j],bytes[j]))
        throw std::invalid_argument("229 Assistant split read-only KV/arena alias");
    auto* partial=static_cast<float*>(w.data);
    if(c.dim==256){segment<256><<<dim3(32,parts),128,0,stream>>>(query,partial,c,bounds,sliding,context,parts,invalid);
        merge<256,32><<<32,32,0,stream>>>(partial,output,c,bounds,sliding,context,parts,invalid);}
    else{segment<512><<<dim3(32,parts),128,0,stream>>>(query,partial,c,bounds,sliding,context,parts,invalid);
        merge<512,256><<<32,256,0,stream>>>(partial,output,c,bounds,sliding,context,parts,invalid);}
    primitive::check_cuda(cudaGetLastError(),"229 Assistant packedQ8 split attention");
}
}
