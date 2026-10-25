#include "attention_global_t2.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
#include <cstdint>
#include "attention_global_merge.cuh"

namespace strata::gemma4::native {
namespace {
__device__ float q8(const void* data,int key,int head,int feature){
    const auto* p=static_cast<const unsigned char*>(data)+
        ((size_t(key)*4+head)*16+feature/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*
        float(reinterpret_cast<const signed char*>(p+2)[feature%32]);
}
__device__ void write_partial(float* partial,int parts,int part,int t,int head,
    int lane,bool valid,float top,float denominator,const float* value){
    float* out=partial+((size_t(t)*32+head)*parts+part)*514;
    if(!lane){out[0]=valid?top:CUDART_NAN_F;out[1]=valid?denominator:CUDART_NAN_F;}
    #pragma unroll
    for(int j=0;j<16;++j)out[2+lane+32*j]=valid?value[j]:CUDART_NAN_F;
}
__global__ __launch_bounds__(512) void segment(const float* query,float* partial,
    CacheView cache,const int* position,int parts){
    __shared__ float key_tile[8][512],value_tile[8][512];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,t=warp/8;
    const int kh=blockIdx.x,head=kh*8+warp%8,part=blockIdx.y,begin=part*128;
    const int base=*position;const long long absolute=static_cast<long long>(base)+t;
    const bool valid=absolute>=0&&absolute<cache.capacity;
    const int qpos=valid?int(absolute):-1;
    float q[16],a[16]{};float top=-CUDART_INF_F,denominator=0;
    #pragma unroll
    for(int j=0;j<16;++j)q[j]=valid?query[(size_t(t)*32+head)*512+lane+32*j]:0;
    // These branches are CTA-uniform; mixed valid/invalid query warps must
    // still cooperate in every load and barrier below.
    const bool any=base<cache.capacity&&static_cast<long long>(base)+1>=0;
    const int last=any?int(min(static_cast<long long>(cache.capacity-1),static_cast<long long>(base)+1)):-1;
    if(!any||begin>last){write_partial(partial,parts,part,t,head,lane,valid,top,denominator,a);return;}
    for(int tile=0;tile<128;tile+=8){
        #pragma unroll
        for(int r=0;r<8;++r){const int key=begin+tile+r;const bool active=key<=last;
            key_tile[r][tid]=active?q8(cache.key,key,kh,tid):0;
            value_tile[r][tid]=active?q8(cache.value,key,kh,tid):0;}
        __syncthreads();
        #pragma unroll
        for(int r=0;r<8;++r){const int key=begin+tile+r;
            // In particular query0 must never perform arithmetic on P+1.
            if(valid&&key<=qpos){float score=0;
                #pragma unroll
                for(int j=0;j<16;++j)score=fmaf(q[j],key_tile[r][lane+32*j],score);
                score=detail234::sum(score);const float next=fmaxf(top,score);
                const float old=expf(top-next),current=expf(score-next);
                #pragma unroll
                for(int j=0;j<16;++j)a[j]=fmaf(current,value_tile[r][lane+32*j],old*a[j]);
                denominator=fmaf(denominator,old,current);top=next;
            }
        }
        __syncthreads();
    }
    write_partial(partial,parts,part,t,head,lane,valid,top,denominator,a);
}
bool overlap(const void* a,size_t x,const void* b,size_t y){
    const auto u=reinterpret_cast<std::uintptr_t>(a),v=reinterpret_cast<std::uintptr_t>(b);
    return u<=v?v-u<x:u-v<y;
}
}
size_t global_t2_partial_bytes(CacheView c){
    if(!c.key||!c.value||c.key==c.value||c.capacity<2||c.capacity>262144||
        c.heads!=4||c.dim!=512||c.window!=0)
        throw std::invalid_argument("234 fixed global T2 cache geometry");
    return size_t((c.capacity+127)/128)*32*2*514*sizeof(float);
}
void shared_global_t2_attention(const float* q,float* out,CacheView c,const int* position,
    int heads,int tokens,GlobalT2Lease w,cudaStream_t stream){
    const size_t bytes=global_t2_partial_bytes(c),query_bytes=size_t(2)*32*512*4;
    const size_t kv_bytes=size_t(c.capacity)*4*16*34;
    if(heads!=32||tokens!=2||!q||!out||!position||!w.owner||!stream||w.stream!=stream||
        !w.data||w.bytes<bytes)throw std::invalid_argument("234 borrowed global T2 lease");
    const void* pointers[]{q,out,c.key,c.value,position,w.data};
    const size_t sizes[]{query_bytes,query_bytes,kv_bytes,kv_bytes,4,bytes};
    for(int i=0;i<6;++i){
        const size_t alignment=i==4?4:(i==2||i==3?2:16);
        if(reinterpret_cast<std::uintptr_t>(pointers[i])%alignment)
            throw std::invalid_argument("234 operand alignment");
        for(int j=i+1;j<6;++j)if(overlap(pointers[i],sizes[i],pointers[j],sizes[j]))
            throw std::invalid_argument("234 operand lifetime alias");
    }
    const int parts=(c.capacity+127)/128;auto* partial=static_cast<float*>(w.data);
    segment<<<dim3(4,parts),512,0,stream>>>(q,partial,c,position,parts);
    detail234::merge<<<dim3(32,2),32,0,stream>>>(partial,out,parts);
    primitive::check_cuda(cudaGetLastError(),"234 shared global T2 attention");
}
GlobalT2Resources shared_global_t2_resources(){
    cudaFuncAttributes r{};
    primitive::check_cuda(cudaFuncGetAttributes(&r,segment),"234 cold segment resources");
    return {512,r.maxThreadsPerBlock,r.numRegs,r.localSizeBytes,r.sharedSizeBytes};
}
}
