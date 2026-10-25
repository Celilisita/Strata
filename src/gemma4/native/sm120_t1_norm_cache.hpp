#pragma once
#include "strata/gemma4/native/decoder_ops.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strata::gemma4::native {
inline bool t1_norm_cache_fusion_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA_T1_NORM_CACHE_FUSION");
        if(!value||std::strcmp(value,"0")==0)return false;
        if(std::strcmp(value,"1")==0)return true;
        throw std::invalid_argument("Gemma T1 norm/cache fusion requires0 or1");
    }();return enabled;
}
// T1 only. Exact in-place or disjoint read/write spans; output Q/K/V remain distinct owned buffers; shared_raw_kv refers to raw
// projection semantics, never to aliasing K and V storage. Table is fresh T1.
void t1_norm_cache_fused(const float*,const float*,const float*,float*,float*,float*,const float*,const float*,CacheView,
    bool shared_raw_kv,float epsilon,const float* table,int table_capacity,
    const int* position,int context,unsigned* invalid,cudaStream_t);
}
#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cmath>
namespace strata::gemma4::native {
namespace t1_norm_cache_detail {
__device__ float sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void norm_store(const float* rq,const float* rk,const float* rv,float* q,float* k,float* v,const float* qw,const float* kw,CacheView c,
    bool shared_raw_kv,float epsilon,const float* table,const int* position,int context,unsigned* invalid){
    __shared__ float partial[2][4],inverse[2],key[512],value[512];
    const int tid=threadIdx.x,lane=tid%32,warp=tid/32,dim=c.dim,head=blockIdx.x;
    const int absolute=*position;
    if(absolute<0||absolute>=context){if(!tid)atomicExch(invalid,1u);return;}
    if(head<32){
        const std::size_t first=std::size_t(head)*dim;float total=0;
        for(int j=tid;j<dim;j+=128)total=__fmaf_rn(rq[first+j],rq[first+j],total);
        total=sum(total);if(!lane)partial[0][warp]=total;__syncthreads();
        if(!tid)inverse[0]=rsqrtf(((partial[0][0]+partial[0][1])+(partial[0][2]+partial[0][3]))/dim+epsilon);
        __syncthreads();
        for(int pair=tid;pair<dim/2;pair+=128){
            float a=rq[first+pair]*inverse[0],b=rq[first+pair+dim/2]*inverse[0];a*=qw[pair];b*=qw[pair+dim/2];
            const float sn=table[pair*2],cs=table[pair*2+1];
            q[first+pair]=__fmaf_rn(-b,sn,a*cs);q[first+pair+dim/2]=__fmaf_rn(a,sn,b*cs);
        }return;
    }
    const int kvhead=head-32;const std::size_t first=std::size_t(kvhead)*dim;
    float rawk[4],rawv[4],totalk=0,totalv=0;
    // Every raw element is held before any thread can publish rotated K.
    for(int j=0;j<dim/128;++j){const int at=tid+j*128;
        rawk[j]=rk[first+at];rawv[j]=shared_raw_kv?rawk[j]:rv[first+at];
        totalk=__fmaf_rn(rawk[j],rawk[j],totalk);if(!shared_raw_kv)totalv=__fmaf_rn(rawv[j],rawv[j],totalv);
    }
    totalk=sum(totalk);if(!shared_raw_kv)totalv=sum(totalv);if(!lane){partial[0][warp]=totalk;partial[1][warp]=totalv;}
    __syncthreads();
    if(!tid){
        inverse[0]=rsqrtf(((partial[0][0]+partial[0][1])+(partial[0][2]+partial[0][3]))/dim+epsilon);
        inverse[1]=shared_raw_kv?inverse[0]:rsqrtf(((partial[1][0]+partial[1][1])+(partial[1][2]+partial[1][3]))/dim+epsilon);
    }__syncthreads();
    for(int pair=tid;pair<dim/2;pair+=128){
        const int j=pair/128;float a=rawk[j]*inverse[0],b=rawk[j+dim/256]*inverse[0];a*=kw[pair];b*=kw[pair+dim/2];
        const float sn=table[pair*2],cs=table[pair*2+1];
        key[pair]=__fmaf_rn(-b,sn,a*cs);key[pair+dim/2]=__fmaf_rn(a,sn,b*cs);
    }
    for(int j=0;j<dim/128;++j)value[tid+j*128]=rawv[j]*inverse[1];
    __syncthreads();
    for(int j=0;j<dim/128;++j){const int at=tid+j*128;const int group=at/32;
        k[first+at]=key[at];v[first+at]=value[at];
        const std::size_t offset=((std::size_t(absolute%c.capacity)*c.heads+kvhead)*(dim/32)+group)*34;
        for(int which=0;which<2;++which){
            const float x=which?value[at]:key[at];float maximum=fabsf(x);
            for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
            const float scale=maximum/127.f;if(!isfinite(x))atomicExch(invalid,1u);
            auto* out=static_cast<unsigned char*>(which?c.value:c.key)+offset;
            if(!lane)*reinterpret_cast<half*>(out)=__float2half_rn(scale);
            reinterpret_cast<signed char*>(out+2)[lane]=static_cast<signed char>(maximum>0?max(-127,min(127,__float2int_rn(x/scale))):0);
        }
    }
}
inline bool overlaps(const void* a,std::size_t n,const void* b,std::size_t m){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    if(n>std::numeric_limits<std::uintptr_t>::max()-x||m>std::numeric_limits<std::uintptr_t>::max()-y)
        throw std::invalid_argument("Gemma T1 norm/cache pointer span overflow");
    return x<y+m&&y<x+n;
}
}
void t1_norm_cache_fused(const float* rq,const float* rk,const float* rv,float* q,float* k,float* v,const float* qw,const float* kw,CacheView c,
    bool shared_raw_kv,float epsilon,const float* table,int table_capacity,
    const int* position,int context,unsigned* invalid,cudaStream_t stream){
    const bool local=c.dim==256&&c.heads==16&&c.window==1024&&!shared_raw_kv;
    const bool global=c.dim==512&&c.heads==4&&c.window==0&&shared_raw_kv;
    if(!rq||!rk||!rv||!q||!k||!v||!qw||!kw||!c.key||!c.value||!table||!position||!invalid||!stream||
       !(epsilon>0)||!std::isfinite(epsilon)||(!local&&!global)||table_capacity<1||table_capacity>8192||
       context<1||context>262144||c.capacity<1||c.capacity>context||(global&&c.capacity!=context))
        throw std::invalid_argument("Gemma T1 norm/cache exact original geometry/lease");
    const std::size_t qbytes=std::size_t(32)*c.dim*4,kvbytes=std::size_t(c.heads)*c.dim*4;
    const std::size_t cachebytes=std::size_t(c.capacity)*c.heads*(c.dim/32)*34;
    const void* buffers[]{q,k,v,c.key,c.value};const std::size_t sizes[]{qbytes,kvbytes,kvbytes,cachebytes,cachebytes};
    for(int a=0;a<5;++a)for(int b=0;b<a;++b)if(t1_norm_cache_detail::overlaps(buffers[a],sizes[a],buffers[b],sizes[b]))
        throw std::invalid_argument("Gemma T1 norm/cache overlapping writable spans");
    const void* raw[]{rq,rk,rv};const std::size_t rawbytes[]{qbytes,kvbytes,kvbytes};
    for(int a=0;a<5;++a)for(int b=0;b<3;++b){if(shared_raw_kv&&b==2)continue;
        if(t1_norm_cache_detail::overlaps(buffers[a],sizes[a],raw[b],rawbytes[b])&&!(a==b&&buffers[a]==raw[b]))
            throw std::invalid_argument("Gemma T1 norm/cache partial or cross-input alias");
    }
    const float* selected=c.dim==256?table:table+std::size_t(table_capacity)*128*2;
    t1_norm_cache_detail::norm_store<<<32+c.heads,128,0,stream>>>(rq,rk,rv,q,k,v,qw,kw,c,shared_raw_kv,epsilon,selected,position,context,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma T1 fused weighted QK norm/RoPE, raw V norm and Q8 cache append");
}
}
#endif
