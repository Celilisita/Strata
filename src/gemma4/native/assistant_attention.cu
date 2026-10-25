#include "strata/gemma4/native/assistant.hpp"
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native::assistant {
namespace {
__device__ float warp_sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__device__ float read(const void* ptr,size_t row,int j){
    const auto* p=static_cast<const unsigned char*>(ptr)+row+(j/32)*34;
    return __half2float(*reinterpret_cast<const half*>(p))*float(reinterpret_cast<const signed char*>(p+2)[j%32]);
}
__global__ void attend(const float* q,float* out,CacheView cache,const Bounds* bounds,bool sliding,int context,unsigned* invalid){
    const Bounds b=*bounds;const int lane=threadIdx.x,head=blockIdx.x,dim=cache.dim;
    const int origin=sliding?b.sliding_origin:b.global_origin,length=sliding?b.sliding_length:b.global_length;
    const int left=sliding?b.sliding_left:origin;
    const bool valid=b.rope_position>=1&&b.rope_position<context&&b.kv_end==b.rope_position-1&&origin>=0&&
        length>=1&&length<=cache.capacity&&int64_t(origin)+length-1==b.kv_end&&left>=origin&&left<=b.kv_end&&
        (sliding?b.kv_end-left+1<=1025:origin==0);
    if(!valid){if(!lane)atomicOr(invalid,2u);for(int j=lane;j<dim;j+=32)out[size_t(head)*dim+j]=NAN;return;}
    const int kvhead=head/(32/cache.heads);float query[16],acc[16]{};
    for(int j=0;j<dim/32;++j){query[j]=q[size_t(head)*dim+lane+j*32];if(!isfinite(query[j]))atomicOr(invalid,4u);}
    float maximum=-INFINITY,denominator=0;
    int slot=left%cache.capacity;
    for(int key=left;key<=b.kv_end;++key){
        const size_t row=(size_t(slot)*cache.heads+kvhead)*(dim/32)*34;float score=0;
        for(int j=0;j<dim/32;++j)score=__fmaf_rn(query[j],read(cache.key,row,lane+j*32),score);
        score=warp_sum(score);const float next=fmaxf(maximum,score),a=__expf(maximum-next),p=__expf(score-next);
        for(int j=0;j<dim/32;++j)acc[j]=__fmaf_rn(p,read(cache.value,row,lane+j*32),a*acc[j]);
        denominator=__fmaf_rn(denominator,a,p);maximum=next;if(++slot==cache.capacity)slot=0;
    }
    for(int j=0;j<dim/32;++j){const float value=acc[j]/denominator;out[size_t(head)*dim+lane+j*32]=value;
        if(!isfinite(value))atomicOr(invalid,4u);}
}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);return x<=y?y-x<na:x-y<nb;
}
}
void cross_attention(const float* query,float* out,CacheView c,const Bounds* bounds,bool sliding,
    int context,unsigned* invalid,cudaStream_t stream){
    if(!query||!out||!c.key||!c.value||!bounds||!invalid||!stream||context<2||context>262144||c.capacity<1||
       c.heads!=(sliding?16:4)||c.dim!=(sliding?256:512)||c.window!=(sliding?1024:0)||
       reinterpret_cast<uintptr_t>(query)%4||reinterpret_cast<uintptr_t>(out)%4||reinterpret_cast<uintptr_t>(bounds)%4||
       reinterpret_cast<uintptr_t>(invalid)%4||reinterpret_cast<uintptr_t>(c.key)%2||reinterpret_cast<uintptr_t>(c.value)%2)
        throw std::invalid_argument("Assistant cross-attention shape/bounds lease");
    const void* spans[]{query,out,c.key,c.value,bounds,invalid};
    const size_t kv=size_t(c.capacity)*c.heads*(c.dim/32)*34;
    const size_t bytes[]{size_t(32)*c.dim*4,size_t(32)*c.dim*4,kv,kv,sizeof(Bounds),sizeof(unsigned)};
    for(int i=0;i<6;++i)for(int j=i+1;j<6;++j)if(overlap(spans[i],bytes[i],spans[j],bytes[j]))
        throw std::invalid_argument("Assistant read-only KV alias");
    attend<<<32,32,0,stream>>>(query,out,c,bounds,sliding,context,invalid);
    primitive::check_cuda(cudaGetLastError(),"Assistant explicit-left/end Q8 cross-attention");
}
}
