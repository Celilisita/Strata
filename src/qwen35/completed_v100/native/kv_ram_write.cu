#include "strata/qwen35/completed_v100/native/kv_ram.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <cuda_fp16.h>
#include <math_constants.h>
#include <algorithm>
namespace strata::completed_v100::qwen35::native {namespace {
__global__ void write(const float *x,unsigned char *host,unsigned char *pool,RamKvView v,const int *positions,int shift,int tokens,bool key){
    const int stride=v.heads*8,block=blockIdx.x*4+threadIdx.x/32;
    if(block>=tokens*stride)return;
    const int token=block/stride,logical=positions[token]-shift,h=(block%stride)/8,dblock=block%8,lane=threadIdx.x%32;
    const float value=x[size_t(block)*32+lane];float maximum=fabsf(value);
    for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    const float delta=maximum/127.f;const int code=maximum?int(roundf(value*(127.f/maximum))):0;
    auto *out=host+(size_t(logical)*stride+block%stride)*34;
    if(!lane)*reinterpret_cast<__half *>(out)=__float2half_rn(delta);
    reinterpret_cast<signed char *>(out+2)[lane]=static_cast<signed char>(max(-127,min(127,code)));
    const int slot=v.page_slot[size_t(h)*v.pages+logical/RamPage];
    if(slot>=0){
        auto *resident=pool+((size_t(slot)*RamPage+logical%RamPage)*v.heads*8+h*8+dblock)*34;
        if(!lane)*reinterpret_cast<__half *>(resident)=__float2half_rn(delta);
        reinterpret_cast<signed char *>(resident+2)[lane]=static_cast<signed char>(max(-127,min(127,code)));
    }
    if(!block&&!lane)atomicAdd(&v.counters->written_bytes,(unsigned long long)(tokens*stride*34));
}
__global__ void summarize(RamKvView v,const int *positions,int shift,int tokens){
    const int start=positions[0]-shift,valid=start+tokens,first=start/RamPage;
    const int page=first+blockIdx.x,h=blockIdx.y,d=threadIdx.x;
    if(page*RamPage>=valid)return;
    float lo=CUDART_INF_F,hi=-CUDART_INF_F;
    for(int t=page*RamPage;t<min((page+1)*RamPage,valid);++t){
        const auto *p=v.host_key+(size_t(t)*v.heads+h)*272+(d/32)*34;
        const float x=__half2float(*reinterpret_cast<const __half *>(p))*float(reinterpret_cast<const signed char *>(p+2)[d%32]);
        lo=fminf(lo,x);hi=fmaxf(hi,x);
    }
    const size_t base=(size_t(h)*v.pages+page)*512;
    if(!d&&v.page_slot[size_t(h)*v.pages+page]>=0)atomicAdd(&v.counters->resident_write_bytes,
        (unsigned long long)((min((page+1)*RamPage,valid)-max(page*RamPage,start))*272*2));
    v.bounds[base+d]=__float2half_rd(lo);v.bounds[base+256+d]=__float2half_ru(hi);
}
__global__ void reset(RamKvView v){
    const int i=blockIdx.x*256+threadIdx.x;
    if(i<v.pages*v.heads){const int p=i%v.pages,h=i/v.pages;v.page_slot[i]=p<v.slots?p:-1;v.page_ids[i]=p;}
    if(i<v.slots*v.heads){const int s=i%v.slots;v.slot_page[i]=s<v.pages?s:-1;v.selected_slot[i]=-1;v.miss[i]=0;if(v.last_use)v.last_use[i]=0;}
    if(i<v.heads&&v.epoch)v.epoch[i]=0;
    if(!i)*v.counters=RamKvCounters{};
}
}
void ram_reset(RamKvView v,cudaStream_t s){
    reset<<<(std::max(v.pages,v.slots)*v.heads+255)/256,256,0,s>>>(v);check_cuda(cudaGetLastError(),"RAM cache epoch reset");
}
void ram_write(const float *x,RamKvView v,bool key,int dim,int heads,int tokens,const int *positions,int shift,cudaStream_t s){
    if(!x||!positions||dim!=256||heads!=v.heads||tokens<1||tokens>8192)throw std::invalid_argument("invalid RAM KV append");
    write<<<(tokens*heads*8+3)/4,128,0,s>>>(x,key?v.host_key:v.host_value,key?v.key:v.value,v,positions,shift,tokens,key);
    check_cuda(cudaGetLastError(),"authoritative RAM and resident Q8 append");
    if(key)ram_summarize(v,positions,shift,tokens,s);
}
void ram_summarize(RamKvView v,const int *positions,int shift,int tokens,cudaStream_t s){
    summarize<<<dim3((tokens+RamPage-2)/RamPage+1,v.heads),256,0,s>>>(v,positions,shift,tokens);
    check_cuda(cudaGetLastError(),"query-aware page summary update");
}

DecodeNode ram_cache_node(void *p){return p==reinterpret_cast<void *>(write)?DecodeNode::CacheWrite:DecodeNode::None;}
}
