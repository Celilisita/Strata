#include "strata/gemma4/native/history.hpp"
#include <math_constants.h>
#include <algorithm>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
__device__ bool append_valid(HistoryView v,const int* position,int tokens){
    const long long p=*position,end=p+tokens;
    return *v.device_epoch==v.binding.epoch&&!*v.invalid&&p>=0&&
        end<=v.logical_capacity&&p<=*v.valid_end;
}
__global__ void pack(const float* key,const float* value,HistoryView v,
    const int* position,int tokens,unsigned* invalid){
    if(!append_valid(v,position,tokens)){
        if(!threadIdx.x){atomicOr(v.invalid,1u);atomicOr(invalid,1u);}return;
    }
    const int lane=threadIdx.x,group=blockIdx.x%16,head=blockIdx.x/16,t=blockIdx.y;
    const int logical=*position+t,page=logical/HistoryPage;
    const size_t source=(size_t(t)*HistoryKvHeads+head)*HistoryDim+group*32+lane;
    const int slot=v.page_slot[size_t(head)*v.pages+page];
    if(slot>=v.slots||slot< -1||
       (slot>=0&&v.slot_page[size_t(head)*v.slots+slot]!=page)){
        if(!lane){atomicOr(v.invalid,2u);atomicOr(invalid,2u);}return;
    }
    for(int which=0;which<2;++which){
        const float x=which?value[source]:key[source];float maximum=fabsf(x);
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
        // Exactly canonical Gemma170 pack; do NOT import Qwen's roundf(x*127/max).
        const float scale=maximum/127.f;
        if(!isfinite(x)){atomicOr(v.invalid,4u);atomicOr(invalid,4u);}
        const int code=maximum>0?max(-127,min(127,__float2int_rn(x/scale))):0;
        auto* raw=(which?v.host_value:v.host_key)+
            (size_t(logical)*HistoryKvHeads+head)*HistoryRowBytes+group*34;
        if(!lane)*reinterpret_cast<__half*>(raw)=__float2half_rn(scale);
        reinterpret_cast<signed char*>(raw+2)[lane]=static_cast<signed char>(code);
        if(slot>=0){
            auto* out=(which?v.value:v.key)+
                (size_t(slot*HistoryPage+logical%HistoryPage)*HistoryKvHeads+head)*HistoryRowBytes+group*34;
            if(!lane)*reinterpret_cast<__half*>(out)=__float2half_rn(scale);
            reinterpret_cast<signed char*>(out+2)[lane]=static_cast<signed char>(code);
        }
    }
}
__global__ void summarize(HistoryView v,const int* position,int tokens){
    if(*v.invalid||*v.device_epoch!=v.binding.epoch)return;
    const int start=*position,end=start+tokens,first=start/HistoryPage;
    const int page=first+blockIdx.x,head=blockIdx.y,d=threadIdx.x;
    if(page>=v.pages||page*HistoryPage>=end)return;
    float lower=CUDART_INF_F,upper=-CUDART_INF_F;
    for(int row=page*HistoryPage;row<min((page+1)*HistoryPage,end);++row){
        const auto* raw=v.host_key+(size_t(row)*HistoryKvHeads+head)*HistoryRowBytes+(d/32)*34;
        const float x=__half2float(*reinterpret_cast<const __half*>(raw))*
            float(reinterpret_cast<const signed char*>(raw+2)[d%32]);
        lower=fminf(lower,x);upper=fmaxf(upper,x);
    }
    const size_t base=(size_t(head)*v.pages+page)*1024;
    v.bounds[base+d]=__float2half_rd(lower);v.bounds[base+HistoryDim+d]=__float2half_ru(upper);
    if(!d&&v.page_slot[size_t(head)*v.pages+page]>=0){
        const int rows=min((page+1)*HistoryPage,end)-max(page*HistoryPage,start);
        atomicAdd(&v.counters->resident_write_bytes,static_cast<unsigned long long>(rows)*HistoryRowBytes*2);
    }
}
__global__ void publish(HistoryView v,const int* position,int tokens,unsigned* invalid){
    if(*v.device_epoch!=v.binding.epoch)atomicOr(v.invalid,8u);
    if(*v.invalid){atomicOr(invalid,*v.invalid);return;}
    *v.valid_end=*position+tokens; // May shrink on speculative rollback/rewrite.
    atomicAdd(&v.counters->written_bytes,static_cast<unsigned long long>(tokens)*HistoryKvHeads*HistoryRowBytes*2);
}
__global__ void reset(HistoryView v){
    const int i=blockIdx.x*256+threadIdx.x;
    if(i<v.pages*HistoryKvHeads){
        const int page=i%v.pages;v.page_slot[i]=page<v.slots?page:-1;v.page_ids[i]=page;
    }
    if(i<v.slots*HistoryKvHeads){const int slot=i%v.slots;
        v.slot_page[i]=slot<v.pages?slot:-1;v.marked[i]=0;v.miss_rank[i]=0;v.free_slot[i]=0;}
    if(i<v.slots*HistoryKvHeads*HistoryQueries){v.selected_page[i]=-1;v.selected_slot[i]=-1;v.miss[i]=0;}
    if(i<HistoryKvHeads*HistoryQueries)v.selected_count[i]=0;
    if(!i){*v.valid_end=0;*v.invalid=0;*v.device_epoch=v.binding.epoch;*v.counters=HistoryCounters{};}
}
bool overlap(const void* a,size_t an,const void* b,size_t bn){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<an:x-y<bn;
}
}
void history_reset(HistoryView v,HistoryBinding live){
    history_detail::validate(v,live);
    const int count=(std::max)(v.pages*HistoryKvHeads,v.slots*HistoryKvHeads*HistoryQueries);
    reset<<<(count+255)/256,256,0,live.stream>>>(v);
    primitive::check_cuda(cudaGetLastError(),"History GPU directory/epoch reset");
}
void history_write(const float* key,const float* value,HistoryView v,const int* position,
    int tokens,unsigned* invalid,HistoryBinding live){
    history_detail::validate(v,live);
    if(!key||!value||!position||!invalid||tokens<1||tokens>8192||
       reinterpret_cast<std::uintptr_t>(key)%4||reinterpret_cast<std::uintptr_t>(value)%4||
       reinterpret_cast<std::uintptr_t>(position)%4||reinterpret_cast<std::uintptr_t>(invalid)%4)
        throw std::invalid_argument("History Gemma append input/baseP/extent");
    const size_t input=size_t(tokens)*HistoryKvHeads*HistoryDim*4;
    const void* pools[]{v.host_key,v.host_value,v.key,v.value};
    const size_t sizes[]{size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.pages)*HistoryPage*HistoryKvHeads*HistoryRowBytes,
        size_t(v.resident)*HistoryKvHeads*HistoryRowBytes,size_t(v.resident)*HistoryKvHeads*HistoryRowBytes};
    for(int i=0;i<4;++i)if(overlap(key,input,pools[i],sizes[i])||overlap(value,input,pools[i],sizes[i])||
        overlap(position,4,pools[i],sizes[i])||overlap(invalid,4,pools[i],sizes[i]))
        throw std::invalid_argument("History append input aliases authoritative/resident pool");
    pack<<<dim3(HistoryKvHeads*16,tokens),32,0,live.stream>>>(key,value,v,position,tokens,invalid);
    primitive::check_cuda(cudaGetLastError(),"History exact Gemma170 Q8 RAM/resident append");
    summarize<<<dim3((tokens+HistoryPage-2)/HistoryPage+1,HistoryKvHeads),HistoryDim,0,live.stream>>>(v,position,tokens);
    primitive::check_cuda(cudaGetLastError(),"History causal page lower/upper bounds");
    publish<<<1,1,0,live.stream>>>(v,position,tokens,invalid);
    primitive::check_cuda(cudaGetLastError(),"History GPU written frontier");
}
}
