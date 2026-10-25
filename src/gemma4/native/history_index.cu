#include "strata/gemma4/native/history.hpp"
#include <cub/device/device_segmented_radix_sort.cuh>
#include <math_constants.h>
#include <cfloat>
#include <algorithm>
#include <stdexcept>
namespace strata::gemma4::native {
namespace {
__device__ float sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_xor_sync(0xffffffff,x,d));return x;}
__global__ void scores(const float* query,HistoryView v,const int* position,int row){
    const int d=threadIdx.x,page=blockIdx.x,head=blockIdx.y,p=*position+row;
    const size_t at=size_t(head)*v.pages+page;
    if(*v.device_epoch!=v.binding.epoch||*v.invalid||p<0||p>=v.logical_capacity||p>=*v.valid_end){
        if(!d){v.scores[at]=-CUDART_INF_F;atomicOr(v.invalid,16u);}return;
    }
    const int live=(p+HistoryPage)/HistoryPage;
    if(page>=live){if(!d)v.scores[at]=-CUDART_INF_F;return;}
    // Partial/future-bearing pages are always recent and mandatory. Bounds are
    // read ONLY for complete causal old pages, never current speculative rows.
    if(page<HistorySinkPages||page>=live-HistoryRecentPages){if(!d)v.scores[at]=CUDART_INF_F;return;}
    const float lower=__half2float(v.bounds[at*1024+d]);
    const float upper=__half2float(v.bounds[at*1024+HistoryDim+d]);
    __shared__ float partial[16];float result=-CUDART_INF_F;
    for(int q=0;q<HistoryHeads/HistoryKvHeads;++q){
        const float x=query[size_t(head*8+q)*HistoryDim+d];
        if(!isfinite(x))atomicOr(v.invalid,32u);
        float bound=x==0?0.f:__fmul_rn(x,x>0?upper:lower);bound=sum(bound);
        if(!(d%32))partial[d/32]=bound;__syncthreads();
        if(d<32){bound=sum(d<16?partial[d]:0.f);if(!d)result=isnan(bound)?CUDART_INF_F:fmaxf(result,bound);}
        __syncthreads();
    }
    // Mandatory pages retain +Inf priority even if a conservative old-page
    // half bound or F32 reduction overflows. Uncertain old bounds rank first
    // among nonmandatory pages, without displacing the mandatory set.
    if(!d)v.scores[at]=isnan(result)?FLT_MAX:fminf(FLT_MAX,fmaxf(-FLT_MAX,result));
}
__global__ void plan(HistoryView v,const int* position,int row){
    const int head=blockIdx.x,tid=threadIdx.x,p=*position+row;
    const size_t slots=size_t(head)*v.slots,pages=size_t(head)*v.pages;
    const size_t selected=(size_t(row)*HistoryKvHeads+head)*v.slots;
    __shared__ int misses,available,bad,count;
    if(!tid){
        bad=*v.invalid||*v.device_epoch!=v.binding.epoch||p<0||p>=v.logical_capacity||p>=*v.valid_end;
        count=bad?0:min((p+HistoryPage)/HistoryPage,v.slots);misses=0;available=0;
        v.selected_count[row*HistoryKvHeads+head]=count;
        if(!head)atomicAdd(&v.counters->select_calls,1ULL);
    }
    for(int rank=tid;rank<v.slots;rank+=256){
        v.marked[slots+rank]=0;v.selected_page[selected+rank]=-1;
        v.selected_slot[selected+rank]=-1;v.miss[selected+rank]=0;
    }
    __syncthreads();if(bad)return;
    const int live=(p+HistoryPage)/HistoryPage;
    for(int rank=tid;rank<count;rank+=256){
        const int page=v.sorted_ids[pages+rank];
        if(page<0||page>=live){atomicExch(&bad,1);atomicOr(v.invalid,64u);continue;}
        const int slot=v.page_slot[pages+page];
        if(slot< -1||slot>=v.slots||(slot>=0&&v.slot_page[slots+slot]!=page)){
            atomicExch(&bad,1);atomicOr(v.invalid,64u);continue;
        }
        v.selected_page[selected+rank]=page;v.selected_slot[selected+rank]=slot;v.miss[selected+rank]=slot<0;
        if(slot>=0)v.marked[slots+slot]=1;
        else v.miss_rank[slots+atomicAdd(&misses,1)]=rank;
    }
    __syncthreads();if(bad)return;
    if(!tid){atomicAdd(&v.counters->selected_pages,static_cast<unsigned long long>(count));
        atomicAdd(&v.counters->hits,static_cast<unsigned long long>(count-misses));
        atomicAdd(&v.counters->misses,static_cast<unsigned long long>(misses));}
    if(!misses)return;
    // Every selected hit is pinned before any unmarked slot can be evicted.
    for(int slot=tid;slot<v.slots;slot+=256)
        if(!v.marked[slots+slot])v.free_slot[slots+atomicAdd(&available,1)]=slot;
    __syncthreads();
    if(available<misses){if(!tid)atomicOr(v.invalid,128u);return;}
    for(int m=tid;m<misses;m+=256){
        const int rank=v.miss_rank[slots+m],page=v.selected_page[selected+rank],slot=v.free_slot[slots+m];
        const int old=v.slot_page[slots+slot];
        if(old>=0&&old<v.pages)v.page_slot[pages+old]=-1;
        v.page_slot[pages+page]=slot;v.slot_page[slots+slot]=page;v.selected_slot[selected+rank]=slot;
    }
}
__global__ void load_pages(HistoryView v,int row){
    const int rank=blockIdx.x,head=blockIdx.y;
    if(*v.invalid||*v.device_epoch!=v.binding.epoch||rank>=v.selected_count[row*HistoryKvHeads+head])return;
    const size_t selected=(size_t(row)*HistoryKvHeads+head)*v.slots+rank;
    if(!v.miss[selected])return;
    const int page=v.selected_page[selected],slot=v.selected_slot[selected],end=*v.valid_end;
    if(page<0||page>=v.pages||slot<0||slot>=v.slots){if(!threadIdx.x)atomicOr(v.invalid,256u);return;}
    constexpr int Chunks=HistoryPage*HistoryRowBytes/16;
    // Load ALL already-written rows of a selected page, including speculative
    // query1 rows. The next query can hit it; attention alone applies P+t mask.
    for(int i=threadIdx.x;i<Chunks*2;i+=256){
        const int which=i/Chunks,local=i%Chunks,within=local/(HistoryRowBytes/16),d=(local%(HistoryRowBytes/16))*16;
        const int logical=page*HistoryPage+within;
        auto* to=(which?v.value:v.key)+(size_t(slot*HistoryPage+within)*HistoryKvHeads+head)*HistoryRowBytes+d;
        if(logical<end&&logical<v.logical_capacity){
            const auto* from=(which?v.host_value:v.host_key)+(size_t(logical)*HistoryKvHeads+head)*HistoryRowBytes+d;
            *reinterpret_cast<uint4*>(to)=*reinterpret_cast<const uint4*>(from);
        }else *reinterpret_cast<uint4*>(to)=make_uint4(0xffffffffu,0xffffffffu,0xffffffffu,0xffffffffu);
    }
    if(!threadIdx.x){const int rows=max(0,min(HistoryPage,end-page*HistoryPage));
        atomicAdd(&v.counters->loaded_bytes,static_cast<unsigned long long>(rows)*HistoryRowBytes*2);}
}
}
size_t history_sort_workspace(HistoryView v){
    if(v.pages<1||v.pages>16384||!v.scores||!v.sorted_scores||!v.page_ids||!v.sorted_ids||!v.offsets)
        throw std::invalid_argument("History cold CUB operands");
    size_t bytes=0;
    primitive::check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,bytes,v.scores,v.sorted_scores,
        v.page_ids,v.sorted_ids,HistoryKvHeads*v.pages,HistoryKvHeads,v.offsets,v.offsets+1),"History cold CUB workspace query");
    return (std::max)(size_t(1),bytes);
}
void history_select(const float* query,HistoryView v,const int* position,int row,HistoryBinding live){
    history_detail::validate(v,live);
    if(!query||!position||row<0||row>=HistoryQueries||reinterpret_cast<std::uintptr_t>(query)%4||
       reinterpret_cast<std::uintptr_t>(position)%4)throw std::invalid_argument("History query/baseP/row selection");
    scores<<<dim3(v.pages,HistoryKvHeads),HistoryDim,0,live.stream>>>(query,v,position,row);
    primitive::check_cuda(cudaGetLastError(),"History Q32/KV4 F32 page upper-bound ranking");
    primitive::check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(v.sort_work,v.sort_bytes,v.scores,v.sorted_scores,
        v.page_ids,v.sorted_ids,HistoryKvHeads*v.pages,HistoryKvHeads,v.offsets,v.offsets+1,0,32,live.stream),"History captured preallocated GPU rank");
    plan<<<HistoryKvHeads,256,0,live.stream>>>(v,position,row);
    primitive::check_cuda(cudaGetLastError(),"History GPU selected residency plan");
    load_pages<<<dim3(v.slots,HistoryKvHeads),256,0,live.stream>>>(v,row);
    primitive::check_cuda(cudaGetLastError(),"History mapped RAM selected Q8 page-in");
}
}
