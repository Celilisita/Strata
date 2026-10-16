#include "strata/qwen35/native/kv_ram.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cuda_fp16.h>
#include <math_constants.h>
namespace strata::qwen35::native { namespace {
// Per KV head, max of the conservative page bounds for its query heads.
// Complete old pages are scored; sinks and the recent64 pages are mandatory.
__global__ void page_scores(const float *query,RamKvView v,const int *positions,int shift,int heads){
    const int valid=positions[0]-shift+1,page=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
    const int live=(valid+RamPage-1)/RamPage,groups=heads/v.heads;
    const size_t i=size_t(h)*v.pages+page;
    if(page>=live){if(!d)v.scores[i]=-CUDART_INF_F;return;}
    if(page<4||page>=live-64){if(!d)v.scores[i]=CUDART_INF_F;return;}
    const float lo=__half2float(v.bounds[i*512+d]),hi=__half2float(v.bounds[i*512+256+d]);
    __shared__ float partial[8];float result=-CUDART_INF_F;
    for(int q=0;q<groups;++q){
        const float x=query[size_t(h*groups+q)*256+d];float score=fmaxf(x*lo,x*hi);
        for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);
        if(!(d%32))partial[d/32]=score;__syncthreads();
        if(d<32){score=d<8?partial[d]:0.f;for(int step=16;step;step/=2)score+=__shfl_xor_sync(0xffffffff,score,step);}
        if(!d)result=fmaxf(result,score);__syncthreads();
    }
    if(!d)v.scores[i]=result;
}
// A selected resident slot cannot be evicted within this query's lease.
__global__ void plan(RamKvView v,const int *positions,int shift){
    const int h=blockIdx.x,live=(positions[0]-shift+RamPage)/RamPage,count=min(live,v.selected_slots);
    const size_t base=size_t(h)*v.slots,pages=size_t(h)*v.pages;
    __shared__ int misses,available,epoch;
    if(!threadIdx.x){misses=0;available=0;epoch=v.keep_queries?++v.epoch[h]:0;atomicAdd(&v.counters->calls,1ULL);atomicAdd(&v.counters->selected_pages,(unsigned long long)count);}
    for(int slot=threadIdx.x;slot<v.slots;slot+=256)v.marked[base+slot]=0;
    __syncthreads();
    for(int rank=threadIdx.x;rank<count;rank+=256){
        const int page=v.sorted_ids[pages+rank],slot=v.page_slot[pages+page];
        v.selected_slot[base+rank]=slot;v.miss[base+rank]=slot<0;
        if(slot>=0){v.marked[base+slot]=1;if(v.last_use)v.last_use[base+slot]=epoch;}
        else v.miss_rank[base+atomicAdd(&misses,1)]=rank;
    }
    __syncthreads();
    if(!threadIdx.x){atomicAdd(&v.counters->misses,(unsigned long long)misses);atomicAdd(&v.counters->hits,(unsigned long long)(count-misses));}
    if(!misses)return;
    for(int slot=threadIdx.x;slot<v.slots;slot+=256)if(!v.marked[base+slot]){
        const bool cold=!v.keep_queries||v.slot_page[base+slot]<0||!v.last_use[base+slot]||v.last_use[base+slot]+v.keep_queries<epoch;
        if(cold)v.free_slot[base+atomicAdd(&available,1)]=slot;
    }
    __syncthreads();
    // Residency hints never change the selected set or make a request fail.
    // The constructor's working-set bound guarantees space; fall back safely
    // if a resumed cache has a shorter or differently aged hint inventory.
    if(available<misses){
        if(!threadIdx.x)available=0;__syncthreads();
        for(int slot=threadIdx.x;slot<v.slots;slot+=256)if(!v.marked[base+slot])v.free_slot[base+atomicAdd(&available,1)]=slot;
        __syncthreads();
    }
    for(int m=threadIdx.x;m<misses;m+=256){
        const int rank=v.miss_rank[base+m],page=v.sorted_ids[pages+rank],slot=v.free_slot[base+m],old=v.slot_page[base+slot];
        if(old>=0)v.page_slot[pages+old]=-1;
        v.page_slot[pages+page]=slot;v.slot_page[base+slot]=page;v.selected_slot[base+rank]=slot;if(v.last_use)v.last_use[base+slot]=epoch;
    }
}
__global__ void load_pages(RamKvView v,const int *positions,int shift){
    const int valid=positions[0]-shift+1,count=min((valid+RamPage-1)/RamPage,v.selected_slots),rank=blockIdx.x,h=blockIdx.y;
    if(rank>=count)return;const size_t selected=size_t(h)*v.slots+rank;
    if(!v.miss[selected])return;
    const int page=v.sorted_ids[size_t(h)*v.pages+rank],slot=v.selected_slot[selected];
    // Speculative Q/K/V for every query are written before the first query's
    // selection. Fill the whole page: a later query can hit this slot without
    // another append and must find the rows it just made logically visible.
    // The attention reader alone enforces the per-query causal boundary.
    const int rows=RamPage,vectors=rows*272/16;
    for(int i=threadIdx.x;i<2*vectors;i+=256){
        const int local=i%vectors,row=local/17,d=(local%17)*16;
        const auto *host=(i<vectors?v.host_key:v.host_value)+(size_t(page*RamPage+row)*v.heads+h)*272+d;
        auto *pool=(i<vectors?v.key:v.value)+(size_t(slot*RamPage+row)*v.heads+h)*272+d;
        *reinterpret_cast<uint4 *>(pool)=*reinterpret_cast<const uint4 *>(host);
    }
    if(!threadIdx.x)atomicAdd(&v.counters->loaded_bytes,(unsigned long long)(rows*272*2));
}
}
size_t ram_sort_workspace(RamKvView v){
    size_t bytes=0;check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr,bytes,v.scores,v.sorted_scores,
        v.page_ids,v.sorted_ids,v.heads*v.pages,v.heads,v.offsets,v.offsets+1),"RAM indexer sort workspace");return bytes;
}
void ram_select(const float *q,RamKvView v,const int *positions,int shift,int heads,cudaStream_t stream){
    page_scores<<<dim3(v.pages,v.heads),256,0,stream>>>(q,v,positions,shift,heads);
    check_cuda(cudaGetLastError(),"query-aware full history page scores");
    check_cuda(cub::DeviceSegmentedRadixSort::SortPairsDescending(v.sort_work,v.sort_bytes,v.scores,v.sorted_scores,
        v.page_ids,v.sorted_ids,v.heads*v.pages,v.heads,v.offsets,v.offsets+1,0,32,stream),"query-aware GPU page selection");
    plan<<<v.heads,256,0,stream>>>(v,positions,shift);check_cuda(cudaGetLastError(),"GPU page residency lease plan");
    load_pages<<<dim3(v.selected_slots,v.heads),256,0,stream>>>(v,positions,shift);check_cuda(cudaGetLastError(),"RAM selected KV page-in");
}
}
