#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <memory>
#include "strata/qwen35/completed_v100/native/decode_graph_patch.hpp"
namespace strata::completed_v100::qwen35::native {
constexpr int RamPage=16;
struct RamKvCounters {
    unsigned long long written_bytes=0,resident_write_bytes=0,selected_pages=0,hits=0,misses=0,loaded_bytes=0,calls=0;
};
// Logical host rows never alias. A device slot is replaceable and has one owner.
struct RamKvView {
    unsigned char *host_key=nullptr,*host_value=nullptr,*key=nullptr,*value=nullptr;
    int host_capacity=0,resident=32768,heads=0,pages=0,slots=2048,selected_tokens=32768,selected_slots=2048,keep_queries=0;
    __half *bounds=nullptr;
    float *scores=nullptr,*sorted_scores=nullptr;
    int *page_ids=nullptr,*sorted_ids=nullptr,*offsets=nullptr;
    int *page_slot=nullptr,*slot_page=nullptr,*selected_slot=nullptr,*miss=nullptr,*marked=nullptr,*miss_rank=nullptr,*free_slot=nullptr;
    RamKvCounters *counters=nullptr;int *last_use=nullptr,*epoch=nullptr;
    void *sort_work=nullptr;size_t sort_bytes=0;
};
// Cold allocation/growth only; all request selection and page-in is on GPU.
class RamKvCache {
public:
    RamKvCache(int capacity,int resident,int heads,cudaStream_t,int selected_tokens=32768,int keep_queries=0);
    ~RamKvCache();
    RamKvCache(const RamKvCache &)=delete;
    RamKvView &view(){return view_;}
    const RamKvView &view()const{return view_;}
    void reserve(int,cudaStream_t);
    void reset(cudaStream_t);
    size_t host_bytes()const{return size_t(view_.host_capacity)*view_.heads*272*2;}
    size_t resident_bytes()const{return size_t(view_.resident)*view_.heads*272*2;}
    RamKvCounters counters(cudaStream_t)const;
private:
    struct Allocations;
    std::unique_ptr<Allocations> storage_;
    RamKvView view_{};
};
void ram_reset(RamKvView,cudaStream_t);
size_t ram_sort_workspace(RamKvView);
void ram_select(const float *query,RamKvView,const int *positions,int shift,int heads,cudaStream_t);
void ram_summarize(RamKvView,const int *,int,int,cudaStream_t);
void ram_write(const float *,RamKvView,bool key,int dim,int heads,int tokens,const int *positions,int shift,cudaStream_t);
DecodeNode ram_cache_node(void *);
DecodeNode ram_decode_node(void *);
}
