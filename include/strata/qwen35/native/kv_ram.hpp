#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <memory>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include "strata/qwen35/native/decode_graph_patch.hpp"
namespace strata::qwen35::native {
constexpr int RamPage=16;
struct RamKvCounters {
    unsigned long long written_bytes=0,resident_write_bytes=0,selected_pages=0,hits=0,misses=0,loaded_bytes=0,calls=0;
};
// host_key/value name the authoritative logical rows: mapped RAM normally,
// or identity GPU rows while every logical page fits the resident directory.
// The owner changes authority only after draining streams and Graph leases.
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
struct PhaseKvStats {
    bool requested=false,admitted=false,migrated=false,failed=false,graphs_cleared=false;
    int resident_rows=0,selected_tokens=0,logical_capacity=0;
    std::uint64_t caches=0,main_caches=0,draft_caches=0,mapped_caches=0,d2h_bytes=0,d2d_bytes=0,summary_bytes=0,epoch=0;
    int pp_bound=0,main_bound=0,draft_bound=0,main_prefix_bound=0;
    std::uintptr_t owner=0;double elapsed_ms=0;
};
inline PhaseKvStats &phase_kv_stats(){static thread_local PhaseKvStats v;return v;}
inline void configure_phase_kv(bool admitted){
    const char *p=std::getenv("STRATA_DENSE_PP_RESIDENT_TG_PAGED");const std::string mode=p?p:"0";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("phase KV mode must be auto,0,1");
    auto &s=phase_kv_stats();s={};s.requested=mode!="0";s.admitted=s.requested&&admitted;
}
inline void begin_phase_kv_request(){
    auto &s=phase_kv_stats();if(s.failed)throw std::runtime_error("phase KV owner failed; reload required");
    const auto epoch=s.epoch,owner=s.owner;const bool requested=s.requested,admitted=s.admitted;
    s={};s.epoch=epoch;s.owner=owner;s.requested=requested;s.admitted=admitted;
}
inline bool phase_kv_will_demote(int prompt,int maximum,bool text,int proposals,bool reusable){
    return phase_kv_stats().admitted&&text&&reusable&&proposals==2&&prompt>8192&&maximum>32768&&maximum<=40960-3;
}
class RamKvCache {
public:
    RamKvCache(int capacity,int resident,int heads,cudaStream_t,int selected_tokens=32768,int keep_queries=0);
    RamKvCache(int capacity,int resident,int heads,cudaStream_t,int selected_tokens,int keep_queries,bool device_rows);
    ~RamKvCache();
    RamKvCache(const RamKvCache &)=delete;
    RamKvView &view(){return view_;}
    const RamKvView &view()const{return view_;}
    void reserve(int,cudaStream_t);
    // Full fresh prefill only: no prior logical rows survive growth.
    // Caller releases graph leases before this call and resets before PP.
    void reserve_discarding(int,cudaStream_t);
    void reserve_discarding_device(int,cudaStream_t);
    bool device_rows()const;
    // Cold exact demotion: caller has released every Graph lease; rows are preserved.
    void preserve_to_host(int required,cudaStream_t);
    void reset(cudaStream_t);
    // Cross-request resume: keep host rows, rebuild identity resident rows.
    void resume(int,cudaStream_t);
    size_t host_bytes()const;
    size_t resident_bytes()const{return size_t(view_.resident)*view_.heads*272*2;}
    RamKvCounters counters(cudaStream_t)const;
private:
    void grow(int,cudaStream_t,bool preserve,bool device_rows);
    struct Allocations;
    std::unique_ptr<Allocations> storage_;
    RamKvView view_{};
};
void ram_reset(RamKvView,cudaStream_t);
void ram_invalidate(RamKvView,cudaStream_t);
size_t ram_sort_workspace(RamKvView);
void ram_select(const float *query,RamKvView,const int *positions,int shift,int heads,cudaStream_t);
void ram_summarize(RamKvView,const int *,int,int,cudaStream_t);
void ram_write(const float *,RamKvView,bool key,int dim,int heads,int tokens,const int *positions,int shift,cudaStream_t);
DecodeNode ram_cache_node(void *);
DecodeNode ram_decode_node(void *);
}
