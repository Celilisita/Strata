#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdint>
#include <memory>
namespace strata::gemma4::native {
constexpr int HistoryPage=16,HistoryDim=512,HistoryKvHeads=4,HistoryHeads=32;
constexpr int HistoryRowBytes=544,HistoryResident=32768,HistoryQueries=3;
constexpr int HistorySinkPages=4,HistoryRecentPages=64;
struct HistoryBinding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct HistoryCounters {
    unsigned long long written_bytes=0,resident_write_bytes=0,select_calls=0;
    unsigned long long selected_pages=0,hits=0,misses=0,loaded_bytes=0;
};
// host_* are mapped DEVICE aliases of authoritative pinned RAM, token-major
// [roundup16(logical)][4][544]. logical_capacity stays the exact request limit.
// Resident pools keep the same layout using physical slots.
struct HistoryView {
    unsigned char *host_key=nullptr,*host_value=nullptr,*key=nullptr,*value=nullptr;
    int logical_capacity=0,resident=0,pages=0,slots=0; // Actual min(roundup16(logical),32768).
    // [KVhead][page][lower512,upper512], outward-rounded half bounds.
    __half* bounds=nullptr;
    float *scores=nullptr,*sorted_scores=nullptr;
    int *page_ids=nullptr,*sorted_ids=nullptr,*offsets=nullptr;
    int *page_slot=nullptr,*slot_page=nullptr,*marked=nullptr,*miss_rank=nullptr,*free_slot=nullptr;
    // Retained [query0..2][KVhead][rank<slots<=2048]; later selection may evict old slots.
    // selected_page remains the authoritative diagnostic ID for that query.
    int *selected_page=nullptr,*selected_slot=nullptr,*miss=nullptr,*selected_count=nullptr;
    // Most recent successful write end (may shrink); NOT a committed cursor.
    // Selection/attention visibility is exclusively *position+query_row.
    int* valid_end=nullptr;unsigned* invalid=nullptr;std::uint64_t* device_epoch=nullptr;
    HistoryCounters* counters=nullptr;
    void* sort_work=nullptr;size_t sort_bytes=0;
    HistoryBinding binding{};
};
class HistoryCache {
public:
    HistoryCache(int logical_capacity,int resident,cudaStream_t);
    ~HistoryCache();
    HistoryCache(const HistoryCache&)=delete;HistoryCache& operator=(const HistoryCache&)=delete;
    HistoryView view()const{return view_;}
    HistoryBinding binding()const{return view_.binding;}
    // CPU aliases for I/O/byte diagnostics only; never model arithmetic.
    const void* host_key()const;const void* host_value()const;
    size_t host_bytes()const;size_t resident_bytes()const;size_t device_bytes()const;
    // Initial identity directory only; caller must keep end<=32768 and must
    // not use this CacheView after selection has replaced any resident page.
    CacheView initial_resident_view()const;
    void reset(); // Cold, same stream; external graphs/leases released first.
private:
    struct Allocations;std::unique_ptr<Allocations> storage_;HistoryView view_{};
};
// GPU position is ONE absolute base; row t is *position+t, never positions[t].
// RAM payload is never shifted. Writes allow causal overwrite after rollback.
void history_write(const float* key,const float* value,HistoryView,const int* position,
    int tokens,unsigned* invalid,HistoryBinding live);
// Selection >32768 is approximate full-history retrieval: preserve all RAM,
// attend selected pages only. <=32768 selects every visible page exactly.
// Per query: GPU page scores(max over its8 Q heads), CUB rank, residency/load,
// original174-style F32 QK/online-softmax/PV/merge. No DP4A query approximation.
void history_attention(const float* query,float* output,HistoryView,const int* position,
    int heads,int tokens,AttentionWorkspace,HistoryBinding live);
// Core-only helpers also used by the dedicated partial unit, not Model hooks.
void history_reset(HistoryView,HistoryBinding live);
size_t history_sort_workspace(HistoryView);
void history_select(const float* query,HistoryView,const int* position,int query_row,HistoryBinding live);
size_t history_attention_bytes();
namespace history_detail { void validate(HistoryView,HistoryBinding); }
}
