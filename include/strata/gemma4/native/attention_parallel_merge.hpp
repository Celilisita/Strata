#pragma once
#include "strata/gemma4/native/history.hpp"
namespace strata::gemma4::native::merge321 {
constexpr int Dim=512,Heads=32,Row=514,Partition=128,MaxParts=2048;
enum class Mode { Parent, Parallel }; // Default remains the frozen parent.
struct Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct Lease {AttentionWorkspace partial{};Binding binding{};};
// All strides are FLOAT ELEMENTS, not bytes. A row is [m,z,numerator512].
// parts is the physical segment count; never derive it from absolute GPU P.
struct Layout {
    int parts=0,queries=0;
    size_t head_stride=0,query_stride=0,output_query_stride=0;
};
// Each query uses *position+query_offset+query_index. Optional history state
// is checked on GPU; valid_end is written extent, not committed cursor.
struct State {
    const int* position=nullptr;int query_offset=0,logical_capacity=0;
    const int* valid_end=nullptr;const unsigned* invalid=nullptr;
    const std::uint64_t* device_epoch=nullptr;
};
Layout layout(int parts,int queries);
size_t partial_bytes(Layout);size_t output_bytes(Layout);
// No allocation/copy/sync. All query/head scratch spans must be disjoint.
// Parent zero-global-weight numerator skip is preserved, including NaN scope.
void merge_aos(float* output,Layout,State,Lease,Binding live);
// Global Q32/KV4/D512/window0, T1..3. Uses byte-identical frozen174 segment;
// candidate merge consumes capacity-derived parts and explicit query stride.
void split_attention_parallel_merge(const float* query,float* output,CacheView,
    const int* position,int heads,int tokens,Lease,Binding live,Mode=Mode::Parent);
// Existing317 select/load and segment, one query consumed before the next.
// Same scratch is reused sequentially; selected_page plans remain retained.
// Supports T1..3 and dynamic resident min(round16(request),32768).
void history_attention_parallel_merge(const float* query,float* output,HistoryView,
    const int* position,int heads,int tokens,AttentionWorkspace,HistoryBinding live,Mode=Mode::Parent);
}
