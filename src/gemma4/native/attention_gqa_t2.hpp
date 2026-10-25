#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::gemma4::native::gqa371 {
constexpr bool DefaultEnabled=false;
constexpr int Heads=32,KvHeads=4,Dim=512,Queries=2,HeadsPerKv=8,Rows=16;
constexpr int PvKeys=512; // Sole PV reduction-tile size; no run-time algorithm search.
constexpr std::size_t RawLimit=134217728,ArenaLimit=462422016,BlasBytes=33554432;
struct Binding {const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;};
struct Lease {
    void* dead_k=nullptr;std::size_t dead_k_bytes=0;
    void* dead_v=nullptr;std::size_t dead_v_bytes=0;
    Binding binding;
};
struct Layout {
    int capacity=0,parts=0;
    std::size_t kv_bytes=0,query=0,score=0,probability=0,partial=0;
    std::size_t maximum=0,denominator=0,query_bad=0,first_bad_key=0,split_partial=0,arena_bytes=0;
    long long kv_stride=0,query_stride=0,score_stride=0,output_stride=0;
    long long pv_kv_stride=0,pv_probability_stride=0,pv_partial_stride=0;
};
Layout layout(int capacity);
// Physical batches are FOUR KV heads. Each has16 independent columns:
// column=t*8+localQhead, original head=KVhead*8+localQhead, t=0,1.
// Qh=RN16(Q), Kh/Vh=RN16(decoded original Q8), no Q8-query approximation.
// P_h=RN16(exp(score-max)); denominator is the sum of UNROUNDED F32 exp.
// QK score stays [4][16][capacity]; P_h is [4][parts][16][PvKeys].
// P_h index=((head*parts+key/PvKeys)*Rows+column)*PvKeys+key%PvKeys.
// PV batches [4*parts][512 rows][16 columns] use K=PvKeys, beta=0.
// split_partial holds every batch; finish sums parts in order in F32, writes
// the merged numerator to original partial[4][16][512], then divides once.
// Every part uses the SAME global maximum and unrounded F32 denominator.
// Capacity is fixed10240 or32768; GPU P and P+t select the live causal prefix.
// Caller has already appended BOTH T2 KV rows, on this same exclusive stream.
// Only staging is zeroed for future rows; authoritative KV is read-only.
// deadK/V each advertise the FULL128MiB allocation; every advertised span is
// disjoint. Their contents and arena are dead after Q8_append, until Out/FFN.
// Draft may alias arena on the same ordered stream, never concurrently.
// Handle must already be bound to live.stream with this owner's blas buffer.
// No allocation/copy/sync; capture/replay/reset ownership remains caller-owned.
// Invalid GPU position/flag or a nonfinite LIVE source produces NaN output.
// Host lease mismatch is rejected before any enqueue. External Graphs must be
// drained/destroyed before changing epoch/storage. No Model hook or admission.
void attention(const float* query,float* output,CacheView,const int* position,
    int heads,int tokens,AttentionWorkspace,Lease,Binding live,const unsigned* invalid=nullptr);
}
