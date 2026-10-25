#pragma once
#include "strata/gemma4/native/verify_round.hpp"
#include <cstddef>
#include <cstdint>
namespace strata::gemma4::native {
// Borrowed views: same Model owner, reset epoch and ordered stream only. Main
// and Draft execute sequentially; the decoded arena cannot be used concurrently.
// Destroy all caller-owned graphs before reset, destruction or policy changes.
// Seeds/pending become usable after prefill; reset invalidates every lease.
// Cold borrowing has no seed/pending/KV visibility and is valid before prefill.
// The arena is stable for the entire Model lifetime; Draft uses it sequentially.
struct MainScratchLease {
    const void* owner=nullptr;cudaStream_t stream=nullptr;
    void* decoded=nullptr;size_t decoded_bytes=0;
};
struct MainDraftLease {
    const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;
    const float* seed=nullptr;const int *position=nullptr,*pending=nullptr;
    const unsigned* invalid=nullptr;void* decoded=nullptr;size_t decoded_bytes=0;
};
// Payloads are read-only. Visible end/left/origin/length are deliberately absent:
// the caller supplies explicit assistant bounds independently of physical slots.
struct MainCacheLease {
    const void* owner=nullptr;std::uint64_t epoch=0;
    const void *key=nullptr,*value=nullptr;int capacity=0,heads=0,dim=0,window=0;
};
struct VerifyPacket {VerifyResponse response;unsigned invalid=0;int rows=0;};
static_assert(sizeof(VerifyResponse)==36&&sizeof(VerifyPacket)==44,"Gemma verify response packet ABI");
static_assert(offsetof(VerifyPacket,invalid)==sizeof(VerifyResponse),"Gemma contiguous response/invalid readback");
struct VerifyInspection {
    const int *inputs=nullptr,*picks=nullptr;const float *features=nullptr,*logits=nullptr;
    const VerifyPacket* packet=nullptr;
};
namespace detail {
// Fixed metadata and invalid aggregation on GPU, including external graph replay.
void begin_main_verify(VerifyPacket*,const unsigned* draft_invalid,int rows,cudaStream_t);
void begin_base_tail(VerifyPacket*,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy,cudaStream_t);
void finish_base_tail(VerifyPacket*,const int* position,const int* pending,
    int capacity,int vocabulary,VerifyPolicy,cudaStream_t);
}
}
