#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
namespace strata::gemma4::native {
// Borrowed stable storage only. Main and Draft use it sequentially on stream.
// Destroy external graphs before reset/release/destruction or policy changes.
struct Sm120MtpScratch {
    const void* owner=nullptr;cudaStream_t stream=nullptr;
    void* decoded=nullptr;std::size_t decoded_bytes=0;
};
struct Sm120MtpLease {
    const void* owner=nullptr;std::uint64_t epoch=0;cudaStream_t stream=nullptr;
    const float* seed=nullptr;const int *position=nullptr,*pending=nullptr;
    const unsigned* invalid=nullptr;void* decoded=nullptr;std::size_t decoded_bytes=0;
};
struct Sm120MtpCacheLease {
    const void* owner=nullptr;std::uint64_t epoch=0;
    const void *key=nullptr,*value=nullptr;int capacity=0,heads=0,dim=0,window=0;
};
struct Sm120TargetRows {const float *features=nullptr,*logits=nullptr;int rows=0;};
namespace detail {
// A fresh target call replaces this metadata. Commit never decides sampling.
struct Sm120TargetState {
    int start=0,rows=0,count=0,position=0,ready=0;unsigned invalid=0;
};
static_assert(sizeof(Sm120TargetState)==24,"SM120 target state ABI");
inline constexpr std::size_t Sm120TargetStorageBytes=
    std::size_t(3)*5376*4+std::size_t(3)*262144*4+5376*4+256;
void begin_sm120_target(Sm120TargetState*,const int* position,int rows,int capacity,
    unsigned* invalid,cudaStream_t);
// count is the number of processed inputs/emitted target IDs, including bonus.
// Seed is normalized row[count-1]; next remains emitted but not yet processed.
// count0 or invalid leaves seed/pending/position unchanged. Rejected suffix KV
// is future to the committed cursor, with window+3 physical ring retention.
void commit_sm120_target(Sm120TargetState*,const float* row_features,float* seed,
    int* pending,int* position,const int* count,const int* next,
    const unsigned* round_invalid,int capacity,unsigned* invalid,cudaStream_t);
}
}
