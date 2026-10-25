#pragma once
#include "strata/gemma4/native/attention_native.hpp"
#include <cstddef>
namespace strata::gemma4::native {
bool whole_history_requested();
std::size_t whole_history_workspace_bytes(int tokens,int key_upper);
void whole_history_attention(const float* query,float* output,CacheView,const int* position,
    int past_upper,int tokens,int maximum_tokens,AttentionWorkspace,cudaStream_t);
namespace whole_history_detail {
constexpr int Query=64,Heads=32,Dim=512,KvHeads=4;
struct Buffers {void *key=nullptr,*value=nullptr,*query=nullptr,*probability=nullptr;
    float *score=nullptr,*partial=nullptr,*den=nullptr;void** pointers=nullptr;unsigned* invalid=nullptr;int keys=0;};
Buffers bind(AttentionWorkspace,int tokens,int key_upper);
void stage(Buffers,const float*,CacheView,const int*,int tokens,cudaStream_t);
void pointers(Buffers,int first,int tokens,cudaStream_t);
void normalize(Buffers,const int*,int first,int queries,int tokens,int capacity,cudaStream_t);
void finish(Buffers,float*,int first,int queries,cudaStream_t);
void propagate(Buffers,float*,int tokens,cudaStream_t);
}
}
