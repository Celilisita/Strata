#pragma once
#include <strata/qwen35/native/kv_ram.hpp>
namespace strata::qwen35::native {
struct PageRowResources {int registers,shared_bytes,local_bytes,binary_version,blocks_per_sm,baseline_thread_limit;};
struct PageRowControl {int registers=0,shared_bytes=0,local_bytes=0,blocks_per_sm=0;};
PageRowResources page_row_resources();
// Cold graph construction only: match and preserve all five arguments of the retained score node.
bool page_row_patch(cudaGraphNode_t,RamKvView,const float*,const int*,int,bool,PageRowControl&);
void page_row_scores(const float*,RamKvView,const int*,int,int,cudaStream_t);
}
