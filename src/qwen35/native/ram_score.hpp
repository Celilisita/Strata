#pragma once
#include <strata/qwen35/native/attention_ram.hpp>
namespace strata::qwen35::native {
struct RamScoreLayout {size_t partial_bytes,score_offset,score_bytes,total_bytes;int cache_keys;};
struct RamScoreResources {int registers,blocks_per_sm,shared_bytes,local_bytes;};
RamScoreLayout ram_score_layout(RamKvView,int heads,int parts);
RamScoreResources ram_score_resources();
void ram_score_validate(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int);
void ram_score_attention(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int,int,cudaStream_t);
}
