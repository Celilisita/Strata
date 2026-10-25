#pragma once
#include <strata/qwen35/native/attention_ram.hpp>
namespace strata::qwen35::native {
struct Ram40ScoreLayout {size_t partial_bytes,score_offset,score_bytes,total_bytes;int cache_keys;};
struct Ram40ScoreResources {int registers,blocks_per_sm,shared_bytes,local_bytes,binary_version;};
Ram40ScoreLayout ram40_score_layout(RamKvView,int heads,int parts);
Ram40ScoreResources ram40_score_resources();
void ram40_score_validate(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int);
void ram40_score_attention(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int,int,cudaStream_t);
}
