#pragma once
#include "strata/qwen35/native/kv_ram.hpp"
namespace strata::qwen35::native {
struct RamBlockResources {int registers,local_bytes,shared_bytes;};
bool ram_block_sort_policy(int pages);
RamBlockResources ram_block_resources();
void ram_sort_control(RamKvView,cudaStream_t);
void ram_block_sort_only(RamKvView,cudaStream_t);
void ram_select_control(const float*,RamKvView,const int*,int,int,cudaStream_t);
void ram_select_block(const float*,RamKvView,const int*,int,int,cudaStream_t);
}
