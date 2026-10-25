#pragma once
#include "strata/qwen35/native/attention_ram.hpp"
namespace strata::qwen35::native {
struct RamSixResources {int registers,local_bytes,shared_bytes;};
RamSixResources ram_six_resources();
void ram_verify_six(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int,cudaStream_t);
void ram_attention_six(const float*,RamKvView,float*,void*,size_t,const int*,int,int,int,DecodeRounding,int,cudaStream_t);
}
