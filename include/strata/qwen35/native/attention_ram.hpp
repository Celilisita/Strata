#pragma once
#include "strata/qwen35/native/kv_ram.hpp"
#include "strata/qwen35/native/attention_decode.hpp"
namespace strata::qwen35::native {
void ram_attention(const float *,RamKvView,float *,void *,size_t,const int *,int,int,int,DecodeRounding,cudaStream_t);
void ram_verify_one(const float *,RamKvView,float *,void *,size_t,const int *,int,int,cudaStream_t);
}
