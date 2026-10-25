#pragma once
#include "strata/qwen35/native/attention_online.hpp"
namespace strata::qwen35::native {
void configure_compact_attention(int query_rows,bool asynchronous);
PrefillOnlineResources compact_attention_resources(int query_rows,bool asynchronous);
void compact_attention(const float *,const void *,const void *,float *,void *,size_t,
    PrefillAttentionShape,int query_rows,bool asynchronous,cudaStream_t);
void configure_pipelined_attention(int mode);
PrefillOnlineResources pipelined_attention_resources(int mode);
void pipelined_attention(const float *,const void *,const void *,float *,void *,size_t,
    PrefillAttentionShape,int mode,cudaStream_t);
}
