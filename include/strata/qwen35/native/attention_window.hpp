#pragma once
#include "strata/qwen35/native/attention_layer.hpp"
#include "strata/qwen35/native/decode_graph_patch.hpp"
namespace strata::qwen35::native {
// Sliding selection is opt-in, changes full attention mathematics, and keeps
// absolute rotary positions. The physical ring includes one PP transaction.
DecodeNode window_cache_node(void *);
DecodeNode window_decode_node(void *);
void q8_window_write(const float *,void *,int,int,int,const int *,int,int,cudaStream_t);
void configure_window_attention();
void window_decode_attention(const float *,const void *,const void *,float *,void *,size_t,
                             AttentionCache,const int *,int,int,int,float,DecodeRounding,cudaStream_t);
void window_verify_attention(const float *,const void *,const void *,float *,void *,size_t,
                             PrefillAttentionShape,const int *,int,int,int,cudaStream_t);
void window_prefill_attention(const float *,const void *,const void *,float *,void *,size_t,
                             PrefillAttentionShape,int,cudaStream_t);
}
