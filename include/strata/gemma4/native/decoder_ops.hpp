#pragma once
#include "strata/gemma4/native/ffn.hpp"
namespace strata::gemma4::native {
struct CacheView {void* key;void* value;int capacity,heads,dim,window;};
void project_vector(primitive::Matrix,const float*,float*,cudaStream_t);
void embed(primitive::Matrix,const int*,float*,int,float,unsigned*,cudaStream_t);
void head_norm_rope(const float*,float*,const float*,int,int,int,float,float,const float*,const int*,bool,cudaStream_t);
void store_kv(const float*,const float*,CacheView,const int*,int,unsigned*,cudaStream_t);
void attention(const float*,float*,CacheView,const int*,int,int,cudaStream_t);
void pick(float*,float*,int*,int*,unsigned*,int,int,float,cudaStream_t);
void advance(int*,int,int,unsigned*,cudaStream_t);
void gelu_product(const float*,const float*,float*,std::size_t,cudaStream_t);
void residual_scale(const float*,const float*,const float*,float*,std::size_t,cudaStream_t);
void softcap_logits(float*,std::size_t,float,cudaStream_t);
void softcap_logits_checked(float*,std::size_t,float,unsigned*,cudaStream_t);
}
