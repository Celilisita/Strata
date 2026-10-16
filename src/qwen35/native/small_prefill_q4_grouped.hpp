#pragma once
#include "strata/qwen35/native/small_prefill.hpp"
#include <thread>

namespace strata::qwen35::native {
void configure_small_prefill_q4_pair();
// Raw single-only scratch schedule; special diagnostic layouts retain five.
int q4_single_partitions(Matrix);
void small_prefill_q4_single_linear(Matrix,const void *,float *,int,cudaStream_t,void *parts=nullptr,size_t bytes=0);
void small_prefill_q4_small_pair_linear(Matrix,Matrix,const void *,float *,float *,cudaStream_t,void *,size_t);
void small_prefill_q4_tensor_bias_linear(Matrix,const void *,float *,int,cudaStream_t,void *,size_t);
struct Q4SingleWorkspace { void *data=nullptr;size_t bytes=0;Matrix weight{};cudaStream_t stream=nullptr;bool ready=false;int device=-1;std::thread::id owner; };
size_t q4_single_workspace_bytes(Matrix);
Q4SingleWorkspace initialize_q4_single_workspace(Matrix,void *,size_t,cudaStream_t);
void small_prefill_q4_fused_bias_linear(Matrix,const void *,float *,int,cudaStream_t,Q4SingleWorkspace);
// Optional Q6 output descriptor must be distinct from the live Q4 input.
void small_prefill_q4_pair_linear(Matrix,Matrix,const void *,float *,float *,float *,int,cudaStream_t,bool=false,void *q6_codes=nullptr,int segmented_processors=0,bool block_affine=false,bool tensor_bias=false);
}
