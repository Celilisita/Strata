#pragma once
#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native {
inline bool pp_rope_table_requested(){
    const char* raw=std::getenv("STRATA_GEMMA_PP_ROPE_TABLE");const std::string value=raw?raw:"0";
    if(value!="0"&&value!="1")throw std::invalid_argument("Gemma PP RoPE table requires0or1");
    if(value=="1"&&gemma_weight_format_requested()!=GemmaWeightFormat::OriginalQuant&&
       gemma_weight_format_requested()!=GemmaWeightFormat::Nvfp4)
        throw std::invalid_argument("Gemma PP RoPE table requires its explicit native weight backend");
    return value=="1";
}
inline std::size_t rope_table_bytes(int capacity){
    if(capacity<1||capacity>8192)throw std::invalid_argument("Gemma RoPE table capacity");
    return std::size_t(capacity)*384*2*sizeof(float);
}
// One ordered-stream, per-forward table: local[capacity][128][sin,cos],
// followed by global[capacity][256][sin,cos]. Never indexed by host past.
void prepare_rope_table(float*,int capacity,int tokens,float local_base,float global_base,
                        const float* frequencies,const int* position,unsigned* invalid,cudaStream_t);
void head_norm_rope_table(const float*,float*,const float*,int dim,int heads,int tokens,
                         float epsilon,const float* table,int capacity,cudaStream_t);
}
