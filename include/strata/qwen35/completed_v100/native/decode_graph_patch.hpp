#pragma once
#include "strata/qwen35/completed_v100/native/attention_decode.hpp"
#include <array>
#include <stdexcept>

namespace strata::completed_v100::qwen35::native {
enum class DecodeNode { None,Position,CacheWrite,Segments,Combine,Record };
DecodeNode decode_token_node(void *function);
DecodeNode decode_cache_node(void *function);
DecodeNode decode_attention_node(void *function);
void update_decode_token(cudaGraphExec_t,cudaGraphNode_t,const cudaKernelNodeParams &,int);
void update_decode_cache(cudaGraphExec_t,cudaGraphNode_t,const cudaKernelNodeParams &,int);
void update_decode_attention(cudaGraphExec_t,cudaGraphNode_t,const cudaKernelNodeParams &,DecodeNode,int,int,DecodeRounding);
template<size_t N> inline std::array<void *,N> borrowed_graph_arguments(const cudaKernelNodeParams &p){
    if(!p.kernelParams||p.extra)throw std::invalid_argument("decode graph requires ordinary captured kernel arguments");
    std::array<void *,N> args{};for(size_t i=0;i<N;++i)args[i]=p.kernelParams[i];return args;
}
}
