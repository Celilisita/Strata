#pragma once
#include "regp.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
struct RegpRequestStats {bool requested=false,admitted=false;std::array<std::uint64_t,4> host_calls{};};
inline RegpRequestStats &regp_stats(){static thread_local RegpRequestStats value;return value;}
inline void configure_regp_request(bool resident_single_P){
    auto &v=regp_stats();v={};const char *raw=std::getenv("STRATA_DENSE_PP_REGISTER_P");const std::string flag=raw?raw:"0";
    if(flag!="0"&&flag!="1")throw std::invalid_argument("PP register-P mode must be0or1");
    v.requested=flag=="1";if(!v.requested||!resident_single_P)return;
    int device=0;check_cuda(cudaGetDevice(&device),"register-P admission device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)return;configure_regp_attention();const auto r=regp_attention_resources();
    if(regp_binary_version()!=120||r.blocks_per_SM<1||r.local_bytes||r.max_threads<256)
        throw std::runtime_error("register-P requires actualSM120 zero-local resident CTA");
    v.admitted=true;
}
inline int regp_case(PrefillAttentionShape s){
    if(s.graph_start||!s.q8||s.dim!=256||s.query_heads!=24||s.kv_heads!=4)return -1;
    if(s.tokens==8192&&(s.query_start==0||s.query_start==8192||s.query_start==16384))return s.query_start/8192;
    return s.tokens==5424&&s.query_start==24576?3:-1;
}
inline bool use_regp_attention(PrefillAttentionShape s){return regp_stats().admitted&&regp_case(s)>=0;}
}
