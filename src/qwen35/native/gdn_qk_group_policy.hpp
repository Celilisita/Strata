#pragma once
#include "gdn_delta_half.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cstdlib>
#include <cstdint>
#include <string>
namespace strata::qwen35::native {
inline bool &gdn_qk_group_active(){static thread_local bool active=false;return active;}
// Counts accepted eager/capture host submissions; graph replay has no host call.
inline std::uint64_t &gdn_qk_group_submissions(){static thread_local std::uint64_t count=0;return count;}
inline void configure_gdn_qk_group_request(bool admitted){
    auto &active=gdn_qk_group_active();active=false;
    const char *flag=std::getenv("STRATA_DENSE_GDN_QK_GROUP");
    if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("GDN QK group mode");
    if(!admitted||(flag&&flag[0]=='0')||!gdn_delta_active())return;
    int device=0;check_cuda(cudaGetDevice(&device),"GDN QK group request device");
    const auto caps=capabilities(device);active=caps.major==12&&caps.minor==0;
}
inline bool use_gdn_qk_group(GdnShape s){
    return gdn_qk_group_active()&&gdn_delta_active()&&s.tokens==8192&&s.key_dim==128&&s.value_dim==128&&
        s.query_heads==16&&s.value_heads==48;
}
}
