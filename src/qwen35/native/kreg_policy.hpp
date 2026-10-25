#pragma once
#include "kreg.hpp"
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
inline bool &gdn_kreg_active(){static thread_local bool active=false;return active;}
// Accepted eager/capture host submissions only; replay keeps no host descriptor.
inline std::uint64_t &gdn_kreg_submissions(){static thread_local std::uint64_t count=0;return count;}
inline void configure_gdn_kreg_request(bool admitted){
    gdn_kreg_active()=false;gdn_kreg_submissions()=0;
    const char *raw=std::getenv("STRATA_DENSE_GDN_KREG");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("GDN K-register mode must be auto,0,1");
    if(mode=="0"||!admitted||!gdn_qk_group_active()||!gdn_delta_active())return;
    kreg_prepare_configure();const auto resources=kreg_prepare_resources().grouped;
    if(resources.blocks<2||resources.local_bytes)throw std::runtime_error("GDN K-register resource admission");
    gdn_kreg_active()=true;
}
inline bool use_gdn_kreg(GdnShape s){return gdn_kreg_active()&&use_gdn_qk_group(s);}
inline void service_gdn_kreg(GdnInputs x,GdnShape s,float *out,float *state,void *scratch,size_t bytes,cudaStream_t stream){
    kreg_run(x,s,out,state,scratch,bytes,stream);
    ++gdn_register_submissions();++gdn_delta_submissions();++gdn_qk_group_submissions();++gdn_kreg_submissions();
}
}
