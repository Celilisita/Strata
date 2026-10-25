#pragma once
#include "rescale_skip_attention.hpp"
#include <cstdlib>
#include <cstdint>
namespace strata::qwen35::native {
inline bool &rescale_skip_active(){static thread_local bool active=false;return active;}
inline std::uint64_t &rescale_skip_submissions(){static thread_local std::uint64_t count=0;return count;}
inline void configure_rescale_skip_request(bool resident,int prompt){
    const char *flag=std::getenv("STRATA_DENSE_SM120_SKIP_IDENTITY_RESCALE");
    rescale_skip_active()=resident&&prompt>=16384&&(!flag||flag[0]!='0');
}
inline void prepare_rescale_skip(){
    static thread_local bool configured=false;
    if(rescale_skip_active()&&!configured){configure_rescale_skip_attention();configured=true;}
}
inline bool use_rescale_skip(PrefillAttentionShape s){
    return rescale_skip_active()&&s.tokens==8192&&(s.query_start==8192||s.query_start==16384);
}
}
