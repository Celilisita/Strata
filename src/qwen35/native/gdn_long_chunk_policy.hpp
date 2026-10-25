#pragma once
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
inline bool gdn_long_flag(const char* key){const char* p=std::getenv(key);const std::string s=p?p:"0";
    if(s!="0"&&s!="1")throw std::invalid_argument(std::string(key)+" must be0or1");return s=="1";
}
inline bool gdn_long_chunk_requested(){static const bool on=gdn_long_flag("STRATA_DENSE_GDN_LONG_CHUNKS");return on;}
inline bool gdn_long_weighted_requested(){static const bool on=[](){
    const bool value=gdn_long_flag("STRATA_DENSE_GDN_LONG_WEIGHTED");
    if(value&&!gdn_long_chunk_requested())throw std::invalid_argument("Long weighted GDN requires long chunks");return value;}();return on;
}
inline bool& gdn_long_profile(){static thread_local bool qwen36=false;return qwen36;}
inline void configure_gdn_long_profile(bool qwen36){
    if((gdn_long_chunk_requested()||gdn_long_weighted_requested())&&!qwen36)
        throw std::invalid_argument("Long GDN candidates require the explicit Qwen3.6 profile");
    gdn_long_profile()=qwen36;
}
// Scope is the measured30000 request only. Per-call services retain exact8192
// geometry; tail5424 uses the existing vector path. No state is zeroed here.
inline bool gdn_fast_request(int prompt){return prompt==8192||(gdn_long_profile()&&prompt==30000&&gdn_long_chunk_requested());}
inline bool gdn_weighted_request(int prompt){return prompt==8192||
    (gdn_long_profile()&&prompt==30000&&gdn_long_chunk_requested()&&gdn_long_weighted_requested());}
}
