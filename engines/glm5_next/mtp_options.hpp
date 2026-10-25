#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline bool mtp_requested() {
    const char* flag=std::getenv("STRATA_GLM53_MTP");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM MTP must be0 or1");
    return true;
}
inline int minimum_cache_routes(int usual) {return mtp_requested()?16:usual;}
inline bool defer_mtp_catchup_requested() {
    const char* flag=std::getenv("STRATA_GLM53_MTP_DEFER_CATCHUP");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM deferred MTP catchup must be0 or1");
    return true;
}
}
