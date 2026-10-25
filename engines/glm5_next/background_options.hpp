#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include "phase_options.hpp"
namespace strata::glm5_next {
inline bool background_requested() {
    const char* value=std::getenv("STRATA_GLM53_BACKGROUND_ADMISSION");
    if(!value||!std::strcmp(value,"0"))return false;
    if(std::strcmp(value,"1"))throw std::runtime_error("GLM background admission must be0 or1");
    if(!phase_cache_requested())throw std::runtime_error("GLM background admission requires the shared phase cache");
    return true;
}
}
