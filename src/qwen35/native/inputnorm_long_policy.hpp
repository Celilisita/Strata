#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::qwen35::native {
inline bool inputnorm_long_parse(const char *value){
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::invalid_argument("STRATA_NVFP4_INPUT_NORM_LONG must be0or1");
}
inline bool inputnorm_long_requested(){
    static const bool enabled=inputnorm_long_parse(std::getenv("STRATA_NVFP4_INPUT_NORM_LONG"));return enabled;
}
// Request admission only: the existing producer still accepts exactly5120x8192.
inline bool inputnorm_request_admit(int prompt){
    if(prompt<1)throw std::invalid_argument("input RMS request span");
    const bool enabled=inputnorm_long_requested();return prompt==8192||(enabled&&prompt>8192);
}
}
