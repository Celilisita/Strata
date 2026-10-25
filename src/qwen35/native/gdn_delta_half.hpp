#pragma once
#include "gdn_float_value.hpp"
#include <cstdint>
#include <cstdlib>
#include <string>
namespace strata::qwen35::native {
struct GdnDeltaResources {int registers,blocks;size_t shared_bytes,local_bytes;};
void gdn_delta_half_configure();
void gdn_delta_half(GdnInputs,GdnShape,GdnFloatValueWork,float*,float*,cudaStream_t);
GdnDeltaResources gdn_delta_half_resources();
inline bool &gdn_delta_active(){static thread_local bool value=false;return value;}
inline uint64_t &gdn_delta_submissions(){static thread_local uint64_t value=0;return value;}
inline void configure_gdn_delta_request(bool admitted){
    const char *mode=std::getenv("STRATA_DENSE_GDN_HALF_DELTA");
    if(mode&&std::string(mode)!="0"&&std::string(mode)!="1")throw std::invalid_argument("GDN halfdelta mode");
    gdn_delta_active()=admitted&&(!mode||mode[0]!='0');
    if(gdn_delta_active())gdn_delta_half_configure();
}

}
