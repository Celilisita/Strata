#pragma once
#include <cerrno>
#include <charconv>
#include <system_error>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
namespace strata::glm5_next {
inline bool phase_cache_requested() {
    const char* value=std::getenv("STRATA_GLM53_PHASE_CACHE");
    if(!value||!std::strcmp(value,"0"))return false;
    if(std::strcmp(value,"1"))throw std::runtime_error("GLM phase cache must be0 or1");
    const char* pp=std::getenv("STRATA_GLM53_STREAM_PP");
    if(!pp||std::strcmp(pp,"1"))throw std::runtime_error("GLM phase cache requires native PP");
    return true;
}
inline size_t tg_headroom_bytes() {
    const char* text=std::getenv("STRATA_GLM53_TG_HEADROOM_MIB");
    if(!text)return size_t(3072)<<20;
    // Open only the authorized 512 MiB reduction from the existing 3 GiB allowance.
    // This qualified range is not a hardware or allocation-success guarantee.
    unsigned long long mib=0;
    const char* end=text+std::strlen(text);
    const auto parsed=std::from_chars(text,end,mib);
    if(parsed.ec!=std::errc{}||parsed.ptr!=end||mib<2560||mib>3072)
        throw std::runtime_error("GLM TG headroom must be an integer in 2560..3072 MiB");
    constexpr size_t mib_bytes=size_t(1)<<20;
    if(mib>std::numeric_limits<size_t>::max()/mib_bytes)
        throw std::runtime_error("GLM TG headroom byte conversion overflow");
    return size_t(mib)*mib_bytes;
}
inline size_t tg_cache_bytes() {
    const char* text=std::getenv("STRATA_GLM53_TG_CACHE_MIB");
    if(!text)return size_t(14336)<<20;
    // A bounded request for this 17.25 GiB slice, not an allocation promise.
    // The unchanged phase admission formula selects the actual mapped budget.
    constexpr long maximum_requested_mib=17664;
    char* end=nullptr;errno=0;const auto mib=std::strtol(text,&end,10);
    if(errno==ERANGE||!*text||*end||mib<4096||mib>maximum_requested_mib)
        throw std::runtime_error("GLM TG cache request must be4096..17664MiB");
    constexpr size_t mib_bytes=size_t(1)<<20;
    const size_t allowance_bytes=tg_headroom_bytes();
    if(static_cast<unsigned long long>(mib)>
       (std::numeric_limits<size_t>::max()-allowance_bytes)/mib_bytes)
        throw std::runtime_error("GLM TG cache request byte conversion overflow");
    return size_t(mib)*mib_bytes;
}
}
