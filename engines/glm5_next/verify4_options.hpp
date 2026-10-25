#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
// Independent whole-verify diagnostic only; never enables normal generation.
inline bool verify4_probe_requested() {
    const char* value=std::getenv("STRATA_GLM53_VERIFY4_PROBE");
    if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
        throw std::runtime_error("GLM four-position probe must be 0 or 1");
    return value&&!std::strcmp(value,"1");
}
}
