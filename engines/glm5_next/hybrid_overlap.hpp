#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline bool hybrid_overlap_requested() {
    const char* value = std::getenv("STRATA_GLM53_HYBRID_OVERLAP");
    if (value && std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("GLM hybrid overlap must be 0 or 1");
    return value && !std::strcmp(value,"1");
}
}
