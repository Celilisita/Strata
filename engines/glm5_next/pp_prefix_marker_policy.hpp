#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
// Graph-build snapshot only. Missing/0 preserves the original attention marker.
inline bool pp_prefix_late_marker_requested() {
    const char* text=std::getenv("STRATA_GLM53_PP_PREFIX_LATE_MARKER");
    if(!text||!std::strcmp(text,"0"))return false;
    if(!std::strcmp(text,"1"))return true;
    throw std::runtime_error("PP prefix late marker must be0 or1");
}
} // namespace strata::glm5_next
