#pragma once
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
// Read once per native GLM PP run, never in a product or expert loop.
inline bool pp_reuse_host_tile_counts_requested() {
    const char* value = std::getenv("STRATA_GLM53_PP_REUSE_HOST_TILE_COUNTS");
    if (!value || !std::strcmp(value, "0")) return false;
    if (!std::strcmp(value, "1")) return true;
    throw std::runtime_error("PP host tile counts option must be0 or1");
}
}
