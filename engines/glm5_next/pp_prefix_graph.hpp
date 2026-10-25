#pragma once
#include <array>
#include <cstdlib>
#include <cstring>
#include "../deepseek_v4/pp_prefetch_graph.hpp"
#include <stdexcept>
#include <numeric>
namespace strata::glm5_next {
inline bool pp_prefix_requested() {
    const char* flag=std::getenv("STRATA_GLM53_PP_PREFIX");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM PP prefix must be0 or1");return true;
}
// MMQ channel strides are expressed in each matrix's quant-block units.
// Padding only between triplets makes the same byte stride integral for both
// original quant types; every learned byte and within-matrix row is unchanged.
inline size_t pp_prefix_payload_stride(const std::array<const ggml_tensor*,3>& source) {
    const size_t unit=std::lcm(ggml_type_size(source[0]->type),ggml_type_size(source[2]->type));
    const size_t bytes=source[0]->nb[2]+source[1]->nb[2]+source[2]->nb[2];return ((bytes+unit-1)/unit)*unit;
}
inline bool pp_prefix_option(const char* name) {
    const char* flag=std::getenv(name);if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM prefix option must be0 or1");return true;
}
inline bool pp_prefix_planar(){return pp_prefix_option("STRATA_GLM53_PP_PREFIX_PLANAR");}
inline bool pp_prefix_true_fences(){return pp_prefix_option("STRATA_GLM53_PP_PREFIX_TRUE_FENCES");}
inline void tag_pp_prefix(ggml_tensor* norm,const deepseek_v4::PpCacheView* view,int layer,int tokens) {
    if(tokens>=2048&&view&&pp_prefix_requested())deepseek_v4::tag_pp_prefetch_marker(norm,view,layer);
}
}
