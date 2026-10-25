#pragma once
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
// PREPARE-only policy; Graphs copy device parameters and never replay these host diagnostics.
inline std::uint64_t &fp4_post_norm_store_host_omissions(){static thread_local std::uint64_t count=0;return count;}
inline bool fp4_post_norm_store_prepare(bool eligible){
    const char *raw=std::getenv("STRATA_NVFP4_SKIP_POST_NORM_STORE");const std::string mode=raw?raw:"auto";
    if(mode!="auto"&&mode!="0"&&mode!="1")throw std::invalid_argument("post norm store mode must be auto,0,1");
    fp4_post_norm_store_host_omissions()=0;return eligible&&mode!="0";
}
inline float *fp4_post_norm_store_output(bool enabled,bool tma,bool merged,int tokens,float *normalized){
    if(enabled&&tma&&merged&&tokens==8192){++fp4_post_norm_store_host_omissions();return nullptr;}
    return normalized;
}
}
