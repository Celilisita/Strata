#pragma once
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
namespace strata::qwen35::native {
// Process-cold immutable policy: captures and eager conditions share one decision.
inline bool rmsconcat_requested(){static const bool enabled=[] {
    const char *value=std::getenv("STRATA_DENSE_MTP_RMS_CONCAT");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::invalid_argument("STRATA_DENSE_MTP_RMS_CONCAT must be 0 or 1");
}();return enabled;}
inline bool rmsconcat_uses(int features,int rows){return rmsconcat_requested()&&features==5120&&rows>=1&&rows<=3;}
inline size_t rmsconcat_copy_bytes(int features,int rows){
    return rmsconcat_uses(features,rows)?0:size_t(2)*features*rows*sizeof(float);
}
inline size_t rmsconcat_prefix_copy_bytes(int features,int tokens,int width){
    size_t copied=0;
    for(int start=0;start<tokens;start+=width){
        const int rows=tokens-start<width?tokens-start:width;
        if(start)copied+=rmsconcat_copy_bytes(features,1);
        if(rows>1)copied+=rmsconcat_copy_bytes(features,rows-1);
        copied+=size_t(features)*sizeof(float); // Retained final target seed per chunk is unchanged.
    }
    return copied;
}
inline std::uint64_t &rmsconcat_host_submissions(){static thread_local std::uint64_t count=0;return count;}
}
