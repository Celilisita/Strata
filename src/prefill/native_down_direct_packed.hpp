#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::prefill::mmq {
inline bool down_direct_packed_flag(const char* v) {
    if(!v||std::strcmp(v,"0")==0)return false;
    if(std::strcmp(v,"1")==0)return true;
    throw std::invalid_argument("PP_DOWN_DIRECT_PACKED requires exactly0 or1");
}
inline bool down_direct_packed_enabled() {
    static const bool on=down_direct_packed_flag(std::getenv("STRATA_QWEN35_PP_DOWN_DIRECT_PACKED"));return on;
}
struct DownDirectPackedResources {
    int cc=0,binary=0,registers=0,threads=256,active_blocks=0,max_threads=0;
    size_t shared=0,local=0;bool eligible=false;
};
DownDirectPackedResources native_down_direct_packed_resources();
// Internal guarded producer; callers prove the exact Down P before any device write.
void native_down_direct_packed(const float* input,void* packed,int entries,const int32_t* ids,void* stream);
} // namespace strata::prefill::mmq
