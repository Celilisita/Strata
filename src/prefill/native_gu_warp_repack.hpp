#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
namespace strata::prefill::mmq {
inline bool gu_warp_repack_flag(const char* value) {
    if(!value||std::strcmp(value,"0")==0)return false;
    if(std::strcmp(value,"1")==0)return true;
    throw std::invalid_argument("PP_GU_WARP_REPACK requires exactly0 or1");
}
struct GuWarpRepackResources {
    int cc=0,binary=0,threads=128,registers=0,active_blocks=0,max_threads=0;
    size_t shared=0,local=0;bool eligible=false;
};
GuWarpRepackResources native_gu_warp_repack_resources();
bool native_gu_warp_repack_supported();
inline bool native_gu_warp_repack_shape(int type,int cols,int outputs,int parts,
    int entries,int source_rows,bool active) {
    return type==12&&cols==2048&&outputs==512&&parts==2&&active&&
        source_rows>=1024&&source_rows<=2048&&entries==source_rows*8;
}
// Source codes/d are canonical native Q8; inverse[original entry] is compact P.
// No F32 activation, new quantization, weight inventory or mathematical tile.
void native_gu_warp_repack(const void* input,void* output,int source_rows,
    const int32_t* inverse,void* stream);
uint64_t native_gu_warp_repack_submissions(); // Host submissions, never graph replay/token counts.
} // namespace strata::prefill::mmq
