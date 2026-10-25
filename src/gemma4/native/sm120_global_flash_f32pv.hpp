#pragma once
#include <cuda_fp16.h>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::gemma4::native::f32pv_detail {
inline bool requested(){
    static const bool enabled=[](){const char* v=std::getenv("STRATA_GEMMA_PP_GLOBAL_FLASH_F32PV");
        if(!v||std::strcmp(v,"0")==0)return false;
        if(std::strcmp(v,"1")==0)return true;
        throw std::invalid_argument("STRATA_GEMMA_PP_GLOBAL_FLASH_F32PV requires0or1");}();return enabled;
}
constexpr int QueryTile=4,Warps=8,Threads=256,Dim=512,Gqa=8,KeyTile=32,Stride=260;
constexpr std::size_t Shared=2*32*Stride*sizeof(half2)+Warps*Gqa*16*sizeof(half);
static_assert(Shared==68608,"F32PV four-query shared allocation");
}
