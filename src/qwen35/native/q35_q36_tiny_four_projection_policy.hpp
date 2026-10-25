#pragma once
#include "q35_four_projection_policy.hpp"
#include "q36_ab_nvfp4_policy.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::qwen35::native {
inline bool q35_q36_tiny_four_projection_requested(){
    static const bool enabled=[](){
        const char* raw=std::getenv("STRATA_DENSE_TINY_FOUR_PROJECTIONS");
        const std::string value=raw?raw:"0";
        if(value!="0"&&value!="1")
            throw std::invalid_argument("Tiny four projections flag must be0/1");
        return value=="1";
    }();
    return enabled;
}
inline bool q35_q36_tiny_four_projection_eligible(const RecurrentWeights& w,bool share,
                                                 bool legacy,int hidden,int tokens){
    if(!q35_q36_tiny_four_projection_requested()||
       (tokens!=1&&tokens!=2&&tokens!=3&&tokens!=4)||!share||legacy||hidden!=5120)return false;
    const char* direct=std::getenv("STRATA_NVFP4_DIRECT_TINY");
    if(!direct||std::string(direct)!="1")return false;
    // The existing Q36 guard binds its configured AB96/nextN profile and views.
    if(q36_ab_four_input_eligible(w,share,legacy,hidden,false))return true;
    if(!q35_four_projection_profile()||!q35_four_projection_requested())return false;
    const auto fp4=[](Matrix m,int rows,bool planes_only){
        return m.data&&m.quant==Quant::NVFP4&&m.columns==5120&&m.rows==rows&&
            (m.layout==BlockLayout::Nvfp4Planes||
             (!planes_only&&m.layout==BlockLayout::Nvfp4WarpFragments));
    };
    // T1/2/4: packed QKV/Z use tiny fallback and all four views use kind1.
    // T3: packed QKV/Z retain kind2, then AB share kind1 (3 codecs to2).
    // Canonical T3 QKV/Z share kind1 with AB (3 to1); weight scales stay independent.
    return fp4(w.qkv,10240,false)&&fp4(w.z,6144,false)&&
        fp4(w.alpha,48,true)&&fp4(w.beta,48,true);
}
}
