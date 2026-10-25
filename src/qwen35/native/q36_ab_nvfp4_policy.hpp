#pragma once
#include "strata/qwen35/native/recurrent.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
inline bool q36_ab_nvfp4_requested(){
    static const bool enabled=[](){
        const char* raw=std::getenv("STRATA_DENSE_Q36_AB_NVFP4");
        const std::string mode=raw?raw:"0";
        if(mode!="0"&&mode!="1")throw std::invalid_argument("Q36 AB NVFP4 flag must be0or1");
        return mode=="1";
    }();
    return enabled;
}
inline bool& q36_ab_nvfp4_profile(){static thread_local bool enabled=false;return enabled;}
inline bool configure_q36_ab_nvfp4(const std::string& variant,bool nextn){
    const bool enabled=q36_ab_nvfp4_requested();
    if(enabled){
        if(variant!="qwen36_27b"||!nextn)
            throw std::invalid_argument("Q36 AB NVFP4 requires explicit Q36 plus nextN8 NVFP4");
        const char* q8=std::getenv("STRATA_DENSE_Q8_PP_SGEMM");
        if(q8&&std::string(q8)!="0")
            throw std::invalid_argument("Q36 AB NVFP4 forbids Q8 PP SGEMM initialization");
    }
    q36_ab_nvfp4_profile()=enabled;
    return enabled;
}
inline bool q36_ab_four_input_eligible(const RecurrentWeights& w,bool share,bool legacy,int hidden,bool planes_only){
    if(!q36_ab_nvfp4_profile()||!q36_ab_nvfp4_requested()||!share||legacy||hidden!=5120)return false;
    const auto fp4=[&](Matrix m,int rows,bool small){return m.data&&m.quant==Quant::NVFP4&&m.columns==5120&&m.rows==rows&&
        (m.layout==BlockLayout::Nvfp4Planes||(!planes_only&&!small&&m.layout==BlockLayout::Nvfp4WarpFragments));};
    return fp4(w.qkv,10240,false)&&fp4(w.z,6144,false)&&fp4(w.alpha,48,true)&&fp4(w.beta,48,true);
}
inline bool q36_ab_four_projection_eligible(const RecurrentWeights& w,bool share,bool legacy,int hidden,int tokens){
    // Tiny T1..4 keep the independent FP4 projection codecs and ordering.
    return tokens>=5&&q36_ab_four_input_eligible(w,share,legacy,hidden,false);
}
inline bool q36_ab_f32_input_eligible(const RecurrentWeights& w,bool share,bool legacy,int hidden){
    // Exact four-plane inventory: all initial projections consume b.norm F32.
    // This removes only the unused legacy encoded copy, never RMS or FP4 work.
    return q36_ab_four_input_eligible(w,share,legacy,hidden,true);
}
}
