#pragma once
#include "strata/qwen35/native/recurrent.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
inline bool q35_four_projection_requested(){
    static const bool enabled=[](){
        const char* raw=std::getenv("STRATA_DENSE_Q35_FOUR_PROJECTIONS");
        const std::string value=raw?raw:"0";
        if(value!="0"&&value!="1")throw std::invalid_argument("Q35 four projections flag must be0/1");
        return value=="1";
    }();
    return enabled;
}
inline bool& q35_four_projection_profile(){static thread_local bool q35=false;return q35;}
inline void configure_q35_four_projection_profile(bool q35){
    if(q35_four_projection_requested()&&!q35)
        throw std::invalid_argument("Four NVFP4 projections require the explicit Qwen3.5 derived profile");
    q35_four_projection_profile()=q35;
}
inline bool q35_four_projection_eligible(const RecurrentWeights& w,bool share,bool legacy,int hidden,int tokens){
    // T1..4 retain their existing mixed direct/compensated/fallback codecs.
    if(tokens<5||!q35_four_projection_profile()||!q35_four_projection_requested()||!share||legacy||hidden!=5120)return false;
    const auto fp4=[](Matrix m,int rows,bool planes_only){return m.data&&m.quant==Quant::NVFP4&&m.columns==5120&&m.rows==rows&&
        (m.layout==BlockLayout::Nvfp4Planes||(!planes_only&&m.layout==BlockLayout::Nvfp4WarpFragments));};
    return fp4(w.qkv,10240,false)&&fp4(w.z,6144,false)&&fp4(w.alpha,48,true)&&fp4(w.beta,48,true);
}
}
