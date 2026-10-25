#pragma once
#include "strata/qwen35/native/recurrent.hpp"
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
inline bool q8_f32_input_requested(){static const bool enabled=[](){
    const char* raw=std::getenv("STRATA_DENSE_Q8_F32_INPUT");const std::string mode=raw?raw:"0";
    if(mode!="0"&&mode!="1")throw std::invalid_argument("Q8 F32 input elision must be0or1");
    return mode=="1";}();return enabled;
}
inline bool q8_f32_input_eligible(const RecurrentWeights& w,bool share,bool legacy,int hidden){
    if(!q8_f32_input_requested()||!share||legacy||hidden!=5120)return false;
    const auto fp4=[](Matrix m,int rows){return m.quant==Quant::NVFP4&&m.columns==5120&&m.rows==rows&&
        (m.layout==BlockLayout::Nvfp4Planes||m.layout==BlockLayout::Nvfp4WarpFragments);};
    const auto q8=[](Matrix m){return m.quant==Quant::Q8&&m.layout==BlockLayout::Gguf&&m.rows==48&&m.columns==5120;};
    return fp4(w.qkv,10240)&&fp4(w.z,6144)&&q8(w.alpha)&&q8(w.beta);
}
}
