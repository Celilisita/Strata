#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35::native {
inline bool nvfp4_span_overlap(const void *a,size_t an,const void *b,size_t bn){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<an:x-y<bn;
}
inline bool nvfp4_float_aligned(const void *p){
    return p&&reinterpret_cast<uintptr_t>(p)%alignof(float)==0;
}
inline void validate_nvfp4_add_output(Nvfp4View w,Nvfp4Input x,const float *residual,float *output,int tokens){
    if(tokens!=8192||w.rows!=5120||w.columns!=17408||!nvfp4_float_aligned(residual)||!nvfp4_float_aligned(output))
        throw std::invalid_argument("NVFP4 residual epilogue shape/alignment");
    const size_t bytes=size_t(tokens)*5120*4;
    if(nvfp4_span_overlap(residual,bytes,output,bytes))
        throw std::invalid_argument("NVFP4 residual epilogue C/D alias");
    const void *operands[]={w.values,w.scales,x.values,x.scales};
    const size_t spans[]={nvfp4_value_bytes(w.rows,w.columns),nvfp4_scale_bytes(w.rows,w.columns),
        nvfp4_value_bytes(tokens,w.columns),nvfp4_scale_bytes(tokens,w.columns)};
    for(int k=0;k<4;++k){
        if(!operands[k]||nvfp4_span_overlap(residual,bytes,operands[k],spans[k])||
           nvfp4_span_overlap(output,bytes,operands[k],spans[k]))
            throw std::invalid_argument("NVFP4 residual epilogue overlaps encoded operand");
    }
}
}
