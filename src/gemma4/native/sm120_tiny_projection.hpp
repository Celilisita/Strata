#pragma once
#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include "strata/qwen35/native/nvfp4.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::gemma4::native {
// Cold process policy: a model never changes representation under live graphs.
inline bool gemma_tiny_compensated_requested(){
    static const bool enabled=[](){
        const char* value=std::getenv("STRATA_GEMMA4_TINY_COMPENSATED");
        if(!value||std::string(value)=="0")return false;
        if(std::string(value)!="1")throw std::invalid_argument("Gemma tiny compensated flag must be0/1");
        return true;
    }();
    return enabled;
}
inline std::size_t gemma_tiny_canonical_scratch(const Nvfp4BodyPlan& plan){
    std::size_t maximum=0;
    for(const auto& selected:plan.tensors){
        const auto& t=*selected.tensor;
        maximum=std::max(maximum,qwen35::native::nvfp4_value_bytes(int(t.shape[1]),int(t.shape[0])));
    }
    return maximum;
}
inline std::size_t gemma_tiny_low_bytes(const Nvfp4BodyPlan& plan){
    return qwen35::native::nvfp4_value_bytes(16,plan.maximum_columns)+
        qwen35::native::nvfp4_scale_bytes(16,plan.maximum_columns);
}
}
