#pragma once
// Fixed constructor capability; active reads only at the original drained boundary.
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
inline bool requested_cpu_gu_q8_reuse() noexcept {
    const char* value=std::getenv("STRATA_GLM53_CPU_GU_Q8_REUSE");
    return value&&std::strcmp(value,"1")==0;
}
// Read only at the original drained cache boundary or initial construction.
inline bool requested_cpu_gu_q8_active() {
    const char* value=std::getenv("STRATA_GLM53_CPU_GU_Q8_ACTIVE");
    if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
        throw std::runtime_error("GU Q8 active flag must be0 or1");
    return !value||std::strcmp(value,"1")==0;
}
inline bool eligible_cpu_gu_q8_reuse(const ggml_tensor* gate,const ggml_tensor* up,int threads) {
    if(threads!=16)return false;
    const ggml_tensor* banks[]{gate,up};
    for(const auto* bank:banks)
        if(!bank||bank->type!=GGML_TYPE_IQ3_S||bank->ne[0]!=4096||bank->ne[1]!=2048||
            bank->ne[2]!=288||bank->ne[3]!=1||!ggml_is_contiguous(bank)||
            !bank->buffer||!ggml_backend_buffer_is_host(bank->buffer)||
            ggml_get_type_traits_cpu(bank->type)->vec_dot_type!=GGML_TYPE_Q8_K)return false;
    return true;
}
} // namespace strata::glm5_next
