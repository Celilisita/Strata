#pragma once
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::prefill {
// Explicit service; callers keep their own projection schedule and weight authority.
// Default0 means no runtime adoption. No device allocation or retained weight copy.
inline unsigned dense_f16_gemm_policy(const char* value) {
    if(!value || !std::strcmp(value,"0"))return 0;
    if(!std::strcmp(value,"12"))return 1u<<12;
    if(!std::strcmp(value,"13"))return 1u<<13;
    if(!std::strcmp(value,"12,13"))return (1u<<12)|(1u<<13);
    throw std::invalid_argument("dense F16 GEMM requires0/12/13/12,13");
}
inline bool dense_f16_gemm_domain(int type,int width,int rows,int outputs) {
    return (type==12 || type==13) && rows>=1024 && rows<=4096 &&
        ((width==2048&&(outputs==4096||outputs==8192)) || (width==4096&&outputs==2048));
}
inline void run_dense_f16_gemm(Gemm& gemm,int type,const void* weight,size_t weight_bytes,
        const float* input,size_t input_bytes,uint16_t* input_half,size_t half_bytes,
        float* output,size_t output_bytes,int width,int rows,int outputs) {
    // The explicit short operator entry tests tails; the runtime domain above is narrower.
    if((type!=12&&type!=13)||(width!=2048&&width!=4096)||rows<1||rows>4096||
       outputs<128||outputs>8192||outputs%128||!gemm.stream()||!gemm.scratch()||
       gemm.scratch_elems()<int64_t(width)*outputs)
        throw std::invalid_argument("dense F16 GEMM shape/prepared scratch");
    const size_t wb=kernels::native_mmvq_weight_bytes(type,width,outputs);
    const size_t xb=size_t(rows)*width*4,hb=size_t(rows)*width*2,yb=size_t(rows)*outputs*4;
    const size_t sb=size_t(width)*outputs*2;
    if(weight_bytes<wb||input_bytes<xb||half_bytes<hb||output_bytes<yb)
        throw std::invalid_argument("dense F16 GEMM extent");
    const std::array<std::pair<uintptr_t,size_t>,5> spans{{{uintptr_t(weight),wb},
        {uintptr_t(input),xb},{uintptr_t(input_half),hb},{uintptr_t(output),yb},{uintptr_t(gemm.scratch()),sb}}};
    for(size_t i=0;i<spans.size();++i) {
        const auto [first,bytes]=spans[i];
        if(!first||first%4||first>(std::numeric_limits<uintptr_t>::max)()-bytes)
            throw std::invalid_argument("dense F16 GEMM pointer/alignment/overflow");
        for(size_t j=0;j<i;++j)if(first<spans[j].first+spans[j].second&&spans[j].first<first+bytes)
            throw std::invalid_argument("dense F16 GEMM borrowed phase aliases");
    }
    // One raw-input cast, one whole-weight stage, one whole-token GEMM.
    // Gemm owns the existing32MiB cuBLAS workspace and borrows this same stream.
    kernels::f32_to_f16_bulk(input,input_half,int64_t(rows)*width,gemm.stream());
    kernels::dequant_f16(type,weight,0,outputs,width,gemm.scratch(),gemm.stream());
    gemm.f16(input_half,gemm.scratch(),output,rows,outputs,width);
}
} // namespace strata::prefill
