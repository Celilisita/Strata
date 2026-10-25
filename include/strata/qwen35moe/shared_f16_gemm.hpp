#pragma once
#include "strata/prefill/gemm.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include <array>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace strata::qwen35moe {
inline bool shared_f16_gemm_policy(const char* value) {
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::invalid_argument("shared F16 GEMM requires0 or1");
}
struct SharedF16Weight {const void* data;size_t bytes;int type;};
inline void run_shared_f16_gemm(prefill::Gemm& gemm,const SharedF16Weight& gate,
        const SharedF16Weight& up,const SharedF16Weight& down,const float* input,
        uint16_t* high,size_t high_bytes,float* gate_out,float* up_out,float* hidden,
        float* output,const float* logits,int rows) {
    if(rows<1024||rows>4096||gate.type!=12||up.type!=12||(down.type!=12&&down.type!=13)||
       !gemm.stream()||!gemm.scratch()||gemm.scratch_elems()<2048*512||high_bytes<size_t(rows)*2048*2)
        throw std::invalid_argument("shared F16 GEMM prepared shape/domain");
    const size_t gu=kernels::native_mmvq_weight_bytes(12,2048,512);
    const size_t d=kernels::native_mmvq_weight_bytes(down.type,512,2048);
    if(gate.bytes<gu||up.bytes<gu||down.bytes<d)throw std::invalid_argument("shared F16 weight extent");
    const std::array<std::pair<uintptr_t,size_t>,10> spans{{
        {uintptr_t(gate.data),gu},{uintptr_t(up.data),gu},{uintptr_t(down.data),d},
        {uintptr_t(input),size_t(rows)*2048*4},{uintptr_t(high),size_t(rows)*2048*2},
        {uintptr_t(gate_out),size_t(rows)*512*4},{uintptr_t(up_out),size_t(rows)*512*4},
        {uintptr_t(hidden),size_t(rows)*512*4},{uintptr_t(output),size_t(rows)*2048*4},
        {uintptr_t(logits),size_t(rows)*257*4}}};
    const uintptr_t scratch=uintptr_t(gemm.scratch());constexpr size_t scratch_bytes=2048*512*2;
    if(!scratch||scratch%4||scratch>(std::numeric_limits<uintptr_t>::max)()-scratch_bytes)
        throw std::invalid_argument("shared F16 GEMM scratch view");
    for(size_t i=0;i<spans.size();++i) {
        const auto [first,bytes]=spans[i];
        if(!first||first%4||first>(std::numeric_limits<uintptr_t>::max)()-bytes||
           (first<scratch+scratch_bytes&&scratch<first+bytes))
            throw std::invalid_argument("shared F16 GEMM source/phase span");
        for(size_t j=0;j<i;++j)if(first<spans[j].first+spans[j].second&&spans[j].first<first+bytes)
            throw std::invalid_argument("shared F16 GEMM aliased phase");
    }
    const auto stream=gemm.stream();
    kernels::f32_to_f16_bulk(input,high,int64_t(rows)*2048,stream);
    kernels::dequant_f16(12,gate.data,0,512,2048,gemm.scratch(),stream);
    gemm.f16(high,gemm.scratch(),gate_out,rows,512,2048);
    kernels::dequant_f16(12,up.data,0,512,2048,gemm.scratch(),stream);
    gemm.f16(high,gemm.scratch(),up_out,rows,512,2048);
    swiglu(gate_out,up_out,hidden,rows*512,stream);
    kernels::f32_to_f16_bulk(hidden,high,int64_t(rows)*512,stream);
    kernels::dequant_f16(down.type,down.data,0,2048,512,gemm.scratch(),stream);
    gemm.f16(high,gemm.scratch(),output,rows,2048,512);
    gate_shared_batch(output,logits,rows,stream);
}
} // namespace strata::qwen35moe
