#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
#include "fp4_compensated.hpp"
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::gemma4::native {
// This private binary opts in to the existing T1/T2/T3 and two long-PP
// extents. No process flag, persistent pointer cache, member or owner is added.
inline bool native_group_codec_tokens(int tokens){return (tokens>=1&&tokens<=3)||tokens==1328||tokens==2048;}
struct NativeGroupOperand {qwen35::native::Nvfp4View weight;float* output=nullptr;};
struct NativeGroupCounts {std::uint64_t projections=0,tiny=0,canonical_materializations=0;};
// Stateless execution cut also usable by a selected-weight checker. The owning
// service admits matrix identity, prepared extents, all spans and stream first.
// Encoded high/low inputs live only across these adjacent ordered projections.
inline NativeGroupCounts execute_native_group(const float* input,std::initializer_list<NativeGroupOperand> group,
        qwen35::native::Nvfp4Input high,qwen35::native::Nvfp4Input low,void* canonical,
        qwen35::native::Nvfp4Gemm& gemm,int tokens,cudaStream_t stream){
    namespace p=qwen35::native;
    if(!input||!stream||!native_group_codec_tokens(tokens)||(group.size()!=2&&group.size()!=3)||!canonical||
       !high.values||!high.scales||!low.values||!low.scales||low.capacity_rows<(tokens<=3?tokens:1)||low.capacity_columns<5376)
        throw std::invalid_argument("Gemma explicit native group extent");
    for(const auto& operand:group){const auto& w=operand.weight;
        if(!w.values||!w.scales||!operand.output||w.columns!=5376||w.rows<1||w.rows%16||w.input_scale!=1||
           !(w.weight_scale>0)||!std::isfinite(w.weight_scale)||high.capacity_rows<tokens||high.capacity_columns<w.columns)
            throw std::invalid_argument("Gemma native group uniform codec operands");}
    const bool tiny=tokens<=3;bool encoded=false;NativeGroupCounts count;
    for(const auto& operand:group){auto view=operand.weight;
        if(!tiny){
            // Preserve every weight unpack and its original ordered scratch
            // lease. This scratch is disjoint from both encoded-input planes.
            p::fp4_unpack_values({view},canonical,stream);view.values=canonical;++count.canonical_materializations;
        }
        if(!encoded){
            if(tiny)p::fp4_compensated_quantize(input,{high,low},tokens,5376,1,stream);
            else p::nvfp4_quantize(input,high,tokens,5376,1,stream);
            encoded=true;
        }
        if(tiny){p::fp4_compensated_linear({view},{high,low},operand.output,tokens,stream);++count.tiny;}
        else gemm.project(view,high,operand.output,tokens,stream);
        ++count.projections;
    }
    return count;
}
}
