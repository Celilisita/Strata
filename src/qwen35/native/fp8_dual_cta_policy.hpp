#pragma once
#include "fp8_dual_cta.hpp"
#include "fp8_p_fusion_policy.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
struct Fp8DualCtaStats {bool requested=false,admitted=false;unsigned case_mask=15;int prompt=0;
    std::array<std::uint64_t,4> host_calls{};std::uint64_t fallback_calls=0;fp8_dual_cta_unit::Resources resource{};};
inline Fp8DualCtaStats& fp8_dual_cta_stats(){static thread_local Fp8DualCtaStats value;return value;}
inline void configure_fp8_dual_cta_request(int prompt){
    configure_fp8_p_fusion_request(prompt);
    auto&v=fp8_dual_cta_stats();v={};v.prompt=prompt;const char*raw=std::getenv("STRATA_DENSE_PP_FP8_DUAL_CTA");
    const std::string flag=raw?raw:"0";
    if(flag!="0"&&flag!="1")throw std::invalid_argument("PP FP8 dualCTA mode must be0or1");v.requested=flag=="1";
}
// Called only after the original FP8 request has passed its complete cold admission.
inline void admit_fp8_dual_cta_request(){
#if CUDART_VERSION >= 12080
    auto&v=fp8_dual_cta_stats();if(!v.requested)return;
    fp8_dual_cta_unit::configure();v.resource=fp8_dual_cta_unit::resources();const auto&r=v.resource;
    if(r.kernel_count!=3||r.binary_version!=120||r.normal_blocks<2||r.diagnostic_blocks<2||r.prepare_blocks<1||r.local_bytes)
        throw std::runtime_error("FP8 dualCTA requires qualifiedSM120 zero-local3entry normal2/diagnostic2/prepare1");
    v.admitted=true;admit_fp8_p_fusion_request(&v,v.case_mask);
#endif
}
inline bool try_fp8_dual_cta_attention(const float*q,const void*k,const void*v,float*out,void*scratch,size_t bytes,
        PrefillAttentionShape s,bool skip,cudaStream_t stream,int original_case){
#if CUDART_VERSION >= 12080
    auto&state=fp8_dual_cta_stats();
    if(state.admitted&&original_case>=0&&original_case<4&&(state.case_mask&(1u<<original_case))&&bytes>=fp8_dual_cta_unit::workspace_bytes(s)){
        if(!try_fp8_p_fusion_attention(&state,q,k,v,out,scratch,bytes,s,skip,stream,original_case))
            fp8_dual_cta_unit::attention(q,k,v,out,scratch,bytes,s,skip,stream);
        ++state.host_calls[original_case];return true;}
    if(state.requested)++state.fallback_calls;
#else
    (void)q;(void)k;(void)v;(void)out;(void)scratch;(void)bytes;(void)s;(void)skip;(void)stream;(void)original_case;
#endif
    return false;
}
inline std::uint64_t fp8_dual_cta_host_submissions(){std::uint64_t n=0;for(auto v:fp8_dual_cta_stats().host_calls)n+=v;return n;}
inline std::ostream& operator<<(std::ostream&out,const Fp8DualCtaStats&value){
    std::uint64_t total=0;for(auto count:value.host_calls)total+=count;
    return out
        << " pp_fp8_dual_requested=" << value.requested
        << " pp_fp8_dual_admitted=" << value.admitted
        << " pp_fp8_dual_mask=" << value.case_mask
        << " pp_fp8_dual_host_calls=" << total
        << " pp_fp8_dual_case0_host_calls=" << value.host_calls[0]
        << " pp_fp8_dual_case1_host_calls=" << value.host_calls[1]
        << " pp_fp8_dual_case2_host_calls=" << value.host_calls[2]
        << " pp_fp8_dual_case3_host_calls=" << value.host_calls[3]
        << " pp_fp8_dual_fallback_host_calls=" << value.fallback_calls
        << " pp_fp8_dual_normal_blocks_per_sm=" << value.resource.normal_blocks
        << " pp_fp8_dual_diagnostic_blocks_per_sm=" << value.resource.diagnostic_blocks
        << " pp_fp8_dual_prepare_blocks_per_sm=" << value.resource.prepare_blocks
        << " pp_fp8_dual_normal_registers=" << value.resource.normal_registers
        << " pp_fp8_dual_diagnostic_registers=" << value.resource.diagnostic_registers
        << " pp_fp8_dual_prepare_registers=" << value.resource.prepare_registers
        << " pp_fp8_dual_local_bytes=" << value.resource.local_bytes;
}
}
