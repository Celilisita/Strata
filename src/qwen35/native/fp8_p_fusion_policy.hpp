#pragma once
#include "fp8_p_fusion.hpp"
#include <array>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <string>

namespace strata::qwen35::native {
struct Fp8PFusionStats {
    bool requested=false,admitted=false;unsigned case_mask=15;int prompt=0;
    std::uintptr_t owner=0;
    std::array<std::uint64_t,4> host_calls{};std::uint64_t fallback_calls=0;
    fp8_p_fusion_unit::Resources resource{};
};
inline Fp8PFusionStats &fp8_p_fusion_stats(){static thread_local Fp8PFusionStats value;return value;}
inline bool fp8_p_fusion_requested(){
    static const bool requested=[](){
        const char *raw=std::getenv("STRATA_DENSE_PP_FP8_P_FUSION");const std::string flag=raw?raw:"0";
        if(flag!="0"&&flag!="1")throw std::invalid_argument("PP FP8 P fusion mode must be0or1");
        return flag=="1";
    }();return requested;
}
inline void configure_fp8_p_fusion_request(int prompt){
    auto &state=fp8_p_fusion_stats();state={};state.prompt=prompt;state.requested=fp8_p_fusion_requested();
}
// The parent calls this only after original FP8 and dual-CTA cold admission succeeded.
inline void admit_fp8_p_fusion_request(const void *parent,unsigned mask){
#if CUDART_VERSION >= 12080
    auto &state=fp8_p_fusion_stats();if(!state.requested)return;
    if(!parent||!mask||(mask&~15u))throw std::logic_error("FP8 P fusion parent request identity/mask");
    fp8_p_fusion_unit::configure();state.resource=fp8_p_fusion_unit::resources();const auto &r=state.resource;
    if(r.kernel_count!=3||r.binary_version!=120||r.normal_blocks<2||r.diagnostic_blocks<2||r.prepare_blocks<1||r.local_bytes)
        throw std::runtime_error("FP8 P fusion requires qualifiedSM120 zero-local3entry normal2/diagnostic2/prepare1");
    state.case_mask=mask;state.owner=reinterpret_cast<std::uintptr_t>(parent);state.admitted=true;
#else
    (void)parent;(void)mask;
#endif
}
inline bool try_fp8_p_fusion_attention(const void *parent,const float *q,const void *k,const void *v,float *out,
        void *scratch,size_t bytes,PrefillAttentionShape shape,bool skip,cudaStream_t stream,int original_case){
#if CUDART_VERSION >= 12080
    auto &state=fp8_p_fusion_stats();
    if(!state.admitted)return false;
    if(state.owner!=reinterpret_cast<std::uintptr_t>(parent))
        throw std::logic_error("FP8 P fusion stale/cross-owner parent request");
    const bool first_three=original_case>=0&&original_case<3&&shape.tokens==8192&&shape.query_start==original_case*8192;
    const bool tail=original_case==3&&shape.tokens==5424&&shape.query_start==24576;
    if((first_three||tail)&&(state.case_mask&(1u<<original_case))&&shape.q8&&!shape.graph_start&&
       shape.dim==256&&shape.query_heads==24&&shape.kv_heads==4&&shape.valid==shape.query_start+shape.tokens&&
       shape.valid<=shape.capacity&&shape.capacity<=40960&&skip==(original_case==1||original_case==2)&&
       bytes>=356515840&&bytes>=fp8_p_fusion_unit::workspace_bytes(shape)){
        fp8_p_fusion_unit::attention(q,k,v,out,scratch,bytes,shape,skip,stream);
        ++state.host_calls[original_case];return true;
    }
    ++state.fallback_calls;
#else
    (void)parent;(void)q;(void)k;(void)v;(void)out;(void)scratch;(void)bytes;(void)shape;(void)skip;(void)stream;(void)original_case;
#endif
    return false;
}
inline std::uint64_t fp8_p_fusion_host_submissions(){
    std::uint64_t total=0;for(auto count:fp8_p_fusion_stats().host_calls)total+=count;return total;
}
inline std::ostream &operator<<(std::ostream &out,const Fp8PFusionStats &state){
    std::uint64_t total=0;for(auto count:state.host_calls)total+=count;
    return out<<" pp_fp8_p_fusion_requested="<<state.requested
        <<" pp_fp8_p_fusion_admitted="<<state.admitted
        <<" pp_fp8_p_fusion_mask="<<state.case_mask
        <<" pp_fp8_p_fusion_owner="<<state.owner
        <<" pp_fp8_p_fusion_host_calls="<<total
        <<" pp_fp8_p_fusion_case0_host_calls="<<state.host_calls[0]
        <<" pp_fp8_p_fusion_case1_host_calls="<<state.host_calls[1]
        <<" pp_fp8_p_fusion_case2_host_calls="<<state.host_calls[2]
        <<" pp_fp8_p_fusion_case3_host_calls="<<state.host_calls[3]
        <<" pp_fp8_p_fusion_fallback_host_calls="<<state.fallback_calls
        <<" pp_fp8_p_fusion_normal_blocks_per_sm="<<state.resource.normal_blocks
        <<" pp_fp8_p_fusion_diagnostic_blocks_per_sm="<<state.resource.diagnostic_blocks
        <<" pp_fp8_p_fusion_prepare_blocks_per_sm="<<state.resource.prepare_blocks
        <<" pp_fp8_p_fusion_normal_registers="<<state.resource.normal_registers
        <<" pp_fp8_p_fusion_diagnostic_registers="<<state.resource.diagnostic_registers
        <<" pp_fp8_p_fusion_prepare_registers="<<state.resource.prepare_registers
        <<" pp_fp8_p_fusion_local_bytes="<<state.resource.local_bytes;
}
}
