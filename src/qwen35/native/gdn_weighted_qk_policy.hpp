#include "gdn_long_chunk_policy.hpp"
#pragma once
#include "gdn_weighted_qk.hpp"
#include "kreg_policy.hpp"
#include "gdn_float_value_service.hpp"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ostream>
#include <stdexcept>
namespace strata::qwen35::native {
struct GdnWeightedQkStats {
    bool requested=false,admitted=false;int last_tokens=0;
    std::uint64_t host_calls=0,fallback_host_calls=0;
    void*arena=nullptr;void*gated_owner=nullptr;
    size_t arena_lease_bytes=0,gated_lease_bytes=0,weighted_bytes=0,workspace_bytes=0;
    gdn_weighted_qk_unit::Resources resource{};
};
inline GdnWeightedQkStats&gdn_weighted_qk_stats(){static thread_local GdnWeightedQkStats value;return value;}
inline bool gdn_weighted_qk_requested(){static const bool enabled=[](){
    const char*p=std::getenv("STRATA_DENSE_GDN_WEIGHTED_QK");
    if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("weightedQK flag must be0or1");
    return p&&p[0]=='1';}();return enabled;
}
// PREPARE only. The existing current-profile KREG admission remains required.
inline void configure_gdn_weighted_qk_request(bool old_kreg,int prompt,bool text,int major,int minor,void*arena,size_t bytes){
    auto&v=gdn_weighted_qk_stats();v={};v.requested=gdn_weighted_qk_requested();
#if CUDART_VERSION >= 12080
    if(!v.requested||!old_kreg||!gdn_weighted_request(prompt)||!text||major!=12||minor!=0)return;
    const GdnShape s{128,128,16,48,8192};
    const size_t needed=gdn_weighted_qk_unit::workspace_bytes(s);
    if(!arena||reinterpret_cast<std::uintptr_t>(arena)%32||bytes<needed||bytes>KregLeaseBytes)return;
    gdn_weighted_qk_unit::configure();v.resource=gdn_weighted_qk_unit::resources();
    const auto&a=v.resource.prepare;const auto&b=v.resource.scan;
    if(v.resource.kernel_count!=2||a.blocks<2||b.blocks<2||a.local_bytes||b.local_bytes||a.binary_version!=120||b.binary_version!=120)
        throw std::runtime_error("weightedQK requires qualified SM120 twoCTA zero-local resources");
    v.arena=arena;v.arena_lease_bytes=bytes;v.workspace_bytes=needed;
    v.weighted_bytes=gdn_weighted_qk_unit::weighted_bytes(s);v.admitted=true;
#else
    (void)old_kreg;(void)prompt;(void)text;(void)major;(void)minor;(void)arena;(void)bytes;
#endif
}
inline bool try_gdn_weighted_qk(GdnInputs x,GdnShape s,float*out,float*next,void*arena,size_t bytes,
        void*dead_gated,size_t gated_bytes,cudaStream_t stream,bool closed){
    auto&v=gdn_weighted_qk_stats();if(!v.admitted)return false;
    if(!closed||!use_gdn_kreg(s)||s.key_dim!=128||s.value_dim!=128||s.query_heads!=16||s.value_heads!=48||s.tokens!=8192||
       arena!=v.arena||bytes!=v.arena_lease_bytes||!dead_gated||reinterpret_cast<std::uintptr_t>(dead_gated)%32||gated_bytes!=v.weighted_bytes){
        ++v.fallback_host_calls;return false;
    }
#if CUDART_VERSION >= 12080
    gdn_weighted_qk_unit::run(x,s,out,next,arena,bytes,dead_gated,gated_bytes,stream);
    // This provider retains the same grouped-KREG/half-delta mathematics.
    ++gdn_register_submissions();++gdn_delta_submissions();++gdn_qk_group_submissions();++gdn_kreg_submissions();
    ++v.host_calls;v.last_tokens=s.tokens;v.gated_owner=dead_gated;v.gated_lease_bytes=gated_bytes;return true;
#else
    (void)x;(void)out;(void)next;(void)stream;return false;
#endif
}
inline std::ostream&operator<<(std::ostream&out,const GdnWeightedQkStats&v){
    return out<<" pp_gdn_weighted_qk_requested="<<int(v.requested)
        <<" pp_gdn_weighted_qk_admitted="<<int(v.admitted)
        <<" pp_gdn_weighted_qk_host_calls="<<v.host_calls
        <<" pp_gdn_weighted_qk_fallback_host_calls="<<v.fallback_host_calls
        <<" pp_gdn_weighted_qk_last_tokens="<<v.last_tokens
        <<" pp_gdn_weighted_qk_arena_lease_bytes="<<v.arena_lease_bytes
        <<" pp_gdn_weighted_qk_gated_lease_bytes="<<v.gated_lease_bytes
        <<" pp_gdn_weighted_qk_gated_owner="<<reinterpret_cast<std::uintptr_t>(v.gated_owner)
        <<" pp_gdn_weighted_qk_workspace_bytes="<<v.workspace_bytes
        <<" pp_gdn_weighted_qk_weighted_bytes="<<v.weighted_bytes
        <<" pp_gdn_weighted_qk_prepare_registers="<<v.resource.prepare.registers
        <<" pp_gdn_weighted_qk_prepare_blocks="<<v.resource.prepare.blocks
        <<" pp_gdn_weighted_qk_prepare_local_bytes="<<v.resource.prepare.local_bytes
        <<" pp_gdn_weighted_qk_scan_registers="<<v.resource.scan.registers
        <<" pp_gdn_weighted_qk_scan_blocks="<<v.resource.scan.blocks
        <<" pp_gdn_weighted_qk_scan_local_bytes="<<v.resource.scan.local_bytes;
}
}
