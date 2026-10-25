#pragma once
#include "fp8_pp.hpp"
#include "fp8_dual_cta_policy.hpp"
#include <strata/qwen35/native/ffn.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35::native {
constexpr unsigned Fp8PpQualifiedMask=15;
struct Fp8PpStats {bool requested=false,admitted=false;unsigned case_mask=Fp8PpQualifiedMask;int prompt=0;
    std::array<std::uint64_t,4> host_calls{};std::uint64_t fallback_calls=0;fp8_pp_unit::Resources resource{};};
inline Fp8PpStats& fp8_pp_stats(){static thread_local Fp8PpStats value;return value;}
inline bool& fp8_pp_closed_scope(){static thread_local bool value=false;return value;}
class Fp8PpScope {bool previous_;public:explicit Fp8PpScope(bool active):previous_(fp8_pp_closed_scope()){fp8_pp_closed_scope()=active;}
    ~Fp8PpScope(){fp8_pp_closed_scope()=previous_;}Fp8PpScope(const Fp8PpScope&)=delete;Fp8PpScope&operator=(const Fp8PpScope&)=delete;};
inline void configure_fp8_pp_request(bool resident_compact,int prompt){
    configure_fp8_dual_cta_request(prompt);
    auto&v=fp8_pp_stats();v={};v.prompt=prompt;const char*raw=std::getenv("STRATA_DENSE_PP_FP8");const std::string flag=raw?raw:"0";
    if(flag!="0"&&flag!="1")throw std::invalid_argument("PP FP8 mode must be0or1");v.requested=flag=="1";
#if CUDART_VERSION >= 12080
    if(!v.requested||!resident_compact)return;int device=0;check_cuda(cudaGetDevice(&device),"FP8 PP admission device");const auto caps=capabilities(device);
    if(caps.major!=12||caps.minor!=0)return;fp8_pp_unit::configure();v.resource=fp8_pp_unit::resources();const auto&r=v.resource;
    if(r.kernel_count!=3||r.binary_version!=120||r.blocks<1||r.local_bytes)throw std::runtime_error("FP8 PP requires qualifiedSM120 resident zero-local3entry provider");
    v.admitted=true;admit_fp8_dual_cta_request();
#else
    (void)resident_compact;
#endif
}
inline int fp8_pp_case(PrefillAttentionShape s){
    if(!s.q8||s.graph_start||s.dim!=256||s.query_heads!=24||s.kv_heads!=4||s.valid!=s.query_start+s.tokens||s.valid>s.capacity||s.capacity>40960)return -1;
    if(s.tokens==8192&&(s.query_start==0||s.query_start==8192||s.query_start==16384))return s.query_start/8192;
    return s.tokens==5424&&s.query_start==24576?3:-1;
}
inline bool try_fp8_pp_attention(const float*q,const void*k,const void*v,float*out,void*scratch,size_t bytes,PrefillAttentionShape s,bool skip,cudaStream_t stream){
#if CUDART_VERSION >= 12080
    auto&state=fp8_pp_stats();const int c=fp8_pp_case(s);
    if(state.admitted&&fp8_pp_closed_scope()&&c>=0&&(state.case_mask&(1u<<c))&&bytes>=356515840&&
       bytes>=fp8_pp_unit::workspace_bytes(s)&&skip==(c==1||c==2)){
        if(!try_fp8_dual_cta_attention(q,k,v,out,scratch,bytes,s,skip,stream,c))
            fp8_pp_unit::attention(q,k,v,out,scratch,bytes,s,skip,stream);
        ++state.host_calls[c];return true;}
    if(state.admitted)++state.fallback_calls;
#else
    (void)q;(void)k;(void)v;(void)out;(void)scratch;(void)bytes;(void)s;(void)skip;(void)stream;
#endif
    return false;
}
inline std::uint64_t fp8_pp_host_submissions(){std::uint64_t n=0;for(auto v:fp8_pp_stats().host_calls)n+=v;return n;}
}
