#pragma once
#include "gdn_output_elision_scope.hpp"
#include <cstdlib>
#include <cstring>
#include <optional>
#include <ostream>
namespace strata::qwen35::native {
struct GdnOutputElisionStats {
    bool requested=false,admitted=false;
    std::uint64_t host_calls=0,full8192_host_calls=0,tail5424_host_calls=0,fallback_host_calls=0;
    std::uint64_t elided_write_bytes=0,elided_read_bytes=0;
    int last_tokens=0;size_t gated_lease_bytes=0;
    const void*last_gated_owner=nullptr;const void*last_encoded_owner=nullptr;
    GdnOutputElisionResources resource{};
};
inline GdnOutputElisionStats&gdn_output_elision_stats(){static thread_local GdnOutputElisionStats v;return v;}
inline bool gdn_output_elision_requested(){static const bool enabled=[](){
    const char*p=std::getenv("STRATA_DENSE_GDN_OUTPUT_ELISION");
    if(p&&std::strcmp(p,"0")&&std::strcmp(p,"1"))throw std::invalid_argument("GDN output elision flag must be 0 or 1");
    return p&&p[0]=='1';}();return enabled;
}
// PREPARE only: resource/device queries are included in the request's profile cost.
inline void configure_gdn_output_elision_request(bool nvfp4,bool text){
    auto&v=gdn_output_elision_stats();v={};v.requested=gdn_output_elision_requested();
#if CUDART_VERSION >= 12080
    if(!v.requested||!nvfp4||!text)return;
    int device=0;cudaDeviceProp p{};
    check_cuda(cudaGetDevice(&device),"GDN output elision cold device");
    check_cuda(cudaGetDeviceProperties(&p,device),"GDN output elision cold device properties");
    if(p.major!=12||p.minor!=0)return;
    v.resource=gdn_output_elision_resources();
    if(v.resource.binary_version!=120||v.resource.blocks<1||v.resource.local_bytes)
        throw std::runtime_error("GDN output elision requires qualified SM120 zero-local resources");
    v.admitted=true;
#else
    (void)nvfp4;(void)text;
#endif
}
// The inner unit-qualified scope and its layout stay byte-identical. Statistics
// live outside it so the exact unit Runtime object can be retained unchanged.
class GdnOutputElisionRequestScope {
    std::optional<GdnOutputElisionScope> lease_;
    float*gated_;int tokens_;size_t gated_bytes_;bool recorded_=false;
public:
    GdnOutputElisionRequestScope(DenseGemm*service,Matrix w,const float*a,const float*gamma,const float*z,
            float*gated,int tokens,float epsilon,cudaStream_t stream,bool closed,size_t gated_bytes)
        :gated_(gated),tokens_(tokens),gated_bytes_(gated_bytes){
        auto&v=gdn_output_elision_stats();
        if(!v.admitted||(tokens!=8192&&tokens!=5424))return;
#if CUDART_VERSION >= 12080
        if(!closed||!service||gated_bytes!=size_t(8192)*6144*4){++v.fallback_host_calls;return;}
        lease_.emplace(true,*service,w,a,gamma,z,gated,tokens,epsilon,stream);
        if(!lease_->deferred())++v.fallback_host_calls;
#else
        (void)service;(void)w;(void)a;(void)gamma;(void)z;(void)epsilon;(void)stream;(void)closed;
#endif
    }
    bool deferred()const{return lease_&&lease_->deferred();}
    void verify(){
        if(!lease_)return;
        lease_->verify();if(!lease_->consumed())return;
        if(recorded_)throw std::logic_error("GDN output elision statistics consumed twice");
        auto&v=gdn_output_elision_stats();++v.host_calls;
        if(tokens_==8192)++v.full8192_host_calls;else ++v.tail5424_host_calls;
        v.last_tokens=tokens_;v.gated_lease_bytes=gated_bytes_;v.last_gated_owner=gated_;
        v.last_encoded_owner=lease_->encoded().values;
        v.elided_write_bytes+=size_t(tokens_)*6144*4;v.elided_read_bytes+=size_t(tokens_)*6144*4;recorded_=true;
    }
};
inline std::ostream&operator<<(std::ostream&out,const GdnOutputElisionStats&v){
    return out<<" pp_gdn_output_elision_requested="<<int(v.requested)
        <<" pp_gdn_output_elision_admitted="<<int(v.admitted)
        <<" pp_gdn_output_elision_host_calls="<<v.host_calls
        <<" pp_gdn_output_elision_full8192_host_calls="<<v.full8192_host_calls
        <<" pp_gdn_output_elision_tail5424_host_calls="<<v.tail5424_host_calls
        <<" pp_gdn_output_elision_fallback_host_calls="<<v.fallback_host_calls
        <<" pp_gdn_output_elision_last_tokens="<<v.last_tokens
        <<" pp_gdn_output_elision_gated_lease_bytes="<<v.gated_lease_bytes
        <<" pp_gdn_output_elision_gated_owner="<<reinterpret_cast<std::uintptr_t>(v.last_gated_owner)
        <<" pp_gdn_output_elision_encoded_owner="<<reinterpret_cast<std::uintptr_t>(v.last_encoded_owner)
        <<" pp_gdn_output_elision_elided_write_bytes="<<v.elided_write_bytes
        <<" pp_gdn_output_elision_elided_read_bytes="<<v.elided_read_bytes
        <<" pp_gdn_output_elision_registers="<<v.resource.registers
        <<" pp_gdn_output_elision_blocks_per_sm="<<v.resource.blocks
        <<" pp_gdn_output_elision_local_bytes="<<v.resource.local_bytes
        <<" pp_gdn_output_elision_binary_version="<<v.resource.binary_version;
}
}
