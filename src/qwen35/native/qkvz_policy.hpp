#pragma once
#include <strata/qwen35/native/nvfp4_runtime.hpp>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <unordered_map>
namespace strata::qwen35::native {
struct QkvzStats {
    bool cold_requested=false,admitted=false;
    std::uint64_t packed_qkv=0,packed_z=0,t3_groups=0,codecs=0,projections=0,fallback_unpacks=0;
};
inline std::unordered_map<const void*,unsigned> &qkvz_publications(){static thread_local std::unordered_map<const void*,unsigned> v;return v;}
inline QkvzStats &qkvz_stats(){static thread_local QkvzStats s;return s;}
// One outer model-load scope only; no live request flag can reinterpret a weight inventory.
inline void configure_qkvz_loading(bool eligible){
    const char *p=std::getenv("STRATA_NVFP4_QKVZ_PACKED");
    if(p&&std::string(p)!="auto"&&std::string(p)!="0"&&std::string(p)!="1")throw std::invalid_argument("QKVZ cold packed mode");
    qkvz_publications().clear();auto &s=qkvz_stats();s={};s.cold_requested=!p||p[0]!='0';s.admitted=eligible&&s.cold_requested;
}
inline bool qkvz_packed_shape(Nvfp4View w){return w.columns==5120&&(w.rows==10240||w.rows==6144);}
inline bool qkvz_tensor(Nvfp4View w,const std::string &name){
    const std::string suffix=w.rows==10240?".attn_qkv.weight":".attn_gate.weight";
    return qkvz_packed_shape(w)&&name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;
}
inline bool qkvz_loading(Nvfp4View w,const std::string &name){return qkvz_stats().admitted&&qkvz_tensor(w,name);}
inline void qkvz_publish(Nvfp4View w){
    if(!qkvz_packed_shape(w)||!w.values||!qkvz_publications().emplace(w.values,w.rows==10240?1u:2u).second)
        throw std::logic_error("QKVZ exact role publication");
    if(w.rows==10240)++qkvz_stats().packed_qkv;else ++qkvz_stats().packed_z;
}
inline unsigned qkvz_published_role(Nvfp4View w){auto i=qkvz_publications().find(w.values);return i==qkvz_publications().end()?0:i->second;}
inline bool qkvz_tiny_fallback(Nvfp4View w,int tokens){return qkvz_packed_shape(w)&&tokens<=4&&tokens!=3;}
inline void qkvz_reset_request(){auto &s=qkvz_stats();s.t3_groups=s.codecs=s.projections=s.fallback_unpacks=0;}
inline void qkvz_record_codec(Nvfp4View w){if(qkvz_packed_shape(w))++qkvz_stats().codecs;}
inline void qkvz_record_projection(Nvfp4View w){if(qkvz_packed_shape(w))++qkvz_stats().projections;}
inline void qkvz_record_unpack(Nvfp4View w){if(qkvz_packed_shape(w))++qkvz_stats().fallback_unpacks;}
// Resolve the existing padded16 LT fallback before capture; one cold unpack per geometry.
template<class Canonical> inline void qkvz_warm_cold(Nvfp4View w,std::unordered_set<unsigned long long>& warmed,
        Nvfp4Gemm &gemm,Nvfp4Input input,float *out,cudaStream_t stream,Canonical canonical){
    if(!qkvz_packed_shape(w))return;
    const auto key=(static_cast<unsigned long long>(w.rows)<<32)|unsigned(w.columns);
    if(warmed.insert(key).second){const auto c=canonical(w);gemm.prepare(c,input,1);gemm.project(c,input,out,1,stream);}
}
inline void qkvz_record_group(const std::array<const Nvfp4View*,4>&v,std::initializer_list<Nvfp4Projection> p,int tokens){
    if(tokens!=3||p.size()!=2||v[0]->input_scale!=v[1]->input_scale)return;
    int mask=0;for(const auto &x:p){if(x.weight.layout!=BlockLayout::Nvfp4WarpFragments||x.weight.columns!=5120)return;
        mask|=x.weight.rows==10240?1:(x.weight.rows==6144?2:0);}
    if(mask==3)++qkvz_stats().t3_groups;
}
}
