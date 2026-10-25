#pragma once
#include "fp4_compensated.hpp"
#include "qkvz_policy.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <string>
#include <vector>
namespace strata::qwen35::native {
static_assert(int(BlockLayout::Gguf)==0&&int(BlockLayout::Q4Tiles)==4&&int(BlockLayout::Nvfp4Planes)==5&&int(BlockLayout::Nvfp4WarpFragments)==6,
    "retained GPU layout values must remain stable");
static_assert(sizeof(Nvfp4View)==32&&sizeof(Nvfp4Input)==24&&sizeof(CompensatedFp4Input)==48,"retained CUDA operand ABI");
class CompensatedFfnLoadScope;
inline CompensatedFfnLoadScope *&compensated_ffn_scope(){static thread_local CompensatedFfnLoadScope *scope=nullptr;return scope;}
inline bool compensated_ffn_shape(Nvfp4View w){
    return (w.rows==17408&&w.columns==5120)||(w.rows==5120&&w.columns==17408);
}
inline bool compensated_ffn_tensor(const std::string &name){
    for(const char *suffix:{".ffn_gate.weight",".ffn_up.weight",".ffn_down.weight"}){
        const std::string end(suffix);if(name.size()>=end.size()&&name.compare(name.size()-end.size(),end.size(),end)==0)return true;
    }
    return false;
}
inline bool compensated_projection_shape(Nvfp4View w){
    return (w.rows==5120&&w.columns==6144)||(w.rows==12288&&w.columns==5120);
}
inline bool compensated_projection_tensor(const std::string &name){
    for(const char *suffix:{".ssm_out.weight",".attn_q.weight"}){
        const std::string end(suffix);if(name.size()>=end.size()&&name.compare(name.size()-end.size(),end.size(),end)==0)return true;
    }
    return false;
}
inline bool compensated_projection_requested(){
    const char *flag=std::getenv("STRATA_NVFP4_COMPENSATED_PROJECTIONS");
    if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("compensated projection mode");
    return !flag||flag[0]!='0';
}
// Cold loader descriptors hand off to Runtime's single authoritative inventory.
// They disappear when this scope ends; request execution never consults them.
class CompensatedFfnLoadScope {
    CompensatedFfnLoadScope *prior_;bool enabled_;std::vector<const void*> published_;
public:
    explicit CompensatedFfnLoadScope(int proposals):prior_(compensated_ffn_scope()),enabled_(false){
        int device=0;check_cuda(cudaGetDevice(&device),"compensated FFN device");const auto caps=capabilities(device);
        const char *flag=std::getenv("STRATA_NVFP4_COMPENSATED_FFN");
        if(flag&&std::string(flag)!="0"&&std::string(flag)!="1")throw std::invalid_argument("compensated FFN mode");
        enabled_=proposals==2&&caps.major==12&&caps.minor==0&&(!flag||flag[0]!='0');if(!prior_)configure_qkvz_loading(enabled_);compensated_ffn_scope()=this;
    }
    ~CompensatedFfnLoadScope(){compensated_ffn_scope()=prior_;if(!prior_)qkvz_publications().clear();}
    CompensatedFfnLoadScope(const CompensatedFfnLoadScope&)=delete;
    bool enabled()const{return enabled_;}
    bool contains(const void *p)const{return std::find(published_.begin(),published_.end(),p)!=published_.end();}
    void publish(const void *p){if(!enabled_||!p||contains(p))throw std::logic_error("compensated load publication");published_.push_back(p);}
};
inline bool compensated_ffn_loading(){auto *p=compensated_ffn_scope();return p&&p->enabled();}
inline bool compensated_ffn_published(const void *p){auto *s=compensated_ffn_scope();return s&&s->contains(p);}
inline void compensated_ffn_publish(const void *p){auto *s=compensated_ffn_scope();if(!s)throw std::logic_error("missing compensated load scope");s->publish(p);}
struct CompensatedFfnCounters {size_t matrices=0,scratch_bytes=0,residual_bytes=0;uint64_t tiny_submissions=0,materialize_submissions=0;};
CompensatedFfnCounters nvfp4_compensated_counters(DenseGemm &);
}
