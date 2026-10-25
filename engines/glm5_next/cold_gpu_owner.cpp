#include "cold_gpu_owner.hpp"
#include "partial_down_binding.hpp"
#include "cold_gpu_protocol.hpp"
#include "cold_gpu_host_mask.hpp"
#include "cold_gpu_receipt.hpp"
#include "../deepseek_v4/expert_staging.hpp"
#include "../deepseek_v4/device_cache_protocol.hpp"
#include "llama-model.h"
#include "llama-moe-cache.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
namespace {
constexpr size_t layers=46;
constexpr std::array<ggml_type,3> types{GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS};
std::array<ggml_tensor*,3> sources(const llama_model& model,int layer) {
    const auto& l=model.layers.at(size_t(layer));
    return {l.ffn_gate_exps,l.ffn_up_exps,l.ffn_down_exps};
}
void check_tensor(const ggml_tensor* tensor,int role,int64_t rows,bool host) {
    if(!tensor||!tensor->data||!tensor->buffer||tensor->type!=types[size_t(role)]||
        tensor->ne[0]!=(role==2?2048:4096)||tensor->ne[1]!=(role==2?4096:2048)||
        tensor->ne[2]!=rows||tensor->ne[3]!=1||tensor->nb[2]!=cold_gpu_stride(role)||
        !ggml_is_contiguous(tensor)||(host&&!ggml_backend_buffer_is_host(tensor->buffer)))
        throw std::runtime_error("GLM cold GPU original/bank layout differs");
    const uintptr_t address=reinterpret_cast<uintptr_t>(tensor->data);
    if(address%16||address>UINTPTR_MAX-uint64_t(rows)*tensor->nb[2])
        throw std::runtime_error("GLM cold GPU tensor address range differs");
}
}
struct ColdGpuOwner::Impl {
    llama_moe_cache& cache;
    uint64_t epoch;
    std::unique_ptr<PartialDownBindingOwner> partial;
    ggml_tensor* tensor=nullptr;
    std::array<ColdGpuLayer,layers> host{};
    std::array<uint64_t,8> counts{};
    bool requested=false,active=false;
    uint32_t maximum=2,reserved=0;
    std::array<uint64_t,2> selected_jobs{};
    uint32_t topology=0;
    std::array<uint64_t,4> topology_counts{};
    std::array<uint64_t,3> repeat_counts{};
    bool conditional4=false;
    std::array<uint64_t,5> conditional4_counts{};
    Impl(llama_moe_cache& value,uint64_t owner,uint32_t next):cache(value),epoch(owner),topology(next) {}
    void publish(){ggml_backend_tensor_set(tensor,host.data(),0,sizeof(host));}
};
bool ColdGpuOwner::requested() {
    const char* value=std::getenv("STRATA_GLM53_COLD_GPU_ROWS");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("STRATA_GLM53_COLD_GPU_ROWS must be 0 or 1");
}
uint32_t ColdGpuOwner::requested_max_rows() {
    const char* value=std::getenv("STRATA_GLM53_COLD_GPU_MAX_ROWS");
    if(!value||!std::strcmp(value,"2"))return 2;
    if(!std::strcmp(value,"4"))return 4;
    throw std::runtime_error("STRATA_GLM53_COLD_GPU_MAX_ROWS must be 2 or 4");
}
uint32_t ColdGpuOwner::topology_requested() {
    const char* value=std::getenv("STRATA_GLM53_COLD_GPU_TOPOLOGY");
    if(!value||!std::strcmp(value,"0"))return 0;
    if(!std::strcmp(value,"1"))return 1;
    if(!std::strcmp(value,"2"))return 2;
    if(!std::strcmp(value,"3"))return 3;
    throw std::runtime_error("STRATA_GLM53_COLD_GPU_TOPOLOGY must be 0, 1, 2 or 3");
}
bool ColdGpuOwner::conditional4_requested() {
    const char* value=std::getenv("STRATA_GLM53_COLD_GPU_CONDITIONAL4");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("STRATA_GLM53_COLD_GPU_CONDITIONAL4 must be 0 or 1");
}
ColdGpuOwner::ColdGpuOwner(const llama_model& model,llama_moe_cache& cache,
    deepseek_v4::ExpertStaging& staging,ggml_context* context,uint64_t epoch):impl_(new Impl(cache,epoch,topology_requested())) {
    if(model.arch!=LLM_ARCH_GLM5_NEXT||model.layers.size()!=layers||!context||!epoch)
        throw std::runtime_error("GLM cold GPU metadata identity differs");
    auto& p=*impl_;p.conditional4=conditional4_requested();
    for(int layer=0;layer<int(layers);++layer)if(cold_gpu_layer_valid(layer)) {
        auto& meta=p.host[size_t(layer)];const auto bank=sources(model,layer);
        for(int role=0;role<3;++role){check_tensor(bank[size_t(role)],role,288,true);meta.stride[role]=cold_gpu_stride(role);}
        // Whole real expert views only. Borrowing cannot register or copy.
        for(int key=1;key<=286;++key){bool complete=true;
            for(int role=0;role<3;++role){const auto* original=bank[size_t(role)];
                const auto* address=static_cast<const char*>(original->data)+size_t(key)*original->nb[2];
                meta.alias[role][key]=staging.borrow_source_alias(address,original->nb[2]);
                complete&=meta.alias[role][key]&&meta.alias[role][key]%16==0&&
                    meta.alias[role][key]<=UINTPTR_MAX-meta.stride[role];}
            if(complete)++p.counts[6];else for(int role=0;role<3;++role)meta.alias[role][key]=0;
        }
    }
    p.tensor=ggml_new_tensor_1d(context,GGML_TYPE_I8,int64_t(sizeof(p.host)));
    ggml_set_name(p.tensor,"strata_glm.cold_gpu_phase_metadata");
}
ColdGpuOwner::~ColdGpuOwner()=default;
bool ColdGpuOwner::requested_value() const{return impl_->requested;}
bool ColdGpuOwner::conditional4_requested_value() const{return impl_->conditional4;}
bool ColdGpuOwner::conditional4_active() const{return impl_->active&&impl_->conditional4&&impl_->reserved==4;}
uint32_t ColdGpuOwner::topology_requested_value() const{return impl_->topology;}
uint32_t ColdGpuOwner::requested_max_rows_value() const{return impl_->maximum;}
void ColdGpuOwner::initialize(){impl_->publish();}
void ColdGpuOwner::release_rows() {
    auto& p=*impl_;
    p.partial.reset(); // Existing quiescence also retires every diagnostic graph reader.
    if(!p.active)return;
    // GPU readers have drained before the caller enters. Invalidate aliases to
    // destination rows before returning those same rows to the single LRU.
    for(auto& meta:p.host){meta.owner_epoch=0;meta.bank_base[0]=meta.bank_base[1]=meta.bank_base[2]=0;
        meta.first_row=meta.bank_rows=0;meta.layer=-1;meta.active_mask=meta.original_missing_mask=0;
        meta.reserved_rows=0;meta.selected_count=0;
        meta.topology_fallback=0;meta.conditional4_policy=0;
        for(auto& key:meta.selected_keys)key=-1;}
    p.publish();p.cache.strata_glm_release_cold_rows();p.active=false;p.reserved=0;p.counts[7]=0;
}
void ColdGpuOwner::refresh(bool requested,bool tg) {
    const uint32_t maximum=requested_max_rows();
    auto& p=*impl_;
    release_rows(); // Existing GPU/CPU/BG quiescence is owned by the caller.
    const uint32_t topology=topology_requested();
    const bool conditional4=conditional4_requested();
    if(conditional4&&(maximum!=4||topology))throw std::runtime_error("GLM conditional cold4 requires max4 and topology0");
    if(topology&&maximum!=2)throw std::runtime_error("GLM cold topology requires maximum two rows");
    p.requested=requested;p.maximum=maximum;p.topology=topology;p.conditional4=conditional4;
    if(!requested||!tg)return;
    ++p.counts[4];
    if(!p.cache.strata_glm_reserve_cold_rows(int(maximum))){++p.counts[5];return;}
    uint32_t actual=0;
    try {
        for(int layer=0;layer<int(layers);++layer)if(cold_gpu_layer_valid(layer)) {
            int32_t first=-1,rows=0;std::array<ggml_tensor*,3> banks{};
            if(!p.cache.strata_glm_cold_rows(layer,first,banks,&rows)||first<0||
               (rows!=2&&rows!=4)||uint32_t(rows)>maximum||(actual&&actual!=uint32_t(rows)))
                throw std::runtime_error("GLM cold GPU reserved rows unavailable");
            actual=uint32_t(rows);auto& meta=p.host[size_t(layer)];meta.reserved_rows=actual;
            meta.topology_fallback=p.topology;meta.conditional4_policy=actual==4&&p.conditional4?1u:0u;
            for(int role=0;role<3;++role){const auto* bank=banks[size_t(role)];
                if(!bank||bank->ne[2]<3||bank->ne[2]>UINT32_MAX)
                    throw std::runtime_error("GLM cold GPU bank extent differs");
                check_tensor(bank,role,bank->ne[2],false);meta.bank_base[role]=reinterpret_cast<uintptr_t>(bank->data);
                if(role&&bank->ne[2]!=banks[0]->ne[2])throw std::runtime_error("GLM cold GPU bank rows differ");}
            meta.owner_epoch=p.epoch;meta.layer=layer;meta.first_row=uint32_t(first);
            meta.bank_rows=uint32_t(banks[0]->ne[2]-1); // Excludes the original padding sentinel.
            if(meta.first_row!=meta.bank_rows-actual||!cold_gpu_layout_valid(meta,p.epoch,layer))
                throw std::runtime_error("GLM cold GPU published reservation differs");
        }
        p.publish();p.active=true;p.reserved=actual;p.counts[7]=39;
        if(actual<maximum)++p.counts[5];
    } catch(...) {
        // No graph can see this partial publication at a quiescent boundary.
        p.cache.strata_glm_release_cold_rows();throw;
    }
}
void ColdGpuOwner::prepare_partial_down_diagnostic(ggml_backend_t gpu,PartialDownPhase* phase,uintptr_t device_phase,uint64_t generation){
    auto& p=*impl_;
    if(p.partial||!p.active||!p.requested||p.reserved!=2||p.topology||!device_layer(24))
        throw std::runtime_error("partial Down owner requires quiescent original cold2");
    p.partial=std::make_unique<PartialDownBindingOwner>(gpu,phase,device_phase,p.host[24],device_layer(24),generation);
}
PartialDownGraphBinding ColdGpuOwner::partial_down_binding(int layer) const {
    return layer==24&&impl_->partial?impl_->partial->binding():PartialDownGraphBinding{};
}
PartialDownPhase* ColdGpuOwner::partial_down_phase(int layer) const {
    return layer==24&&impl_->partial?impl_->partial->phase():nullptr;
}
uint64_t ColdGpuOwner::partial_down_generation() const{return impl_->partial?impl_->partial->generation():0;}
uintptr_t ColdGpuOwner::device_layer(int layer) const {
    const auto& p=*impl_;
    if(!p.active||!p.requested||!cold_gpu_layer_valid(layer))return 0;
    return reinterpret_cast<uintptr_t>(p.tensor->data)+size_t(layer)*sizeof(ColdGpuLayer);
}
uint32_t ColdGpuOwner::validate_ticket(int layer,int routes,uint32_t missing,const int32_t* directory,
    const deepseek_v4::DeviceMissPacket& packet) {
    auto& p=*impl_;
    const bool attached=routes==16&&packet.native_request==5&&device_layer(layer)!=0;
    // This word was padding in the original packet and is untouched by the
    // feature-off GPU template. Other paths must ignore its stale contents.
    if(!attached)return 0;
    const auto& meta=p.host[size_t(layer)];
    const auto receipt=validate_cold_gpu_ticket_receipt(meta,p.epoch,layer,missing,directory,packet);
    const auto& selected=receipt.selected;const uint32_t mask=selected.mask;
    // Native5 host tickets only: all-hit device fast paths do not visit here.
    if(meta.conditional4_policy==1){auto& c=p.conditional4_counts;++c[0];
        const bool shape=cold_gpu_conditional4_shape(packet.ids,missing);
        if(shape){++c[1];++c[selected.count==4?2:3];}
        else if(selected.count==2)++c[4];}
    if(!mask){++p.counts[3];return 0;}
    const uint32_t occurrences=receipt.occurrences;
    ++p.counts[0];p.counts[1]+=occurrences;p.counts[2]+=selected.count*(meta.stride[0]+meta.stride[1]+meta.stride[2]);
    if(selected.count>=2)++p.selected_jobs[selected.count==4?1:0];
    if(selected.topology){++p.topology_counts[0];p.topology_counts[1]+=selected.count;p.topology_counts[2]+=occurrences;
        p.topology_counts[3]+=selected.count==1;
        if(selected.topology==2){++p.repeat_counts[0];p.repeat_counts[1]+=occurrences-2;p.repeat_counts[2]+=occurrences==4;}}
    return mask;
}
std::array<uint64_t,8> ColdGpuOwner::metrics() const{return impl_->counts;}
std::array<uint64_t,2> ColdGpuOwner::selected_job_counts() const{return impl_->selected_jobs;}
std::array<uint64_t,4> ColdGpuOwner::topology_metrics() const{return impl_->topology_counts;}
std::array<uint64_t,3> ColdGpuOwner::repeat_priority_metrics() const{return impl_->repeat_counts;}
std::array<uint64_t,5> ColdGpuOwner::conditional4_metrics() const{return impl_->conditional4_counts;}
uint64_t ColdGpuOwner::reserved_rows() const{return impl_->active?impl_->reserved:0;}
size_t ColdGpuOwner::metadata_bytes() const{return sizeof(impl_->host);}
}
