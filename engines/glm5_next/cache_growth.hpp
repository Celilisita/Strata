#pragma once
#include "phase_growth.hpp"
#include "cache_budget.hpp"
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>
namespace strata::glm5_next {
// Growth only. Caller has retired graph readers, admissions and borrowed rows.
// Prepare every potentially-throwing host allocation before mapping any bank.
template<class Impl> int grow_phase(Impl& p,size_t budget,bool(*admissible)(void*),void* user,PhaseMapAudit& audit) {
    if(p.devices.size()!=1||(p.groups.size()!=3&&p.groups.size()!=4))
        throw std::runtime_error("GLM growth requires the single-device inventory");
    auto& d=p.devices.at(0);
    if(d.phase_banks.size()!=3*p.groups.size())throw std::runtime_error("GLM growth bank ownership differs");
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
    const auto grow=reinterpret_cast<PhaseGrowBatch>(ggml_backend_reg_get_proc_address(reg,"strata_phase_buffer_grow_batch"));
    if(!grow)throw std::runtime_error("GLM transactional growth operation unavailable");
    struct Prepared {int count=0;std::vector<int32_t> keys,prev,next;};
    std::array<Prepared,4> prepared;
    std::array<ggml_backend_buffer_t,12> buffers{};std::array<size_t,12> sizes{};
    const size_t alignment=ggml_backend_buft_get_alignment(d.buft);
    const auto allocation_size=[&](const auto& group,int slots) {
        size_t result=0;for(auto* t:group.ref)result+=((t->nb[2]*(slots+1)+alignment-1)/alignment)*alignment;
        return result;
    };
    bool changed=false;
    for(size_t gi=0;gi<p.groups.size();++gi) {
        const auto& g=p.groups[gi];const auto& lru=g.lru;auto& next=prepared[gi];
        if(g.banks.size()!=3||lru.n_expert!=288||g.native.slots||g.n_slots!=lru.n_slots)
            throw std::runtime_error("GLM growth has outstanding borrowed/native rows");
        const auto share=cache_budget(p.groups,g,budget,d.host_bytes,
            size_t(double(budget)*g.host_bytes/d.host_bytes),allocation_size,p.n_expert_used);
        int low=minimum_cache_routes(p.n_expert_used),high=int(g.layers.size())*lru.n_expert;
        while(low<high){const int mid=low+(high-low+1)/2;if(allocation_size(g,mid)<=share)low=mid;else high=mid-1;}
        next.count=std::max(low,lru.n_slots);changed|=next.count>lru.n_slots;
        next.keys=lru.key_of;next.keys.resize(next.count,-1);
        next.prev=lru.prev;next.next=lru.next;next.prev.resize(next.count);next.next.resize(next.count);
        for(int s=lru.n_slots;s<next.count;++s){next.prev[s]=s==lru.n_slots?-1:s-1;next.next[s]=s+1<next.count?s+1:lru.head;}
        if(next.count>lru.n_slots)next.prev[lru.head]=next.count-1;
        for(size_t role=0;role<3;++role) {
            buffers[3*gi+role]=d.phase_banks[3*gi+role].get();
            sizes[3*gi+role]=g.banks[role]->nb[2]*(next.count+1);
        }
    }
    // A no-op still reads exact VMM mapping via the backend transaction.
    const int result=grow(d.backend,buffers.data(),sizes.data(),d.phase_banks.size(),admissible,user,&audit);
    if(result!=growth_committed)return result;
    if(changed)for(size_t gi=0;gi<p.groups.size();++gi) {
        auto& g=p.groups[gi];auto& lru=g.lru;auto& next=prepared[gi];
        // All following operations are no-throw, and retain every key's old slot.
        lru.key_of.swap(next.keys);lru.prev.swap(next.prev);lru.next.swap(next.next);
        if(next.count>lru.n_slots)lru.head=lru.n_slots;
        g.n_slots=lru.n_slots=next.count;g.native.keys.fill(-1);
        for(auto* bank:g.banks){bank->ne[2]=next.count+1;bank->nb[3]=bank->nb[2]*bank->ne[2];}
        for(auto& entry:p.bindings)if(p.layers[entry.second.il].ig==int(gi)) {
            auto* view=entry.second.cached;view->ne[2]=next.count;view->nb[3]=view->nb[2]*view->ne[2];
        }
    }
    d.buf_size=audit.allocated_after;
    return growth_committed;
}
}
