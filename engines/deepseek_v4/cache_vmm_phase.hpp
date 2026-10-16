#pragma once
#include "ggml-backend.h"
#include <algorithm>
#include <stdexcept>
namespace strata::deepseek_v4 {
using PhaseCreate=ggml_backend_buffer_t (*)(ggml_backend_t,size_t,size_t);
using PhaseResize=bool (*)(ggml_backend_buffer_t,size_t);
using PhaseSwap=void (*)(ggml_backend_t,void*,void*,void*,size_t,int,int);
template<class Impl,class Device> void allocate_phase_banks(Impl& p,Device& d) {
    if(p.groups.size()!=1||p.groups[0].banks.size()!=3)throw std::runtime_error("elastic cache requires one native three-bank group");
    auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
    const auto create=reinterpret_cast<PhaseCreate>(ggml_backend_reg_get_proc_address(reg,"strata_phase_buffer_new"));
    if(!create)throw std::runtime_error("Strata native VMM cache factory unavailable");
    auto& g=p.groups[0];d.buf_size=0;
    for(auto* bank:g.banks){auto* buffer=create(d.backend,ggml_nbytes(bank),bank->nb[2]*(g.layers.size()*g.lru.n_expert+g.native.slots+1));
        if(!buffer)throw std::runtime_error("native phase cache mapping allocation failed");
        d.phase_banks.emplace_back(buffer);ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(ggml_backend_tensor_alloc(buffer,bank,ggml_backend_buffer_get_base(buffer))!=GGML_STATUS_SUCCESS)throw std::runtime_error("native VMM bank tensor attachment failed");
        d.buf_size+=ggml_backend_buffer_get_size(buffer);}
    for(auto& entry:p.bindings)if(ggml_backend_view_init(entry.second.cached)!=GGML_STATUS_SUCCESS)throw std::runtime_error("native VMM view attachment failed");
}
template<class Impl> size_t resize_vmm_cache(Impl& p,int permanent) {
    auto& g=p.groups.at(0);auto& d=p.devices[g.id];auto& lru=g.lru;
    const int old=lru.n_slots;if(old==permanent)return 0;
    if(permanent<6||permanent>int(g.layers.size())*lru.n_expert||d.phase_banks.size()!=3)throw std::runtime_error("native VMM resize capacity differs");
    auto* reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(d.backend));
    const auto resize=reinterpret_cast<PhaseResize>(ggml_backend_reg_get_proc_address(reg,"strata_phase_buffer_resize"));
    const auto swap=reinterpret_cast<PhaseSwap>(ggml_backend_reg_get_proc_address(reg,"strata_phase_swap_rows"));
    if(!resize||!swap)throw std::runtime_error("native VMM phase operations unavailable");
    ggml_backend_synchronize(d.backend);
    if(permanent<old){
        auto selected=lru.key_of;
        std::stable_sort(selected.begin(),selected.end(),[&](int a,int b){return a>=0&&(b<0||(lru.frequency&&lru.frequency->score(a)>lru.frequency->score(b)));});
        selected.resize(permanent);
        for(int slot=0;slot<permanent;++slot)if(selected[slot]>=0){
            const auto from=std::find(lru.key_of.begin(),lru.key_of.end(),selected[slot])-lru.key_of.begin();
            if(from==slot)continue;
            if(from>=old)throw std::runtime_error("elastic cache lost compacted key");
            swap(d.backend,g.banks[0]->data,g.banks[1]->data,g.banks[2]->data,g.banks[0]->nb[2],slot,int(from));
            std::swap(lru.key_of[slot],lru.key_of[from]);
        }
        ggml_backend_synchronize(d.backend);
    }
    for(size_t role=0;role<3;++role)if(!resize(d.phase_banks[role].get(),g.banks[role]->nb[2]*(permanent+g.native.slots+1)))throw std::runtime_error("native VMM physical resize failed");
    g.n_slots=permanent+g.native.slots;lru.n_slots=permanent;lru.key_of.resize(permanent,-1);
    for(auto* map:lru.slot_map)if(map)std::fill_n(map,lru.n_expert,-1);
    for(int slot=0;slot<permanent;++slot)if(lru.key_of[slot]>=0)lru.slot_map[lru.key_of[slot]/lru.n_expert][lru.key_of[slot]%lru.n_expert]=slot;
    lru.prev.resize(permanent);lru.next.resize(permanent);
    for(int slot=0;slot<permanent;++slot){lru.prev[slot]=slot-1;lru.next[slot]=slot+1<permanent?slot+1:-1;}
    lru.head=0;lru.tail=permanent-1;g.native.keys.fill(-1);d.buf_size=0;
    for(size_t role=0;role<3;++role){auto* bank=g.banks[role];bank->ne[2]=g.n_slots+1;bank->nb[3]=bank->nb[2]*bank->ne[2];d.buf_size+=ggml_backend_buffer_get_size(d.phase_banks[role].get());}
    for(auto& entry:p.bindings){auto* cached=entry.second.cached;cached->ne[2]=g.n_slots;cached->nb[3]=cached->nb[2]*cached->ne[2];}
    return 0; // Existing weights never cross PCIe during capacity transitions.
}
}
