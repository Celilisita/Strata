#pragma once
#include "pp_plan_receipt.hpp"
#include "pp_stream_graph.hpp"
#include "pp_stream_plan.hpp"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
namespace strata::deepseek_v4 {
class PpPlanReceiptStore {
    std::unique_ptr<PpPlanReceiptPod> data_;
    std::array<bool,42> done_{};
    uint64_t seen_=0;
    bool active_=false, read_=false;
    void invalid() { if(data_)data_->flags|=pp_receipt_invalid; }
public:
    static constexpr uint32_t no_layer=std::numeric_limits<uint32_t>::max();
    bool active() const { return active_; }
    bool begin(uint64_t nonce,bool enabled) {
        if(!nonce)return false;
        if(!enabled&&data_&&data_->request_nonce==nonce) {
            active_=false;data_->flags|=pp_receipt_retired;
            if(!(data_->flags&pp_receipt_sealed))data_->flags|=pp_receipt_incomplete;
            return true;
        }
        if(active_||(data_&&!read_))return false;
        if(!enabled){data_.reset();return true;}
        if(!data_)data_=std::make_unique<PpPlanReceiptPod>();
        std::memset(data_.get(),0,sizeof(PpPlanReceiptPod));done_={};seen_=0;read_=false;active_=true;
        data_->request_nonce=nonce;data_->version=1;data_->expected_tokens=8192;
        data_->flags=pp_receipt_requested;
        return true;
    }
    const PpPlanReceiptPod* read() {
        if(!data_||!(data_->flags&pp_receipt_retired))return nullptr;
        read_=true;return data_.get();
    }
    uint32_t plan(const PpStreamSources& s,const PpStreamPlan& p,const int32_t* ids,
                  int prefix,const uint64_t* missing,uint32_t states) {
        if(!active_)return no_layer;
        if(p.tokens!=8192||p.topk!=8||s.gate->ne[2]!=288||p.groups.size()>288||p.selected_experts>288) {
            invalid();return no_layer;
        }
        size_t nb=0;for(const auto& g:p.groups)nb+=g.bounds.size();
        if(data_->layer_count>=42||data_->group_count+p.groups.size()>pp_plan_receipt_groups||
           data_->bound_count+nb>pp_plan_receipt_bounds) {
            data_->flags|=pp_receipt_overflow;++data_->capacity_errors;return no_layer;
        }
        const char* name=s.gate->name;char* end=nullptr;
        if(std::strncmp(name,"blk.",4)){invalid();return no_layer;}
        const long layer=std::strtol(name+4,&end,10);
        if(end==name+4||std::strcmp(end,".ffn_gate_exps.weight")||layer<3||layer>44||(seen_&(uint64_t(1)<<layer))) {
            invalid();return no_layer;
        }
        const uint32_t at=data_->layer_count++;auto& out=data_->layers[at];seen_|=uint64_t(1)<<layer;
        out.layer=int32_t(layer);out.tokens=p.tokens;out.topk=p.topk;out.experts=288;out.window_experts=s.window_experts;
        out.prefix_boundary=uint32_t(prefix);out.groups_begin=data_->group_count;out.groups_count=uint32_t(p.groups.size());
        out.bounds_begin=data_->bound_count;out.bounds_count=uint32_t(nb);
        out.flags=(uint32_t(s.gate->type)&255)|((uint32_t(s.up->type)&255)<<8)|
                  ((uint32_t(s.down->type)&255)<<16)|(states<<24);
        std::memcpy(&out.clamp_bits,&s.clamp,sizeof(float));
        const ggml_tensor* sources[]{s.gate,s.up,s.down};
        for(int role=0;role<3;++role) {
            out.role_expert_bytes[role]=sources[role]->nb[2];
            std::memcpy(out.source_names[role],sources[role]->name,64);
            if(!std::memchr(out.source_names[role],0,64))invalid();
        }
        std::copy_n(missing,3,out.prefix_missing);
        const int32_t* directory=s.cache?static_cast<const int32_t*>(s.cache->directory->data):nullptr;
        for(int key=0;key<288;++key)out.directory[key]=directory?directory[key]:-1;
        // These are per-call counts from existing host IDs, not cumulative prefill gauges.
        for(int pair=0;pair<p.tokens*p.topk;++pair) {
            const int key=ids[pair];if(key<0||key>=288){invalid();continue;}++out.route_counts[key];
        }
        if(layer==24) {
            if(data_->full_ids_captures)invalid();
            else {
                std::copy_n(ids,8192*8,data_->layer24_route_ids);
                data_->full_ids_layer=24;data_->full_ids_tokens=8192;
                data_->full_ids_count=8192*8;data_->full_ids_captures=1;
            }
        }
        for(const auto& g:p.groups) {
            auto& record=data_->groups[data_->group_count++];
            record={g.first_expert,g.experts,g.first_row,g.rows,g.max_rows,g.cache_first_slot,
                    data_->bound_count,PpPlanSourceKind::unknown};
            std::copy(g.bounds.begin(),g.bounds.end(),data_->bounds+data_->bound_count);
            data_->bound_count+=uint32_t(g.bounds.size());
        }
        return at;
    }
    void kind(uint32_t layer,size_t group,PpPlanSourceKind value,bool separate) {
        if(!active_||layer==no_layer)return;
        if(layer>=data_->layer_count||group>=data_->layers[layer].groups_count){invalid();return;}
        auto& record=data_->groups[data_->layers[layer].groups_begin+group];
        if(record.source_kind!=PpPlanSourceKind::unknown){invalid();return;}
        const bool expected=value==PpPlanSourceKind::direct_cache||value==PpPlanSourceKind::prefix_planar||
                            value==PpPlanSourceKind::source_planar;
        if(expected!=separate)invalid();record.source_kind=value;
    }
    void done(uint32_t layer){if(active_&&layer!=no_layer)done_[layer]=true;}
    void seal() {
        if(!active_)return;
        bool good=data_->layer_count==42&&data_->full_ids_captures==1&&data_->full_ids_layer==24&&
            data_->full_ids_tokens==8192&&data_->full_ids_count==8192*8;
        for(uint32_t i=0;i<data_->layer_count;++i)good&=done_[i];
        for(uint32_t i=0;i<data_->group_count;++i)
            if(data_->groups[i].source_kind==PpPlanSourceKind::unknown){good=false;++data_->source_kind_unset;}
        if(good)data_->flags|=pp_receipt_sealed;else data_->flags|=pp_receipt_incomplete;
        active_=false; // Seal ends this capture; later shared-backend PP is outside this receipt.
    }
};
}
