#include "native_read_ahead.hpp"
#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::deepseek_v4 {
namespace {uint64_t now(){return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());}}
NativeReadAhead::NativeReadAhead(ggml_backend_t backend,const llama_model& model,Mxfp4CpuService& cpu,DeviceMissPacket* real,
    uint32_t(*ready)(const DeviceMissPacket*),const std::atomic<bool>& stop,const std::atomic<bool>& active)
    :model_(model),cpu_(cpu),real_(real),ready_(ready),stop_(stop),active_(active) {
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Packet=RouterPrefetchPacket*(*)(ggml_backend_t);
    auto get=reinterpret_cast<Packet>(ggml_backend_reg_get_proc_address(reg,"strata_router_prefetch_packet"));
    if(!get||!(packet_=get(backend))||!ready_)throw std::runtime_error("Native read-ahead mapped producer absent");
    const char* select=std::getenv("STRATA_DSV4_PREFETCH_SELECTED_TWO");select_two_=select&&!std::strcmp(select,"1");
}
bool NativeReadAhead::guard(void* user){auto& s=*static_cast<NativeReadAhead*>(user);
    return !s.stop_.load(std::memory_order_acquire)&&s.active_.load(std::memory_order_acquire)&&
        s.ready_(s.real_)==s.guard_ticket_&&(s.guard_hash_?s.packet_->hash_sequence:s.packet_->sequence)==s.guard_sequence_;
}
bool NativeReadAhead::read(const RouterPrefetchRoutes& route,uint32_t ticket,bool hash){
    if(route.layer>=43||route.tokens<1||route.tokens>2||route.epoch==0||(route.masks[0]|route.masks[1])>63)
        throw std::runtime_error("Read-ahead descriptor invalid");
    if(!(route.masks[0]|route.masks[1])||!cpu_.idle())return false;
    for(int t=0;t<2;++t)for(int rank=0;rank<6;++rank)if(route.masks[t]&(1u<<rank)){
        const int id=route.ids[t*6+rank];if(id<0||id>=256)throw std::runtime_error("Read-ahead immutable expert key invalid");}
    guard_ticket_=ticket;guard_hash_=hash;guard_sequence_=hash?hash_seen_:seen_;
    if(!guard(this))return false;
    const auto& layer=model_.layers[route.layer];
    const Mxfp4CpuBank bank{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps,model_.hparams.swiglu_clamp_exp[route.layer]};
    const auto start=now();const auto bytes=cpu_.read_ahead_pair(bank,route.ids,route.masks,route.source==1?1.2:.33,
        !hash&&select_two_?route.selected:nullptr,guard,this);
    const auto end=now();++stats_.jobs;stats_.bytes+=bytes;stats_.wall_ns+=end-start;
    if(hash){++stats_.hash_jobs;stats_.hash_bytes+=bytes;}
    if(!guard(this))++stats_.canceled;
    if(epochs_[route.layer]!=route.epoch){touched_[route.layer].fill(false);epochs_[route.layer]=route.epoch;}
    if(bytes)for(int t=0;t<2;++t)for(int rank=0;rank<6;++rank)if(route.masks[t]&(1u<<rank)){
        const int id=route.ids[t*6+rank];if(!hash&&select_two_&&id!=route.selected[0]&&id!=route.selected[1])continue;
        if(!touched_[route.layer][id]){touched_[route.layer][id]=true;++stats_.predicted_ids;}}
    return true;
}
bool NativeReadAhead::poll(uint32_t ticket){
    if(!cpu_.idle())return false;
    const uint32_t sequence=packet_->sequence;
    if(sequence!=seen_){std::atomic_thread_fence(std::memory_order_acquire);const auto route=packet_->learned[sequence&1];
        std::atomic_thread_fence(std::memory_order_acquire);if(packet_->sequence!=sequence)return false;
        seen_=sequence;return read(route,ticket,false);}
    const uint32_t sequence_hash=packet_->hash_sequence;
    if(sequence_hash!=hash_seen_){std::atomic_thread_fence(std::memory_order_acquire);
        for(int layer=0;layer<3;++layer)hashes_[layer]=packet_->hash[sequence_hash&1][layer];
        std::atomic_thread_fence(std::memory_order_acquire);if(packet_->hash_sequence!=sequence_hash)return false;
        hash_seen_=sequence_hash;hash_next_=hashes_[0].source==1?0:std::max(0,last_hash_done_+1);}
    if(hash_next_<3){const int layer=hash_next_;
        if(layer>last_hash_done_+1&&hashes_[0].source!=1)return false;
        hash_next_=hashes_[0].source==1?3:hash_next_+1;return read(hashes_[layer],ticket,true);}
    return false;
}
void NativeReadAhead::observe(const DeviceMissPacket& packet){
    if(packet.layer==43){last_hash_done_=-1;return;}
    if(packet.layer<3)last_hash_done_=int(packet.layer);
    if(packet.layer>=43)return;
    const int tokens=packet.native_request==3?2:1;
    for(int id=0;id<256;++id)if(touched_[packet.layer][id]){
        bool correct=false;for(int r=0;r<tokens*6;++r)correct|=(packet.cpu_mask&(1u<<r))&&packet.ids[r]==id;
        stats_.correct_ids+=correct;
    }
    touched_[packet.layer].fill(false);
}
}
