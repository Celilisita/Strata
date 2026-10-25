#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include <chrono>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::submit_nvfp4_admission(size_t il,Metrics& metrics){
    if(!nvfp4_||!nvfp4_->tg_admission||!decode_phase_)return;
    auto& plan=*nvfp4_->tg_admission;
    if(!plan.opened())return;
    const auto before=std::chrono::steady_clock::now();
    try{
        const auto cost=plan.submit(il,prefetch_cache(),host_readonly_,[&](Nvfp4TgAdmission::Copy r,cudaStream_t copy){
            const size_t layer=size_t(r.key)/256;const int id=r.key%256;
            const auto view=nvfp4_->inventory.view(layer,id);
            auto* target=expert_cache_.data.slot(size_t(r.slot));
            const size_t bytes=Nvfp4ExpertInventory::matrix_bytes;
            const auto* pinned=layer<pinned_experts_.layers.size()&&pinned_experts_.layers[layer].data?
                &pinned_experts_.layers[layer]:nullptr;
            if(pinned){
                cuda_check(cudaMemcpyAsync(target,pinned->data+size_t(id)*pinned->stride,3*bytes,
                    cudaMemcpyHostToDevice,copy),"typed TG packed admission H2D");
                return HostCopyCost{3*bytes,1,1,3*bytes,0};
            }
            HostCopyCost transferred;
            transferred+=host_readonly_.copy_to_device(target,view.gate,bytes,copy,"typed TG gate admission",r.key,0);
            transferred+=host_readonly_.copy_to_device(target+bytes,view.up,bytes,copy,"typed TG up admission",r.key,1);
            transferred+=host_readonly_.copy_to_device(target+2*bytes,view.down,bytes,copy,"typed TG down admission",r.key,2);
            return transferred;
        });
        metrics.expert_h2d+=cost.payload.bytes;metrics.expert_h2d_calls+=cost.payload.calls;
        metrics.expert_h2d_logical_copies+=cost.payload.logical_copies;
        metrics.expert_h2d_registered_bytes+=cost.payload.registered_bytes;
        metrics.expert_h2d_pageable_bytes+=cost.payload.pageable_bytes;
        metrics.control_h2d+=cost.directory_bytes;
        if(cost.rows)metrics.expert_enqueue_ms+=std::chrono::duration<double,std::milli>(
            std::chrono::steady_clock::now()-before).count();
    }catch(...){plan.invalidate();batch_failed_=true;throw;}
}
void Runtime::wait_nvfp4_admission(size_t il){
    if(!nvfp4_||!nvfp4_->tg_admission||!decode_phase_)return;
    auto& plan=*nvfp4_->tg_admission;if(!plan.opened())return;
    try{plan.wait(il,prefetch_cache(),host_readonly_,stream_);}
    catch(...){plan.invalidate();batch_failed_=true;throw;}
}
void Runtime::flush_nvfp4_admission(Metrics& metrics){
    if(!nvfp4_||!nvfp4_->tg_admission||!nvfp4_->tg_admission->opened())return;
    auto& state=*nvfp4_;auto& plan=*state.tg_admission;
    const uint64_t before_bytes=metrics.expert_h2d,before_control=metrics.control_h2d;
    try{
        state.tg_pending=plan.dirty_mask()!=0;
        if(state.tg_pending)cuda_check(cudaEventRecord(state.tg_admit_begin,plan.stream()),"typed TG final flush begin");
        // Last-token planning is useful learning too. Flush it rather than
        // silently dropping it or moving its transfer cost outside the receipt.
        for(size_t il=0;il<expert_sources_.size();++il)submit_nvfp4_admission(il,metrics);
        for(size_t il=0;il<expert_sources_.size();++il)wait_nvfp4_admission(il);
        if(state.tg_pending){
            cuda_check(cudaEventRecord(state.tg_admit_end,plan.stream()),"typed TG final flush end");
            cuda_check(cudaStreamWaitEvent(stream_,state.tg_admit_end,0),"typed TG finish joins final flush");
        }
        state.tg_tail_bytes=metrics.expert_h2d-before_bytes;
        state.tg_tail_control=metrics.control_h2d-before_control;
    }catch(...){plan.invalidate();batch_failed_=true;throw;}
}
}
