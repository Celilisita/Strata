#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/nvfp4_tg_controls.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include <array>
#include <limits>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::batch_moe_nvfp4_device(size_t il,const LayerWeights& layer,Metrics& metrics) {
    if(!nvfp4_ || !nvfp4_->device_tg || !tg_dynamic_ || !decode_phase_ ||
       il>=expert_sources_.size() || batch_.expert_q8.bytes()<sizeof(Nvfp4TgControls))
        throw std::invalid_argument("unprepared NVFP4 device TG");
    auto& b=batch_;using namespace prefill::mmq;
    auto* controls=b.expert_q8.as<Nvfp4TgControls>();
    void* packed=b.qkv.data();size_t bytes=b.qkv.bytes();
    if(b.phase_arena.data()) {
        const size_t offset=size_t(b.capacity)*20480*sizeof(float);
        if(offset>b.phase_arena.bytes())throw std::runtime_error("NVFP4 TG packed arena");
        packed=b.phase_arena.as<uint8_t>()+offset;bytes=b.phase_arena.bytes()-offset;
    }
    route_project(layer.router.f32(),b.norm.as<float>(),b.logits.as<float>(),2048,257,stream_);
    auto* args=dynamic_args_.as<core::NativeMissArgs>()+il;
    route_top8_continue(b.logits.as<float>(),b.ids.as<int32_t>(),b.weights.as<float>(),args,stream_);
    const auto gu=encode_native_nvfp4(b.norm.as<float>(),packed,bytes,1,2048,controls->sources,stream_);
    native_grouped_nvfp4(controls->gu,8,2,controls->outputs,b.expert_gu.as<float>(),
        8,1,512,controls->tiles,8,8,gu,stream_);
    swiglu(b.expert_gu.as<float>(),b.expert_gu.as<float>()+8*512,b.expert_hidden.as<float>(),
        8*512,stream_,b.down_addresses.as<unsigned long long>(),512);
    // Miss hidden rows are deliberately unwritten. The masked encoder must not read them.
    const auto down=encode_native_nvfp4_masked(b.expert_hidden.as<float>(),packed,bytes,8,512,
        controls->sources+8,b.down_addresses.as<unsigned long long>(),stream_);
    native_grouped_nvfp4(controls->down,8,1,controls->outputs,b.expert_parts.as<float>(),
        8,1,2048,controls->tiles,8,8,down,stream_);
    batch_quantize_q8(b.norm.as<float>(),1,2048);
    batch_project_native(layer.shared_gate,b.shared_gate.as<float>(),1,2048,512);
    batch_project_native(layer.shared_up,b.shared_up.as<float>(),1,2048,512);
    swiglu(b.shared_gate.as<float>(),b.shared_up.as<float>(),b.shared_hidden.as<float>(),512,stream_);
    batch_quantize_q8(b.shared_hidden.as<float>(),1,512);
    batch_project_native(layer.shared_down,b.shared_output.as<float>(),1,512,2048);
    gate_shared_batch(b.shared_output.as<float>(),b.logits.as<float>(),1,stream_);
    moe_combine_continue(b.expert_parts.as<float>(),b.weights.as<float>(),b.shared_output.as<float>(),
        b.mixer.as<float>(),b.hidden.as<float>(),args,stream_);
    metrics.expert_launches+=13; // GU2+SwiGLU1+Down2+shared7+combine1; router remains separately submitted.
    ++dynamic_layers_;dynamic_entries_+=8;
}

// Called only after head_token's ordinary stream synchronization. Current-token
// GPU/CPU consumers have finished, so the old snapshot can be replaced safely.
void Runtime::replay_nvfp4_routes(Metrics& metrics) {
    auto& state=*nvfp4_;auto& cache=expert_cache_;
    const auto* ledger=miss_service_->route_ledger();
    const size_t layers=expert_sources_.size();
    if(!ledger || state.tg_token==(std::numeric_limits<uint32_t>::max)())
        throw std::runtime_error("invalid NVFP4 token ledger");
    const uint32_t token=state.tg_token+1;
    // Validate every record before the first directory mutation/copy.
    try {validate_nvfp4_routes(ledger,layers,decode_epoch_,token);}
    catch(...){batch_failed_=true;throw;}
    if(state.tg_admission) {
        auto& transfer=*state.tg_admission;
        try {
            transfer.begin_plan(prefetch_cache(),host_readonly_,decode_epoch_,token);
            Metrics reservations;
            for(size_t il=0;il<layers;++il)for(int id:ledger->layers[il].ids)
                (void)expert_view(il,id,expert_sources_[il],reservations);
            transfer.seal(prefetch_cache(),host_readonly_);
        }catch(...){transfer.invalidate();batch_failed_=true;throw;}
        state.tg_token=token;state.tg_pending=transfer.pending();
        state.tg_tail_bytes=state.tg_tail_control=0;
        metrics.control_d2h+=layers*sizeof(core::NativeRouteRecord);
        ++metrics.host_continuations;
        return; // Reservations are not transfers. Actual costs belong to the next STEP/finish flush.
    }
    Metrics admission;
    bool needs_copy=false;
    for(size_t il=0;il<layers;++il)for(int id:ledger->layers[il].ids)
        needs_copy|=cache.key_to_slot[il*256+size_t(id)]<0;
    try {
        if(needs_copy)cuda_check(cudaEventRecord(state.tg_admit_begin,stream_),"mark NVFP4 admission begin");
        // Replay hits AND misses in original layer/rank order. This keeps the
        // authoritative expert_view learning; GPU hit ages are not merged again.
        for(size_t il=0;il<layers;++il)for(int id:ledger->layers[il].ids)
            (void)expert_view(il,id,expert_sources_[il],admission);
        if(admission.expert_cache_misses) {
            state.tg_directory=cache.key_to_slot;
            if(state.tg_directory.size()*sizeof(int32_t)!=cache.resident_slots.bytes())
                throw std::runtime_error("NVFP4 decode directory extent changed");
            // Payload copies precede directory publication in this SAME stream.
            // The host vector is unchanged until the next normal head sync.
            cuda_check(cudaMemcpyAsync(cache.resident_slots.data(),state.tg_directory.data(),
                cache.resident_slots.bytes(),cudaMemcpyHostToDevice,stream_),"publish learned NVFP4 directory");
            metrics.control_h2d+=cache.resident_slots.bytes();state.tg_pending=true;
            cuda_check(cudaEventRecord(state.tg_admit_end,stream_),"mark NVFP4 admission end");
        } else state.tg_pending=false;
    } catch(...) {
        // An incomplete payload is never reused as a future device hit. This
        // runtime is poisoned, and existing cancellation/reset drains its stream.
        batch_failed_=true;
        for(size_t k=0;k<cache.key_to_slot.size();++k)if(k>=size_t(resident_layers_)*256)cache.key_to_slot[k]=-1;
        for(size_t s=size_t(resident_layers_)*256;s<cache.slot_to_key.size();++s){cache.slot_to_key[s]=-1;cache.age[s]=0;}
        throw;
    }
    state.tg_token=token;
    state.tg_tail_bytes=admission.expert_h2d;
    state.tg_tail_control=admission.expert_cache_misses?cache.resident_slots.bytes():0;
    metrics.control_d2h+=layers*sizeof(core::NativeRouteRecord);
    ++metrics.host_continuations;
    metrics.expert_h2d+=admission.expert_h2d;
    metrics.expert_h2d_calls+=admission.expert_h2d_calls;
    metrics.expert_h2d_logical_copies+=admission.expert_h2d_logical_copies;
    metrics.expert_h2d_registered_bytes+=admission.expert_h2d_registered_bytes;
    metrics.expert_h2d_pageable_bytes+=admission.expert_h2d_pageable_bytes;
    metrics.expert_enqueue_ms+=admission.expert_enqueue_ms;
    // Admission hits/misses are NOT a second count of current-token execution hits/misses.
}
}
