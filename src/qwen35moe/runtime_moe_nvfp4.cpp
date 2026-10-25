#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/shared_f16_gemm.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include <stdexcept>
#include <cstring>

namespace strata::qwen35moe {
void Runtime::batch_moe_nvfp4(size_t il,const LayerWeights& layer,int count,
    const std::string& reference,Metrics& metrics) {
    if(!nvfp4_ || count<1 || count>batch_.capacity || count>8192 ||
       il>=nvfp4_->inventory.layers() || checkpoint_sink_ || paired_session_ || sessions_ ||
       mtp_enabled_ || verify_state_ || verify_history_ || batch_.expert_stage.data() ||
       expert_cache_.slots<size_t(resident_layers_+1)*256 || dense_profile_.active())
        throw std::invalid_argument("NVFP4 ordinary grouped PP domain");
    auto& b=batch_;auto& fp4=*nvfp4_;
    if(!fp4.provider)fp4.provider=std::make_unique<prefill::mmq::NativeGroupedNvfp4Provider>(stream_);
    const auto ids=batch_route(layer,count,reference,metrics,false);
    const int entries=count*8;
    if(ids.size()!=size_t(entries))throw std::runtime_error("NVFP4 route count");
    core::ExpertRetention::LayerLease lease(expert_cache_.retention,il,
        expert_sources_.size(),256,ids,true);
    auto event=[&](int index){if(!decode_phase_ && !prefill_events_.empty())
        cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_+il][index],stream_),"NVFP4 PP event");};
    event(2);
    prefetch_routes(il,ids); // Actual prefix triplets become ready before any typed cache view/consumer.
    // The full router has already drained the previous compute readers. Close
    // that request before a new reservation can change its directory/epoch.
    finish_gu_copy(metrics);
    static const int copy_wave=[] {
        const char* raw=std::getenv("STRATA_QWEN35_NVFP4_GU_COPY_PIPELINE");
        if(!raw||!std::strcmp(raw,"0"))return 0;
        if(!std::strcmp(raw,"32"))return 32;
        throw std::invalid_argument("NVFP4_GU_COPY_PIPELINE requires0 or32");
    }();
    uint64_t reserved=0;
    const auto& cache=expert_cache_;
    const bool eligible=copy_wave&&!decode_phase_&&(count==1280||count==4096)&&
        prefill_config_.prefix_demand&&!prefill_config_.prefix_stage&&gu_copy_prefix_layer_==il&&
        reference.empty()&&!visual_table()&&!cache.retention.enabled()&&
        !prefill_config_.expert_copy_runs&&!prefill_config_.expert_copy_stage&&
        !prefill_config_.expert_prefetch&&!prefill_config_.mapped_pp&&
        cache.slots>=size_t(resident_layers_)*256+512&&
        !(il<pinned_experts_.layers.size()&&pinned_experts_.layers[il].data);
    if(eligible) {
        std::array<bool,256> requested{};size_t unique=0,cold=0;
        for(int id:ids){if(id<0||id>=256)throw std::runtime_error("NVFP4 GU route ID");
            if(!requested[size_t(id)]){requested[size_t(id)]=true;++unique;
                cold+=cache.key_to_slot[il*256+size_t(id)]<0;}}
        if(unique>64&&cold&&cold<unique) {
            if(!gu_copy_)gu_copy_=std::make_unique<ExpertGuCopyPipeline>();
            const auto source=fp4.inventory.view(il,0);
            const auto layout=Nvfp4ExpertInventory::layout();
            try{gu_copy_pending_=gu_copy_->prepare(prefetch_cache(),il,
                {{source.gate,source.up,source.down},{layout.gu,layout.gu,layout.down}},host_readonly_,ids);
            }catch(...){batch_failed_=true;throw;}
            if(gu_copy_pending_)reserved=cold;
        }
    }
    fp4.gate_scales.resize(size_t(entries));fp4.up_scales.resize(size_t(entries));
    fp4.down_scales.resize(size_t(entries));
    for(int e=0;e<entries;++e) {
        const auto view=expert_view(il,ids[size_t(e)],expert_sources_[il],metrics);
        b.host_expert_addresses[size_t(e)]=reinterpret_cast<unsigned long long>(view.gate);
        b.host_expert_addresses[size_t(e+entries)]=reinterpret_cast<unsigned long long>(view.up);
        b.host_down_addresses[size_t(e)]=reinterpret_cast<unsigned long long>(view.down);
        b.host_expert_rows[size_t(e)]=e/8;b.host_down_rows[size_t(e)]=e;
        fp4.gate_scales[size_t(e)]=view.gate_scale;fp4.up_scales[size_t(e)]=view.up_scale;
        fp4.down_scales[size_t(e)]=view.down_scale;
    }
    event(3);
    void* packed=b.qkv.data();size_t packed_bytes=b.qkv.bytes();
    if(b.phase_arena.data()) {
        const size_t offset=size_t(b.capacity)*20480*sizeof(float);
        if(offset>b.phase_arena.bytes())throw std::runtime_error("NVFP4 packed arena offset");
        packed=b.phase_arena.as<uint8_t>()+offset;packed_bytes=b.phase_arena.bytes()-offset;
    }
    prefill::mmq::NativeGroupedMmqWork gu;
    try{gu=gu_copy_pending_?fp4.provider->run_gu_pipeline(b.host_expert_addresses.data(),
        b.host_expert_addresses.data()+entries,fp4.gate_scales.data(),fp4.up_scales.data(),
        ids.data(),b.host_expert_rows.data(),entries,count,b.norm.as<float>(),
        b.expert_q8.data(),b.expert_q8.bytes(),packed,packed_bytes,b.expert_gu.as<float>(),copy_wave,
        [&](const prefill::mmq::NativeNvfp4Matrix* matrices,int n,bool last){
            // Explicit address adapter: NVFP4 matrices carry scales and use a
            // different ABI. The owner sees only original cache triplet bases.
            std::array<prefill::mmq::NativeMmqMatrix,32> address{};
            if(n<1||n>32)throw std::runtime_error("NVFP4 GU wave extent");
            for(int i=0;i<n;++i)address[size_t(i)]={matrices[i].gate,matrices[i].up,matrices[i].begin,matrices[i].end};
            gu_copy_->ready(prefetch_cache(),address.data(),n,last,stream_);
        }):fp4.provider->run(b.host_expert_addresses.data(),
        b.host_expert_addresses.data()+entries,fp4.gate_scales.data(),fp4.up_scales.data(),
        ids.data(),b.host_expert_rows.data(),entries,count,b.norm.as<float>(),
        b.expert_q8.data(),b.expert_q8.bytes(),packed,packed_bytes,
        b.expert_gu.as<float>(),2048,512,64);
    }catch(...){if(gu_copy_pending_)batch_failed_=true;throw;}
    if(gu_copy_pending_){const auto& copied=gu_copy_->cost();
        if(copied.cold!=reserved||metrics.expert_cache_hits<reserved)
            throw std::runtime_error("NVFP4 GU reserved miss accounting");
        metrics.expert_cache_hits-=reserved;metrics.expert_cache_misses+=reserved;
        metrics.expert_h2d+=copied.bytes;metrics.expert_h2d_calls+=copied.calls;
        metrics.expert_h2d_logical_copies+=copied.logical_copies;
        metrics.expert_h2d_registered_bytes+=copied.registered_bytes;
        metrics.expert_h2d_pageable_bytes+=copied.pageable_bytes;
        metrics.expert_enqueue_ms+=copied.enqueue_ms;
    }
    swiglu(b.expert_gu.as<float>(),b.expert_gu.as<float>()+size_t(entries)*512,
        b.expert_hidden.as<float>(),entries*512,stream_);
    event(5);
    const auto down=fp4.provider->run(b.host_down_addresses.data(),nullptr,
        fp4.down_scales.data(),nullptr,ids.data(),b.host_down_rows.data(),entries,entries,
        b.expert_hidden.as<float>(),b.expert_q8.data(),b.expert_q8.bytes(),
        packed,packed_bytes,b.expert_parts.as<float>(),512,2048,32);
    event(6);
    // Shared tensors retain their original GGUF format and their existing
    // qualified service. Only the routed weights/activations use NVFP4 here.
    static const bool raw_shared=shared_f16_gemm_policy(std::getenv("STRATA_QWEN35_SHARED_F16_GEMM"));
    const bool shared=raw_shared&&!decode_phase_&&prefill_tc_&&count>=1024&&count<=4096&&
        layer.shared_gate.type==12&&layer.shared_up.type==12&&
        (layer.shared_down.type==12||layer.shared_down.type==13);
    uint64_t shared_submissions=0;
    if(shared) {
        b.half_width=0;b.half_count=0;b.clear_raw_input();
        run_shared_f16_gemm(b.gemm,{layer.shared_gate.ptr(),layer.shared_gate.data.bytes(),layer.shared_gate.type},
            {layer.shared_up.ptr(),layer.shared_up.data.bytes(),layer.shared_up.type},
            {layer.shared_down.ptr(),layer.shared_down.data.bytes(),layer.shared_down.type},
            b.norm.as<float>(),b.input_half.as<uint16_t>(),b.input_half.bytes(),b.shared_gate.as<float>(),
            b.shared_up.as<float>(),b.shared_hidden.as<float>(),b.shared_output.as<float>(),b.logits.as<float>(),count);
        shared_submissions=10; // 7 known CUDA launches + 3 vendor API submissions, not private kernel count.
    } else {
        batch_quantize_q8(b.norm.as<float>(),count,2048);
        batch_project_native(layer.shared_gate,b.shared_gate.as<float>(),count,2048,512);
        batch_project_native(layer.shared_up,b.shared_up.as<float>(),count,2048,512);
        swiglu(b.shared_gate.as<float>(),b.shared_up.as<float>(),b.shared_hidden.as<float>(),count*512,stream_);
        batch_quantize_q8(b.shared_hidden.as<float>(),count,512);
        batch_project_native(layer.shared_down,b.shared_output.as<float>(),count,512,2048);
        gate_shared_batch(b.shared_output.as<float>(),b.logits.as<float>(),count,stream_);
        // Preserve the legacy logical-submission estimate for this unchanged branch.
        const bool bulk=!decode_phase_&&std::getenv("STRATA_QWEN35_BULK_QUANT")&&
            std::string(std::getenv("STRATA_QWEN35_BULK_QUANT"))=="1";
        shared_submissions=2*(bulk?1:size_t((count+7)/8))+2+
            (prefill_tc_&&!decode_phase_&&count>=1024?18+9*size_t((count+1023)/1024-1):size_t((count+7)/8)*3);
    }
    moe_combine_batch(b.expert_parts.as<float>(),b.weights.as<float>(),b.shared_output.as<float>(),
        b.mixer.as<float>(),b.hidden.as<float>(),count,stream_);
    event(4);
    metrics.control_h2d+=gu.control_bytes+down.control_bytes;
    metrics.expert_launches+=gu.launches+down.launches+2+shared_submissions;
    if(!reference.empty()) {
        batch_probe_.shared=batch_stage_error(reference,"ffn_shexp_gated-0",b.shared_output.as<float>(),2048);
        batch_probe_.moe_out=batch_stage_error(reference,"l_out-0",b.hidden.as<float>(),2048);
    }
}
} // namespace strata::qwen35moe
