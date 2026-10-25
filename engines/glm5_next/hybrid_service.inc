// Included once by the shared device-cache worker implementation.
void DeviceExpertCache::service_hybrid(int32_t layer,strata::glm5_next::CausalHostSample* causal){
    auto* packet=dma_->packet();
    if(packet->native_request==5)++hybrid_overlapped_services_;
    if(!hybrid_||!hybrid_active_||layer<3||layer>=int(model_.layers.size())||!originals_[layer])
        throw std::runtime_error("hybrid miss has no active inventory/provider");
    const auto* slots=static_cast<const int32_t*>(originals_[layer]->data);
    auto& count=last_miss_counts_[layer];count=0;
    const int routes=int(packet->routed_count);
    if(routes==32){service_hybrid_four(layer);return;}
    if(routes!=8&&routes!=16)throw std::runtime_error("hybrid real route occurrence count differs");
    std::array<int32_t,16> selected{},ranks{};std::array<uint8_t,16> positions{};
    int unique=0;uint32_t missing_mask=0;
    for(int rank=0;rank<routes;++rank){
        const int id=packet->ids[rank];
        if(id<0||id>=288)throw std::runtime_error("hybrid expert key invalid");
        if(slots[id]<0){last_miss_ids_[layer][count]=id;selected[count]=id;ranks[count]=rank;positions[count]=uint8_t(rank/8);
            bool seen=false;for(uint32_t i=0;i<count;++i)seen|=selected[i]==id;
            if(!seen)++unique;++count;missing_mask|=1u<<rank;}
    }
    if(!count)throw std::runtime_error("hybrid cold request has no missing expert");
    // Preserve the GPU's positive tail-row receipts until the pure-host mirror
    // validates selection, epoch and the unchanged resident directory.
    const uint32_t gpu_mask=cold_gpu_mask(layer,routes,missing_mask,slots);
    for(int rank=0;rank<routes;++rank)if(!(gpu_mask&(1u<<rank)))packet->reply_slots[rank]=slots[packet->ids[rank]];
    activation_control_bytes_+=uint64_t(routes/8)*4096*sizeof(float);
    const bool cpu_preferred=routes==16?hybrid_->prefer_cpu_pair(layer,int(count),unique):hybrid_->prefer_cpu(layer,int(count));
    if(packet->native_request==5&&!cpu_preferred)throw std::runtime_error("Overlapped CPU request differs from measured provider choice");
    if(cpu_preferred){
        uint32_t cpu_count=0;
        for(uint32_t i=0;i<count;++i)if(!(gpu_mask&(1u<<ranks[i]))){
            selected[cpu_count]=selected[i];ranks[cpu_count]=ranks[i];positions[cpu_count]=positions[i];++cpu_count;}
        std::array<float,8192> activation{};
        std::memcpy(activation.data(),packet->activation,size_t(routes/8)*4096*sizeof(float));
        // This bounded scratch belongs only to the serial miss worker. No
        // second resident expert store or GPU placeholder computation exists.
        std::array<float,16*4096> output{};
        const auto start=std::chrono::steady_clock::now();
        if(causal)causal->compute_begin(start);
        strata::glm5_next::PartialDownHostReceipt partial;
        if(routes==16)if(auto* phase=cold_gpu_?cold_gpu_->partial_down_phase(layer):nullptr)
            partial=strata::glm5_next::partial_down_service(hybrid_->partial_down_diagnostic_service(),*packet,*phase,
                owner_epoch_,cold_gpu_->partial_down_generation(),selected.data(),positions.data(),ranks.data(),cpu_count);
        if(partial.selected)cpu_bytes_+=uint64_t(10)*(2*strata::glm5_next::cold_gpu_stride(0))+uint64_t(6)*strata::glm5_next::cold_gpu_stride(2);
        else if(cpu_count)cpu_bytes_+=routes==16?hybrid_->compute_pair(layer,selected.data(),positions.data(),int(cpu_count),activation.data(),output.data()):
            hybrid_->compute(layer,selected.data(),int(cpu_count),activation.data(),output.data());
        // A validated GPU-only ticket still needs a causal closure marker;
        // it invokes no CPU provider and contributes no CPU job or duration.
        const auto end=cpu_count?std::chrono::steady_clock::now():start;
        if(causal)causal->compute_end(end);
        cpu_ns_+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(end-start).count());
        if(cpu_count){++cpu_jobs_;cpu_routes_+=cpu_count;}
        if(partial.selected)contribution_bytes_+=partial.published_mid_bytes+partial.contribution_bytes;
        else {
            packet->cpu_mask=0;
            for(uint32_t i=0;i<cpu_count;++i){
                std::memcpy(packet->contribution+size_t(ranks[i])*4096,output.data()+size_t(i)*4096,4096*sizeof(float));
                packet->reply_slots[ranks[i]]=-1;packet->cpu_mask|=1u<<ranks[i];
            }
            contribution_bytes_+=uint64_t(cpu_count)*4096*sizeof(float);
        }
    }else{
        if(gpu_mask)throw std::runtime_error("GLM cold GPU ticket lacks CPU-only provider contract");
        // A GPU miss may reuse any free row; settle background leases first.
        finish_background(true);
        ++hybrid_gpu_services_;
        cache_.strata_fill_selected(layer,packet->ids,size_t(routes),upload,this,&planning_ns_,frequency_!=nullptr,(1u<<routes)-1);
        project_directory();
        slots=static_cast<const int32_t*>(originals_[layer]->data);
        for(int rank=0;rank<routes;++rank)packet->reply_slots[rank]=slots[packet->ids[rank]];
        packet->cpu_mask=0;
    }
    // The existing worker release fence publishes this response only after
    // CPU results or completed payload+directory DMA. The next layer depends
    // on contribution consumption, so this one ticket cannot be overwritten.
}
