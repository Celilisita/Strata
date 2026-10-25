// Included after hybrid_service.inc in the existing serial miss worker.
void DeviceExpertCache::service_hybrid_four(int32_t layer) {
    auto* packet=dma_->packet();
    if(packet->native_request!=5||packet->routed_count!=32||!hybrid_four_scratch_||
       !hybrid_->four_positions_prepared())
        throw std::runtime_error("four-position cold service is not prepared overlap diagnostic");
    auto& scratch=*hybrid_four_scratch_;
    auto& count=last_miss_counts_[layer];count=0;
    const auto* slots=static_cast<const int32_t*>(originals_[layer]->data);
    for(int rank=0;rank<32;++rank) {
        const int id=packet->ids[rank];
        if(id<0||id>=288)throw std::runtime_error("four-position expert key invalid");
        packet->reply_slots[rank]=slots[id];
        if(slots[id]<0) {
            last_miss_ids_[layer][count]=id;scratch.selected[count]=id;
            scratch.ranks[count]=rank;scratch.positions[count]=uint8_t(rank/8);++count;
        }
    }
    if(!count)throw std::runtime_error("four-position cold service has no missing routes");
    std::memcpy(scratch.activation.data(),packet->activation,sizeof(scratch.activation));
    activation_control_bytes_+=sizeof(scratch.activation);
    const auto start=std::chrono::steady_clock::now();
    cpu_bytes_+=hybrid_->compute_four(layer,scratch.selected.data(),scratch.positions.data(),int(count),
        scratch.activation.data(),scratch.output.data());
    cpu_ns_+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());
    ++cpu_jobs_;cpu_routes_+=count;packet->cpu_mask=0;
    for(uint32_t i=0;i<count;++i) {
        const int rank=scratch.ranks[i];
        std::memcpy(packet->contribution+size_t(rank)*4096,scratch.output.data()+size_t(i)*4096,4096*sizeof(float));
        packet->reply_slots[rank]=-1;packet->cpu_mask|=uint32_t(1)<<rank;
    }
    contribution_bytes_+=uint64_t(count)*4096*sizeof(float);
    // No fill, lease or directory update. The original worker publishes its
    // release response only after every original-rank contribution is ready.
}
