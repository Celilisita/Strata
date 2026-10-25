void DeviceExpertCache::settle_background() {
    ggml_backend_synchronize(backend_);
    std::lock_guard<std::mutex> lock(service_mutex_);finish_background(true);
}

// One definition, included by the existing authoritative cache service.
bool DeviceExpertCache::finish_background(bool wait) {
    if(pending_admission_.empty())return true;
    if(wait)dma_->wait_background();else if(!dma_->background_ready())return false;
    const char* verify=std::getenv("STRATA_GLM53_BACKGROUND_VERIFY");
    if(model_.arch==LLM_ARCH_GLM5_NEXT&&verify&&!std::strcmp(verify,"1"))
        background_verify_bytes_+=cache_.strata_verify_background(pending_admission_);
    cache_.strata_commit_background(pending_admission_);
    for(const auto& item:pending_admission_)if(size_t(item.group)<background_layout_commits_.size())
        ++background_layout_commits_[item.group];
    pending_admission_.clear();project_directory();++background_commits_;
    return true;
}

void DeviceExpertCache::idle() {
    const bool glm=model_.arch==LLM_ARCH_GLM5_NEXT;
    const char* enabled=std::getenv("STRATA_DSV4_TG_BACKGROUND_ADMISSION");
    if(!active_||(glm?!strata::glm5_next::background_requested():
        (!native_tg_cpu_only()||!enabled||std::strcmp(enabled,"1"))))return;
    const char* stage=std::getenv("STRATA_DSV4_BG_STAGE_UNPINNED");
    const bool staged=stage&&!std::strcmp(stage,"1");
    // Caller consumed completed logits. Old bank readers have retired.
    // Pending destinations stay absent from the maps until the event completes.
    std::lock_guard<std::mutex> lock(service_mutex_);
    if(!finish_background(false))return;
    const char* cadence=std::getenv("STRATA_DSV4_BG_ADMISSION_PERIOD");
    unsigned period=16;
    if(cadence){char* end=nullptr;const long value=std::strtol(cadence,&end,10);
        if(!*cadence||*end||value<1||value>64)throw std::runtime_error("background cadence outside1..64");period=unsigned(value);}
    if(++idle_tokens_%period)return;
    if(background_eligible_.empty()||background_staging_mode_!=staged) {
        background_staging_mode_=staged;
        background_eligible_.assign(model_.layers.size()*experts_,0);
        for(int key=0;key<int(model_.layers.size())*experts_;++key) {
            const int il=key/experts_;if(!originals_[il])continue;
            const auto& layer=model_.layers[il];bool eligible=true;
            for(auto* source:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps}) {
                if(!source){eligible=false;break;}
                const auto part=staging_?staging_->transport_segment(static_cast<const char*>(source->data)+(key%experts_)*source->nb[2],source->nb[2]):ExpertSourcePin::Segment{0,false};
                if((!part.registered||part.bytes<source->nb[2])&&!(staged&&(glm||il<43)))eligible=false;
            }
            background_eligible_[key]=uint8_t(eligible);
        }
    }
    read_frequency();pending_admission_=cache_.strata_plan_background(background_eligible_.data());
    if(pending_admission_.empty())return;
    for(const auto& item:pending_admission_) {
        const auto& layer=model_.layers[item.key/experts_];bool needs_stage=false;
        for(auto* source:{layer.ffn_gate_exps,layer.ffn_up_exps,layer.ffn_down_exps}) {
            const auto part=staging_?staging_->transport_segment(static_cast<const char*>(source->data)+(item.key%experts_)*source->nb[2],source->nb[2]):ExpertSourcePin::Segment{0,false};
            needs_stage|=!part.registered||part.bytes<source->nb[2];
        }
        if(needs_stage)++background_staged_experts_;
    }
    const auto staged_before=dma_->metrics().host_copy_bytes;
    background_bytes_+=cache_.strata_upload_background(pending_admission_,upload,this);
    background_staged_bytes_+=dma_->metrics().host_copy_bytes-staged_before;
    dma_->seal_background();
}

std::array<uint64_t,4> DeviceExpertCache::probe_background() {
    if(model_.arch!=LLM_ARCH_GLM5_NEXT||!active_||!glm_tg_phase_||!frequency_||
       !strata::glm5_next::background_requested())throw std::runtime_error("background probe lacks the GLM cache authority");
    const auto bytes=background_verify_bytes_;const auto counts=background_layout_commits_;
    {
        std::lock_guard<std::mutex> lock(service_mutex_);finish_background(true);
        std::fill(frequency_host_.begin(),frequency_host_.end(),0u);
        for(size_t il=0;il<model_.layers.size();++il)if(originals_[il]) {
            const auto* map=static_cast<const int32_t*>(originals_[il]->data);
            for(int id=0;id<experts_;++id)frequency_host_[il*experts_+id]=map[id]<0?1024:1;
        }
        frequency_->import_state(frequency_host_.data(),frequency_host_.size());
        ggml_backend_tensor_set(frequency_gpu_,frequency_host_.data(),0,frequency_host_.size()*sizeof(uint32_t));
        idle_tokens_=0;
    }
    idle();
    {std::lock_guard<std::mutex> lock(service_mutex_);finish_background(true);}
    return {background_verify_bytes_-bytes,background_layout_commits_[0]-counts[0],
        background_layout_commits_[1]-counts[1],background_layout_commits_[2]-counts[2]};
}
