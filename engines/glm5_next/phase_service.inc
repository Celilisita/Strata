// Included once by the authoritative device-cache service implementation.
void DeviceExpertCache::glm_phase_transition(bool to_tg) {
    if(model_.arch!=LLM_ARCH_GLM5_NEXT||!strata::glm5_next::phase_cache_requested()||to_tg==glm_tg_phase_)return;
    const auto registration=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    auto inspect=reinterpret_cast<CudaPoolInspect>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_inspect"));
    auto retire=reinterpret_cast<CudaPoolRetire>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_retire"));
    CudaPoolRetirement retired;
    if(!inspect||!retire||!inspect(backend_,&retired)||!retire(backend_,&retired))
        throw std::runtime_error("GLM phase transition requires retired graphs and a quiescent CUDA pool");
    // Caller retired GPU/CPU readers; pending admissions still own row leases.
    finish_background(true);
    if(cold_gpu_)cold_gpu_->release_rows();
    const auto before=cache_.strata_cost_snapshot();
    size_t target=to_tg?strata::glm5_next::tg_cache_bytes():0;
    if(to_tg) {
        glm_capacity_={};glm_tg_calls_=0;glm_capacity_pending_=false;
        const auto started=std::chrono::steady_clock::now();
        const size_t requested=target,allowance=strata::glm5_next::tg_headroom_bytes();
        glm_capacity_.requested=requested;glm_capacity_.retired_pool=retired.released_bytes;
        DeviceMemoryBudget resource;
        size_t free=0;
        for(int sample=0;sample<11;++sample) {
            resource=device_memory_budget(backend_);
            const size_t local_free=resource.wddm_available?
                (resource.local_budget>resource.local_usage?size_t(resource.local_budget-resource.local_usage):0):resource.cuda_free;
            free=std::min(size_t(resource.cuda_free),local_free);
            const size_t growth=requested>before.backend_alloc_bytes?requested-before.backend_alloc_bytes:0;
            if(free>=growth+allowance||sample==10)break;
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        const size_t available=before.weight_tensor_bytes+(free>allowance?free-allowance:0);
        const size_t rounded=(available/(256ull<<20))*(256ull<<20);
        target=std::min(requested,std::max(cache_.strata_glm_pp_budget(),rounded));
        last_phase_budget_={requested,target,uint64_t(resource.cuda_free),resource.local_budget,resource.local_usage,
            uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count()),retired.released_bytes};
    }
    const auto start=std::chrono::steady_clock::now();
    if(to_tg)glm_capacity_admit(target);else cache_.strata_resize_glm_phase(target);
    phase_resize_ns_+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());
    const auto resized=cache_.strata_cost_snapshot();
    if(resized.resident_keys<before.resident_keys)phase_evicted_keys_+=before.resident_keys-resized.resident_keys;
    if(to_tg) {
        const auto seed=std::chrono::steady_clock::now();
        uint64_t removed=0;phase_seed_bytes_+=cache_.strata_seed_glm_phase(upload,this,&removed);phase_evicted_keys_+=removed;
        project_directory(); // Commits and waits on the payload DMA stream.
        phase_seed_ns_+=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-seed).count());
    }
    const char* verify=std::getenv("STRATA_GLM53_PHASE_VERIFY");
    if(verify&&!std::strcmp(verify,"1"))phase_verify_bytes_+=cache_.strata_verify_glm_phase();
    if(to_tg) {
        using Reserve=bool(*)(ggml_backend_t,size_t,bool);
        auto reserve=reinterpret_cast<Reserve>(ggml_backend_reg_get_proc_address(registration,"strata_device_cache_reserve"));
        if(!reserve||!reserve(backend_,128ull<<20,true))throw std::runtime_error("GLM TG preallocated kernel workspace unavailable");
    }
    glm_tg_phase_=to_tg;
    // Resize/seed may replace backing allocations even with the same flag.
    if(cold_gpu_)cold_gpu_->refresh(strata::glm5_next::ColdGpuOwner::requested(),to_tg&&enabled_);
    ++phase_transitions_;
    if(!to_tg)project_directory();
}

void DeviceExpertCache::phase_probe_reverse_prior() {
    if(model_.arch!=LLM_ARCH_GLM5_NEXT||!glm_tg_phase_||!frequency_)
        throw std::runtime_error("phase probe lacks the GLM TG authority");
    ggml_backend_synchronize(backend_);std::lock_guard<std::mutex> lock(service_mutex_);
    std::fill(frequency_host_.begin(),frequency_host_.end(),0u);
    for(size_t il=0;il<model_.layers.size();++il)if(originals_[il]) {
        const auto* map=static_cast<const int32_t*>(originals_[il]->data);
        for(int e=0;e<experts_;++e)if(map[e]>=0)frequency_host_[il*experts_+e]=uint32_t(map[e]+1);
    }
    frequency_->import_state(frequency_host_.data(),frequency_host_.size());
    ggml_backend_tensor_set(frequency_gpu_,frequency_host_.data(),0,frequency_host_.size()*sizeof(uint32_t));
}
