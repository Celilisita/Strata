// Included once by the authoritative device-cache service.
// Admission headroom is checked at this boundary; later workspace, cold rows
// and NextN allocations may consume it. It is not a perpetual free guarantee.
void DeviceExpertCache::glm_capacity_admit(size_t target) {
    using namespace strata::glm5_next;
    auto& out=glm_capacity_;out.headroom=tg_headroom_bytes();
    const auto snapshot=cache_.strata_cost_snapshot();
    const size_t floor=cache_.strata_glm_pp_budget();
    struct Check {ggml_backend_t backend;size_t margin;DeviceMemoryBudget resource;bool error=false;};
    Check check{backend_,size_t(out.headroom),{}};
    const auto acceptable=+[](void* opaque)->bool {
        auto& c=*static_cast<Check*>(opaque);
        try {c.resource=device_memory_budget(c.backend);}catch(...){c.error=true;return false;}
        const auto local=c.resource.wddm_available?
            (c.resource.local_budget>c.resource.local_usage?c.resource.local_budget-c.resource.local_usage:0):c.resource.cuda_free;
        return c.resource.cuda_free>=c.margin&&local>=c.margin;
    };
    size_t accepted=std::max(floor,size_t(out.admitted));
    target=std::max(target,accepted);
    PhaseMapAudit audit;
    for(int attempt=0;attempt<4;++attempt) {
        ++out.attempts;
        const int result=cache_.strata_grow_glm_phase(target,acceptable,&check,audit);
        out.reason=uint32_t(result);out.driver_error=audit.driver_error;
        if(check.error){glm_capacity_poisoned_=true;throw std::runtime_error("GLM capacity telemetry failed; no continuation");}
        out.allocated=audit.allocated_after;
        if(result==growth_committed){accepted=target;break;}
        ++out.refused;
        if((result!=growth_allocation_refused&&result!=growth_headroom_refused)||!audit.rollback_verified) {
            glm_capacity_poisoned_=true;throw std::runtime_error("GLM capacity transaction failed; owner cannot continue");
        }
        // A verified old mapping remains usable. Probe a bounded smaller tail,
        // never remap/reseed old payload and never continue an uncertain tail.
        if(target<=accepted+(256ull<<20))break;
        target=accepted+((target-accepted)/2/(256ull<<20))*(256ull<<20);
    }
    out.admitted=accepted;out.pp_floor_limited=accepted==floor&&out.refused>0;
    const auto actual=cache_.strata_cost_snapshot();out.weight_bytes=actual.weight_tensor_bytes;out.slots=actual.permanent_slots;
    const auto resource=device_memory_budget(backend_);
    out.cuda_free=resource.cuda_free;
    out.local_free=resource.wddm_available?(resource.local_budget>resource.local_usage?resource.local_budget-resource.local_usage:0):resource.cuda_free;
    out.unknown_cuda_dxgi_gap=resource.wddm_available&&out.cuda_free!=out.local_free;
    last_phase_budget_[1]=accepted;
    // A refusal cannot manufacture physical bytes from backend buffer size.
    // This field comes exclusively from the checked VMM per-chunk mapping audit.
    if(!out.allocated&&snapshot.backend_alloc_bytes)out.reason=growth_invalid_owner;
}

bool DeviceExpertCache::glm_capacity_due(bool single_token) {
    using namespace strata::glm5_next;
    if(!enabled_||!single_token||!glm_tg_phase_)return false;
    if(glm_capacity_pending_)return true;
    ++glm_tg_calls_;
    const uint64_t checkpoint=128ull<<glm_capacity_.rechecks;
    if(glm_capacity_.rechecks>=4||glm_tg_calls_<checkpoint)return false;
    ++glm_capacity_.rechecks;
    if(glm_capacity_.admitted>=glm_capacity_.requested){glm_capacity_.reason=growth_requested_reached;return false;}
    const auto resource=device_memory_budget(backend_);
    const size_t local=resource.wddm_available?(resource.local_budget>resource.local_usage?resource.local_budget-resource.local_usage:0):resource.cuda_free;
    const size_t free=std::min(size_t(resource.cuda_free),local),margin=tg_headroom_bytes();
    glm_capacity_.cuda_free=resource.cuda_free;glm_capacity_.local_free=local;
    glm_capacity_.unknown_cuda_dxgi_gap=resource.wddm_available&&resource.cuda_free!=local;
    if(free<=margin+(256ull<<20)){glm_capacity_.reason=growth_no_reported_room;return false;}
    const size_t available=cache_.strata_cost_snapshot().weight_tensor_bytes+free-margin;
    const size_t target=std::min(size_t(glm_capacity_.requested),size_t((available/(256ull<<20))*(256ull<<20)));
    if(target<=glm_capacity_.admitted){glm_capacity_.reason=growth_no_reported_room;return false;}
    glm_capacity_pending_=true;return true;
}

void DeviceExpertCache::glm_capacity_recheck() {
    using namespace strata::glm5_next;
    worker_active_.store(false,std::memory_order_release);
    ggml_backend_synchronize(backend_);
    std::lock_guard<std::mutex> lock(service_mutex_);
    if(cpu_)cpu_->wait();finish_background(true);read_frequency();
    const auto registration=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_));
    auto inspect=reinterpret_cast<CudaPoolInspect>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_inspect"));
    auto retire=reinterpret_cast<CudaPoolRetire>(ggml_backend_reg_get_proc_address(registration,"strata_cuda_pool_retire"));
    CudaPoolRetirement retired;
    if(!inspect||!retire||!inspect(backend_,&retired)||!retire(backend_,&retired))
        throw std::runtime_error("GLM readmission requires retired graphs and no CUDA pool loans");
    glm_capacity_.retired_pool+=retired.released_bytes;
    if(cold_gpu_)cold_gpu_->release_rows();
    const auto resource=device_memory_budget(backend_);
    const size_t local=resource.wddm_available?(resource.local_budget>resource.local_usage?resource.local_budget-resource.local_usage:0):resource.cuda_free;
    const size_t free=std::min(size_t(resource.cuda_free),local),margin=tg_headroom_bytes();
    const size_t old=cache_.strata_cost_snapshot().weight_tensor_bytes;
    const size_t available=old+(free>margin?free-margin:0);
    const size_t target=std::min(size_t(glm_capacity_.requested),size_t((available/(256ull<<20))*(256ull<<20)));
    glm_capacity_admit(target);project_directory();
    if(cold_gpu_)cold_gpu_->refresh(ColdGpuOwner::requested(),enabled_);
    glm_capacity_pending_=false;
    // Existing worker admission may populate new empty slots; no full-cache seed.
}
