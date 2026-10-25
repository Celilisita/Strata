#pragma once
#include "../../../../../engines/glm5_next/phase_growth.hpp"
#include <array>
#include <limits>
// Called only after graph, CPU, DMA/background and borrowed-tail retirement.
// All old VMM bases stay fixed. Sizes publish only after every new tail commits.
static int strata_phase_buffer_grow_batch(ggml_backend_t backend,ggml_backend_buffer_t* buffers,
    const size_t* bytes,size_t count,bool(*admissible)(void*),void* user,
    strata::glm5_next::PhaseMapAudit* report) {
    using namespace strata::glm5_next;
    if(!report)return growth_invalid_owner;*report={};
    struct Saved {strata::phase_vmm::VmmRange* range=nullptr;size_t size=0;int64_t chunks=0,target=0;};
    std::array<Saved,12> saved{};
    if(!backend||!ggml_backend_is_cuda(backend)||!buffers||!bytes||!admissible||!count||count>saved.size())return growth_invalid_owner;
    auto* owner=static_cast<ggml_backend_cuda_context*>(backend->context);
    ggml_cuda_set_device(owner->device);const size_t gran=strata::phase_vmm::vmm_granularity();
    if(!gran||cudaPeekAtLastError()!=cudaSuccess)return growth_sync_failed;
    for(size_t i=0;i<count;++i) {
        auto* b=buffers[i];if(!b||b->buft!=ggml_backend_cuda_buffer_type(owner->device))return growth_invalid_owner;
        for(size_t j=0;j<i;++j)if(buffers[j]==b)return growth_invalid_owner;
        auto* c=static_cast<ggml_backend_cuda_buffer_context*>(b->context);
        if(!c||c->device!=owner->device||!c->phase_range||bytes[i]>std::numeric_limits<size_t>::max()-gran+1)
            return growth_invalid_owner;
        auto& v=*c->phase_range;auto& s=saved[i];s={&v,b->size,v.mapped_count(),int64_t((bytes[i]+gran-1)/gran)};
        if(s.size!=size_t(s.chunks)*gran||s.target<s.chunks||s.target>v.chunks())return growth_invalid_owner;
        for(int64_t k=0;k<v.chunks();++k)if(v.mapped(k)!=(k<s.chunks))return growth_invalid_owner;
        report->allocated_before+=s.size;report->requested_mapping+=size_t(s.target)*gran;
    }
    report->allocated_after=report->allocated_before;
    const auto rollback=[&] {
        bool ok=true;for(size_t i=count;i-->0;)ok=saved[i].range->trim_tail_checked(saved[i].chunks)&&ok;
        report->rollback_verified=false;
        if(!ok){report->reason=growth_rollback_failed;return false;}
        if(cudaDeviceSynchronize()!=cudaSuccess||cudaPeekAtLastError()!=cudaSuccess) {
            report->reason=growth_sync_failed;return false;
        }
        report->rollback_verified=true;return true;
    };
    for(size_t i=0;i<count;++i) {
        const int error=saved[i].range->grow_tail_checked(saved[i].target);
        if(error) {
            report->bank=uint32_t(i);report->driver_error=error;
            report->reason=error==2?growth_allocation_refused:growth_driver_failed;
            if(!rollback())return report->reason;
            return report->reason;
        }
    }
    if(!admissible(user)) {
        report->reason=growth_headroom_refused;if(!rollback())return report->reason;return report->reason;
    }
    if(cudaDeviceSynchronize()!=cudaSuccess||cudaPeekAtLastError()!=cudaSuccess) {
        report->reason=growth_sync_failed;rollback();return report->reason;
    }
    for(size_t i=0;i<count;++i)buffers[i]->size=size_t(saved[i].target)*gran;
    report->allocated_after=report->requested_mapping;report->reason=growth_committed;
    return growth_committed;
}
