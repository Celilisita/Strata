#pragma once
#include "partial_down_protocol.hpp"
#include "cold_gpu_protocol.hpp"
#include "ggml-backend.h"
#include <array>
#include <climits>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
// Borrows the existing authority's metadata, mapped phase and two physical rows.
// No weight storage, registration, stream or packet is owned here.
class PartialDownBindingOwner {
public:
    PartialDownBindingOwner(ggml_backend_t gpu,PartialDownPhase* phase,uintptr_t device_phase,
        const ColdGpuLayer& meta,uintptr_t device_metadata,uint64_t generation):gpu_(gpu),phase_(phase){
        if(!gpu||!phase||reinterpret_cast<uintptr_t>(phase)%64||!device_phase||device_phase%64||!device_metadata||device_metadata%64||
            !generation||!cold_gpu_layout_valid(meta,meta.owner_epoch,24,2)||meta.topology_fallback||meta.first_row>uint32_t(INT32_MAX-1))
            throw std::runtime_error("partial Down owner capability differs");
        const auto registry=ggml_backend_dev_backend_reg(ggml_backend_get_device(gpu));
        if(!registry||std::strcmp(ggml_backend_reg_name(registry),"CUDA"))throw std::runtime_error("partial Down requires the original CUDA provider");
        context_=ggml_init({4*ggml_tensor_overhead(),nullptr,true});
        if(!context_)throw std::runtime_error("partial Down bridge metadata allocation failed");
        try {
            binding_.bridge=ggml_new_tensor_1d(context_,GGML_TYPE_I64,8);
            binding_.indices=ggml_new_tensor_1d(context_,GGML_TYPE_I32,8);
            for(auto* tensor:{binding_.bridge,binding_.indices}){ggml_set_input(tensor);ggml_set_output(tensor);}
            storage_=ggml_backend_alloc_ctx_tensors(context_,gpu_);
            if(!storage_||ggml_backend_buffer_is_host(storage_))throw std::runtime_error("partial Down bridge requires CUDA storage");
            descriptor_.owner_epoch=meta.owner_epoch;descriptor_.bank_generation=generation;
            descriptor_.phase=device_phase;descriptor_.metadata=device_metadata;descriptor_.first_row=meta.first_row;
            const int32_t first=int32_t(meta.first_row);
            const std::array<int32_t,8> indices{first,first+1,first+1,first,57,234,234,57};
            *phase_=PartialDownPhase{};phase_->owner_epoch=meta.owner_epoch;phase_->bank_generation=generation;
            ggml_backend_tensor_set(binding_.bridge,&descriptor_,0,sizeof(descriptor_));
            ggml_backend_tensor_set(binding_.indices,indices.data(),0,sizeof(indices));
            ggml_backend_synchronize(gpu_);
        }catch(...){release();throw;}
    }
    ~PartialDownBindingOwner(){release();}
    PartialDownBindingOwner(const PartialDownBindingOwner&)=delete;
    PartialDownBindingOwner& operator=(const PartialDownBindingOwner&)=delete;
    PartialDownGraphBinding binding()const noexcept{return binding_;}
    PartialDownPhase* phase()const noexcept{return phase_;}
    uint64_t generation()const noexcept{return descriptor_.bank_generation;}
    size_t device_buffer_bytes()const noexcept{return storage_?ggml_backend_buffer_get_size(storage_):0;}
    size_t metadata_bytes()const noexcept{return context_?ggml_used_mem(context_):0;}
private:
    void release()noexcept {
        if(gpu_)ggml_backend_synchronize(gpu_);
        if(storage_)ggml_backend_buffer_free(storage_);storage_=nullptr;
        if(context_)ggml_free(context_);context_=nullptr;binding_={};
    }
    ggml_backend_t gpu_;PartialDownPhase* phase_;ggml_context* context_=nullptr;
    ggml_backend_buffer_t storage_=nullptr;PartialDownGraphBinding binding_;PartialDownBridge descriptor_;
};
}
