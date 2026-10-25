#pragma once
#include "strata_glm_kv_window.cuh"
#include <array>
#include <mutex>
#include <vector>

namespace {
constexpr int glm_kv_registry_max = 128;
__device__ __constant__ StrataGlmKvWindow glm_kv_registry[glm_kv_registry_max];
__device__ __constant__ int glm_kv_registry_count;
}

// Only this F32 attention translation unit owns these constant descriptors.
// Updates occur at tensor creation/destruction, outside graph capture/compute.
void strata_glm_kv_attention_update(int device,const StrataGlmKvWindow& view,bool remove) {
    static std::mutex mutex;
    static std::array<std::vector<StrataGlmKvWindow>,GGML_CUDA_MAX_DEVICES> devices;
    std::lock_guard<std::mutex> guard(mutex);
    ggml_cuda_set_device(device);
    auto& entries=devices.at(device);
    auto it=std::find_if(entries.begin(),entries.end(),[&](const StrataGlmKvWindow& x){return x.host==view.host;});
    if(remove) {if(it!=entries.end())entries.erase(it);}
    else {GGML_ASSERT(it==entries.end());GGML_ASSERT(entries.size()<glm_kv_registry_max);entries.push_back(view);}
    const int count=int(entries.size());
    if(count)CUDA_CHECK(cudaMemcpyToSymbol(glm_kv_registry,entries.data(),entries.size()*sizeof(view)));
    CUDA_CHECK(cudaMemcpyToSymbol(glm_kv_registry_count,&count,sizeof(count)));
}

template<class T> static __device__ __forceinline__ const StrataGlmKvWindow* strata_glm_kv_attention_window(const T* ptr) {
    const auto* p=reinterpret_cast<const unsigned char*>(ptr);
    const auto address=reinterpret_cast<uintptr_t>(p);
    for(int i=0;i<glm_kv_registry_count;++i) {
        const auto& v=glm_kv_registry[i];
        const auto start=reinterpret_cast<uintptr_t>(v.host);
        if(address>=start&&address-start<uint64_t(v.logical_bytes))
            return &glm_kv_registry[i];
    }
    return nullptr;
}
