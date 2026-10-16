#include "device_cache_budget.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#ifdef STRATA_DEEPSEEK_DEVICE_CACHE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#endif
#endif
namespace strata::deepseek_v4 {
DeviceMemoryBudget device_memory_budget(ggml_backend_t backend) {
    DeviceMemoryBudget result;
#ifdef STRATA_DEEPSEEK_DEVICE_CACHE_CUDA
    auto* device=ggml_backend_get_device(backend);auto* reg=ggml_backend_dev_backend_reg(device);
    if(std::strcmp(ggml_backend_reg_name(reg),"CUDA"))return result;
    int ordinal=-1;for(size_t i=0;i<ggml_backend_reg_dev_count(reg);++i)
        if(ggml_backend_reg_dev_get(reg,i)==device)ordinal=int(i);
    int previous=-1;
    if(ordinal<0||cudaGetDevice(&previous)!=cudaSuccess||cudaSetDevice(ordinal)!=cudaSuccess)
        throw std::runtime_error("memory budget CUDA device selection failed");
    size_t free=0,total=0;const auto status=cudaMemGetInfo(&free,&total);
    if(previous!=ordinal)cudaSetDevice(previous);
    if(status!=cudaSuccess)throw std::runtime_error("memory budget CUDA query failed");
    result.cuda_free=free;result.cuda_total=total;
#ifdef _WIN32
    char luid[8]{};unsigned node_mask=0;CUdevice cu_device;
    if(cuDeviceGet(&cu_device,ordinal)!=CUDA_SUCCESS||cuDeviceGetLuid(luid,&node_mask,cu_device)!=CUDA_SUCCESS)return result;
    // Match the selected CUDA adapter by LUID, including CUDA_VISIBLE_DEVICES.
    // LoadLibraryEx searches System32; no task DLL or temporary file is loaded.
    static HMODULE dxgi=LoadLibraryExW(L"dxgi.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    using Factory=HRESULT(WINAPI*)(REFIID,void**);
    const auto create=dxgi?reinterpret_cast<Factory>(GetProcAddress(dxgi,"CreateDXGIFactory1")):nullptr;
    IDXGIFactory1* factory=nullptr;
    if(!create||FAILED(create(__uuidof(IDXGIFactory1),reinterpret_cast<void**>(&factory))))return result;
    for(UINT i=0;;++i){IDXGIAdapter1* adapter=nullptr;if(factory->EnumAdapters1(i,&adapter)==DXGI_ERROR_NOT_FOUND)break;
        if(!adapter)break;DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);
        if(!std::memcmp(&desc.AdapterLuid,luid,sizeof(luid))){IDXGIAdapter3* budget=nullptr;
            if(SUCCEEDED(adapter->QueryInterface(__uuidof(IDXGIAdapter3),reinterpret_cast<void**>(&budget)))){
                UINT node=0;while(node<31&&!(node_mask&(1u<<node)))++node;
                DXGI_QUERY_VIDEO_MEMORY_INFO local{},nonlocal{};
                if(SUCCEEDED(budget->QueryVideoMemoryInfo(node,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&local))&&
                   SUCCEEDED(budget->QueryVideoMemoryInfo(node,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nonlocal))){
                    result.wddm_available=true;result.local_budget=local.Budget;result.local_usage=local.CurrentUsage;result.nonlocal_usage=nonlocal.CurrentUsage;}
                budget->Release();}
            adapter->Release();break;}adapter->Release();}
    factory->Release();
#endif
#else
    (void)backend;
#endif
    return result;
}
void trace_device_memory_budget(ggml_backend_t backend,const char* phase,int slots) {
    const char* enabled=std::getenv("STRATA_DSV4_MEMORY_BUDGET_TRACE");if(!enabled||std::strcmp(enabled,"1"))return;
    const auto b=device_memory_budget(backend);
    std::fprintf(stderr,"strata_memory_budget phase=%s cache_slots=%d cuda_free=%llu local_budget=%llu local_usage=%llu nonlocal_usage=%llu wddm=%d\n",
        phase,slots,(unsigned long long)b.cuda_free,(unsigned long long)b.local_budget,(unsigned long long)b.local_usage,(unsigned long long)b.nonlocal_usage,int(b.wddm_available));
}
int pp_cache_slots_with_headroom(int slots) {
    const char* setting=std::getenv("STRATA_DSV4_PP_CACHE_HEADROOM_MIB");if(!setting)return slots;
    char* end=nullptr;const auto mib=std::strtoul(setting,&end,10);
    if(!*setting||*end||mib>2048)throw std::runtime_error("PP cache headroom must be0..2048 MiB");
    constexpr uint64_t expert_bytes=13369344;
    const int retire=int(((uint64_t(mib)<<20)+expert_bytes-1)/expert_bytes);
    if(slots-retire<6)throw std::runtime_error("PP headroom exceeds expert cache capacity");
    return slots-retire;
}
}
