#include "expert_source_pin_backend.hpp"

#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
#include <cuda_runtime_api.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <dxgi1_4.h>
#include <cuda.h>
#include <cstring>
#endif
#endif

namespace strata::deepseek_v4 {

SourcePinBackendInfo source_pin_backend_info(const void* data, std::size_t bytes) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    unsigned flags = cudaHostRegisterPortable | cudaHostRegisterReadOnly;
#ifdef _WIN32
    // Only a file view originally allocated WRITECOPY may omit ReadOnly.
    // Windows reports materialized COW pages as READWRITE and MEM_MAPPED;
    // requiring current WRITECOPY incorrectly selects unsupported ReadOnly.
    // Retain ReadOnly for ordinary shared READWRITE/READONLY allocations.
    auto* cursor = static_cast<const char*>(data);
    std::size_t left = bytes;
    bool private_source = true;
    const void* allocation = nullptr;
    while (left) {
        MEMORY_BASIC_INFORMATION region{};
        if (!VirtualQuery(cursor, &region, sizeof(region)) || region.State != MEM_COMMIT ||
            region.Type != MEM_MAPPED || region.AllocationProtect != PAGE_WRITECOPY ||
            (region.Protect != PAGE_WRITECOPY && region.Protect != PAGE_READWRITE) ||
            !region.AllocationBase || (allocation && allocation != region.AllocationBase)) {
            private_source = false;
            break;
        }
        allocation = region.AllocationBase;
        const auto available = static_cast<std::size_t>(
            static_cast<const char*>(region.BaseAddress) + region.RegionSize - cursor);
        const auto step = available < left ? available : left;
        if (!step) { private_source = false; break; }
        cursor += step;
        left -= step;
    }
    if (private_source) flags = cudaHostRegisterPortable;
#endif
    return {true, flags};
#else
    (void) data; (void) bytes; return {false, 0};
#endif
}

bool register_readonly_source(void* data, std::size_t bytes, uint32_t& error, bool mapped) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    const auto flags = source_pin_backend_info(data, bytes).flags | (mapped ? cudaHostRegisterMapped : 0);
    const auto result = cudaHostRegister(data, bytes, flags);
    if (result == cudaSuccess) return true;
    error = static_cast<uint32_t>(result);
    (void) cudaGetLastError();
#else
    (void) data;
    (void) bytes;
    (void) mapped;
    error = 801; // cudaErrorNotSupported; CPU builds cannot register CUDA sources.
#endif
    return false;
}

void unregister_readonly_source(void* data) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    if (cudaHostUnregister(data) != cudaSuccess) (void) cudaGetLastError();
#else
    (void) data;
#endif
}

namespace {
bool source_registration_device(int device, uint32_t& error) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    int current=-1;
    const auto result=cudaGetDevice(&current);
    if(result!=cudaSuccess) {error=uint32_t(result);return false;}
    if(device<0||current!=device) {error=uint32_t(cudaErrorInvalidDevice);return false;}
    return true;
#else
    (void)device;error=801;return false;
#endif
}
}
bool synchronize_source_registration(int device, SourcePinApiResult& status) noexcept {
    status={};
    if(!source_registration_device(device,status.primary)){status.stage="selected_device";return false;}
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    status.primary=uint32_t(cudaDeviceSynchronize());
    status.pending=uint32_t(cudaPeekAtLastError());
#endif
    return status.primary==0;
}
bool unregister_source_checked(int device, void* data, SourcePinApiResult& status) noexcept {
    status={};
    if(!source_registration_device(device,status.primary)){status.stage="selected_device";return false;}
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    status.primary=uint32_t(cudaHostUnregister(data));
    status.pending=uint32_t(cudaPeekAtLastError());
#else
    (void)data;
#endif
    return status.primary==0; // Ownership follows this call, never the last-error slot.
}
bool register_source_checked(int device, void* data, std::size_t bytes, bool mapped, SourcePinApiResult& status) noexcept {
    status={};
    if(!source_registration_device(device,status.primary)){status.stage="selected_device";return false;}
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    const auto flags=source_pin_backend_info(data,bytes).flags|(mapped?cudaHostRegisterMapped:0);
    status.primary=uint32_t(cudaHostRegister(data,bytes,flags));
    status.pending=uint32_t(cudaPeekAtLastError());
#else
    (void)data;(void)bytes;(void)mapped;
#endif
    return status.primary==0;
}
uintptr_t source_alias_checked(int device, void* data, SourcePinApiResult& status) noexcept {
    status={};
    if(!source_registration_device(device,status.primary)){status.stage="selected_device";return 0;}
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    void* alias=nullptr;
    status.primary=uint32_t(cudaHostGetDevicePointer(&alias,data,0));
    status.pending=uint32_t(cudaPeekAtLastError());
    return status.primary==0?reinterpret_cast<uintptr_t>(alias):0;
#else
    (void)data;return 0;
#endif
}
bool clear_source_registration_error(int device, uint32_t expected, uint32_t& observed) noexcept {
    // Last-error consumption is host-thread local and does not select a device.
    (void)device;observed=0;
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    observed=uint32_t(cudaPeekAtLastError());
    if(!observed)return true;
    if((expected!=2&&expected!=801)||observed!=expected)return false; // Unknown stays pending.
    observed=uint32_t(cudaGetLastError());
    return observed==expected;
#else
    (void)expected;return false;
#endif
}

namespace {
#if defined(STRATA_DEEPSEEK_SOURCE_PIN_CUDA) && defined(_WIN32)
template<class T> struct SourceBudgetCom {
    T* p=nullptr;
    ~SourceBudgetCom(){if(p)p->Release();}
    void** out(){return reinterpret_cast<void**>(&p);}
};
#endif
}
bool query_source_alias_budget(int device, SourceAliasBudget& info) noexcept {
    info={};
#if defined(STRATA_DEEPSEEK_SOURCE_PIN_CUDA) && defined(_WIN32)
    info.required=true;info.stage="selected_device";
    if(!source_registration_device(device,info.error))return false;
    CUdevice selected;
    auto status=cuDeviceGet(&selected,device);
    if(status!=CUDA_SUCCESS){info.error=uint32_t(status);return false;}
    char bytes[8]{};unsigned mask=0;info.stage="selected_luid";
    status=cuDeviceGetLuid(bytes,&mask,selected);
    if(status!=CUDA_SUCCESS){info.error=uint32_t(status);return false;}
    if(!mask||(mask&(mask-1))){info.error=ERROR_INVALID_DATA;return false;}
    while(!(mask&(1u<<info.node)))++info.node;
    LUID luid{};static_assert(sizeof(luid)==sizeof(bytes));std::memcpy(&luid,bytes,sizeof(luid));
    info.luid_low=uint32_t(luid.LowPart);info.luid_high=uint32_t(luid.HighPart);
    // System32-only load; QueryVideoMemoryInfo reports the calling process.
    static HMODULE dxgi=LoadLibraryExW(L"dxgi.dll",nullptr,LOAD_LIBRARY_SEARCH_SYSTEM32);
    info.stage="system32_dxgi";
    if(!dxgi){info.error=GetLastError();return false;}
    using Create=HRESULT(WINAPI*)(REFIID,void**);
    auto create=reinterpret_cast<Create>(GetProcAddress(dxgi,"CreateDXGIFactory1"));
    if(!create){info.error=GetLastError();return false;}
    SourceBudgetCom<IDXGIFactory1> f1;info.stage="factory1";
    HRESULT result=create(__uuidof(IDXGIFactory1),f1.out());
    if(FAILED(result)){info.error=uint32_t(result);return false;}
    SourceBudgetCom<IDXGIFactory4> f4;info.stage="factory4";
    result=f1.p->QueryInterface(__uuidof(IDXGIFactory4),f4.out());
    if(FAILED(result)){info.error=uint32_t(result);return false;}
    SourceBudgetCom<IDXGIAdapter3> adapter;info.stage="exact_luid_adapter";
    result=f4.p->EnumAdapterByLuid(luid,__uuidof(IDXGIAdapter3),adapter.out());
    if(FAILED(result)){info.error=uint32_t(result);return false;}
    DXGI_ADAPTER_DESC1 desc{};info.stage="adapter_identity";
    result=adapter.p->GetDesc1(&desc);
    if(FAILED(result)){info.error=uint32_t(result);return false;}
    if(std::memcmp(&desc.AdapterLuid,&luid,sizeof(luid))){info.error=ERROR_INVALID_DATA;return false;}
    DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal{};info.stage="own_nonlocal_query";
    result=adapter.p->QueryVideoMemoryInfo(info.node,DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL,&nonlocal);
    if(FAILED(result)){info.error=uint32_t(result);return false;}
    info.budget=nonlocal.Budget;info.usage=nonlocal.CurrentUsage;info.stage="ok";
#else
    (void)device; // Linux/CPU: no DXGI requirement and no new admission failure.
#endif
    return true;
}

bool selected_source_cuda_uuid(int device, char* output, std::size_t capacity, uint32_t& error) noexcept {
    error=0;if(output&&capacity)output[0]='\0';
    if(!output||capacity<41){error=1;return false;}
    if(!source_registration_device(device,error))return false;
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    cudaDeviceProp properties{};
    const auto status=cudaGetDeviceProperties(&properties,device);
    if(status!=cudaSuccess){error=uint32_t(status);return false;}
    static constexpr char hex[]="0123456789abcdef";
    output[0]='G';output[1]='P';output[2]='U';output[3]='-';size_t at=4;
    for(int i=0;i<16;++i) {
        if(i==4||i==6||i==8||i==10)output[at++]='-';
        const auto byte=static_cast<unsigned char>(properties.uuid.bytes[i]);
        output[at++]=hex[byte>>4];output[at++]=hex[byte&15];
    }
    output[at]='\0';return true;
#else
    return false;
#endif
}
int source_alias_current_device() noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    int device = -1;
    if (cudaGetDevice(&device) == cudaSuccess) return device;
    (void) cudaGetLastError();
#endif
    return -1;
}

uintptr_t registered_source_alias(void* data, uint32_t& error) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    void* alias = nullptr;
    const auto result = cudaHostGetDevicePointer(&alias, data, 0);
    if (result == cudaSuccess) return reinterpret_cast<uintptr_t>(alias);
    error = static_cast<uint32_t>(result);
    (void) cudaGetLastError();
#else
    (void) data; error = 801;
#endif
    return 0;
}

bool drain_source_alias_readers(int device, uint32_t& error) noexcept {
#ifdef STRATA_DEEPSEEK_SOURCE_PIN_CUDA
    int prior = -1;
    auto result = cudaGetDevice(&prior);
    if (result == cudaSuccess && prior != device) result = cudaSetDevice(device);
    if (result == cudaSuccess) result = cudaDeviceSynchronize();
    if (prior >= 0 && prior != device) {
        const auto restored = cudaSetDevice(prior);
        if (result == cudaSuccess) result = restored;
    }
    if (result == cudaSuccess) return true;
    error = static_cast<uint32_t>(result);
    (void) cudaGetLastError();
#else
    (void) device; error = 801;
#endif
    return false;
}

} // namespace strata::deepseek_v4
