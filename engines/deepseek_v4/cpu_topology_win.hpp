#pragma once
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif
namespace strata::deepseek_v4 {
struct CpuCoreBinding {uint16_t group=0;uint8_t logical=0;uint64_t mask=0;};
struct CpuCcd {uint16_t group=0;uint64_t mask=0;uint32_t cache_bytes=0;std::vector<CpuCoreBinding> cores;};
inline std::vector<CpuCcd> cpu_ccd_topology(){
#ifdef _WIN32
    DWORD bytes=0;GetLogicalProcessorInformationEx(RelationAll,nullptr,&bytes);
    std::vector<unsigned char> storage(bytes);
    if(!bytes||!GetLogicalProcessorInformationEx(RelationAll,reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()),&bytes))throw std::runtime_error("CPU topology query failed");
    std::vector<CpuCcd> result;std::vector<GROUP_AFFINITY> cores;
    for(DWORD offset=0;offset<bytes;){auto* info=reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()+offset);
        if(!info->Size||offset+info->Size>bytes)throw std::runtime_error("CPU topology record invalid");
        if(info->Relationship==RelationCache&&info->Cache.Level==3)result.push_back({info->Cache.GroupMask.Group,uint64_t(info->Cache.GroupMask.Mask),info->Cache.CacheSize,{}});
        if(info->Relationship==RelationProcessorCore){if(info->Processor.GroupCount!=1)throw std::runtime_error("CPU core spans multiple groups");cores.push_back(info->Processor.GroupMask[0]);}
        offset+=info->Size;
    }
    std::sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.cache_bytes>b.cache_bytes;});
    for(auto& ccd:result){for(const auto& core:cores)if(core.Group==ccd.group&&(uint64_t(core.Mask)&ccd.mask)){
        if((uint64_t(core.Mask)&ccd.mask)!=uint64_t(core.Mask))throw std::runtime_error("Physical core spans CCD caches");
        uint8_t logical=0;while(logical<64&&!(uint64_t(core.Mask)&(1ull<<logical)))++logical;
        ccd.cores.push_back({core.Group,logical,1ull<<logical});}
        std::sort(ccd.cores.begin(),ccd.cores.end(),[](const auto& a,const auto& b){return a.logical<b.logical;});}
    return result;
#else
    return {};
#endif
}
inline void cpu_bind_current(const CpuCoreBinding& core){
#ifdef _WIN32
    GROUP_AFFINITY affinity{};affinity.Group=core.group;affinity.Mask=KAFFINITY(core.mask);
    if(!SetThreadGroupAffinity(GetCurrentThread(),&affinity,nullptr))throw std::runtime_error("SetThreadGroupAffinity failed");
    GROUP_AFFINITY actual{};if(!GetThreadGroupAffinity(GetCurrentThread(),&actual)||actual.Group!=core.group||uint64_t(actual.Mask)!=core.mask)throw std::runtime_error("Worker affinity verification failed");
#else
    (void)core;throw std::runtime_error("CPU group affinity requires Windows");
#endif
}
inline int cpu_current_logical(){
#ifdef _WIN32
    PROCESSOR_NUMBER p{};GetCurrentProcessorNumberEx(&p);return p.Group==0?int(p.Number):-1;
#else
    return -1;
#endif
}
}
