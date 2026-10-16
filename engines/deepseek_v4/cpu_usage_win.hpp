#pragma once
#include "cpu_topology_win.hpp"
#include <array>
#include <ostream>
namespace strata::deepseek_v4 {
struct CpuUsageSnapshot {std::array<uint64_t,64> idle{},total{};std::array<bool,64> parked{};size_t count=0;};
inline CpuUsageSnapshot cpu_usage_snapshot(){
    CpuUsageSnapshot out;
#ifdef _WIN32
    struct Perf {LARGE_INTEGER idle,kernel,user,dpc,interrupt;ULONG interrupts;};
    using Query=LONG(WINAPI*)(ULONG,PVOID,ULONG,PULONG);
    const auto query=reinterpret_cast<Query>(GetProcAddress(GetModuleHandleW(L"ntdll.dll"),"NtQuerySystemInformation"));
    std::array<Perf,64> values{};ULONG bytes=0;
    if(!query||query(8,values.data(),ULONG(sizeof(values)),&bytes)<0||bytes%sizeof(Perf))throw std::runtime_error("Per-core performance query failed");
    out.count=bytes/sizeof(Perf);if(out.count>64)throw std::runtime_error("CPU usage snapshot bound exceeded");
    for(size_t i=0;i<out.count;++i){out.idle[i]=uint64_t(values[i].idle.QuadPart);out.total[i]=uint64_t(values[i].kernel.QuadPart+values[i].user.QuadPart);}
    ULONG size=0;GetSystemCpuSetInformation(nullptr,0,&size,nullptr,0);std::vector<unsigned char> storage(size);
    if(!size||!GetSystemCpuSetInformation(reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage.data()),size,&size,nullptr,0))throw std::runtime_error("CPU parking query failed");
    for(ULONG offset=0;offset<size;){auto* s=reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(storage.data()+offset);
        if(!s->Size||offset+s->Size>size)throw std::runtime_error("CPU set record invalid");
        if(s->Type==CpuSetInformation&&s->CpuSet.Group==0&&s->CpuSet.LogicalProcessorIndex<64)out.parked[s->CpuSet.LogicalProcessorIndex]=s->CpuSet.Parked!=0;
        offset+=s->Size;}
#endif
    return out;
}
inline void cpu_usage_json(std::ostream& out,const CpuUsageSnapshot& first,const CpuUsageSnapshot& last,
    const std::array<uint64_t,64>& parked_samples,uint64_t samples){
    if(first.count!=last.count)throw std::runtime_error("CPU topology changed during measurement");out<<'[';
    for(size_t i=0;i<first.count;++i){const auto total=last.total[i]-first.total[i],idle=last.idle[i]-first.idle[i];if(i)out<<',';
        out<<"{\"logical\":"<<i<<",\"busy_percent\":"<<(total?100.0*double(total-std::min(total,idle))/total:0)<<",\"parked_at_start\":"<<(first.parked[i]?"true":"false")<<",\"parked_sample_percent\":"<<(samples?100.0*parked_samples[i]/samples:0)<<'}';}out<<']';
}
}
