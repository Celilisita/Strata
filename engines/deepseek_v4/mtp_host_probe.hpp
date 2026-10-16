#pragma once
#include <cstdint>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif
namespace strata::deepseek_v4 {
struct MtpHostProbe {
    uint64_t wall_ns=0,process_faults=0,main_cpu_100ns=0,process_cpu_100ns=0,worker_cpu_100ns=0;
};
inline MtpHostProbe mtp_host_probe(uint64_t worker_cpu_100ns=0) {
    MtpHostProbe s{};s.worker_cpu_100ns=worker_cpu_100ns;
#ifdef _WIN32
    LARGE_INTEGER counter{},frequency{};QueryPerformanceCounter(&counter);QueryPerformanceFrequency(&frequency);
    const auto ticks=uint64_t(counter.QuadPart),hz=uint64_t(frequency.QuadPart);
    s.wall_ns=(ticks/hz)*1000000000ull+(ticks%hz)*1000000000ull/hz;
    PROCESS_MEMORY_COUNTERS_EX memory{};memory.cb=sizeof(memory);
    if(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&memory),sizeof(memory)))
        s.process_faults=memory.PageFaultCount;
    FILETIME created{},exited{},kernel{},user{};
    const auto file_ticks=[](const FILETIME& value){return (uint64_t(value.dwHighDateTime)<<32)|value.dwLowDateTime;};
    if(GetThreadTimes(GetCurrentThread(),&created,&exited,&kernel,&user))s.main_cpu_100ns=file_ticks(kernel)+file_ticks(user);
    if(GetProcessTimes(GetCurrentProcess(),&created,&exited,&kernel,&user))s.process_cpu_100ns=file_ticks(kernel)+file_ticks(user);
#endif
    return s;
}
}
