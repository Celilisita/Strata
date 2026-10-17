#pragma once
#include <cstdio>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#endif
namespace strata::deepseek_v4 {
// Cold allocation phases only; never called in per-token expert scheduling.
inline void runtime_memory(const char* stage){
#ifdef _WIN32
    MEMORYSTATUSEX memory{};memory.dwLength=sizeof(memory);
    if(!GlobalMemoryStatusEx(&memory))return;
    static HMODULE library=LoadLibraryW(L"psapi.dll");
    using Performance=BOOL(WINAPI*)(PPERFORMANCE_INFORMATION,DWORD);
    using Process=BOOL(WINAPI*)(HANDLE,PPROCESS_MEMORY_COUNTERS,DWORD);
    static const auto performance=reinterpret_cast<Performance>(GetProcAddress(library,"GetPerformanceInfo"));
    static const auto process=reinterpret_cast<Process>(GetProcAddress(library,"GetProcessMemoryInfo"));
    PERFORMANCE_INFORMATION p{};p.cb=sizeof(p);
    PROCESS_MEMORY_COUNTERS_EX own{};own.cb=sizeof(own);
    if(performance)performance(&p,sizeof(p));
    if(process)process(GetCurrentProcess(),reinterpret_cast<PPROCESS_MEMORY_COUNTERS>(&own),sizeof(own));
    std::fprintf(stderr,"strata_memory stage=%s available=%llu physical_total=%llu commit=%llu commit_limit=%llu rss=%llu private=%llu kernel_nonpaged=%llu\n",stage,
        (unsigned long long)memory.ullAvailPhys,(unsigned long long)memory.ullTotalPhys,
        (unsigned long long)(p.CommitTotal*p.PageSize),(unsigned long long)(p.CommitLimit*p.PageSize),
        (unsigned long long)own.WorkingSetSize,(unsigned long long)own.PrivateUsage,
        (unsigned long long)(p.KernelNonpaged*p.PageSize));
    std::fflush(stderr);
#else
    (void)stage;
#endif
}
} // namespace strata::deepseek_v4
