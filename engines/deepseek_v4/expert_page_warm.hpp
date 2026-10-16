#pragma once
#include <chrono>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace strata::deepseek_v4 {
struct ExpertWarmRange {const uint8_t* data;size_t bytes;};
struct ExpertWarmMetrics {uint64_t bytes=0,pages=0,locked_bytes=0,lock_failures=0,checksum=0;
    uint64_t working_set_requested_bytes=0,working_set_error=0;bool working_set_adjusted=false;
    double read_ms=0,lock_ms=0;};
// Keeps successfully locked read-only GGUF pages until the model is released.
// Failed locks still get a full page touch; no extra copy of model weights.
class ExpertPageWarm {
    struct Lock {void* data;size_t bytes;};
    std::vector<Lock> locks_;
    std::vector<ExpertWarmRange> read_ranges_,lock_ranges_;
    ExpertWarmMetrics metrics_;
#ifdef _WIN32
    SIZE_T original_working_set_min_=0,original_working_set_max_=0;
    bool working_set_changed_=false;
#endif
public:
    explicit ExpertPageWarm(std::vector<ExpertWarmRange> ranges):
        read_ranges_(ranges),lock_ranges_(std::move(ranges)){}
    ExpertPageWarm(std::vector<ExpertWarmRange> read_ranges,std::vector<ExpertWarmRange> lock_ranges):
        read_ranges_(std::move(read_ranges)),lock_ranges_(std::move(lock_ranges)){}
    ExpertPageWarm(const ExpertPageWarm&)=delete;
    ExpertPageWarm& operator=(const ExpertPageWarm&)=delete;
    ~ExpertPageWarm(){
#ifdef _WIN32
        for(const auto& lock:locks_)VirtualUnlock(lock.data,lock.bytes);
        if(working_set_changed_)SetProcessWorkingSetSize(GetCurrentProcess(),original_working_set_min_,original_working_set_max_);
#endif
    }
    void read_pages(){
        using Clock=std::chrono::steady_clock;const auto begin=Clock::now();
#ifdef _WIN32
        SYSTEM_INFO info{};GetSystemInfo(&info);const size_t page=info.dwPageSize;
#else
        const long size=sysconf(_SC_PAGESIZE);const size_t page=size>0?size_t(size):4096;
#endif
        for(const auto& range:read_ranges_){
            metrics_.bytes+=range.bytes;
            const volatile uint8_t* data=range.data;
            for(size_t i=0;i<range.bytes;i+=page){metrics_.checksum+=data[i];++metrics_.pages;}
        }
        metrics_.read_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    }
    void lock_pages(){
        using Clock=std::chrono::steady_clock;const auto begin=Clock::now();
#ifdef _WIN32
        SYSTEM_INFO info{};GetSystemInfo(&info);const uintptr_t page=info.dwPageSize;
        constexpr size_t chunk=16ull<<20;
        // Keep the late trunk layers and MTP source resident first. Other
        // mapped pages are read during load and can use the available RAM.
        constexpr size_t lock_budget=24ull<<30;
        size_t requested=0;
        for(auto it=lock_ranges_.rbegin();it!=lock_ranges_.rend()&&requested<lock_budget;++it)
            requested+=std::min(it->bytes,lock_budget-requested);
        if(requested){
            const HANDLE process=GetCurrentProcess();
            metrics_.working_set_requested_bytes=requested+(512ull<<20);
            if(GetProcessWorkingSetSize(process,&original_working_set_min_,&original_working_set_max_)&&
                SetProcessWorkingSetSize(process,metrics_.working_set_requested_bytes,192ull<<30)){
                working_set_changed_=true;metrics_.working_set_adjusted=true;
            }else metrics_.working_set_error=GetLastError();
        }
        size_t attempted=0;
        for(auto it=lock_ranges_.rbegin();it!=lock_ranges_.rend()&&attempted<lock_budget;++it){
            const auto& range=*it;
            const auto start=reinterpret_cast<uintptr_t>(range.data);
            const auto first=(start+page-1)/page*page;
            const auto last=(start+range.bytes)/page*page;
            for(auto cursor=first;cursor<last&&attempted<lock_budget;){
                const size_t size=std::min<size_t>({chunk,last-cursor,lock_budget-attempted});
                if(VirtualLock(reinterpret_cast<void*>(cursor),size)){
                    locks_.push_back({reinterpret_cast<void*>(cursor),size});metrics_.locked_bytes+=size;
                }else ++metrics_.lock_failures;
                attempted+=size;
                cursor+=size;
            }
        }
#endif
        metrics_.lock_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
    }
    const ExpertWarmMetrics& metrics()const{return metrics_;}
};
}
