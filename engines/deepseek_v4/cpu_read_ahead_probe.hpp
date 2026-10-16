#pragma once
#include "cpu_topology_win.hpp"
#include "mxfp4_cpu_service.hpp"
#include <immintrin.h>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>

namespace strata::deepseek_v4 {
using ReadClock = std::chrono::steady_clock;
inline double read_ms(ReadClock::time_point start) {
    return std::chrono::duration<double,std::milli>(ReadClock::now()-start).count();
}
struct ReadSpan { const unsigned char* data; size_t bytes; };
inline std::vector<ReadSpan> expert_spans(const Mxfp4CpuBank& bank, const std::vector<int32_t>& ids) {
    std::vector<ReadSpan> spans;
    for(int id:ids) for(auto* t:{bank.gate,bank.up,bank.down})
        spans.push_back({static_cast<const unsigned char*>(t->data)+id*t->nb[2],t->nb[2]});
    return spans;
}
inline void flush_spans(const std::vector<ReadSpan>& spans) {
    for(const auto& s:spans) for(size_t offset=0;offset<s.bytes;offset+=64)
        _mm_clflush(s.data+offset);
    _mm_mfence();
}
// Persistent test readers only. Production pool and its placement are unchanged.
class ReadAheadProbePool {
    struct alignas(64) Lane {uint64_t bytes=0,checksum=0;uint32_t l3_bytes=0;};
    std::vector<std::thread> workers_;
    std::vector<Lane> lanes_;
    std::atomic<int> ready_{0},done_{0};
    std::atomic<uint64_t> epoch_{0};
    std::atomic<bool> stop_{false},failed_{false};
    std::mutex mutex_;
    std::condition_variable wake_,finished_;
    std::vector<ReadSpan> spans_;
    ReadClock::time_point start_,deadline_;
    bool bounded_=false;
    static uint32_t l3_size() {
        int registers[4]{};
        for(int sub=0;sub<8;++sub) {
            __cpuidex(registers,0x8000001d,sub);
            if(!(registers[0]&31))break;
            if(((registers[0]>>5)&7)==3) {
                const uint64_t bytes=uint64_t((registers[1]&4095)+1)*
                    (((registers[1]>>12)&1023)+1)*(((registers[1]>>22)&1023)+1)*
                    (uint64_t(uint32_t(registers[2]))+1);
                return uint32_t(bytes);
            }
        }
        return 0;
    }
    void worker(int lane,CpuCoreBinding binding) {
        try {cpu_bind_current(binding);lanes_[lane].l3_bytes=l3_size();}
        catch(...) {failed_=true;++ready_;return;}
        ++ready_;uint64_t seen=0;
        while(true) {
            {
                std::unique_lock<std::mutex> lock(mutex_);
                wake_.wait(lock,[&]{return epoch_.load(std::memory_order_acquire)!=seen||stop_.load();});
                if(stop_)return;
                seen=epoch_.load(std::memory_order_acquire);
            }
            auto& result=lanes_[lane];result.bytes=0;
            __m512i sums[8];for(auto& sum:sums)sum=_mm512_setzero_si512();
            size_t total=0;for(const auto& span:spans_)total+=span.bytes;
            const size_t begin=total*lane/lanes_.size(),end=total*(lane+1)/lanes_.size();
            size_t prefix=0;bool expired=false;
            for(const auto& span:spans_) {
                const size_t first=begin>prefix?std::min(span.bytes,begin-prefix):0;
                const size_t last=end>prefix?std::min(span.bytes,end-prefix):0;
                for(size_t chunk=first;chunk<last;chunk+=65536) {
                    if(bounded_&&ReadClock::now()>=deadline_) {expired=true;break;}
                    const size_t limit=std::min(chunk+65536,last);
                    size_t offset=chunk;
                    for(;offset+512<=limit;offset+=512)for(int s=0;s<8;++s)
                        sums[s]=_mm512_xor_si512(sums[s],_mm512_loadu_si512(span.data+offset+s*64));
                    for(;offset+64<=limit;offset+=64)
                        sums[0]=_mm512_xor_si512(sums[0],_mm512_loadu_si512(span.data+offset));
                    result.bytes+=offset-chunk;
                }
                prefix+=span.bytes;if(expired)break;
            }
            for(int s=1;s<8;++s)sums[0]=_mm512_xor_si512(sums[0],sums[s]);
            alignas(64) uint64_t sum[8];_mm512_store_si512(sum,sums[0]);result.checksum^=sum[0];
            if(done_.fetch_add(1,std::memory_order_acq_rel)+1==int(lanes_.size())) {
                std::lock_guard<std::mutex> lock(mutex_);finished_.notify_one();
            }
        }
    }
public:
    explicit ReadAheadProbePool(const CpuCcd& ccd,int threads):lanes_(threads) {
        for(int i=0;i<threads;++i)workers_.emplace_back(&ReadAheadProbePool::worker,this,i,ccd.cores.at(i));
        while(ready_.load()!=threads)_mm_pause();
        if(failed_) { {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}wake_.notify_all();for(auto& w:workers_)w.join();throw std::runtime_error("Read-ahead test reader binding failed");}
    }
    ~ReadAheadProbePool() { {std::lock_guard<std::mutex> lock(mutex_);stop_=true;}wake_.notify_all();for(auto& w:workers_)w.join();}
    void start(const std::vector<ReadSpan>& spans,double window_ms=0) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            spans_=spans;bounded_=window_ms>0;done_=0;start_=ReadClock::now();
            deadline_=start_+std::chrono::duration_cast<ReadClock::duration>(std::chrono::duration<double,std::milli>(window_ms));
            epoch_.fetch_add(1,std::memory_order_release);
        }
        wake_.notify_all();
    }
    std::pair<double,uint64_t> wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        finished_.wait(lock,[&]{return done_.load(std::memory_order_acquire)==int(lanes_.size());});
        const double ms=read_ms(start_);uint64_t bytes=0;for(const auto& lane:lanes_)bytes+=lane.bytes;
        return {ms,bytes};
    }
    uint32_t l3_bytes() const {return lanes_.front().l3_bytes;}
};
}
