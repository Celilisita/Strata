#include "cpu_topology_win.hpp"
#include "cpu_usage_win.hpp"
#include <immintrin.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <thread>
#include <string>
using namespace strata::deepseek_v4;
#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((noinline))
#endif
// Eight independent streams and accumulators keep enough loads in flight.
NOINLINE uint64_t read_streams(const unsigned char* p,size_t bytes){
    const size_t stream=bytes/8;__m512i sums[8];for(auto& sum:sums)sum=_mm512_setzero_si512();
    for(size_t offset=0;offset<stream;offset+=64)for(int s=0;s<8;++s)sums[s]=_mm512_xor_si512(sums[s],_mm512_load_si512(p+s*stream+offset));
    for(int s=1;s<8;++s)sums[0]=_mm512_xor_si512(sums[0],sums[s]);alignas(64) uint64_t out[8];_mm512_store_si512(out,sums[0]);return out[0];
}
int main(int argc,char** argv)try{
    if(argc!=3)throw std::runtime_error("Expected power-plan label and report path");
    const auto topology=cpu_ccd_topology();if(topology.size()!=2||topology[0].group||topology[1].group||topology[0].cores.size()!=8||topology[1].cores.size()!=8)throw std::runtime_error("Expected two8-core CCDs in group0");
    constexpr size_t bytes=4ull<<30;auto* memory=static_cast<unsigned char*>(VirtualAlloc(nullptr,bytes,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));if(!memory)throw std::runtime_error("4GiB stream buffer unavailable");
    std::memset(memory,0x5a,bytes);std::ofstream out(argv[2]);out<<"{\"whole_model_loaded\":false,\"power_plan\":\""<<argv[1]<<"\",\"working_set_bytes\":"<<bytes<<",\"streams_per_thread\":8,\"ccds\":[";
    for(size_t c=0;c<topology.size();++c){if(c)out<<',';out<<"{\"ccd\":"<<c<<",\"l3_bytes\":"<<topology[c].cache_bytes<<",\"logical_mask\":"<<topology[c].mask<<",\"physical_core_logicals\":[";for(size_t i=0;i<8;++i){if(i)out<<',';out<<int(topology[c].cores[i].logical);}out<<"]}";}out<<"],\"cases\":[";
    for(int test=0;test<3;++test){std::vector<CpuCoreBinding> bindings;if(test<2)bindings=topology[test].cores;else{bindings=topology[0].cores;bindings.insert(bindings.end(),topology[1].cores.begin(),topology[1].cores.end());}
        std::atomic<int> ready{0};std::atomic<bool> go{false},stop{false},failed{false};std::vector<std::thread> workers;
        struct alignas(64) Result{uint64_t passes=0,checksum=0;int first=-1,last=-1;};std::vector<Result> results(bindings.size());
        for(size_t lane=0;lane<bindings.size();++lane)workers.emplace_back([&,lane]{try{cpu_bind_current(bindings[lane]);}catch(...){failed=true;++ready;return;}results[lane].first=cpu_current_logical();++ready;while(!go.load(std::memory_order_acquire))_mm_pause();
            const size_t part=bytes/bindings.size();while(!stop.load(std::memory_order_relaxed)){results[lane].checksum^=read_streams(memory+lane*part,part);++results[lane].passes;}results[lane].last=cpu_current_logical();});
        while(ready.load()!=int(bindings.size()))_mm_pause();if(failed){go=true;stop=true;for(auto& w:workers)w.join();throw std::runtime_error("CCD binding failed");}
        auto first=cpu_usage_snapshot();std::array<uint64_t,64> parked{};uint64_t samples=0;const auto start=std::chrono::steady_clock::now();go.store(true,std::memory_order_release);
        while(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()<2){Sleep(100);auto snap=cpu_usage_snapshot();for(size_t i=0;i<snap.count;++i)parked[i]+=snap.parked[i];++samples;}
        stop=true;for(auto& w:workers)w.join();const auto seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();const auto last=cpu_usage_snapshot();uint64_t read_bytes=0;for(const auto& r:results)read_bytes+=r.passes*(bytes/bindings.size());
        if(test)out<<',';out<<"{\"case\":\""<<(test==0?"ccd0_8":test==1?"ccd1_8":"both_8plus8")<<"\",\"read_bytes\":"<<read_bytes<<",\"seconds\":"<<seconds<<",\"gbps\":"<<read_bytes/seconds/1e9<<",\"workers\":[";
        for(size_t i=0;i<results.size();++i){if(i)out<<',';out<<"{\"bound_logical\":"<<int(bindings[i].logical)<<",\"first_logical\":"<<results[i].first<<",\"last_logical\":"<<results[i].last<<",\"passes\":"<<results[i].passes<<'}';}out<<"],\"logical_usage\":";cpu_usage_json(out,first,last,parked,samples);out<<'}';out.flush();
    }out<<"],\"passed\":true}";if(!out)throw std::runtime_error("CCD bandwidth receipt write failed");VirtualFree(memory,0,MEM_RELEASE);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
