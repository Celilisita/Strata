#include "cpu_weight_fixture.hpp"
#include "mxfp4_cpu_service.hpp"
#include <cuda_runtime_api.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using namespace strata::deepseek_v4;
namespace {
void checked(cudaError_t code){if(code!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(code));}
int deadline_probe(const CpuWeightFixture& fixture,Mxfp4CpuService& pool,const char* report){
    int32_t ids[12]{};const uint32_t masks[2]{3,0};
    for(int id=0;id<64;++id){ids[0]=id;const uint32_t one[2]{1,0};const int32_t select[2]{id,-1};
        pool.read_ahead_pair(fixture.bank,ids,one,0,select);}
    struct Result{double window;int selected,complete,first_complete;uint64_t bytes;double median_us;};
    std::vector<Result> results;std::array<int,32> order{};std::iota(order.begin(),order.end(),0);
    for(double window:{.35,.55})for(int count:{1,2}){
        std::shuffle(order.begin(),order.end(),std::mt19937(unsigned(window*1000)+unsigned(count)));
        int complete=0,first_complete=0;uint64_t bytes=0;std::vector<double> elapsed;
        for(int index:order){ids[0]=2*index;ids[1]=2*index+1;const int32_t select[2]{ids[0],count==2?ids[1]:-1};
            bool finished=false;uint64_t first_bytes=0;const auto start=std::chrono::steady_clock::now();
            const auto touched=pool.read_ahead_pair(fixture.bank,ids,masks,window,select,nullptr,nullptr,&finished,&first_bytes);
            elapsed.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());
            bytes+=touched;complete+=finished&&touched==uint64_t(count)*13369344ull;
            first_complete+=first_bytes==13369344ull;}
        std::sort(elapsed.begin(),elapsed.end());results.push_back({window,count,complete,first_complete,bytes,elapsed[elapsed.size()/2]});}
    std::ofstream out(report);out<<"{\"whole_model_loaded\":false,\"real_mxfp4_experts\":true,\"workers\":8,\"trials_each\":32,\"cases\":[";
    for(size_t i=0;i<results.size();++i){const auto& r=results[i];if(i)out<<',';
        out<<"{\"window_ms\":"<<r.window<<",\"selected_experts\":"<<r.selected<<",\"fully_completed\":"<<r.complete
           <<",\"first_expert_completed\":"<<r.first_complete
           <<",\"read_bytes\":"<<r.bytes<<",\"median_wall_us\":"<<r.median_us<<'}';}
    out<<"]}";return out?0:2;
}
}
int main(int argc,char** argv)try{
    if(argc!=3&&argc!=4)throw std::runtime_error("Expected MXFP4 shard and report");
    CpuWeightFixture fixture(argv[1]);Mxfp4CpuService pool(8);
    if(argc==4&&std::string(argv[3])=="--deadline")return deadline_probe(fixture,pool,argv[2]);
    const int32_t ids[12]{0,1,2,3,4,5,0,1,2,3,4,5};const uint32_t masks[2]{63,0};const int32_t selected[2]{0,1};
    pool.read_ahead_pair(fixture.bank,ids,masks,0,selected);
    void* device=nullptr;cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t execution=nullptr;
    checked(cudaMalloc(&device,4096));checked(cudaStreamCreate(&stream));checked(cudaGraphCreate(&graph,0));
    cudaMemsetParams fill{};fill.dst=device;fill.pitch=4096;fill.value=0;fill.elementSize=4;fill.width=1024;fill.height=1;
    cudaGraphNode_t node;checked(cudaGraphAddMemsetNode(&node,graph,nullptr,0,&fill));
    checked(cudaGraphInstantiate(&execution,graph,0));
    std::atomic<bool> reading{false},stop{false};std::atomic<uint64_t> bytes{0};
    std::thread reader([&]{while(!stop.load(std::memory_order_acquire)){
        if(reading.load(std::memory_order_acquire))
            bytes.fetch_add(pool.read_ahead_pair(fixture.bank,ids,masks,.6,selected),std::memory_order_relaxed);
        else std::this_thread::sleep_for(std::chrono::milliseconds(1));}});
    auto phase=[&](int samples){std::vector<double> elapsed;elapsed.reserve(samples);
        for(int i=0;i<samples;++i){const auto start=std::chrono::steady_clock::now();
            checked(cudaGraphLaunch(execution,stream));checked(cudaStreamSynchronize(stream));
            elapsed.push_back(std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count());}
        std::sort(elapsed.begin(),elapsed.end());return std::array<double,3>{elapsed[samples/2],elapsed[size_t(samples*.95)],
            std::accumulate(elapsed.begin(),elapsed.end(),0.)/samples};};
    const auto idle=phase(1000);reading.store(true,std::memory_order_release);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));const auto busy=phase(1000);
    std::array<double,3> pinned{};bool pinned_valid=false;
    std::array<int,5> cores{8,10,16,18,30};std::array<std::array<double,3>,5> pinned_cases{};
#ifdef _WIN32
    for(size_t i=0;i<cores.size();++i){
        const DWORD_PTR prior=SetThreadAffinityMask(GetCurrentThread(),DWORD_PTR(1)<<cores[i]);
        if(prior){pinned_cases[i]=phase(1000);pinned_valid=true;SetThreadAffinityMask(GetCurrentThread(),prior);}
    }
    pinned=pinned_cases.back();
#endif
    const auto active_after=phase(1000);
    reading.store(false,std::memory_order_release);stop.store(true,std::memory_order_release);reader.join();const auto idle_after=phase(1000);
    checked(cudaGraphExecDestroy(execution));checked(cudaGraphDestroy(graph));checked(cudaStreamDestroy(stream));checked(cudaFree(device));
    std::ofstream out(argv[2]);out<<"{\"whole_model_loaded\":false,\"cpu_workers\":8,\"worker_affinity\":\"unbound\",\"graph_nodes\":1,\"samples_per_phase\":1000,\"read_bytes\":"<<bytes.load()
        <<",\"idle_median_us\":"<<idle[0]<<",\"active_median_us\":"<<busy[0]<<",\"pinned_core30_median_us\":"<<(pinned_valid?pinned[0]:-1)
        <<",\"active_after_median_us\":"<<active_after[0]<<",\"idle_after_median_us\":"<<idle_after[0]
        <<",\"idle_p95_us\":"<<idle[1]<<",\"active_p95_us\":"<<busy[1]
        <<",\"pinned_p95_us\":"<<(pinned_valid?pinned[1]:-1)<<",\"single_main_thread_pinning_tested\":"<<(pinned_valid?"true":"false")<<",\"pin_cases\":[";
    for(size_t i=0;i<cores.size();++i){if(i)out<<',';out<<"{\"logical_core\":"<<cores[i]
        <<",\"median_us\":"<<pinned_cases[i][0]<<",\"p95_us\":"<<pinned_cases[i][1]<<'}';}
    out<<"]}";
    return out?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
