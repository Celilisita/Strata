#include <cuda/atomic>
#include <cuda_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
struct alignas(64) Packet {
    alignas(64) uint32_t request=0;
    alignas(64) uint32_t response=0;
    alignas(64) uint32_t failure=0;
};
using Word=cuda::atomic_ref<uint32_t,cuda::thread_scope_system>;
void check(cudaError_t error) {
    if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
}
__global__ void roundtrips(Packet* p,unsigned rounds,unsigned long long timeout_ticks) {
    if(threadIdx.x)return;
    for(unsigned i=1;i<=rounds;++i) {
        Word(p->request).store(i,cuda::memory_order_release);
        const auto start=clock64();
        while(Word(p->response).load(cuda::memory_order_acquire)!=i) {
            if(clock64()-start>timeout_ticks) {
                Word(p->failure).store(i,cuda::memory_order_release);return;
            }
            __nanosleep(64);
        }
    }
}
struct Sample {double gpu_ms,host_ms;};
Sample run(Packet* host,Packet* device,int clock_khz,unsigned rounds) {
    host->request=host->response=host->failure=0;
    cudaEvent_t start=nullptr,end=nullptr;
    check(cudaEventCreate(&start));check(cudaEventCreate(&end));
    std::atomic<bool> complete{false};
    std::thread responder([&] {
        for(unsigned i=1;i<=rounds;++i) {
            while(Word(host->request).load(cuda::memory_order_acquire)!=i) {
                if(complete.load(std::memory_order_relaxed)||
                   Word(host->failure).load(cuda::memory_order_acquire))return;
                _mm_pause();
            }
            Word(host->response).store(i,cuda::memory_order_release);
        }
    });
    const auto before=std::chrono::steady_clock::now();
    check(cudaEventRecord(start));
    roundtrips<<<1,32>>>(device,rounds,(unsigned long long)clock_khz*200);
    check(cudaGetLastError());
    check(cudaEventRecord(end));
    const auto sync=cudaEventSynchronize(end);
    complete.store(true,std::memory_order_release);
    responder.join();
    check(sync);
    const auto after=std::chrono::steady_clock::now();
    float elapsed=0;check(cudaEventElapsedTime(&elapsed,start,end));
    check(cudaEventDestroy(start));check(cudaEventDestroy(end));
    if(host->failure||host->response!=rounds)throw std::runtime_error("Mapped CPU/GPU handshake did not complete");
    return {double(elapsed),std::chrono::duration<double,std::milli>(after-before).count()};
}
double median(std::vector<double> x){std::sort(x.begin(),x.end());return x[x.size()/2];}
}
int main(int argc,char** argv)try {
    if(argc!=2)throw std::runtime_error("Expected Y-temp report path");
    int mapped=0,clock=0;
    check(cudaDeviceGetAttribute(&mapped,cudaDevAttrCanMapHostMemory,0));
    check(cudaDeviceGetAttribute(&clock,cudaDevAttrClockRate,0));
    if(!mapped||clock<1)throw std::runtime_error("Mapped host signal unavailable");
    Packet* host=nullptr,*device=nullptr;
    check(cudaHostAlloc(reinterpret_cast<void**>(&host),sizeof(Packet),cudaHostAllocMapped));
    check(cudaHostGetDevicePointer(reinterpret_cast<void**>(&device),host,0));
    constexpr unsigned rounds=512;
    run(host,device,clock,rounds);
    std::vector<double> gpu,wall;
    for(int i=0;i<9;++i){const auto s=run(host,device,clock,rounds);gpu.push_back(s.gpu_ms*1000/rounds);wall.push_back(s.host_ms*1000/rounds);}
    check(cudaFreeHost(host));
    std::ofstream out(argv[1]);
    out<<std::setprecision(12)<<"{\"whole_model_loaded\":false,\"mapped_host_signal\":true,"
       <<"\"one_GPU_request_and_one_CPU_response_each_round\":true,\"rounds_per_sample\":"<<rounds
       <<",\"samples\":"<<gpu.size()<<",\"gpu_event_roundtrip_us\":[";
    for(size_t i=0;i<gpu.size();++i){if(i)out<<',';out<<gpu[i];}
    out<<"],\"gpu_event_median_us\":"<<median(gpu)<<",\"host_wall_median_us\":"<<median(wall)
       <<",\"passed\":true}\n";
    if(!out)throw std::runtime_error("Signal probe report write failed");
    return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
