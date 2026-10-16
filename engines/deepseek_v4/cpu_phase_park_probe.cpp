#include "mxfp4_cpu_service.hpp"
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <thread>
using strata::deepseek_v4::Mxfp4CpuService;
int main(){
#ifdef _WIN32
    if(_putenv_s("STRATA_DSV4_CPU_SPIN_WAIT","1"))return 2;
#else
    if(setenv("STRATA_DSV4_CPU_SPIN_WAIT","1",1))return 2;
#endif
    Mxfp4CpuService pool(8);
    pool.set_spin_active(false);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    const auto parked_before=pool.worker_cpu_100ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto parked=pool.worker_cpu_100ns()-parked_before;
    pool.set_spin_active(true);
    const auto spin_before=pool.worker_cpu_100ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(250));
    const auto spinning=pool.worker_cpu_100ns()-spin_before;
    pool.set_spin_active(false);
    std::cout<<"{\"parked_worker_cpu_ms\":"<<parked/10000.0
             <<",\"spinning_worker_cpu_ms\":"<<spinning/10000.0
             <<",\"passed\":"<<(spinning>parked+1000000?"true":"false")<<"}\n";
    return spinning>parked+1000000?0:1;
}
