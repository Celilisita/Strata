#pragma once
#include <chrono>
#include <algorithm>
#include <cstdlib>
#include <fstream>
#include <string>
#include <cstdint>
#include <stdexcept>
namespace strata::deepseek_v4 {
// Optional CPU API timing. CUDA event spans include gaps between submissions;
// these counters distinguish driver enqueue stalls from GPU kernel execution.
struct PpHostTrace {
    std::string path=[] {const char* value=std::getenv("STRATA_DSV4_PP_HOST_TRACE");return value?std::string(value):std::string();}();
    uint64_t product_ns=0,copy_ns=0,product_max_ns=0,copy_max_ns=0,products=0,copies=0,epoch=0;
    template<class F> void product(F f){if(path.empty()){f();return;}auto start=std::chrono::steady_clock::now();f();
        const auto ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());product_ns+=ns;product_max_ns=std::max(product_max_ns,ns);++products;}
    template<class F> void copy(F f){if(path.empty()){f();return;}auto start=std::chrono::steady_clock::now();f();
        const auto ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count());copy_ns+=ns;copy_max_ns=std::max(copy_max_ns,ns);++copies;}
    void save(uint64_t bytes,uint64_t reused){if(path.empty())return;std::ofstream out(path,std::ios::app);
        out<<"{\"epoch\":"<<++epoch<<",\"host_product_enqueue_ns\":"<<product_ns<<",\"host_copy_enqueue_ns\":"<<copy_ns
           <<",\"max_product_enqueue_ns\":"<<product_max_ns<<",\"max_copy_enqueue_ns\":"<<copy_max_ns
           <<",\"products\":"<<products<<",\"copies\":"<<copies<<",\"h2d_bytes\":"<<bytes<<",\"gpu_reused_bytes\":"<<reused<<"}\n";
        if(!out)throw std::runtime_error("PP host enqueue timing receipt failed");product_ns=copy_ns=product_max_ns=copy_max_ns=products=copies=0;}
};
}
