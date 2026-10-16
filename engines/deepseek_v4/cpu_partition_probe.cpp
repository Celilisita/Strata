#include "mtp_sidecar.hpp"
#include "mxfp4_cpu_service.hpp"
#include "mxfp4_fast_dot.hpp"
#include "checkpoint.hpp"
#include "benchmark_fault_guard.hpp"
#include "llama.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <thread>
#include <string>
using namespace strata::deepseek_v4;
static void cost_setting(int cost){
#ifdef _WIN32
    _putenv_s("STRATA_DSV4_CPU_SPIN_WAIT","0");_putenv_s("STRATA_DSV4_CPU_VNNI","1");_putenv_s("STRATA_DSV4_CPU_DUAL_COST_PCT",std::to_string(cost).c_str());
#else
    setenv("STRATA_DSV4_CPU_SPIN_WAIT","0",1);setenv("STRATA_DSV4_CPU_VNNI","1",1);setenv("STRATA_DSV4_CPU_DUAL_COST_PCT",std::to_string(cost).c_str(),1);
#endif
}
int main(int argc,char** argv)try{
    benchmark_fault_guard();if(argc!=4)throw std::runtime_error("Expected MTP GGUF, activation directory and report path");
    llama_backend_init();if(!mxfp4_fast_dot_supported())throw std::runtime_error("VNNI unavailable");
    auto* device=ggml_backend_dev_by_name("CUDA0");if(!device)throw std::runtime_error("CUDA0 missing");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    MtpSidecar mtp(argv[1],backend.get());const auto fixture=read_checkpoint_data(argv[2],"ffn_norm-0");
    if(fixture.metadata.type!=GGML_TYPE_F32||fixture.data.size()<8192*sizeof(float))throw std::runtime_error("Real activation pair missing");
    std::array<float,8192> x;std::memcpy(x.data(),fixture.data.data(),sizeof(x));
    Mxfp4CpuBank bank{mtp.tensor("blk.43.ffn_gate_exps.weight"),mtp.tensor("blk.43.ffn_up_exps.weight"),mtp.tensor("blk.43.ffn_down_exps.weight"),10};
    uint64_t checksum=0;for(auto* t:{bank.gate,bank.up,bank.down}){const auto* p=static_cast<const volatile uint8_t*>(t->data);for(size_t i=0;i<ggml_nbytes(t);i+=4096)checksum+=p[i];}
    std::vector<std::vector<float>> reference(6*11);std::vector<float> actual(49152),single(3*4096),single_reference;bool passed=true;
    std::ofstream out(argv[3]);out<<"{\"whole_model_loaded\":false,\"baseline\":\"current8-worker VNNI200-percent dual-cost pool\",\"only_one_pool_alive\":true,\"passes\":[";
    // Sequential ABCCBA uses one pool at a time; all pools keep normal condition-variable waits.
    const int costs[]{200,110,100,100,110,200};
    for(int pass=0;pass<6;++pass){const int cost=costs[pass];cost_setting(cost);Mxfp4CpuService pool(8);
        const int32_t single_ids[]{17,63,191};pool.compute(bank,x.data(),single_ids,3,single.data());
        if(!pass)single_reference=single;else for(size_t i=0;i<single.size();++i)passed&=std::isfinite(single[i])&&single[i]==single_reference[i];
        if(pass)out<<',';out<<"{\"dual_cost_percent\":"<<cost<<",\"cases\":[";
        for(int test=0;test<6;++test){const int common=(test%3)*3;const bool partial=test>=3;std::array<double,9> times{};double worst=0;
            for(int repetition=0;repetition<11;++repetition){std::array<int32_t,12> ids;for(int rank=0;rank<6;++rank){ids[rank]=(repetition*23+rank)%256;ids[rank+6]=(repetition*23+(rank<common?rank:rank+6))%256;}
                const uint32_t masks[]{partial?7u:63u,partial?14u:63u};
                std::this_thread::sleep_for(std::chrono::microseconds(250)); // Representative GPU work between CPU requests; excluded from timing.
                const auto begin=std::chrono::steady_clock::now();pool.compute_pair(bank,x.data(),ids.data(),masks,actual.data());
                if(repetition>=2)times[repetition-2]=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
                auto& expected=reference[test*11+repetition];if(!pass)expected=actual;
                double error=0,norm=0;for(size_t i=0;i<actual.size();++i){if(!std::isfinite(actual[i]))throw std::runtime_error("Nonfinite contribution");const double d=double(actual[i])-expected[i];error+=d*d;norm+=double(expected[i])*expected[i];}
                worst=std::max(worst,std::sqrt(error/std::max(norm,1e-30)));passed&=worst<=1e-6;
            }
            std::sort(times.begin(),times.end());if(test)out<<',';out<<"{\"common_experts\":"<<common<<",\"partial\":"<<(partial?"true":"false")<<",\"median_ms\":"<<times[4]<<",\"relative_rms\":"<<worst<<'}';
        }out<<"]}";out.flush();
    }out<<"],\"page_touch_checksum\":"<<checksum<<",\"passed\":"<<(passed?"true":"false")<<"}\n";
    if(!out)throw std::runtime_error("Wake probe receipt write failed");return passed?0:2;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
