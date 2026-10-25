#pragma once
#include "strata/qwen35moe/nvfp4_expert_inventory.hpp"
#include "strata/prefill/native_grouped_nvfp4.hpp"
#include <memory>
#include <cstdlib>
#include "strata/qwen35moe/device_buffer.hpp"
#include "strata/qwen35moe/nvfp4_tg_admission.hpp"
#include <cuda_runtime.h>
namespace strata::qwen35moe {
struct Nvfp4ExpertState {
    Nvfp4ExpertInventory inventory;
    bool cpu_tg=false;
    bool device_tg=false;
    std::unique_ptr<Nvfp4TgAdmission> tg_admission;
    DeviceBuffer tg_globals; // allocated before the expert-cache budget is chosen
    std::vector<float> tg_scale_host;
    std::array<std::vector<float>,3> tg_cpu_scales;
    std::vector<int32_t> tg_directory;
    uint32_t tg_token=0;
    bool tg_pending=false;
    cudaEvent_t tg_admit_begin=nullptr,tg_admit_end=nullptr;
    uint64_t tg_tail_bytes=0,tg_tail_control=0;
    std::unique_ptr<prefill::mmq::NativeGroupedNvfp4Provider> provider;
    std::vector<float> gate_scales,up_scales,down_scales;
    explicit Nvfp4ExpertState(const GgufFile& file,const Spec& spec):inventory(file,spec) {
        const char* raw=std::getenv("STRATA_QWEN35_NVFP4_TG");
        const std::string value=raw?raw:"grouped";
        if(value!="grouped"&&value!="cpu"&&value!="device")throw std::invalid_argument("NVFP4_TG requiresgrouped,cpu ordevice");
        cpu_tg=value=="cpu";
        device_tg=value=="device";
        const bool overlap=nvfp4_expert_flag(std::getenv("STRATA_QWEN35_NVFP4_TG_ADMISSION_OVERLAP"));
        if(overlap&&!device_tg)throw std::invalid_argument("NVFP4 TG admission overlap requires device TG");
        if(overlap)tg_admission=std::make_unique<Nvfp4TgAdmission>(inventory.layers());
        if(device_tg) {
            tg_scale_host.resize(inventory.layers()*256*3);
            for(auto& v:tg_cpu_scales)v.resize(inventory.layers()*256);
            for(size_t l=0;l<inventory.layers();++l)for(int e=0;e<256;++e) {
                const auto v=inventory.view(l,e);const size_t key=l*256+size_t(e);
                const float s[3]={v.gate_scale,v.up_scale,v.down_scale};
                for(int i=0;i<3;++i){tg_scale_host[key*3+size_t(i)]=s[i];tg_cpu_scales[size_t(i)][key]=s[i];}
            }
            tg_globals.reset(tg_scale_host.size()*sizeof(float));
            const auto error=cudaMemcpy(tg_globals.data(),tg_scale_host.data(),tg_globals.bytes(),cudaMemcpyHostToDevice);
            if(error!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(error));
            auto status=cudaEventCreate(&tg_admit_begin);
            if(status==cudaSuccess)status=cudaEventCreate(&tg_admit_end);
            if(status!=cudaSuccess){if(tg_admit_begin)cudaEventDestroy(tg_admit_begin);throw std::runtime_error(cudaGetErrorString(status));}
        }
    }
    ~Nvfp4ExpertState(){if(tg_admit_begin)cudaEventDestroy(tg_admit_begin);if(tg_admit_end)cudaEventDestroy(tg_admit_end);}
};
} // namespace strata::qwen35moe
