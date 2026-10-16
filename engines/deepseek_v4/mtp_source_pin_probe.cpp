#include "mtp_sidecar.hpp"
#include "expert_source_pin.hpp"
#include "llama.h"
#include <array>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::deepseek_v4;
int main(int argc,char** argv) try {
    if(argc!=3)throw std::runtime_error("Expected MTP GGUF and report path");
    llama_backend_init();
    auto* device=ggml_backend_dev_by_name("CUDA0");
    if(!device)throw std::runtime_error("CUDA0 unavailable");
    std::unique_ptr<ggml_backend,decltype(&ggml_backend_free)> backend(ggml_backend_dev_init(device,nullptr),ggml_backend_free);
    if(!backend)throw std::runtime_error("CUDA backend unavailable");
    MtpSidecar sidecar(argv[1],backend.get());
    std::vector<ggml_tensor*> sources;
    for(const char* role:{"gate","up","down"})
        sources.push_back(sidecar.tensor(std::string("blk.43.ffn_")+role+"_exps.weight"));
    const uint64_t budget=3422552064ull;
    ExpertSourcePin pin(backend.get(),sources,budget);
    std::array<uint32_t,3> registered_by_role{};
    uint32_t complete=0;
    for(int expert=0;expert<256;++expert){
        bool all=true;
        for(size_t role=0;role<sources.size();++role){
            auto* tensor=sources[role];const auto* p=static_cast<const char*>(tensor->data)+expert*tensor->nb[2];
            const auto segment=pin.segment(p,tensor->nb[2]);
            const bool registered=segment.registered&&segment.bytes>=tensor->nb[2];
            registered_by_role[role]+=registered;all&=registered;
        }
        complete+=all;
    }
    const auto metrics=pin.metrics();
    std::ofstream out(argv[2]);
    out<<"{\"whole_model_loaded\":false,\"real_mtp_weights\":true,\"budget_bytes\":"<<budget
       <<",\"registered_bytes\":"<<metrics.registered_bytes
       <<",\"registration_attempts\":"<<metrics.attempts
       <<",\"registration_failures\":"<<metrics.failures
       <<",\"complete_expert_triplets\":"<<complete
       <<",\"role_complete_experts\":["<<registered_by_role[0]<<','<<registered_by_role[1]<<','<<registered_by_role[2]<<"]}\n";
    if(!out||complete<240)throw std::runtime_error("MTP source registration is insufficient for background admission");
    return 0;
}catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
