#pragma once
#include <vector>
#include "request.hpp"
#include <functional>
namespace strata::glm5_next {
struct MtpGeneration {
    Json metrics;
    bool eos=false,cancelled=false;
};
using MtpTargetSample=std::function<llama_token(int)>;
MtpGeneration generate_mtp(deepseek_v4::Runtime&,const MtpTargetSample&,const std::vector<llama_token>&,
    int limit,int cancel_after,const std::function<void(llama_token)>& emit,const std::function<void()>& on_warm,const std::function<bool()>& should_cancel = {});
MtpGeneration generate_mtp(deepseek_v4::Runtime&,llama_sampler*,const std::vector<llama_token>&,
    int limit,int cancel_after,const std::function<void(llama_token)>& emit,const std::function<void()>& on_warm,const std::function<bool()>& should_cancel = {});
}
