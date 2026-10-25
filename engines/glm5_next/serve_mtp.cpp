#include "serve_mtp.hpp"
#include "mtp_generate.hpp"
#include "serve_mtp_sampling.hpp"
#include "../deepseek_v4/serve_finish.hpp"
#include "llama-context.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
namespace strata::glm5_next {
int serve_mtp(deepseek_v4::Runtime& runtime,const deepseek_v4::ServeRequest& request,
    const std::shared_ptr<std::atomic<bool>>& cancel,std::chrono::steady_clock::time_point request_start,
    const deepseek_v4::ServeMetrics& counters,KernelActivity& activity) {
    using Clock=std::chrono::steady_clock;
    auto milliseconds=[](Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    const auto& input=request.ids;
    ServeMtpSampling sampler(request,input);
    double pp_ms=0;Clock::time_point tg_start{};KernelActivityStats pp_activity{};int emitted=0;
    auto result=generate_mtp(runtime,MtpTargetSample([&](int index){
        return sampler.pick(llama_get_logits_ith(runtime.context(),index),llama_vocab_n_tokens(runtime.vocab()),
                            [&](llama_token token){return runtime.is_eog(token);});
    }),input,request.limit,0,[&](llama_token token){
        sampler.commit(token);std::printf("T %d\n",token);++emitted;
    },[&]{
        pp_ms=milliseconds(request_start);pp_activity=activity.snapshot();
        std::printf("PP %zu %zu %.3f %.3f\n",input.size(),input.size(),pp_ms,input.size()*1000.0/std::max(.001,pp_ms));
        tg_start=Clock::now();
    },[&]{return cancel->load();});
    llama_synchronize(runtime.context());deepseek_v4::settle_device_background(runtime.context());
    const double tg_ms=milliseconds(tg_start);counters.report(runtime);
    if(activity.active()) {
        const auto counts=activity.finish();
        std::fprintf(stderr,"strata_kernel_activity pp_kernels=%llu tg_kernels=%llu dropped=%llu errors=%llu profiled=1\n",
            (unsigned long long)pp_activity.kernels,(unsigned long long)(counts.kernels-pp_activity.kernels),
            (unsigned long long)counts.dropped,(unsigned long long)counts.errors);
        if(counts.dropped||counts.errors||!counts.kernels)throw std::runtime_error("GLM serving kernel activity incomplete");
    }
    const auto accepted=result.metrics.value("accepted",uint64_t(0)),offered=result.metrics.value("proposals",uint64_t(0));
    result.metrics["target_policy"]="normal serve_sample / serve_rng / causal ServeHistory";
    result.metrics["target_sample_calls"]=sampler.draws();
    result.metrics["actual_output_tokens"]=emitted;
    result.metrics["temperature"]=request.temperature;
    std::fprintf(stderr,"strata_glm_mtp %s\n",result.metrics.dump().c_str());std::fflush(stderr);
    const char* finish=result.cancelled?"cancel":result.eos?"stop":"limit";
    deepseek_v4::report_serve_finish(finish,size_t(emitted),input.size(),input.size(),pp_ms,tg_ms);
    std::printf("DONE %d %zu %.3f %.3f %s %llu %llu 0\n",emitted,input.size(),pp_ms,tg_ms,
        finish,(unsigned long long)accepted,(unsigned long long)offered);
    return 0;
}
}
