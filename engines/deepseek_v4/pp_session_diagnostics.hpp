#pragma once
#include "session_gate.hpp"
#include "request_gpu_profile.hpp"
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
namespace strata::deepseek_v4 {
inline void pp_session_diagnostics(Runtime& runtime,const std::vector<llama_token>& input,const std::filesystem::path& output) {
    const char* enabled=std::getenv("STRATA_DSV4_PP_DIAGNOSTICS");if(!enabled||std::strcmp(enabled,"1"))return;
    const char* prior=std::getenv("STRATA_DSV4_PP_FIXED_EDGES");const std::string restore=prior?prior:"1";
    auto set=[](const char* value){
#ifdef _WIN32
        _putenv_s("STRATA_DSV4_PP_FIXED_EDGES",value);
#else
        setenv("STRATA_DSV4_PP_FIXED_EDGES",value,1);
#endif
    };
    const char* profile_only=std::getenv("STRATA_DSV4_PP_PROFILE_ONLY");const int count=profile_only&&!std::strcmp(profile_only,"1")?1:3;
    std::ofstream report(output/"pp-diagnostics.json");report<<std::setprecision(16)
        <<"{\"same_loaded_model\":true,\"extra_pp_requests_before_main_measurement\":"<<count<<",\"diagnostic_only\":true,\"input_tokens_each\":8192,\"variants\":[";
    try {
        const char* variants[]{"1","0","1"};
        for(int i=0;i<count;++i){set(variants[i]);const std::string name="pp-diagnostic-"+std::to_string(i);
            {std::ofstream progress(output/"progress.json");progress<<"{\"phase\":\""<<name<<"\"}";}
            SessionRequest result;
            {RequestGpuProfile activity(i==0);result=session_request_count(runtime,input,output/(name+"-failure.json"),1);}
            if(result.pp_tokens!=8192||result.output.size()!=1||result.tg_tokens)throw std::runtime_error("PP diagnostic workload differs");
            if(i)report<<',';report<<"{\"fixed_edges\":"<<variants[i]<<",\"matches_normal_warmup_for_pp_comparison\":"<<(i==0?"true":"false")<<",\"data\":";
            session_write_request(report,result);report<<'}';report.flush();
            std::cerr<<"strata_pp_diagnostic fixed_edges="<<variants[i]<<" pp_ms="<<result.pp_ms<<'\n'<<std::flush;
        }
    }catch(...){set(restore.c_str());throw;}
    set(restore.c_str());report<<"]}\n";if(!report)throw std::runtime_error("PP diagnostics receipt failed");
}
}
