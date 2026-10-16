#include "runtime.hpp"
#include "device_cache.hpp"
#include "native_tg_timing.hpp"
#include "benchmark_fault_guard.hpp"
#include "request_gpu_profile.hpp"
#include "llama-context.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>
using namespace strata::deepseek_v4;
void env(const char* name,const char* value){
#ifdef _WIN32
    if(_putenv_s(name,value))throw std::runtime_error("Diagnostic environment failed");
#else
    if(setenv(name,value,1))throw std::runtime_error("Diagnostic environment failed");
#endif
}
int run(int argc,char** argv){
    if(argc!=3&&argc!=4&&argc!=5)throw std::runtime_error("Expected model path, new report path, optional2-position count, optional ab/combo/trio/quad/quad-smoke");
    const int positions=argc>=4?std::stoi(argv[3]):1;if(positions!=1&&positions!=2)throw std::runtime_error("Position count must be1 or2");
    const bool quad=argc==5&&std::string(argv[4])=="quad";
    const bool quad_smoke=argc==5&&std::string(argv[4])=="quad-smoke";
    const bool stagger_split=argc==5&&std::string(argv[4])=="stagger-split";
    const bool trio=quad||(argc==5&&std::string(argv[4])=="trio");
    const bool combo=(argc==5&&std::string(argv[4])=="combo")||trio;
    const bool ab=argc==5&&(std::string(argv[4])=="ab"||combo);
    if(argc==5&&!ab&&!quad_smoke&&!stagger_split)throw std::runtime_error("Fourth argument must be ab, combo, trio, quad, quad-smoke or stagger-split");
    env("STRATA_DSV4_DEVICE_CACHE","1");env("STRATA_DSV4_NATIVE_TG","1");
    env("STRATA_DSV4_TG_TIMING","1");env("STRATA_DSV4_STREAM_PP","0");
    env("STRATA_DSV4_SOURCE_PIN_MIB","0");env("STRATA_DSV4_SOURCE_PRIVATE_MAPPING","0");
    env("STRATA_DSV4_EXPERT_FREQUENCY","0");env("STRATA_DSV4_NATIVE_CPU_THREADS","4");
    if (!std::getenv("GGML_CUDA_GRAPH_OPT")) env("GGML_CUDA_GRAPH_OPT","0");
    env("STRATA_DSV4_TG_CPU_ONLY","1");env("STRATA_DSV4_TG_CACHE_GROW_MIB","0");env("STRATA_DSV4_LAST_LOGIT_ONLY","0");
    if(quad_smoke){env("STRATA_DSV4_FUSED_COMPRESS","1");env("STRATA_DSV4_HC_COEFF_FUSION","1");
        env("STRATA_DSV4_FUSED_SHARED","1");env("STRATA_DSV4_FUSED_QKV","1");}
    Options options;options.model=argv[1];options.n_ctx=12288;options.n_batch=8;options.n_ubatch=8;
    options.n_threads=4;options.moe_cache_mib=positions==2?8192:4096;options.pinned_experts=1;
    std::string error;auto runtime=Runtime::load(options,error);
    if(!runtime)throw std::runtime_error(error);
    const auto tokens=runtime->tokenize("The",true,true);
    if(tokens.empty())throw std::runtime_error("Actual tokenizer input unavailable");
    const std::vector<llama_token> input(positions,tokens[0]);
    auto evaluate_positions=[&](int count){return runtime->decode_tokens(input.data(),count,0,false,error,true)==0;};
    auto evaluate=[&]{return evaluate_positions(positions);};
    if(quad_smoke){
        enable_device_expert_cache(runtime->context(),false);
        if(!evaluate())throw std::runtime_error(error);
        llama_synchronize(runtime->context());
        enable_device_expert_cache(runtime->context(),true);runtime->reset();
        {RequestGpuProfile capture(true);if(!evaluate())throw std::runtime_error(error);
            llama_synchronize(runtime->context());}
        verify_device_expert_cache(runtime->context());
        const auto first=runtime->greedy_at(0),second=runtime->greedy_at(1);
        std::ofstream report(argv[2]);report<<"{\"whole_model_loaded\":false,\"actual_nonexpert_weights\":true,"
            <<"\"two_position_fused_graph_passed\":true,\"first_output\":"<<first
            <<",\"second_output\":"<<second<<"}\n";
        if(!report)throw std::runtime_error("Fused graph smoke report failed");return 0;
    }
    enable_device_expert_cache(runtime->context(),false);
    if(!evaluate())throw std::runtime_error(error);
    llama_synchronize(runtime->context());
    if(stagger_split){
        runtime->reset();
        if(!evaluate_positions(1))throw std::runtime_error(error);
        llama_synchronize(runtime->context());
    }
    enable_device_expert_cache(runtime->context(),true);
    const auto scheduler=runtime->context()->get_sched();ggml_backend_t backend=nullptr;
    for(int i=0;i<ggml_backend_sched_get_n_backends(scheduler);++i){auto* candidate=ggml_backend_sched_get_backend(scheduler,i);
        const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(candidate));
        if(std::string(ggml_backend_reg_name(reg))=="CUDA"){backend=candidate;break;}}
    if(!backend)throw std::runtime_error("Selected CUDA backend unavailable");
    const auto reg=ggml_backend_dev_backend_reg(ggml_backend_get_device(backend));
    using Begin=bool(*)(ggml_backend_t);using Read=bool(*)(ggml_backend_t,NativeTgTimingResult*);
    const auto begin=reinterpret_cast<Begin>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_timing_begin"));
    const auto read=reinterpret_cast<Read>(ggml_backend_reg_get_proc_address(reg,"strata_native_tg_timing_read"));
    if(!begin||!read)throw std::runtime_error("Physical GPU timing unavailable");
    auto middle=[](std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];};
    struct Result {double whole,expert,nonexpert;uint64_t warmup_cpu_misses,timed_cpu_misses;};
    auto sample=[&](bool profile,int count){
        std::vector<double> whole,expert,nonexpert;
        uint64_t warmup_cpu_misses=0,timed_cpu_misses=0;
        for(int repeat=0;repeat<12;++repeat){
            runtime->reset();llama_synchronize(runtime->context());
            const auto before=device_expert_cache_metrics(runtime->context());
            RequestGpuProfile capture(profile&&repeat==4);
            if(!begin(backend)||!evaluate_positions(count))throw std::runtime_error(error.empty()?"Timing begin failed":error);
            llama_synchronize(runtime->context());NativeTgTimingResult timing{};
            if(!read(backend,&timing))throw std::runtime_error("Incomplete GPU timing coverage");
            const auto after=device_expert_cache_metrics(runtime->context());
            verify_device_expert_cache(runtime->context());
            const uint64_t cpu_misses=after.host_services-before.host_services;
            if(stagger_split&&repeat<4)warmup_cpu_misses+=cpu_misses;
            if(stagger_split&&repeat>=4)timed_cpu_misses+=cpu_misses;
            const bool stagger_cache_ok=stagger_split&&cpu_misses<=1&&
                after.gpu[1]-before.gpu[1]+cpu_misses==43&&
                after.cpu_routes-before.cpu_routes<=1;
            if(after.gpu[0]-before.gpu[0]!=43||
               (stagger_split?!stagger_cache_ok:after.gpu[1]-before.gpu[1]!=43)||
               (!stagger_split&&cpu_misses)||after.transport.bytes!=before.transport.bytes||
               (!stagger_split&&after.cpu_routes!=before.cpu_routes))throw std::runtime_error(
                   "Physical probe was not true all-hit: gpu0="+std::to_string(after.gpu[0]-before.gpu[0])+
                   " gpu1="+std::to_string(after.gpu[1]-before.gpu[1])+
                   " host="+std::to_string(after.host_services-before.host_services)+
                   " h2d="+std::to_string(after.transport.bytes-before.transport.bytes)+
                   " cpu="+std::to_string(after.cpu_routes-before.cpu_routes));
            if(repeat>=4){whole.push_back(timing.whole_gpu_ns/1e6);expert.push_back(timing.expert_ns/1e6);
                nonexpert.push_back((timing.whole_gpu_ns-timing.expert_ns)/1e6);}
        }
        return Result{middle(whole),middle(expert),middle(nonexpert),warmup_cpu_misses,timed_cpu_misses};
    };
    if(stagger_split){
        if(positions!=2)throw std::runtime_error("Stagger split needs a two-position input capacity");
        env("STRATA_DSV4_FUSED_COMPRESS","1");env("STRATA_DSV4_HC_COEFF_FUSION","1");
        env("STRATA_DSV4_FUSED_SHARED","1");env("STRATA_DSV4_FUSED_QKV","1");
        runtime->context()->strata_invalidate_graph_for_fusion_switch();
        const Result single_a=sample(false,1),pair=sample(false,2),single_b=sample(false,1);
        const double single=(single_a.nonexpert+single_b.nonexpert)/2;
        std::ofstream report(argv[2]);report<<std::setprecision(12)
            <<"{\"whole_model_loaded\":false,\"actual_dense_weights\":true,"
            <<"\"max_one_cpu_miss_per_43_layers\":true,\"same_load_aba\":true,\"layers\":43,"
            <<"\"past_tokens\":0,\"samples_per_case\":8,\"fused_gpu_path\":true,"
            <<"\"single_a_nonexpert_ms\":"<<single_a.nonexpert
            <<",\"pair_nonexpert_ms\":"<<pair.nonexpert
            <<",\"single_b_nonexpert_ms\":"<<single_b.nonexpert
            <<",\"two_single_minus_pair_nonexpert_ms\":"<<2*single-pair.nonexpert
            <<",\"single_a_whole_gpu_ms\":"<<single_a.whole
            <<",\"pair_whole_gpu_ms\":"<<pair.whole
            <<",\"single_b_whole_gpu_ms\":"<<single_b.whole
            <<",\"two_single_minus_pair_whole_gpu_ms\":"<<single_a.whole+single_b.whole-pair.whole
            <<",\"warmup_cpu_misses\":["<<single_a.warmup_cpu_misses<<','<<pair.warmup_cpu_misses<<','<<single_b.warmup_cpu_misses<<']'
            <<",\"timed_cpu_misses\":["<<single_a.timed_cpu_misses<<','<<pair.timed_cpu_misses<<','<<single_b.timed_cpu_misses<<']'
            <<"}\n";
        if(!report)throw std::runtime_error("Stagger dense split report failed");
        return 0;
    }
    if(ab){env("STRATA_DSV4_FUSED_COMPRESS","0");if(combo)env("STRATA_DSV4_HC_COEFF_FUSION","0");if(trio)env("STRATA_DSV4_FUSED_SHARED","0");if(quad)env("STRATA_DSV4_FUSED_QKV","0");}
    const Result a=sample(!ab,positions);
    Result b{},recovery{};
    if(ab){env("STRATA_DSV4_FUSED_COMPRESS","1");if(combo)env("STRATA_DSV4_HC_COEFF_FUSION","1");if(trio)env("STRATA_DSV4_FUSED_SHARED","1");if(quad)env("STRATA_DSV4_FUSED_QKV","1");runtime->context()->strata_invalidate_graph_for_fusion_switch();b=sample(false,positions);
        env("STRATA_DSV4_FUSED_COMPRESS","0");if(combo)env("STRATA_DSV4_HC_COEFF_FUSION","0");if(trio)env("STRATA_DSV4_FUSED_SHARED","0");if(quad)env("STRATA_DSV4_FUSED_QKV","0");runtime->context()->strata_invalidate_graph_for_fusion_switch();recovery=sample(false,positions);}
    std::ofstream report(argv[2]);report<<std::setprecision(12)
        <<"{\"scope\":\"real dense and selected expert weights;43 layers;fixed actual token reset;physical component only\",\"positions_per_call\":"<<positions<<','
        <<"\"full_expert_model_loaded\":false,\"source_pin_bytes\":0,\"context_capacity\":12288,\"actual_past_tokens\":0,"
        <<"\"all_hit\":true,\"expert_h2d_bytes\":0,\"host_miss_services\":0,\"cpu_expert_routes\":0,"
        <<"\"cuda_graph_optimizer\":"<<(std::getenv("GGML_CUDA_GRAPH_OPT")&&std::strcmp(std::getenv("GGML_CUDA_GRAPH_OPT"),"1")==0?"true":"false")<<','
        <<"\"layers_each\":43,\"warmups\":4,\"samples\":8,\"whole_gpu_median_ms\":"<<a.whole
        <<",\"routed_expert_gpu_median_ms\":"<<a.expert<<",\"nonexpert_gpu_median_ms\":"<<a.nonexpert
        <<",\"nonexpert_includes_route_shared_mhc_attention_dense_and_exposed_submission_gaps\":true,"
        <<"\"eight_k_context_attention_increment_not_measured\":true,\"performance_target_met\":false";
    if(ab)report<<(quad?",\"fused_compressor_hc_shared_qkv_aba\":{":(trio?",\"fused_compressor_hc_shared_aba\":{":(combo?",\"fused_compressor_and_hc_aba\":{":",\"fused_compressor_aba\":{")))
        <<"\"a_nonexpert_ms\":"<<a.nonexpert<<",\"b_nonexpert_ms\":"<<b.nonexpert
        <<",\"a_recovery_nonexpert_ms\":"<<recovery.nonexpert
        <<",\"a_whole_ms\":"<<a.whole<<",\"b_whole_ms\":"<<b.whole
        <<",\"a_recovery_whole_ms\":"<<recovery.whole<<"}";
    report<<"}\n";
    if(!report)throw std::runtime_error("Physical GPU report failed");
    std::cout<<"GPU all-hit nonexpert physical measurement complete\n";return 0;
}
int main(int argc,char** argv){benchmark_fault_guard();try{return run(argc,argv);}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
