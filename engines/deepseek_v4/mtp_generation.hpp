#pragma once
#include "runtime.hpp"
#include "device_cache.hpp"
#include "request_gpu_profile.hpp"
#include "mtp_host_probe.hpp"
#include <array>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <vector>
#include <stdexcept>
namespace strata::deepseek_v4 {
struct MtpVerificationRecord {
    int32_t position,pending,draft,target,bonus;bool accepted;int positions_verified=2;
    double wall_ms=0,draft_ms=0,verify_ms=0,post_ms=0,draft_input_ms=0,draft_cpu_ms=0;
    float top1_probability=0,top1_top2_logit_gap=0;bool confidence_measured=false;
    MtpHostProbe before_draft{},after_draft{},after_verify{};
    std::array<double,8> draft_input_parts_ms{};
};
void mtp_invalidate_target_graph(Runtime& runtime);
struct MtpPhaseMetrics {
    uint64_t proposals=0,accepted=0,verification_calls=0,single_position_calls=0,generated_outputs=0;
    uint64_t steady_verifications=0,steady_outputs=0;
    double wall_ms=0,steady_wall_ms=0,draft_ms=0,verify_ms=0,catchup_ms=0,rollback_ms=0,switch_ms=0;
};
struct MtpGenerationMetrics {
    std::vector<MtpVerificationRecord> records;
    uint64_t proposals=0,accepted=0,verification_calls=0,physical_target_positions=0;
    double initialization_ms=0,draft_ms=0,verify_ms=0,catchup_ms=0,rollback_ms=0;
    MtpStageMetrics draft_stages;
    std::vector<double> steady_verification_wall_ms;
    bool fusion_phase_aba=false,fusion_initially_enabled=false,input_phase_aba=false,input_initially_async=false;
    std::array<MtpPhaseMetrics,3> phases{};
};
inline void mtp_set_fusion_phase(bool enabled) {
    const char* names[]={"STRATA_DSV4_FUSED_COMPRESS","STRATA_DSV4_HC_COEFF_FUSION",
                         "STRATA_DSV4_FUSED_SHARED","STRATA_DSV4_FUSED_QKV"};
    for(const char* name:names){
#ifdef _WIN32
        if(_putenv_s(name,enabled?"1":"0"))throw std::runtime_error("MTP fusion phase switch failed");
#else
        if(setenv(name,enabled?"1":"0",1))throw std::runtime_error("MTP fusion phase switch failed");
#endif
    }
}
inline void mtp_generate(Runtime& runtime,const std::vector<llama_token>& input,std::vector<llama_token>& output,
    int limit,bool stop_on_eog,const std::filesystem::path& progress,MtpGenerationMetrics& m,const std::filesystem::path& receipt) {
    using Clock=std::chrono::steady_clock;auto elapsed=[](auto start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();};
    auto start=Clock::now();runtime.mtp_begin(input);m.initialization_ms+=elapsed(start);
    const char* phase_ab=std::getenv("STRATA_DSV4_FUSION_PHASE_ABA");
    m.fusion_phase_aba=input.size()==8192&&limit==2048&&phase_ab&&!std::strcmp(phase_ab,"1");
    const char* input_ab=std::getenv("STRATA_DSV4_MTP_INPUT_ABA");
    m.input_phase_aba=input.size()==8192&&limit==2048&&input_ab&&!std::strcmp(input_ab,"1");
    if(m.input_phase_aba&&m.fusion_phase_aba)throw std::runtime_error("Only one TG A/B/A axis can change per request");
    const char* input_async=std::getenv("STRATA_DSV4_MTP_ASYNC_INPUT");
    m.input_initially_async=input_async&&!std::strcmp(input_async,"1");
    const char* fusion=std::getenv("STRATA_DSV4_FUSED_COMPRESS");
    m.fusion_initially_enabled=fusion&&!std::strcmp(fusion,"1");
    int current_phase=0;
    const char* route_trace=std::getenv("STRATA_DSV4_TG_ROUTE_TRACE");
    const char* host_trace=std::getenv("STRATA_DSV4_HOST_STALL_TRACE");
    const bool trace_host=host_trace&&!std::strcmp(host_trace,"1");
    const char* sample_one=std::getenv("STRATA_DSV4_MTP_SAMPLE_SINGLE");
    const bool sample_single=sample_one&&!std::strcmp(sample_one,"1");
    auto host_snapshot=[&](){return trace_host?mtp_host_probe(runtime.cpu_worker_time_100ns()):MtpHostProbe{};};
    if(input.size()==8192&&route_trace&&!std::strcmp(route_trace,"1"))
        write_device_routing_seed(runtime.context(),receipt.string()+"-routing-seed.json");
    llama_token pending=output.back();std::string error;
    while(output.size()<size_t(limit)) {
        int phase_index=0;
        if(m.fusion_phase_aba||m.input_phase_aba){
            phase_index=output.size()>=size_t(1+2*(limit-1)/3)?2:output.size()>=size_t(1+(limit-1)/3)?1:0;
            if(phase_index!=current_phase){
                start=Clock::now();
                if(m.fusion_phase_aba){mtp_set_fusion_phase(phase_index==1);mtp_invalidate_target_graph(runtime);}
                if(m.input_phase_aba)runtime.mtp_set_async_input(phase_index==1);
                m.phases[phase_index].switch_ms+=elapsed(start);current_phase=phase_index;
            }
        }
        auto& phase=m.phases[phase_index];const size_t outputs_before=output.size();const auto loop_start=Clock::now();
        auto finish_phase=[&](bool two_position=true){
            const double duration=elapsed(loop_start);const uint64_t emitted=output.size()-outputs_before;
            phase.wall_ms+=duration;phase.generated_outputs+=emitted;
            if(two_position&&!m.records.empty())m.records.back().wall_ms=duration;
            if(two_position&&phase.verification_calls>5){phase.steady_wall_ms+=duration;phase.steady_outputs+=emitted;++phase.steady_verifications;
                m.steady_verification_wall_ms.push_back(duration);}
        };
        if(size_t(limit)-output.size()==1) {
            start=Clock::now();if(runtime.decode_tokens(&pending,1,runtime.position(),true,error))throw std::runtime_error(error);
            const double verify=elapsed(start);m.verify_ms+=verify;phase.verify_ms+=verify;++phase.single_position_calls;++m.physical_target_positions;
            auto next=runtime.greedy(!stop_on_eog);if(!stop_on_eog||!runtime.is_eog(next))output.push_back(next);finish_phase(false);break;
        }
        const auto position=runtime.position();const auto draft_before=runtime.mtp_stage_metrics();
        const auto host_before=host_snapshot();
        start=Clock::now();const auto draft=runtime.mtp_draft(pending);
        const auto confidence=runtime.mtp_confidence();
        const auto draft_after=runtime.mtp_stage_metrics();
        const double draft_time=elapsed(start);m.draft_ms+=draft_time;phase.draft_ms+=draft_time;++m.proposals;++phase.proposals;
        const auto host_after_draft=host_snapshot();
        const bool one_position=sample_single&&m.proposals%16==0;
        const llama_token verify[]{pending,draft};start=Clock::now();
        const char* one_profile=std::getenv("STRATA_DSV4_TG_PROFILE_ONE_VERIFY");
        {
            RequestGpuProfile capture(limit==33&&input.size()==8192&&m.verification_calls==5&&
                                      one_profile&&!std::strcmp(one_profile,"1"),"STRATA_DSV4_TG_ACTIVITY_PROFILE");
            if(runtime.decode_tokens(verify,one_position?1:2,position,false,error,true))throw std::runtime_error(error);
        }
        const double verify_time=elapsed(start);m.verify_ms+=verify_time;phase.verify_ms+=verify_time;
        const auto host_after_verify=host_snapshot();
        ++m.verification_calls;++phase.verification_calls;m.physical_target_positions+=one_position?1:2;
        const auto target=runtime.greedy_at(0,!stop_on_eog);
        m.records.push_back({position,pending,draft,target,-1,draft==target});
        m.records.back().positions_verified=one_position?1:2;
        m.records.back().draft_ms=draft_time;m.records.back().verify_ms=verify_time;
        m.records.back().draft_input_ms=draft_after.host_ms[0]-draft_before.host_ms[0];
        for(size_t part=0;part<8;++part)m.records.back().draft_input_parts_ms[part]=
            draft_after.input_sub_ms[part]-draft_before.input_sub_ms[part];
        m.records.back().draft_cpu_ms=double(draft_after.cpu_miss_compute_ns-draft_before.cpu_miss_compute_ns)/1e6;
        m.records.back().top1_probability=confidence.top1_probability;
        m.records.back().top1_top2_logit_gap=confidence.top1_top2_logit_gap;
        m.records.back().confidence_measured=confidence.measured;
        m.records.back().before_draft=host_before;m.records.back().after_draft=host_after_draft;
        m.records.back().after_verify=host_after_verify;
        if(one_position){
            ++phase.single_position_calls;
            if(draft==target){++m.accepted;++phase.accepted;}
            runtime.mtp_single_advance();
            if(stop_on_eog&&runtime.is_eog(target)){finish_phase();break;}
            output.push_back(target);pending=target;finish_phase();continue;
        }
        if(draft==target) {
            ++m.accepted;++phase.accepted;if(stop_on_eog&&runtime.is_eog(draft)){finish_phase();break;}output.push_back(draft);
            const auto bonus=runtime.greedy_at(1,!stop_on_eog);m.records.back().bonus=bonus;if(stop_on_eog&&runtime.is_eog(bonus)){finish_phase();break;}
            if(output.size()<size_t(limit))output.push_back(bonus);
            start=Clock::now();runtime.mtp_accept_catchup(draft);const double catchup=elapsed(start);
            m.catchup_ms+=catchup;phase.catchup_ms+=catchup;pending=bonus;
            m.records.back().post_ms=catchup;
        }else {
            start=Clock::now();runtime.rollback(position+1);const double rollback=elapsed(start);
            m.rollback_ms+=rollback;phase.rollback_ms+=rollback;
            m.records.back().post_ms=rollback;
            if(stop_on_eog&&runtime.is_eog(target)){finish_phase();break;}output.push_back(target);pending=target;
        }
        finish_phase();
        if(output.size()%32<=1){std::ofstream out(progress);out<<"{\"generated_outputs\":"<<output.size()<<",\"position\":"<<runtime.position()<<",\"mtp_proposals\":"<<m.proposals<<",\"mtp_accepted\":"<<m.accepted<<'}';if(!out)throw std::runtime_error("MTP progress write failed");}
    }
    if(input.size()==8192&&route_trace&&!std::strcmp(route_trace,"1"))
        write_device_routing_seed(runtime.context(),receipt.string()+"-routing-final.json");
}
}
