#pragma once
#include "generation_quality.hpp"
#include "pp_session_diagnostics.hpp"
#include "long_prefix_compare.hpp"
#include <fstream>

namespace strata::deepseek_v4 {
inline int prefix_free_pair(Runtime& runtime, const Options& options, const PrefixReference& reference,
        const std::filesystem::path& output, std::ostream& report, std::string& phase, bool stream_pp, bool hybrid_tg,
        bool candidate_only = false) {
    const auto quality_input = generation_recall_input(runtime, reference.input);
    std::ofstream fixture(output / "free-generation-inputs.json");
    fixture << "{\"sampling\":\"greedy\",\"performance_ignore_eog\":true,\"quality_ignore_eog\":false,\"performance_input_ids\":[";
    for (size_t i = 0; i < reference.input.size(); ++i) { if (i) fixture << ','; fixture << reference.input[i]; }
    fixture << "],\"recall_input_ids\":[";
    for (size_t i = 0; i < quality_input.size(); ++i) { if (i) fixture << ','; fixture << quality_input[i]; }
    fixture << "],\"expected_fields\":{\"alpha\":\"NORTH-7319\",\"beta\":\"CIRCUIT-2846\",\"sum\":42,\"direction\":\"host_to_device\"}}";
    fixture.close();
    if (!fixture) throw std::runtime_error("free generation fixture write failed");
    const char* single_pp=std::getenv("STRATA_DSV4_SINGLE_PP");
    const char* tg_aa=std::getenv("STRATA_DSV4_TG_AA");
    const char* cache_only_aba=std::getenv("STRATA_DSV4_MTP_CACHE_ONLY_ABA");
    const char* bg_stage_aba=std::getenv("STRATA_DSV4_BG_STAGE_ABA");
    const char* pp_park_aba=std::getenv("STRATA_DSV4_PP_WORKER_PARK_ABA");
    const char* fusion_aba=std::getenv("STRATA_DSV4_FUSION_PHASE_ABA");
    if(candidate_only&&((single_pp&&!std::strcmp(single_pp,"1"))||(fusion_aba&&!std::strcmp(fusion_aba,"1")))){
        const bool aba=fusion_aba&&!std::strcmp(fusion_aba,"1");
        const bool aa=tg_aa&&!std::strcmp(tg_aa,"1");
        const bool mtp_aba=cache_only_aba&&!std::strcmp(cache_only_aba,"1");
        const bool bg_aba=bg_stage_aba&&!std::strcmp(bg_stage_aba,"1");
        const bool park_aba=pp_park_aba&&!std::strcmp(pp_park_aba,"1");
        if(mtp_aba&&aba)throw std::runtime_error("MTP cache-only A/B/A cannot share a fusion phase switch");
        if(bg_aba&&(mtp_aba||aba))throw std::runtime_error("Background staging A/B/A requires its own phase switch");
        if(park_aba&&(mtp_aba||bg_aba||aba))throw std::runtime_error("PP worker parking A/B/A requires its own phase switch");
        if(mtp_aba){
#ifdef _WIN32
            if(_putenv_s("STRATA_DSV4_MTP_CACHE_ONLY","0"))throw std::runtime_error("MTP cache-only warmup mode failed");
#else
            if(setenv("STRATA_DSV4_MTP_CACHE_ONLY","0",1))throw std::runtime_error("MTP cache-only warmup mode failed");
#endif
        }
        if(bg_aba){
#ifdef _WIN32
            if(_putenv_s("STRATA_DSV4_BG_STAGE_UNPINNED","0"))throw std::runtime_error("Background stage warmup mode failed");
#else
            if(setenv("STRATA_DSV4_BG_STAGE_UNPINNED","0",1))throw std::runtime_error("Background stage warmup mode failed");
#endif
        }
        const char* warmup_first=std::getenv("STRATA_DSV4_WARMUP_FIRST");
        const bool warm=warmup_first&&!std::strcmp(warmup_first,"1");
        phase=aba?"strata_native_hybrid_single_pp_fusion_aba":"strata_native_hybrid_single_pp_fused";
        {std::ofstream progress(output/"progress.json");progress<<"{\"phase\":"<<session_json(phase)<<'}';}
        SessionRequest warmup;
        if(warm){
            warmup=session_request_count(runtime,reference.input,output/"warmup-failure.json",33);
            if(warmup.pp_tokens!=8192||warmup.output.size()!=33)throw std::runtime_error("warmup workload incomplete");
            std::ofstream warmup_receipt(output/"warmup-request.json");session_write_request(warmup_receipt,warmup);
            if(!warmup_receipt)throw std::runtime_error("warmup receipt write failed");
        }
        report<<"{\"name\":\"strata_native_hybrid\",\"requests\":[";
        const int requests=mtp_aba||bg_aba||park_aba?3:(aa?2:1);
        for(int request=0;request<requests;++request){
            if(park_aba){
#ifdef _WIN32
                if(_putenv_s("STRATA_DSV4_PP_WORKER_PARK",request==1?"0":"1"))throw std::runtime_error("PP worker parking phase switch failed");
#else
                if(setenv("STRATA_DSV4_PP_WORKER_PARK",request==1?"0":"1",1))throw std::runtime_error("PP worker parking phase switch failed");
#endif
            }
            if(mtp_aba){
#ifdef _WIN32
                if(_putenv_s("STRATA_DSV4_MTP_CACHE_ONLY",request==1?"1":"0"))throw std::runtime_error("MTP cache-only phase switch failed");
#else
                if(setenv("STRATA_DSV4_MTP_CACHE_ONLY",request==1?"1":"0",1))throw std::runtime_error("MTP cache-only phase switch failed");
#endif
            }
            if(bg_aba){
#ifdef _WIN32
                if(_putenv_s("STRATA_DSV4_BG_STAGE_UNPINNED",request==1?"1":"0"))throw std::runtime_error("Background stage phase switch failed");
#else
                if(setenv("STRATA_DSV4_BG_STAGE_UNPINNED",request==1?"1":"0",1))throw std::runtime_error("Background stage phase switch failed");
#endif
            }
            const std::string scored_phase=mtp_aba||bg_aba||park_aba?phase+"_aba"+std::to_string(request+1):aa?phase+(request?"_aa2":"_aa1"):phase;
            const auto result=session_request_count(runtime,reference.input,output/(scored_phase+"-failure.json"),2048);
            if(result.output.size()!=2048||result.pp_tokens!=8192||result.tg_tokens!=2047||result.mtp.fusion_phase_aba!=aba)
                throw std::runtime_error("single-PP fusion workload incomplete");
            if(request)report<<',';
            report<<"{\"phase\":"<<session_json(scored_phase)<<",\"mtp_cache_only\":"<<(mtp_aba&&request==1?"true":"false")
                  <<",\"pp_worker_park_enabled\":"<<(park_aba&&request==1?"false":"true")
                  <<",\"background_staging_enabled\":"<<(bg_aba&&request==1?"true":"false")<<",\"measured\":true,\"data\":";
            session_write_request(report,result);
            report<<",\"text_statistics\":";generation_text_receipt(runtime,result,output,scored_phase,report);
            report<<'}';report.flush();
            std::ofstream receipt(output/(scored_phase+"-request.json"));session_write_request(receipt,result);
            if(!receipt||!report)throw std::runtime_error("Single-PP fusion receipt write failed");
        }
        bool bounded_quality=false;
        report<<']';
        if(mtp_aba||bg_aba||park_aba){
            const auto quality=session_request_count(runtime,quality_input,output/"mtp-quality-failure.json",128,true);
            report<<",\"quality_request\":";session_write_request(report,quality);
            report<<",\"quality_text\":";generation_text_receipt(runtime,quality,output,"mtp_quality",report);
            report<<",\"quality_result\":";bounded_quality=generation_recall_check(runtime.detokenize(quality.output),report);
        }
        report<<"}],\"input_tokens\":8192,\"generated_outputs_each\":2048,"
              <<"\"pp_evaluations\":"<<(warm?1:0)+requests<<",\"measured_pp_evaluations\":"<<requests
              <<",\"warmup_first\":"<<(warm?"true":"false")<<",\"tg_aa\":"<<(aa?"true":"false")<<",\"mtp_cache_only_aba\":"<<(mtp_aba?"true":"false")
              <<",\"background_stage_aba\":"<<(bg_aba?"true":"false")<<",\"pp_worker_park_aba\":"<<(park_aba?"true":"false")
              <<",\"cold_warmup_pp_tps\":"<<(warm?warmup.pp_tokens*1000.0/warmup.pp_ms:0)
              <<",\"independent_autoregressive_generation\":true,"
              <<"\"output_id_equality_required\":false,\"teacher_forcing\":false,"
              <<"\"quality_request_skipped_to_preserve_single_pp\":"<<(mtp_aba||bg_aba||park_aba?"false":"true")
              <<",\"bounded_quality_passed\":"<<(mtp_aba||bg_aba||park_aba?(bounded_quality?"true":"false"):"null")<<','
              <<"\"finite_fixed_workload_passed\":true,\"fully_tuned_bare_llama_comparison\":false}\n";
        if(!report)throw std::runtime_error("Single-PP fusion report write failed");
        return 0;
    }
    std::array<SessionRequest, 2> measured;
    bool quality_passed = true;
    const int first_mode = candidate_only ? 1 : 0;
    for (int mode = first_mode; mode < 2; ++mode) {
        const std::string name = mode ? (hybrid_tg ? "strata_native_hybrid" : "strata_device_frequency") : "canonical_cpu_directory_pinned";
        phase = name + "_setup";
        session_set_device_cache(mode != 0); session_set_frequency(mode != 0);
        session_set_native(mode && stream_pp, mode && hybrid_tg);
        // Retain the same registered model and source budget across contexts.
        if (mode && !candidate_only) {
            std::string error;
            if (!runtime.reconfigure_context(options, error)) throw std::runtime_error(error);
        }
        if (mode > first_mode) report << ',';
        report << "{\"name\":" << session_json(name) << ",\"native_stream_pp\":" << (mode && stream_pp ? "true" : "false")
               << ",\"native_hybrid_tg\":" << (mode && hybrid_tg ? "true" : "false") << ",\"requests\":[";
        report.flush();
        for (int request = 0; request < 2; ++request) {
            const bool warmup = request == 0;
            if(!warmup&&candidate_only){pp_session_diagnostics(runtime,reference.input,output);
                const char* profile=std::getenv("STRATA_DSV4_PP_PROFILE_ONLY");if(profile&&!std::strcmp(profile,"1")) {
                    report<<"],\"diagnostic_only\":true}],\"profile_only\":true,\"performance_target_met\":false}\n";return 0;}}

            phase = name + (warmup ? "_kernel_source_warmup" : "_free_2048");
            { std::ofstream progress(output / "progress.json"); progress << "{\"phase\":" << session_json(phase) << '}'; }
            const auto result = session_request_count(runtime, reference.input, output / (phase + "-failure.json"), warmup ? 33 : 2048);
            if (result.output.size() != size_t(warmup ? 33 : 2048) || result.pp_tokens != 8192 || result.tg_tokens != (warmup ? 32 : 2047))
                throw std::runtime_error("free generation workload incomplete");
            if (request) report << ',';
            report << "{\"phase\":" << session_json(phase) << ",\"measured\":" << (warmup ? "false" : "true") << ",\"data\":";
            session_write_request(report, result);
            report << ",\"text_statistics\":"; generation_text_receipt(runtime, result, output, phase, report);
            report << '}'; report.flush();
            std::ofstream receipt(output / (phase + "-request.json"));
            session_write_request(receipt, result);
            if (!receipt) throw std::runtime_error("free request receipt write failed");
            if (!warmup) measured[mode] = result;
        }
        phase = name + "_recall";
        const auto result = session_request_count(runtime, quality_input, output / (phase + "-failure.json"), 128, true);
        report << "],\"quality_request\":"; session_write_request(report, result);
        report << ",\"quality_text\":"; generation_text_receipt(runtime, result, output, phase, report);
        report << ",\"quality_result\":";
        quality_passed &= generation_recall_check(runtime.detokenize(result.output), report);
        report << '}'; report.flush();
    }
    const double pp_ratio = candidate_only ? 0 : measured[0].pp_ms / measured[1].pp_ms;
    const double tg_ratio = candidate_only ? 0 : measured[0].tg_ms / measured[1].tg_ms;
    report << "],\"input_tokens\":8192,\"generated_outputs_each\":2048,\"independent_autoregressive_generation\":true"
           << ",\"output_id_equality_required\":false,\"full_logits_equality_required\":false,\"teacher_forcing\":false"
           << ",\"finite_fixed_workload_passed\":true,\"bounded_quality_passed\":" << (quality_passed ? "true" : "false")
           << ",\"output_ids_equal_diagnostic\":" << (measured[0].output == measured[1].output ? "true" : "false")
           << ",\"external_reference_comparison_required\":" << (candidate_only ? "true" : "false")
           << ",\"pp_ratio\":" << (candidate_only ? "null" : std::to_string(pp_ratio))
           << ",\"tg_ratio\":" << (candidate_only ? "null" : std::to_string(tg_ratio))
           << ",\"pinned_directory_comparison_2x\":" << (pp_ratio >= 2 && tg_ratio >= 2 ? "true" : "false")
           << ",\"warmup\":\"8192 PP plus 33 free outputs per mode; kernel/source warm, not full cache steady state\""
           << ",\"fully_tuned_bare_llama_comparison\":false,\"performance_target_met\":false}\n";
    if (!report) throw std::runtime_error("free generation report write failed");
    return quality_passed ? 0 : 2;
}
}
