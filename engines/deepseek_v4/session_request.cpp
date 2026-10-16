#include "session_gate.hpp"
#include <iostream>
#include "device_cache_probe_report.hpp"
#include "tg_diagnostics.hpp"
#include "request_gpu_profile.hpp"
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
SessionRequest::PhaseCost phase_cost(const ExpertStagingMetrics& a, const DeviceCacheMetrics& ca,
                                     const ExpertStagingMetrics& b, const DeviceCacheMetrics& cb) {
    SessionRequest::PhaseCost p;
    p.weight_bytes = b.bytes - a.bytes + cb.transport.bytes - ca.transport.bytes + cb.pp.payload_bytes - ca.pp.payload_bytes;
    p.weight_calls = b.calls - a.calls + cb.transport.calls - ca.transport.calls + cb.pp.dma_calls - ca.pp.dma_calls;
    p.source_cpu_copy_ns = b.host_copy_wall_ns - a.host_copy_wall_ns + cb.transport.host_copy_wall_ns - ca.transport.host_copy_wall_ns +
                          cb.pp.source_cpu_copy_ns - ca.pp.source_cpu_copy_ns;
    p.lookups = cb.gpu[0] - ca.gpu[0]; p.total_routes=p.lookups*6; p.initial_hit_routes = cb.gpu[2] - ca.gpu[2];
    p.host_services = cb.host_services - ca.host_services;
    p.native_pp_layers = cb.pp.layers - ca.pp.layers; p.native_pp_groups = cb.pp.groups - ca.pp.groups;
    p.pp_cache_reused_bytes = cb.pp.cache_reused_bytes - ca.pp.cache_reused_bytes;
    p.pp_timing.device_alloc_ns = cb.pp.device_alloc_ns-ca.pp.device_alloc_ns;
    p.pp_timing.device_alloc_calls = cb.pp.device_alloc_calls-ca.pp.device_alloc_calls;
    p.pp_timing.staging_wait_ns = cb.pp.staging_wait_ns-ca.pp.staging_wait_ns;
    p.pp_timing.cache_copy_calls = cb.pp.cache_copy_calls-ca.pp.cache_copy_calls;
    p.pp_timing.staged_bytes = cb.pp.staged_bytes-ca.pp.staged_bytes;
    p.pp_timing.timed_layers = cb.pp.timed_layers-ca.pp.timed_layers;
    p.pp_timing.gpu_layer_ns = cb.pp.gpu_layer_ns-ca.pp.gpu_layer_ns;
    p.pp_timing.gpu_weight_wait_ns = cb.pp.gpu_weight_wait_ns-ca.pp.gpu_weight_wait_ns;
    p.pp_timing.gpu_products_ns = cb.pp.gpu_products_ns-ca.pp.gpu_products_ns;
    p.pp_timing.gpu_prelude_ns = cb.pp.gpu_prelude_ns-ca.pp.gpu_prelude_ns;
    p.pp_timing.gpu_intergroup_ns = cb.pp.gpu_intergroup_ns-ca.pp.gpu_intergroup_ns;
    p.pp_timing.gpu_tail_ns = cb.pp.gpu_tail_ns-ca.pp.gpu_tail_ns;
    p.pp_timing.prefetch_bytes=cb.pp.prefetch_bytes-ca.pp.prefetch_bytes;
    p.pp_timing.prefetch_consumed_bytes=cb.pp.prefetch_consumed_bytes-ca.pp.prefetch_consumed_bytes;
    p.pp_timing.prefetch_launches=cb.pp.prefetch_launches-ca.pp.prefetch_launches;
    p.cpu_routes = cb.cpu_routes - ca.cpu_routes;
    p.cpu_logical_weight_bytes = cb.cpu_logical_weight_bytes - ca.cpu_logical_weight_bytes;
    p.cpu_compute_ns = cb.cpu_compute_ns - ca.cpu_compute_ns;
    p.native_slot_h2d_bytes = cb.dma_costs.native_slot_h2d_bytes - ca.dma_costs.native_slot_h2d_bytes;
    return p;
}
}
SessionRequest session_request(Runtime& runtime, const std::vector<llama_token>& input,
                               const std::filesystem::path& failure_path) {
    return session_request_count(runtime, input, failure_path, 32);
}
SessionRequest session_request_count(Runtime& runtime, const std::vector<llama_token>& input,
                               const std::filesystem::path& failure_path, int output_count,
                               bool stop_on_eog) {
    if (output_count < 1 || output_count > 2048) throw std::runtime_error("request output count exceeds gate bound");
    if(runtime.mtp_enabled())std::cerr<<"strata_mtp_stage=request_initial_sync\n"<<std::flush;
    llama_synchronize(runtime.context());
    if(runtime.mtp_enabled())std::cerr<<"strata_mtp_stage=request_reset\n"<<std::flush;
    runtime.reset();
    llama_synchronize(runtime.context());
    if(runtime.mtp_enabled())std::cerr<<"strata_mtp_stage=request_ready\n"<<std::flush;
    const auto transport_before = runtime.expert_transport_metrics();
    const auto cache_before = device_expert_cache_metrics(runtime.context());
    ExpertStagingMetrics pp_transport;
    DeviceCacheMetrics pp_cache;
    SessionRequest result;
    TgDiagnostics diagnostics(runtime,failure_path);
    const auto start = std::chrono::steady_clock::now();
    std::string error;
    try {
        if (!runtime.evaluate(input, error)) throw std::runtime_error(error);
        for (int i = 0; i < output_count; ++i) {
            const llama_token token = runtime.greedy(!stop_on_eog);
            if (!i) {
                result.ttft_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
                // First logits have completed PP. One boundary snapshot separates
                // actual phase transfers; no per-token/per-layer observer added.
                llama_synchronize(runtime.context());
                const auto prefill_perf=runtime.performance();result.pp_ms=prefill_perf.t_p_eval_ms;result.pp_tokens=prefill_perf.n_p_eval;
                pp_transport = runtime.expert_transport_metrics();
                pp_cache = device_expert_cache_metrics(runtime.context());
                if(pp_cache.read_ahead.jobs!=cache_before.read_ahead.jobs)
                    throw std::runtime_error("PP unexpectedly executed TG read-ahead");
                result.pp_cost = phase_cost(transport_before, cache_before, pp_transport, pp_cache);
            }
            if (stop_on_eog && runtime.is_eog(token)) break;
            result.output.push_back(token);
            if(runtime.mtp_enabled()&&output_count>1){
                const char* one_profile=std::getenv("STRATA_DSV4_TG_PROFILE_ONE_VERIFY");
                RequestGpuProfile capture(output_count==33&&!(one_profile&&!std::strcmp(one_profile,"1")),"STRATA_DSV4_TG_ACTIVITY_PROFILE");
                mtp_generate(runtime,input,result.output,output_count,stop_on_eog,
                    failure_path.parent_path()/"generation-progress.json",result.mtp,failure_path);
                result.mtp.draft_stages=runtime.mtp_stage_metrics();break;}
            if ((i + 1) % 32 == 0) {
                std::ofstream progress(failure_path.parent_path() / "generation-progress.json");
                progress << "{\"position\":" << runtime.position() << ",\"generated_outputs\":" << i + 1 << '}';
                if (!progress) throw std::runtime_error("request progress write failed");
            }
            if (i + 1 != output_count) {
                diagnostics.before(runtime.position());
                if(!runtime.evaluate({token},error))throw std::runtime_error(error);
                diagnostics.after();
            }
        }
    } catch (const std::exception& failure) {
        std::string reason = failure.what();
        try { verify_device_expert_cache(runtime.context()); }
        catch (const std::exception& cache_failure) { reason += "; " + std::string(cache_failure.what()); }
        const auto perf = runtime.performance();
        const auto cache = device_expert_cache_metrics(runtime.context());
        std::ofstream failed(failure_path);
        failed << "{\"error\":" << session_json(reason) << ",\"position\":" << runtime.position()
               << ",\"pp_tokens\":" << perf.n_p_eval << ",\"tg_tokens\":" << perf.n_eval
               << ",\"pp_ms\":" << perf.t_p_eval_ms << ",\"tg_ms\":" << perf.t_eval_ms
               << ",\"ttft_ms\":" << result.ttft_ms
               << ",\"host_miss_services\":" << cache.host_services << ",\"output_ids\":[";
        for (size_t i = 0; i < result.output.size(); ++i) { if (i) failed << ','; failed << result.output[i]; }
        failed << "],\"api_costs\":";
        write_cache_api_costs(failed, cache.dma_costs);
        failed << "}\n";
        throw std::runtime_error(reason);
    }
    llama_synchronize(runtime.context());
    result.request_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    diagnostics.save(failure_path);
    if(runtime.mtp_enabled()){
        std::ofstream trace(failure_path.string()+"-mtp-verification.json");trace<<"{\"whole_model_acceptance\":true,\"physical_target_routes_include_rejected_drafts\":true,\"steps\":[";
        auto host=[&](const MtpHostProbe& p){trace<<"{\"wall_ns\":"<<p.wall_ns<<",\"process_faults\":"<<p.process_faults
            <<",\"main_cpu_100ns\":"<<p.main_cpu_100ns<<",\"process_cpu_100ns\":"<<p.process_cpu_100ns
            <<",\"worker_cpu_100ns\":"<<p.worker_cpu_100ns<<'}';};
        for(size_t i=0;i<result.mtp.records.size();++i){const auto& r=result.mtp.records[i];if(i)trace<<',';trace<<"{\"position\":"<<r.position<<",\"pending\":"<<r.pending<<",\"draft\":"<<r.draft<<",\"target\":"<<r.target<<",\"bonus\":"<<r.bonus<<",\"positions_verified\":"<<r.positions_verified<<",\"accepted\":"<<(r.accepted?"true":"false")
            <<",\"wall_ms\":"<<r.wall_ms<<",\"draft_ms\":"<<r.draft_ms<<",\"verify_ms\":"<<r.verify_ms<<",\"post_ms\":"<<r.post_ms
            <<",\"draft_input_ms\":"<<r.draft_input_ms<<",\"draft_cpu_ms\":"<<r.draft_cpu_ms<<",\"draft_input_parts_ms\":[";
            for(size_t part=0;part<r.draft_input_parts_ms.size();++part){if(part)trace<<',';trace<<r.draft_input_parts_ms[part];}
            trace<<']'
            <<",\"top1_probability\":"<<r.top1_probability<<",\"top1_top2_logit_gap\":"<<r.top1_top2_logit_gap
            <<",\"confidence_measured\":"<<(r.confidence_measured?"true":"false")<<",\"before_draft\":";host(r.before_draft);
            trace<<",\"after_draft\":";host(r.after_draft);trace<<",\"after_verify\":";host(r.after_verify);trace<<'}';}
        trace<<"]}";if(!trace)throw std::runtime_error("MTP verification trace write failed");
    }
    verify_device_expert_cache(runtime.context());
    const auto perf = runtime.performance();
    if(runtime.mtp_enabled()){result.tg_ms=result.request_ms-result.ttft_ms;result.tg_tokens=int32_t(result.output.size())-1;}
    else {result.pp_ms=perf.t_p_eval_ms;result.pp_tokens=perf.n_p_eval;result.tg_ms=perf.t_eval_ms;result.tg_tokens=perf.n_eval;}
    const auto after = runtime.expert_transport_metrics();
    result.transport.bytes = after.bytes - transport_before.bytes;
    result.transport.calls = after.calls - transport_before.calls;
    result.transport.direct_bytes = after.direct_bytes - transport_before.direct_bytes;
    result.transport.direct_calls = after.direct_calls - transport_before.direct_calls;
    result.transport.host_copy_wall_ns = after.host_copy_wall_ns - transport_before.host_copy_wall_ns;
    result.transport.slab_wait_wall_ns = after.slab_wait_wall_ns - transport_before.slab_wait_wall_ns;
    result.cache = device_expert_cache_metrics(runtime.context());
    result.tg_cost = phase_cost(pp_transport, pp_cache, after, result.cache);
    if(runtime.mtp_enabled())result.tg_cost.total_routes=43ull*6*result.mtp.physical_target_positions+6ull*result.mtp.proposals;
    result.cache.pp.payload_bytes -= cache_before.pp.payload_bytes;
    result.cache.pp.dma_calls -= cache_before.pp.dma_calls;
    result.cache.pp.layers -= cache_before.pp.layers;
    result.cache.pp.groups -= cache_before.pp.groups;
    result.cache.pp.source_cpu_copy_ns -= cache_before.pp.source_cpu_copy_ns;
    result.cache.pp.cache_reused_bytes -= cache_before.pp.cache_reused_bytes;
    result.cache.pp.cache_copy_calls -= cache_before.pp.cache_copy_calls;
    result.cache.cpu_jobs -= cache_before.cpu_jobs;
    result.cache.cpu_routes -= cache_before.cpu_routes;
    result.cache.cpu_logical_weight_bytes -= cache_before.cpu_logical_weight_bytes;
    result.cache.cpu_compute_ns -= cache_before.cpu_compute_ns;
    result.cache.dma_costs.native_slot_h2d_bytes -= cache_before.dma_costs.native_slot_h2d_bytes;
    result.cache.dma_costs.native_slot_publish_calls -= cache_before.dma_costs.native_slot_publish_calls;
    result.cache.capacity.plans -= cache_before.capacity.plans;
    result.cache.capacity.planned_fills -= cache_before.capacity.planned_fills;
    result.cache.capacity.planned_evictions -= cache_before.capacity.planned_evictions;
    result.cache.capacity.repeated_fills -= cache_before.capacity.repeated_fills;
    result.cache.capacity.refills_within43_miss_plans -= cache_before.capacity.refills_within43_miss_plans;
    result.cache.capacity.incoming_score_not_greater -= cache_before.capacity.incoming_score_not_greater;
    result.cache.capacity.transient_fills -= cache_before.capacity.transient_fills;
    result.cache.capacity.reused_transient_routes -= cache_before.capacity.reused_transient_routes;
    result.cache.capacity.admission_bypasses -= cache_before.capacity.admission_bypasses;
    for (size_t i = 0; i < result.cache.gpu.size(); ++i) result.cache.gpu[i] -= cache_before.gpu[i];
    result.cache.host_services -= cache_before.host_services;
    auto& ra=result.cache.read_ahead;const auto& rb=cache_before.read_ahead;
    ra.jobs-=rb.jobs;ra.bytes-=rb.bytes;ra.wall_ns-=rb.wall_ns;ra.canceled-=rb.canceled;ra.cancel_tail_ns-=rb.cancel_tail_ns;
    ra.predicted_ids-=rb.predicted_ids;ra.correct_ids-=rb.correct_ids;ra.hash_jobs-=rb.hash_jobs;ra.hash_bytes-=rb.hash_bytes;
    ra.completed_jobs-=rb.completed_jobs;ra.canceled_after_completion-=rb.canceled_after_completion;
    ra.completed_first_expert_jobs-=rb.completed_first_expert_jobs;
    ra.selected_one_jobs-=rb.selected_one_jobs;ra.selected_two_jobs-=rb.selected_two_jobs;
    ra.observed_windows-=rb.observed_windows;ra.unobserved_windows-=rb.unobserved_windows;
    for(size_t i=0;i<ra.layer_window_ns.size();++i){
        ra.layer_window_ns[i]-=rb.layer_window_ns[i];ra.layer_read_ns[i]-=rb.layer_read_ns[i];
        ra.layer_window_count[i]-=rb.layer_window_count[i];
    }
    result.cache.background_bytes-=cache_before.background_bytes;result.cache.background_commits-=cache_before.background_commits;
    result.cache.background_staged_bytes-=cache_before.background_staged_bytes;
    result.cache.background_staged_experts-=cache_before.background_staged_experts;
    result.cache.directory_h2d_bytes -= cache_before.directory_h2d_bytes;
    result.cache.transport.bytes -= cache_before.transport.bytes;
    result.cache.transport.calls -= cache_before.transport.calls;
    result.cache.transport.direct_bytes -= cache_before.transport.direct_bytes;
    result.cache.transport.direct_calls -= cache_before.transport.direct_calls;
    result.cache.service_wall_ns -= cache_before.service_wall_ns;
    result.cache.transport.host_copy_wall_ns -= cache_before.transport.host_copy_wall_ns;
    result.cache.transport.slab_wait_wall_ns -= cache_before.transport.slab_wait_wall_ns;
    result.cache.dma_costs.submit_ns -= cache_before.dma_costs.submit_ns;
    result.cache.dma_costs.commit_wait_ns -= cache_before.dma_costs.commit_wait_ns;
    result.cache.dma_costs.frequency_d2h_bytes -= cache_before.dma_costs.frequency_d2h_bytes;
    result.cache.dma_costs.frequency_read_ns -= cache_before.dma_costs.frequency_read_ns;
    return result;
}
}
