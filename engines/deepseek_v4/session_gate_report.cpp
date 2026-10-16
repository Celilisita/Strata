#include "session_gate.hpp"
#include "device_cache_probe_report.hpp"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace strata::deepseek_v4 {
std::string session_json(const std::string& value) {
    std::ostringstream result;
    result << '"';
    for (unsigned char c : value) {
        if (c == '"' || c == '\\') result << '\\' << c;
        else if (c < 32) result << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
        else result << c;
    }
    result << '"';
    return result.str();
}
void session_write_request(std::ostream& out, const SessionRequest& r) {
    auto verify_samples=r.mtp.steady_verification_wall_ms;
    std::sort(verify_samples.begin(),verify_samples.end());
    const auto quantile=[&](double q){return verify_samples.empty()?0.0:verify_samples[std::min(verify_samples.size()-1,size_t(q*(verify_samples.size()-1)))];};
    out << "{\"mtp\":{\"proposals\":" << r.mtp.proposals << ",\"accepted\":" << r.mtp.accepted << ",\"acceptance\":" << (r.mtp.proposals ? double(r.mtp.accepted)/r.mtp.proposals : 0)
        << ",\"verification_calls\":" << r.mtp.verification_calls << ",\"physical_target_positions\":" << r.mtp.physical_target_positions
        << ",\"initialization_ms\":" << r.mtp.initialization_ms << ",\"draft_ms\":" << r.mtp.draft_ms << ",\"verify_ms\":" << r.mtp.verify_ms
        << ",\"catchup_ms\":" << r.mtp.catchup_ms << ",\"rollback_ms\":" << r.mtp.rollback_ms
        << ",\"draft_stages\":{\"calls\":" << r.mtp.draft_stages.calls
        << ",\"input_prepare_ms\":" << r.mtp.draft_stages.host_ms[0]
        << ",\"input_detail_ms\":{\"window_check\":" << r.mtp.draft_stages.input_sub_ms[0]
        << ",\"mask_build\":" << r.mtp.draft_stages.input_sub_ms[1]
        << ",\"mask_upload\":" << r.mtp.draft_stages.input_sub_ms[2]
        << ",\"hidden_upload\":" << r.mtp.draft_stages.input_sub_ms[3]
        << ",\"embedding_prepare_upload\":" << r.mtp.draft_stages.input_sub_ms[4]
        << ",\"ids_upload\":" << r.mtp.draft_stages.input_sub_ms[5]
        << ",\"positions_upload\":" << r.mtp.draft_stages.input_sub_ms[6]
        << ",\"host_staging_or_fallback_ids\":" << r.mtp.draft_stages.input_sub_ms[7] << '}'
        << ",\"front_graph_submit_ms\":" << r.mtp.draft_stages.host_ms[1]
        << ",\"finish_graph_submit_ms\":" << r.mtp.draft_stages.host_ms[2]
        << ",\"draft_token_read_ms\":" << r.mtp.draft_stages.host_ms[3]
        << ",\"kv_commit_ms\":" << r.mtp.draft_stages.host_ms[4]
        << ",\"cpu_miss_jobs\":" << r.mtp.draft_stages.cpu_miss_jobs
        << ",\"cpu_miss_service_ns\":" << r.mtp.draft_stages.cpu_miss_compute_ns << '}'
        << ",\"steady_verification_wall\":{\"samples\":" << verify_samples.size()
        << ",\"min_ms\":" << quantile(0) << ",\"p10_ms\":" << quantile(.1)
        << ",\"median_ms\":" << quantile(.5) << ",\"p90_ms\":" << quantile(.9)
        << ",\"max_ms\":" << quantile(1) << '}'
        << ",\"fusion_phase_aba\":" << (r.mtp.fusion_phase_aba?"true":"false")
        << ",\"input_phase_aba\":" << (r.mtp.input_phase_aba?"true":"false") << ",\"fusion_phases\":[";
    for(size_t i=0;i<r.mtp.phases.size();++i){
        const auto& p=r.mtp.phases[i];if(i)out<<',';
        const bool fusion_enabled=r.mtp.fusion_phase_aba ? i==1 : r.mtp.fusion_initially_enabled;
        const bool input_async=r.mtp.input_phase_aba ? i==1 : r.mtp.input_initially_async;
        out<<"{\"phase\":"<<i<<",\"fusion_enabled\":"<<(fusion_enabled?"true":"false")
           <<",\"input_async\":"<<(input_async?"true":"false")
           <<",\"proposals\":"<<p.proposals<<",\"accepted\":"<<p.accepted
           <<",\"verification_calls\":"<<p.verification_calls
           <<",\"single_position_calls\":"<<p.single_position_calls
           <<",\"generated_outputs\":"<<p.generated_outputs
           <<",\"wall_ms\":"<<p.wall_ms<<",\"draft_ms\":"<<p.draft_ms
           <<",\"verify_ms\":"<<p.verify_ms<<",\"catchup_ms\":"<<p.catchup_ms
           <<",\"rollback_ms\":"<<p.rollback_ms<<",\"switch_ms\":"<<p.switch_ms
           <<",\"steady_verifications\":"<<p.steady_verifications
           <<",\"steady_outputs\":"<<p.steady_outputs
           <<",\"steady_wall_ms\":"<<p.steady_wall_ms<<'}';
    }
    out<<"]},";
    out << std::setprecision(16) << "\"pp_ms\":" << r.pp_ms << ",\"tg_ms\":" << r.tg_ms
        << ",\"request_ms\":" << r.request_ms << ",\"ttft_ms\":" << r.ttft_ms << ",\"generated_outputs\":" << r.output.size() << ",\"pp_tokens\":" << r.pp_tokens
        << ",\"tg_tokens\":" << r.tg_tokens << ",\"pp_tps\":" << r.pp_tokens * 1000.0 / r.pp_ms
        << ",\"tg_tps\":" << r.tg_tokens * 1000.0 / r.tg_ms
        << ",\"expert_h2d_bytes\":" << r.transport.bytes + r.cache.transport.bytes + r.cache.pp.payload_bytes
        << ",\"expert_h2d_calls\":" << r.transport.calls + r.cache.transport.calls + r.cache.pp.dma_calls
        << ",\"expert_direct_bytes\":" << r.transport.direct_bytes + r.cache.transport.direct_bytes
        << ",\"expert_cpu_copy_ns\":" << r.transport.host_copy_wall_ns + r.cache.transport.host_copy_wall_ns + r.cache.pp.source_cpu_copy_ns
        << ",\"expert_slab_wait_ns\":" << r.transport.slab_wait_wall_ns + r.cache.transport.slab_wait_wall_ns
        << ",\"device_cache_directory_h2d_bytes\":" << r.cache.directory_h2d_bytes
        << ",\"host_miss_services\":" << r.cache.host_services
        << ",\"miss_service_ns\":" << r.cache.service_wall_ns
        << ",\"miss_memcpy_ns\":" << r.cache.transport.host_copy_wall_ns
        << ",\"miss_slab_wait_ns\":" << r.cache.transport.slab_wait_wall_ns
        << ",\"miss_submit_ns\":" << r.cache.dma_costs.submit_ns
        << ",\"miss_commit_wait_ns\":" << r.cache.dma_costs.commit_wait_ns
        << ",\"history_d2h_bytes\":" << r.cache.dma_costs.frequency_d2h_bytes
        << ",\"history_read_ns\":" << r.cache.dma_costs.frequency_read_ns
        << ",\"background_admission_weight_bytes\":" << r.cache.background_bytes << ",\"idle_directory_commits\":" << r.cache.background_commits
        << ",\"background_staged_bytes\":" << r.cache.background_staged_bytes << ",\"background_staged_experts\":" << r.cache.background_staged_experts
        << ",\"native_cpu_jobs\":" << r.cache.cpu_jobs << ",\"native_cpu_routes\":" << r.cache.cpu_routes
        << ",\"cpu_logical_weight_bytes_not_dma\":" << r.cache.cpu_logical_weight_bytes
        << ",\"cpu_compute_ns\":" << r.cache.cpu_compute_ns
        << ",\"native_slot_control_h2d_bytes\":" << r.cache.dma_costs.native_slot_h2d_bytes
        << ",\"read_ahead\":{\"jobs\":"<<r.cache.read_ahead.jobs<<",\"bytes\":"<<r.cache.read_ahead.bytes
        <<",\"wall_ns_overlapped\":"<<r.cache.read_ahead.wall_ns<<",\"canceled\":"<<r.cache.read_ahead.canceled
        <<",\"predicted_ids_descriptor_not_exact_touched_rows\":"<<r.cache.read_ahead.predicted_ids<<",\"correct_missing_ids\":"<<r.cache.read_ahead.correct_ids
        <<",\"hash_jobs\":"<<r.cache.read_ahead.hash_jobs<<",\"hash_bytes\":"<<r.cache.read_ahead.hash_bytes
        <<",\"completed_jobs\":"<<r.cache.read_ahead.completed_jobs
        <<",\"completed_first_expert_jobs\":"<<r.cache.read_ahead.completed_first_expert_jobs
        <<",\"canceled_after_completion\":"<<r.cache.read_ahead.canceled_after_completion
        <<",\"selected_one_jobs\":"<<r.cache.read_ahead.selected_one_jobs
        <<",\"selected_two_jobs\":"<<r.cache.read_ahead.selected_two_jobs
        <<",\"observed_windows\":"<<r.cache.read_ahead.observed_windows
        <<",\"unobserved_windows\":"<<r.cache.read_ahead.unobserved_windows
        <<",\"window_by_layer\":[";
    for(size_t i=0;i<r.cache.read_ahead.layer_window_ns.size();++i){
        if(i)out<<',';
        out<<"{\"layer\":"<<i<<",\"count\":"<<r.cache.read_ahead.layer_window_count[i]
           <<",\"available_ns\":"<<r.cache.read_ahead.layer_window_ns[i]
           <<",\"read_ns\":"<<r.cache.read_ahead.layer_read_ns[i]<<'}';
    }
    out<<"]}"
        << ",\"device_cache_counters\":[";
    for (size_t i = 0; i < r.cache.gpu.size(); ++i) { if (i) out << ','; out << r.cache.gpu[i]; }
    out << "],\"phase_transfer_measurements\":{";
    for (int i = 0; i < 2; ++i) {
        const auto& p = i ? r.tg_cost : r.pp_cost;
        if (i) out << ',';
        out << (i ? "\"tg\":{" : "\"pp\":{") << "\"weight_h2d_bytes\":" << p.weight_bytes
            << ",\"weight_h2d_calls\":" << p.weight_calls << ",\"source_cpu_copy_ns\":" << p.source_cpu_copy_ns
            << ",\"requested_routes\":" << p.total_routes << ",\"lookups\":" << p.lookups << ",\"initial_hit_routes\":" << p.initial_hit_routes
            << ",\"initial_missing_routes\":" << p.total_routes - p.initial_hit_routes
            << ",\"host_miss_services\":" << p.host_services << ",\"native_pp_layers\":" << p.native_pp_layers
            << ",\"native_pp_groups\":" << p.native_pp_groups
            << ",\"pp_cache_reused_bytes\":" << p.pp_cache_reused_bytes
            << ",\"pp_device_alloc_ns\":" << p.pp_timing.device_alloc_ns
            << ",\"pp_device_alloc_calls\":" << p.pp_timing.device_alloc_calls
            << ",\"pp_staging_wait_ns\":" << p.pp_timing.staging_wait_ns
            << ",\"pp_staged_bytes\":" << p.pp_timing.staged_bytes
            << ",\"pp_cache_copy_calls\":" << p.pp_timing.cache_copy_calls
            << ",\"pp_timed_layers\":" << p.pp_timing.timed_layers
            << ",\"pp_gpu_layer_ns\":" << p.pp_timing.gpu_layer_ns
            << ",\"pp_gpu_weight_wait_ns\":" << p.pp_timing.gpu_weight_wait_ns
            << ",\"pp_gpu_products_ns\":" << p.pp_timing.gpu_products_ns
            << ",\"pp_gpu_prelude_ns\":" << p.pp_timing.gpu_prelude_ns
            << ",\"pp_gpu_intergroup_ns\":" << p.pp_timing.gpu_intergroup_ns
            << ",\"pp_gpu_tail_ns\":" << p.pp_timing.gpu_tail_ns
            << ",\"pp_prefetch_bytes\":" << p.pp_timing.prefetch_bytes
            << ",\"pp_prefetch_consumed_bytes\":" << p.pp_timing.prefetch_consumed_bytes
            << ",\"pp_prefetch_launches\":" << p.pp_timing.prefetch_launches
            << ",\"cpu_routes\":" << p.cpu_routes << ",\"cpu_logical_weight_bytes_not_dma\":" << p.cpu_logical_weight_bytes
            << ",\"cpu_compute_ns\":" << p.cpu_compute_ns << ",\"native_slot_control_h2d_bytes\":" << p.native_slot_h2d_bytes << '}';
    }
    const auto& c = r.cache.capacity;
    out << "},\"cache_capacity_and_fill_audit\":{\"permanent_slots\":" << c.permanent_slots
        << ",\"transient_slots_in_same_bank\":" << c.transient_slots
        << ",\"model_expert_keys\":" << c.model_expert_keys << ",\"resident_keys\":" << c.resident_keys
        << ",\"weight_tensor_bytes\":" << c.weight_tensor_bytes << ",\"backend_alloc_bytes\":" << c.backend_alloc_bytes
        << ",\"invalid_directory_links\":" << c.invalid_directory_links << ",\"miss_service_plans\":" << c.plans
        << ",\"planned_fills\":" << c.planned_fills << ",\"planned_evictions\":" << c.planned_evictions
        << ",\"repeated_fills\":" << c.repeated_fills << ",\"refills_within43_miss_plans\":" << c.refills_within43_miss_plans
        << ",\"incoming_score_not_greater\":" << c.incoming_score_not_greater
        << ",\"transient_fills\":" << c.transient_fills << ",\"reused_transient_routes\":" << c.reused_transient_routes
        << ",\"admission_bypasses\":" << c.admission_bypasses
        << ",\"full_route_reuse_distance_measured\":false},\"output_ids\":[";
    for (size_t i = 0; i < r.output.size(); ++i) { if (i) out << ','; out << r.output[i]; }
    out << "],\"device_api_costs_context_max\":";
    write_cache_api_costs(out, r.cache.dma_costs);
    out << '}';
}
void session_write_layer(std::ostream& out, const LayerGateResult& r) {
    out << std::setprecision(16) << "{\"token\":" << r.token << ",\"exact_tensors\":" << r.compared
        << ",\"all_hit\":" << r.all_hit << ",\"misses\":" << r.misses
        << ",\"hit_routes\":" << r.hit_routes << ",\"expert_h2d_bytes\":" << r.bytes
        << ",\"candidate_ms\":" << r.candidate_ms << ",\"control_after_fill_ms\":" << r.control_ms
        << ",\"service_ns\":" << r.service_ns << ",\"memcpy_ns\":" << r.memcpy_ns
        << ",\"slab_wait_ns\":" << r.slab_wait_ns << ",\"submit_ns\":" << r.submit_ns
        << ",\"commit_wait_ns\":" << r.commit_wait_ns << '}';
}
}
