#pragma once
// Diagnostic-only fixed counters. No payload owner, packet, clock correlation or trace.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
namespace strata::glm5_next {
inline constexpr size_t causal_phase_first_reserved_layer=55,causal_phase_last_reserved_layer=62;
inline constexpr size_t causal_phase_publish_primary=173,causal_phase_publish_nextn=174,causal_phase_mismatches=175;
inline constexpr size_t causal_phase_primary_stats=176,causal_phase_nextn_stats=185;
inline constexpr size_t causal_phase_scratch_publish=194,causal_phase_scratch_tag=195,causal_phase_scratch_epoch=196;
inline constexpr uint32_t causal_phase_sample_period=16;
static_assert(8+3*55==causal_phase_publish_primary&&8+3*62==causal_phase_scratch_publish&&causal_phase_scratch_epoch+1==197);
#if defined(__CUDACC__)
#define STRATA_GLM53_PHASE_HD __host__ __device__
#else
#define STRATA_GLM53_PHASE_HD
#endif
STRATA_GLM53_PHASE_HD constexpr bool causal_phase_sample_ticket(uint32_t ticket){return ticket!=0&&(ticket&15u)==0;}
STRATA_GLM53_PHASE_HD constexpr bool causal_phase_layer_valid(int layer){return layer>=3&&layer<46;}
STRATA_GLM53_PHASE_HD constexpr int causal_phase_group(int layer){return layer==45?1:0;}
STRATA_GLM53_PHASE_HD constexpr uint64_t causal_phase_tag(uint32_t ticket,int layer){return (uint64_t(ticket)<<32)|uint32_t(layer);}
#undef STRATA_GLM53_PHASE_HD
inline bool causal_phase_reserved_layer(size_t layer){return layer>=causal_phase_first_reserved_layer&&layer<=causal_phase_last_reserved_layer;}
inline bool causal_phases_requested() {
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_GLM53_CAUSAL_PHASES");
        if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))throw std::runtime_error("GLM causal phases must be0 or1");
        return value&&!std::strcmp(value,"1");
    }();return enabled;
}
// Maxima are counter-epoch maxima; only sums/counts/publication/error counters use deltas.
// Scratch describes the last sampled publication and is never differenced.
template<class Json,size_t N>
Json causal_gpu_json(const std::array<uint64_t,N>& after,const std::array<uint64_t,N>& before) {
    static_assert(N>=200);bool reset=false,paired=true;
    const auto delta=[&](size_t i){if(after[i]<before[i]){reset=true;return uint64_t(0);}return after[i]-before[i];};
    const uint64_t mismatches=delta(causal_phase_mismatches);Json groups=Json::array();
    const char* phases[]{"lookup_release_to_join_entry","join_entry_to_response_acquire","lookup_release_to_response_acquire"};
    for(int group=0;group<2;++group) {
        const uint64_t published=delta(causal_phase_publish_primary+group);Json intervals=Json::object();
        const size_t base=group?causal_phase_nextn_stats:causal_phase_primary_stats;
        for(size_t phase=0;phase<3;++phase) {
            const size_t at=base+3*phase;const uint64_t sum=delta(at),count=delta(at+2);paired&=count==published;
            intervals[phases[phase]]={{"sum_ns",sum},{"count",count},{"mean_ns",count?double(sum)/count:0.0},
                {"max_ns_counter_epoch",after[at+1]},{"counter_indices",{at,at+1,at+2}}};
        }
        groups.push_back({{"cohort",group?"NextN45":"primary3to44"},{"sampled_publications",published},{"intervals",intervals}});
    }
    Json raw_before=Json::array(),raw_after=Json::array();
    for(size_t i=causal_phase_publish_primary;i<causal_phase_scratch_publish;++i){raw_before.push_back(before[i]);raw_after.push_back(after[i]);}
    return {{"enabled",true},{"passed",!reset&&paired&&!mismatches},{"counter_reset",reset},{"paired_counts_passed",paired},
        {"identity_or_clock_order_mismatches",mismatches},{"groups",groups},{"sample_period",causal_phase_sample_period},
        {"sample_rule","nonzero service5 cold ticket sequence%16==0; routes8/16 only"},{"clock","GPU globaltimer only; no host/GPU subtraction"},
        {"P","after mapped activation/IDs payload and fences, immediately before sequence release; timestamp precedes diagnostic scratch stores"},
        {"J","representative CTA0 thread0 join poll entry, after original ticket/status validation"},
        {"R","representative CTA0 thread0 after successful response acquire loop and packet error check"},
        {"interpretation","PtoJ contains cached/shared work and scheduling gaps, not pure kernel active time. JtoR is representative response-observation residency, not CPU compute. R is device observation, not host release time."},
        {"sampling_limitations","Deterministic ticket sampling; not a random sample or per-layer latency histogram. GPU globaltimer may span WDDM descheduling."},
        {"max_scope","counter epoch, not request maximum"},{"raw_interval_counter_first",causal_phase_publish_primary},
        {"raw_interval_counters_before",raw_before},{"raw_interval_counters_after",raw_after},
        {"last_publication_scratch",{{"publish_ns",after[causal_phase_scratch_publish]},{"ticket_layer_tag",after[causal_phase_scratch_tag]},
            {"owner_epoch",after[causal_phase_scratch_epoch]},{"scope","last publication in counter epoch; never differenced"}}},
        {"reserved_counter_layers",{55,56,57,58,59,60,61,62}},{"new_GPU_allocations",0},{"new_streams",0},{"new_kernels",0},
        {"diagnostic_failure_policy","nonzero mismatch, reset or unmatched publication/phase counts fails this diagnostic; no production status/math mutation"}};
}
}
