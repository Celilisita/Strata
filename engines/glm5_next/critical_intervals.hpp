#pragma once
#include "../deepseek_v4/device_cache_protocol.hpp"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::glm5_next {
// Validated GLM has primary layers0..44 and optional NextN45. Reuse the
// existing layer63 counter triplet; no buffer, packet or allocation grows.
inline constexpr size_t critical_counter_layer=63;
inline constexpr size_t critical_join_sum=8+3*critical_counter_layer;
inline constexpr size_t critical_join_max=critical_join_sum+1;
inline constexpr size_t critical_join_count=critical_join_sum+2;
static_assert(deepseek_v4::device_cache_max_layers==64&&
              deepseek_v4::device_cache_counter_count==200&&critical_join_count==199);

inline bool critical_intervals_requested() {
    // Process option: immutable across captured CUDA graphs and requests.
    static const bool enabled=[] {
        const char* value=std::getenv("STRATA_GLM53_CRITICAL_INTERVALS");
        if(value&&std::strcmp(value,"0")&&std::strcmp(value,"1"))
            throw std::runtime_error("GLM critical intervals must be0 or1");
        return value&&!std::strcmp(value,"1");
    }();
    return enabled;
}
struct CriticalJoinInterval {
    uint64_t sum_ns=0,count=0,max_ns_counter_epoch=0;
    bool counter_reset=false;
};
template<size_t N>
CriticalJoinInterval critical_join_delta(const std::array<uint64_t,N>& after,
                                         const std::array<uint64_t,N>& before) {
    static_assert(N>critical_join_count);
    CriticalJoinInterval result;
    result.counter_reset=after[critical_join_sum]<before[critical_join_sum]||
                         after[critical_join_count]<before[critical_join_count];
    result.sum_ns=after[critical_join_sum]-(result.counter_reset?0:before[critical_join_sum]);
    result.count=after[critical_join_count]-(result.counter_reset?0:before[critical_join_count]);
    // A cumulative maximum cannot be differenced into a request maximum.
    result.max_ns_counter_epoch=after[critical_join_max];
    return result;
}
template<class Json,size_t N>
Json critical_join_json(const std::array<uint64_t,N>& after,const std::array<uint64_t,N>& before) {
    const auto sample=critical_join_delta(after,before);
    return {{"enabled",true},{"scope","representative_CTA0_response_poll"},
        {"sum_ns",sample.sum_ns},{"count",sample.count},
        {"mean_ns",sample.count?double(sample.sum_ns)/sample.count:0.0},
        {"max_ns_counter_epoch",sample.max_ns_counter_epoch},{"counter_reset",sample.counter_reset},
        {"interpretation","sampled join residency lower bound; not CPU compute or complete GPU kernel time"},
        {"eligible_path","overlapped GLM joins, primary and NextN together"},
        {"limitations","CTA0 may start late; globaltimer interval may include WDDM descheduling"},
        {"max_scope","current counter epoch, not per-request maximum"},
        {"zero_wait","response already acquired at first poll records zero"},
        {"clock","GPU globaltimer; no host/GPU clock subtraction"},
        {"counter_indices",{critical_join_sum,critical_join_max,critical_join_count}}};
}

// Existing service timers can overlap one another and the observed GPU poll.
// Export their deltas separately. Never add/subtract them as exclusive wall.
template<class Transport,class Cache>
void report_critical_intervals(const Transport& before_transport,const Cache& before,
                              const Transport& after_transport,const Cache& after) {
    bool reset=false;
    auto delta=[&](uint64_t a,uint64_t b){if(a<b){reset=true;return a;}return a-b;};
    const auto cpu=delta(after.cpu_compute_ns,before.cpu_compute_ns);
    const auto service=delta(after.service_wall_ns,before.service_wall_ns);
    const auto planning=delta(after.planning_ns,before.planning_ns);
    const auto projection=delta(after.projection_ns,before.projection_ns);
    const auto phase_resize=delta(after.phase_resize_ns,before.phase_resize_ns);
    const auto phase_seed=delta(after.phase_seed_ns,before.phase_seed_ns);
    const auto copy=delta(after_transport.host_copy_wall_ns,before_transport.host_copy_wall_ns)+
                    delta(after.transport.host_copy_wall_ns,before.transport.host_copy_wall_ns);
    const auto slab=delta(after_transport.slab_wait_wall_ns,before_transport.slab_wait_wall_ns)+
                    delta(after.transport.slab_wait_wall_ns,before.transport.slab_wait_wall_ns);
    const auto submit=delta(after.dma_costs.submit_ns,before.dma_costs.submit_ns);
    const auto commit=delta(after.dma_costs.commit_wait_ns,before.dma_costs.commit_wait_ns);
    const auto budget=delta(after.dma_costs.enqueue_budget_wait_ns,before.dma_costs.enqueue_budget_wait_ns);
    const auto miss=delta(after.dma_costs.enqueue_miss_wait_ns,before.dma_costs.enqueue_miss_wait_ns);
    const auto frequency=delta(after.dma_costs.frequency_read_ns,before.dma_costs.frequency_read_ns);
    const auto preupload=delta(after.dma_costs.graph_preupload_ns,before.dma_costs.graph_preupload_ns);
    const auto join=critical_join_delta(after.gpu,before.gpu);
    std::fprintf(stderr,"strata_critical_intervals cpu_compute_ns=%llu host_service_ns=%llu planning_ns=%llu projection_ns=%llu phase_resize_ns=%llu phase_seed_ns=%llu host_copy_ns=%llu slab_wait_ns=%llu dma_submit_ns=%llu directory_wait_ns=%llu enqueue_budget_wait_ns=%llu enqueue_miss_wait_ns=%llu frequency_read_ns=%llu graph_preupload_ns=%llu join_CTA0_poll_sum_ns=%llu join_CTA0_poll_count=%llu join_CTA0_poll_max_ns_counter_epoch=%llu counter_reset=%d service_intervals_overlap=1 join_is_sampled_residency_lower_bound=1\n",
        (unsigned long long)cpu,(unsigned long long)service,(unsigned long long)planning,
        (unsigned long long)projection,(unsigned long long)phase_resize,(unsigned long long)phase_seed,
        (unsigned long long)copy,(unsigned long long)slab,(unsigned long long)submit,
        (unsigned long long)commit,(unsigned long long)budget,(unsigned long long)miss,
        (unsigned long long)frequency,(unsigned long long)preupload,(unsigned long long)join.sum_ns,
        (unsigned long long)join.count,(unsigned long long)join.max_ns_counter_epoch,int(reset||join.counter_reset));
    std::fflush(stderr);
}
} // namespace strata::glm5_next
