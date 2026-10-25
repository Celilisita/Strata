#pragma once
#include "runtime.hpp"
#include "device_cache.hpp"
#include "../glm5_next/critical_intervals.hpp"
#include "../glm5_next/phase_capacity_report.hpp"
#include <cstdio>
namespace strata::deepseek_v4 {
struct ServeMetrics {
    ExpertStagingMetrics transport;
    DeviceCacheMetrics cache;
    bool critical=false;
    explicit ServeMetrics(Runtime& r):transport(r.expert_transport_metrics()),cache(device_expert_cache_metrics(r.context())){
        if(std::getenv("STRATA_GLM53_CRITICAL_INTERVALS")&&r.metadata("general.architecture")=="glm5-next")
            critical=strata::glm5_next::critical_intervals_requested();
    }
    void report(Runtime& r)const{
        const auto t=r.expert_transport_metrics();const auto c=device_expert_cache_metrics(r.context());bool reset=false;
        const auto delta=[&](uint64_t now,uint64_t before){if(now<before){reset=true;return now;}return now-before;};
        const auto h2d=delta(t.bytes,transport.bytes)+delta(c.transport.bytes,cache.transport.bytes);
        const auto copies=delta(t.host_copy_bytes,transport.host_copy_bytes)+delta(c.transport.host_copy_bytes,cache.transport.host_copy_bytes);
        const auto cpu=delta(c.cpu_logical_weight_bytes,cache.cpu_logical_weight_bytes);
        const auto jobs=delta(c.cpu_jobs,cache.cpu_jobs),routes=delta(c.cpu_routes,cache.cpu_routes);
        const auto background=delta(c.background_bytes,cache.background_bytes);
        auto gpu=c.gpu;for(size_t i=0;i<gpu.size();++i)gpu[i]=delta(c.gpu[i],cache.gpu[i]);
        std::fprintf(stderr,"strata_request_counters expert_h2d_bytes=%llu host_copy_bytes=%llu cpu_logical_weight_bytes=%llu cpu_jobs=%llu cpu_routes=%llu background_bytes=%llu counter_reset=%d gpu=[",
            (unsigned long long)h2d,(unsigned long long)copies,(unsigned long long)cpu,
            (unsigned long long)jobs,(unsigned long long)routes,
            (unsigned long long)background,int(reset));
        for(size_t i=0;i<gpu.size();++i)std::fprintf(stderr,"%s%llu",i?",":"",(unsigned long long)gpu[i]);
        std::fprintf(stderr,"] cpu_service_threads=%d phase_budget=[",c.cpu_service_threads);
        for(size_t i=0;i<c.last_phase_budget.size();++i)
            std::fprintf(stderr,"%s%llu",i?",":"",(unsigned long long)c.last_phase_budget[i]);
        std::fputs("]\n",stderr);std::fflush(stderr);
        strata::glm5_next::report_phase_capacity(c.glm_capacity);
        if(critical)strata::glm5_next::report_critical_intervals(transport,cache,t,c);
    }
};
} // namespace strata::deepseek_v4
