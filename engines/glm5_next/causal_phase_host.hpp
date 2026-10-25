#pragma once
#include "causal_phases.hpp"
#include <array>
#include <chrono>
#include <cstdint>
#include <limits>

namespace strata::glm5_next {
inline uint64_t causal_host_ns(std::chrono::steady_clock::time_point value) noexcept {
    const auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(value.time_since_epoch()).count();
    return ns>0?uint64_t(ns):0;
}
inline uint64_t causal_host_now_ns() noexcept {
    return causal_host_ns(std::chrono::steady_clock::now());
}
struct CausalHostInterval { uint64_t sum_ns=0,max_ns=0,count=0; };
struct CausalHostGroup {
    uint64_t sampled_tickets=0,completed=0,identity_mismatches=0,clock_order_mismatches=0;
    uint64_t incomplete_samples=0,service_errors=0,notice_to_reply_sum_ns=0;
    std::array<CausalHostInterval,4> intervals{};
};
struct CausalHostSnapshot {
    bool enabled=false;
    uint64_t owner_epoch=0,unclassified_identity_mismatches=0;
    std::array<CausalHostGroup,2> groups{};
};
// One worker-local ticket. No packet pointer survives the response release.
struct CausalHostSample {
    bool active=false,identity_valid=false,finished=false;
    uint32_t ticket=0,routes=0,compute_calls=0;
    int layer=-1,group=-1;
    uint64_t owner_epoch=0,notice_ns=0,service_start_ns=0,compute_begin_ns=0,compute_end_ns=0;
    void compute_begin(std::chrono::steady_clock::time_point value) noexcept {
        if(active){++compute_calls;compute_begin_ns=causal_host_ns(value);}
    }
    void compute_end(std::chrono::steady_clock::time_point value) noexcept {
        if(active)compute_end_ns=causal_host_ns(value);
    }
};
// Embedded in the existing DeviceExpertCache stats owner. All mutations and
// snapshots use its service_mutex_; there is no separate lock or reset API.
// A new owner starts a new epoch. Default-off never reads a clock here.
class CausalHostStats {
public:
    explicit CausalHostStats(bool enabled=false) noexcept { data_.enabled=enabled; }
    bool candidate(uint32_t ticket) const noexcept {
        return data_.enabled&&causal_phase_sample_ticket(ticket);
    }
    CausalHostSample begin(uint32_t ticket,uint64_t owner,uint64_t packet_owner,int layer,
        uint32_t request,uint32_t routes,uint64_t notice,uint64_t service_start) const noexcept {
        CausalHostSample out;
        if(!candidate(ticket)||request!=5||(routes!=8&&routes!=16))return out;
        out.active=true;out.ticket=ticket;out.owner_epoch=owner;out.layer=layer;out.routes=routes;
        out.notice_ns=notice;out.service_start_ns=service_start;
        out.identity_valid=owner!=0&&owner==packet_owner&&causal_phase_layer_valid(layer);
        if(causal_phase_layer_valid(layer))out.group=causal_phase_group(layer);
        return out;
    }
    void complete(CausalHostSample& sample,bool identity_before_release,bool service_error,
                  uint64_t native_reply_call_return_upper_bound) noexcept {
        if(!sample.active)return;
        if(sample.group<0||sample.group>1){++data_.unclassified_identity_mismatches;return;}
        auto& group=data_.groups[size_t(sample.group)];++group.sampled_tickets;
        const bool duplicate=sample.finished;sample.finished=true;
        if(!sample.identity_valid||!identity_before_release||duplicate){++group.identity_mismatches;return;}
        if(service_error){++group.service_errors;return;}
        if(sample.compute_calls!=1||!sample.compute_begin_ns||!sample.compute_end_ns){++group.incomplete_samples;return;}
        const std::array<uint64_t,5> points{sample.notice_ns,sample.service_start_ns,sample.compute_begin_ns,
            sample.compute_end_ns,native_reply_call_return_upper_bound};
        if(!points[0]){++group.clock_order_mismatches;return;}
        uint64_t closure=0;
        std::array<uint64_t,4> elapsed{};
        for(size_t i=0;i<elapsed.size();++i){
            if(points[i+1]<points[i]){++group.clock_order_mismatches;return;}
            elapsed[i]=points[i+1]-points[i];closure+=elapsed[i];
        }
        if(closure!=points.back()-points.front()||
            group.notice_to_reply_sum_ns>(std::numeric_limits<uint64_t>::max)()-closure){
            ++group.clock_order_mismatches;return;
        }
        for(size_t i=0;i<elapsed.size();++i)
            if(group.intervals[i].sum_ns>(std::numeric_limits<uint64_t>::max)()-elapsed[i]){
                ++group.clock_order_mismatches;return;
            }
        ++group.completed;group.notice_to_reply_sum_ns+=closure;
        for(size_t i=0;i<elapsed.size();++i){auto& interval=group.intervals[i];
            interval.sum_ns+=elapsed[i];if(elapsed[i]>interval.max_ns)interval.max_ns=elapsed[i];++interval.count;}
    }
    CausalHostSnapshot snapshot(uint64_t owner_epoch) const noexcept {
        auto out=data_;out.owner_epoch=owner_epoch;return out;
    }
private:
    CausalHostSnapshot data_{};
};

template<class Json>
Json causal_host_json(const CausalHostSnapshot& after,const CausalHostSnapshot& before) {
    bool reset=after.owner_epoch!=before.owner_epoch;
    const auto decreased=[](const CausalHostGroup& a,const CausalHostGroup& b){
        if(a.sampled_tickets<b.sampled_tickets||a.completed<b.completed||a.identity_mismatches<b.identity_mismatches||
            a.clock_order_mismatches<b.clock_order_mismatches||a.incomplete_samples<b.incomplete_samples||
            a.service_errors<b.service_errors||a.notice_to_reply_sum_ns<b.notice_to_reply_sum_ns)return true;
        for(size_t i=0;i<a.intervals.size();++i)
            if(a.intervals[i].sum_ns<b.intervals[i].sum_ns||a.intervals[i].count<b.intervals[i].count)return true;
        return false;
    };
    for(size_t g=0;g<after.groups.size();++g)reset|=decreased(after.groups[g],before.groups[g]);
    reset|=after.unclassified_identity_mismatches<before.unclassified_identity_mismatches;
    const auto delta=[&](uint64_t a,uint64_t b){return a-(reset?0:b);};
    const char* names[]{"notice_to_service_lock_acquired","service_start_to_CPU_compute_begin",
        "CPU_compute_existing_timer","compute_end_to_native_reply_call_return_upper_bound"};
    Json groups=Json::array();bool passed=after.enabled&&!reset;
    for(size_t g=0;g<after.groups.size();++g){const auto& a=after.groups[g];const auto& b=before.groups[g];
        const auto samples=delta(a.sampled_tickets,b.sampled_tickets),completed=delta(a.completed,b.completed);
        const auto identity=delta(a.identity_mismatches,b.identity_mismatches),clock=delta(a.clock_order_mismatches,b.clock_order_mismatches);
        const auto incomplete=delta(a.incomplete_samples,b.incomplete_samples),errors=delta(a.service_errors,b.service_errors);
        Json intervals=Json::object();uint64_t closure=0;bool counts=true;
        for(size_t i=0;i<a.intervals.size();++i){const auto sum=delta(a.intervals[i].sum_ns,b.intervals[i].sum_ns);
            const auto count=delta(a.intervals[i].count,b.intervals[i].count);counts&=count==completed;closure+=sum;
            intervals[names[i]]={{"sum_ns",sum},{"count",count},{"mean_ns",count?double(sum)/count:0.0},
                {"max_ns_counter_epoch",a.intervals[i].max_ns}};}
        const auto total=delta(a.notice_to_reply_sum_ns,b.notice_to_reply_sum_ns);
        const bool valid=counts&&samples==completed&&closure==total&&!identity&&!clock&&!incomplete&&!errors;
        passed&=valid;
        groups.push_back({{"group",g?"NextN45":"primary3_to44"},{"sampled_tickets",samples},{"completed_samples",completed},
            {"identity_mismatches",identity},{"clock_order_mismatches",clock},{"incomplete_samples",incomplete},{"service_errors",errors},
            {"interval_counts_passed",counts},{"closure_passed",closure==total},{"notice_to_reply_sum_ns",total},
            {"passed",valid},{"intervals",intervals}});
    }
    const auto unclassified=delta(after.unclassified_identity_mismatches,before.unclassified_identity_mismatches);
    passed&=!unclassified;
    return {{"enabled",after.enabled},{"passed",passed},{"owner_epoch",after.owner_epoch},{"counter_reset",reset},
        {"unclassified_identity_mismatches",unclassified},{"sample_rule","nonzero ticket sequence modulo16 equals0; overlapped GLM routes8/16 only"},
        {"clock","host steady_clock only; no subtraction from GPU timestamps"},{"groups",groups},
        {"max_scope","current stats-owner epoch; cumulative maximum is not a per-request maximum"},
        {"release_boundary","native_reply_call_return_upper_bound: timestamp immediately after the actual release call returns, before the worker's existing final fence; not GPU visibility or consumption"},
        {"scope","Four adjacent host intervals partition sampled notice-to-release-call-return only. Not full request critical path; no raw trace or added owner/store/pool/stream."}};
}
} // namespace strata::glm5_next
