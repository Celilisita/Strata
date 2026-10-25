#pragma once
#include "pp_stream_plan.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

namespace strata::deepseek_v4 {
// A request option, read once at the native PP run boundary.
inline bool pp_miss_coalesce_requested() {
    const char* value=std::getenv("STRATA_GLM53_PP_MISS_COALESCE");
    if(!value||!std::strcmp(value,"0"))return false;
    if(!std::strcmp(value,"1"))return true;
    throw std::runtime_error("PP miss coalesce option must be0 or1");
}
struct PpMissCoalesceResult {
    size_t groups_before=0,groups_after=0;
    bool changed()const{return groups_after<groups_before;}
};
inline void pp_validate_miss_group(const PpStreamGroup& group,const int32_t* directory) {
    if(group.source_keys.empty())return;
    if(group.experts<1||group.experts>32||group.rows<1||group.rows>8192||group.cache_first_slot>=0||
        group.source_keys.size()!=size_t(group.experts)||group.bounds.size()!=size_t(group.experts+1)||
        group.bounds.front()!=0||group.bounds.back()!=group.rows||group.source_runs.empty())
        throw std::runtime_error("PP coalesced group shape differs");
    int local=0,max_rows=0;
    for(const auto& run:group.source_runs) {
        if(run.local_first!=local||run.count<1||run.count>group.experts-local)
            throw std::runtime_error("PP coalesced source run differs");
        for(int j=0;j<run.count;++j,++local) {
            const int key=group.source_keys[local],rows=group.bounds[local+1]-group.bounds[local];
            if(key!=run.original_first+j||key<0||key>=288||(directory&&directory[key]>=0)||
                (local&&key<=group.source_keys[local-1])||rows<1)
                throw std::runtime_error("PP coalesced source key/row projection differs");
            max_rows=std::max(max_rows,rows);
        }
    }
    if(local!=group.experts||max_rows!=group.max_rows||group.first_expert!=group.source_keys.front())
        throw std::runtime_error("PP coalesced local rows differ");
}

// The caller classifies actual source ownership (cache/prefix/packed/planar).
// This function only concatenates owned-miss keys and their existing row spans.
// It never writes sorted_pairs/inverse, so token/rank and per-key order survive.
template<class IsOwnedMiss>
PpMissCoalesceResult pp_coalesce_miss_plan(PpStreamPlan& plan,int window_experts,
    const int32_t* directory,IsOwnedMiss is_owned_miss) {
    const size_t before=plan.groups.size();
    PpMissCoalesceResult result{before,before};
    if(plan.tokens!=8192||plan.topk!=8||window_experts!=32)return result;
    std::vector<PpStreamGroup> next;
    PpStreamGroup pending;
    int row=0;
    uint64_t selected=0;
    auto flush=[&] {
        if(!pending.experts)return;
        pending.first_expert=pending.source_keys.front();
        pending.first_row=row;row+=pending.rows;
        next.push_back(std::move(pending));pending=PpStreamGroup{};
    };
    for(const auto& group:plan.groups) {
        if(group.first_row!=row+pending.rows||group.experts<1||
            group.bounds.size()!=size_t(group.experts+1)||group.bounds.front()!=0||
            group.bounds.back()!=group.rows||!group.source_keys.empty()||!group.source_runs.empty())
            throw std::runtime_error("PP miss coalesce original group metadata differs");
        selected+=group.experts;
        bool owned=is_owned_miss(group)&&group.cache_first_slot<0;
        if(owned)for(int j=0;j<group.experts;++j) {
            const int key=group.first_expert+j;
            if(key<0||key>=288||(directory&&directory[key]>=0)){owned=false;break;}
        }
        if(!owned) {
            flush();next.push_back(group);row+=group.rows;continue;
        }
        for(int j=0;j<group.experts;++j) {
            const int key=group.first_expert+j,count=group.bounds[j+1]-group.bounds[j];
            if(count<1||count>8192)throw std::runtime_error("PP miss coalesce row span differs");
            if(pending.experts&&(pending.experts==window_experts||pending.rows+count>8192))flush();
            if(!pending.experts)pending.bounds.push_back(0);
            if(pending.source_runs.empty()||key!=pending.source_runs.back().original_first+pending.source_runs.back().count)
                pending.source_runs.push_back({key,1,pending.experts});
            else ++pending.source_runs.back().count;
            pending.source_keys.push_back(key);++pending.experts;pending.rows+=count;
            pending.max_rows=std::max(pending.max_rows,count);pending.bounds.push_back(pending.rows);
        }
    }
    flush();
    if(row!=plan.tokens*plan.topk||selected!=plan.selected_experts||
        plan.sorted_pairs.size()!=size_t(row)||plan.inverse.size()!=size_t(row))
        throw std::runtime_error("PP miss coalesce lost selected route work");
    if(next.size()>=before)return result;
    int max_rows=0;
    for(const auto& group:next){pp_validate_miss_group(group,directory);max_rows=std::max(max_rows,group.rows);}
    plan.groups=std::move(next);plan.max_group_rows=max_rows;
    result.groups_after=plan.groups.size();return result;
}
} // namespace strata::deepseek_v4
