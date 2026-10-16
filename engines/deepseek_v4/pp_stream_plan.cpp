#include "pp_stream_plan.hpp"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
PpStreamPlan make_pp_stream_plan(const int32_t* ids, int tokens, int topk,
                               int experts, int window_experts, size_t triplet_bytes, const int32_t* cache_slots) {
    if (!ids || tokens < 1 || tokens > 8192 || topk < 1 || topk > 6 || experts != 256 ||
        window_experts < 1 || window_experts > 64 || !triplet_bytes)
        throw std::runtime_error("PP streaming geometry outside bounded D4 slice");
    PpStreamPlan p;
    p.tokens = tokens; p.topk = topk;
    std::vector<std::vector<int32_t>> rows(experts);
    for (int t = 0; t < tokens; ++t) {
        bool seen[256] = {};
        for (int rank = 0; rank < topk; ++rank) {
            const int pair = t * topk + rank, expert = ids[pair];
            if (expert < 0 || expert >= experts || seen[expert])
                throw std::runtime_error("PP route contains invalid or duplicate selected expert");
            seen[expert] = true; rows[expert].push_back(pair);
        }
    }
    p.inverse.resize(tokens * topk);
    std::vector<int> cached;
    bool all_cached=cache_slots!=nullptr;
    const char* direct_setting=std::getenv("STRATA_DSV4_PP_DIRECT_CACHE_GROUPS");
    const bool direct_groups=cache_slots&&!(direct_setting&&std::strcmp(direct_setting,"0")==0);
    for(int e=0;e<experts;++e) if(!rows[e].empty()) {
        if(!cache_slots||cache_slots[e]<0)all_cached=false;
        else cached.push_back(e);
    }
    if(all_cached||direct_groups) {
        std::sort(cached.begin(),cached.end(),[&](int a,int b){return cache_slots[a]<cache_slots[b];});
        for(size_t next=0;next<cached.size();) {
            PpStreamGroup g;g.first_expert=cached[next];g.cache_first_slot=cache_slots[cached[next]];
            g.first_row=int(p.sorted_pairs.size());g.bounds.push_back(0);
            while(next<cached.size()&&g.experts<window_experts) {
                const int e=cached[next];
                if(g.experts&&(cache_slots[e]!=g.cache_first_slot+g.experts||g.rows+int(rows[e].size())>8192))break;
                for(int pair:rows[e]) {p.inverse[pair]=int(p.sorted_pairs.size());p.sorted_pairs.push_back(pair);}
                g.max_rows=std::max(g.max_rows,int(rows[e].size()));g.rows+=int(rows[e].size());
                ++g.experts;++next;g.bounds.push_back(g.rows);
            }
            p.selected_experts+=g.experts;p.max_group_rows=std::max(p.max_group_rows,g.rows);p.groups.push_back(std::move(g));
        }
        if(all_cached) {
            p.payload_bytes=p.selected_experts*triplet_bytes;
            if(p.sorted_pairs.size()!=p.inverse.size())throw std::runtime_error("PP cached grouping lost route work");
            return p;
        }
    }
    for (int e = 0; e < experts;) {
        if (rows[e].empty() || (direct_groups&&cache_slots[e]>=0)) { ++e; continue; }
        // Consecutive active experts let three pitched DMA calls transfer a
        // window. Never upload an unselected expert just to fill a window.
        PpStreamGroup g;
        g.first_expert = e; g.first_row = int(p.sorted_pairs.size());
        g.bounds.push_back(0);
        while (e < experts && !rows[e].empty() && (!direct_groups||cache_slots[e]<0) && g.experts < window_experts) {
            // Bound activation scratch as well as expert bytes. A single
            // selected expert has at most one row per token, so this never
            // splits/reloads an expert even for highly skewed router outputs.
            if (g.experts && g.rows + int(rows[e].size()) > 8192) break;
            g.max_rows = std::max(g.max_rows, int(rows[e].size()));
            for (const auto pair : rows[e]) {
                p.inverse[pair] = int(p.sorted_pairs.size());
                p.sorted_pairs.push_back(pair);
            }
            g.rows += int(rows[e].size()); ++g.experts; ++e;
            g.bounds.push_back(g.rows);
        }
        p.selected_experts += g.experts;
        p.max_group_rows = std::max(p.max_group_rows, g.rows);
        if(cache_slots&&cache_slots[g.first_expert]>=0) {
            g.cache_first_slot=cache_slots[g.first_expert];
            for(int rank=1;rank<g.experts;++rank)
                if(cache_slots[g.first_expert+rank]!=g.cache_first_slot+rank){g.cache_first_slot=-1;break;}
        }
        p.groups.push_back(std::move(g));
    }
    p.payload_bytes = p.selected_experts * triplet_bytes;
    if (p.sorted_pairs.size() != p.inverse.size())
        throw std::runtime_error("PP route grouping lost work");
    return p;
}
}
