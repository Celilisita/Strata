#pragma once
#include "../deepseek_v4/pp_stream_graph.hpp"
#include "../deepseek_v4/pp_stream_plan.hpp"
#include <array>
namespace strata::deepseek_v4 {
struct PpWeightsView {
    bool separate=false;
    std::array<const void*,3> role{};
    size_t gate_stride=0,down_stride=0;
};
template<class Window>PpWeightsView pp_weights(const PpStreamSources& s,const PpStreamGroup& g,const Window& w) {
    PpWeightsView v;v.separate=g.cache_first_slot>=0||w.planar[0];
    if(g.cache_first_slot>=0)for(int r=0;r<3;++r)v.role[r]=static_cast<const char*>(s.cache->bank[r]->data)+size_t(g.cache_first_slot)*s.cache->bank[r]->nb[2];
    else if(w.planar[0])v.role=w.planar;
    else {v.role[0]=w.payload();v.role[2]=static_cast<const char*>(w.payload())+(w.packed?2*s.gate->nb[2]:size_t(s.window_experts)*2*s.gate->nb[2]);}
    v.gate_stride=v.separate?s.gate->nb[2]:w.packed?w.packed_stride:2*s.gate->nb[2];
    v.down_stride=v.separate?s.down->nb[2]:w.packed?w.packed_stride:s.down->nb[2];return v;
}
}
