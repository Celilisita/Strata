#pragma once
// Post-route, owned PP window only. Prefix/cache borrowing keeps its old path.
#include "../deepseek_v4/pp_stream_graph.hpp"
#include "../deepseek_v4/pp_stream_plan.hpp"
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace strata::glm5_next {
inline bool pp_source_window_planar_requested(const deepseek_v4::PpStreamSources& source) {
    // The option never changes ordinary D4's layout or option parsing.
    if(!source.gate||source.gate->ne[2]!=288)return false;
    const char* flag=std::getenv("STRATA_GLM53_PP_SOURCE_WINDOW_PLANAR");
    if(!flag||!std::strcmp(flag,"0"))return false;
    if(std::strcmp(flag,"1"))throw std::runtime_error("GLM PP source-window planar must be0 or1");
    return true;
}

template<class Window>void pp_bind_source_window_planar(const deepseek_v4::PpStreamSources& source,
    const deepseek_v4::PpStreamGroup& group,Window& window) {
    const auto* gate=source.gate;const auto* up=source.up;const auto* down=source.down;
    if(!gate||!up||!down||gate->ne[2]!=288||up->ne[2]!=288||down->ne[2]!=288||
        gate->ne[0]!=4096||gate->ne[1]!=2048||!ggml_are_same_shape(gate,up)||gate->type!=up->type||
        down->ne[0]!=2048||down->ne[1]!=4096||gate->nb[2]!=up->nb[2]||
        !ggml_is_contiguous(gate)||!ggml_is_contiguous(up)||!ggml_is_contiguous(down)||
        source.window_experts<1||source.window_experts>64||group.experts<1||
        uint32_t(group.experts)>source.window_experts||group.first_expert<0||
        group.first_expert>288-group.experts||group.cache_first_slot>=0||window.borrowed||window.planar[0])
        throw std::runtime_error("GLM PP owned source-window planar shape/ownership differs");
    const size_t half=gate->nb[2],d_stride=down->nb[2],limit=std::numeric_limits<size_t>::max();
    if(!half||!d_stride||half>(limit-d_stride)/2)throw std::runtime_error("GLM PP window stride overflow");
    const size_t triplet=2*half+d_stride;
    if(source.window_experts>limit/triplet)throw std::runtime_error("GLM PP window allocation overflow");
    const size_t count=source.window_experts,bytes=count*triplet;
    if(!window.device.p||window.device.bytes<bytes)throw std::runtime_error("GLM PP source planes exceed existing window");
    // Fixed capacity, rather than the current group size, keeps the original
    // Down offset and total window allocation exactly unchanged.
    const auto* base=static_cast<const char*>(window.device.p);
    window.planar={base,base+count*half,base+count*2*half};
    window.owned_planar=true;
}

template<class Window>bool pp_window_borrows_prefix(const Window& window) {
    // Source planes are readers of their own two-window storage. They must
    // record done, and must never retire a separately borrowed prefix lease.
    return window.borrowed||(window.planar[0]&&!window.owned_planar);
}
} // namespace strata::glm5_next
