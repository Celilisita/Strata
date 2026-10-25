#pragma once
#include <array>
#include "../deepseek_v4/pp_stream_graph.hpp"
#include "../deepseek_v4/pp_stream_plan.hpp"
#include "pp_weights_view.cuh"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::glm5_next {
inline bool pp_transport_verify_requested(){
    const char* p=std::getenv("STRATA_GLM53_PP_TRANSPORT_VERIFY");
    if(!p||!std::strcmp(p,"0"))return false;
    if(std::strcmp(p,"1"))throw std::runtime_error("PP transport verify must be0 or1");
    return true;
}
// Diagnostic only. Caller snapshots the strict flag once/run and invokes only
// after all owned-window copies and ready recording, before any new reader.
// Copy-stream synchronization, D2H and CPU comparison are deliberately paid.
template<class Window,class Metrics>
void pp_transport_verify_owned(const deepseek_v4::PpStreamSources& s,
    const deepseek_v4::PpStreamGroup& g,const Window& w,cudaStream_t copy,Metrics& m) {
    if(g.cache_first_slot>=0||w.borrowed||(!w.owned_planar&&w.planar[0])||w.packed)
        throw std::runtime_error("PP transport verifier requires an actual owned window");
    if(g.experts<1||g.experts>s.window_experts||s.window_experts!=32||
        (!g.source_keys.empty()&&g.source_keys.size()!=size_t(g.experts)))
        throw std::runtime_error("PP transport verifier local-key geometry differs");
    const std::array<const ggml_tensor*,3> source{s.gate,s.up,s.down};
    const auto view=deepseek_v4::pp_weights(s,g,w);
    const void* role[3]{view.role[0],view.separate?view.role[1]:
        static_cast<const char*>(view.role[0])+s.gate->nb[2],view.role[2]};
    const size_t pitch[3]{view.gate_stride,view.gate_stride,view.down_stride};
    size_t largest=0;for(const auto* t:source)largest=(std::max)(largest,t->nb[2]);
    if(largest>4456448u)throw std::runtime_error("PP transport verifier bounded scratch exceeded");
    std::vector<unsigned char> bytes(largest);
    for(int local=0;local<g.experts;++local){
        const int key=g.source_keys.empty()?g.first_expert+local:g.source_keys[local];
        if(key<0||key>=s.gate->ne[2])throw std::runtime_error("PP transport verifier expert outside source");
        for(int r=0;r<3;++r){
            const size_t width=source[r]->nb[2];
            const char* gpu=static_cast<const char*>(role[r])+size_t(local)*pitch[r];
            const char* original=static_cast<const char*>(source[r]->data)+size_t(key)*width;
            CUDA_CHECK(cudaMemcpyAsync(bytes.data(),gpu,width,cudaMemcpyDeviceToHost,copy));
            CUDA_CHECK(cudaStreamSynchronize(copy));
            m.transport_verify_D2H_bytes+=width;
            if(std::memcmp(bytes.data(),original,width))
                throw std::runtime_error("PP owned transport original payload differs");
            m.transport_verified_bytes+=width;
        }
    }
    ++m.transport_verified_groups;
}
} // namespace strata::glm5_next
