#pragma once
#include "../deepseek_v4/device_cache_protocol.hpp"
#include "causal_phases.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_set>
namespace strata::glm5_next {
// Schedule the existing shared FFN between GPU resident products and the
// contribution join, like D4 finish_native_tg. Tensor math stays unchanged.
inline void hybrid_shared_before_join(ggml_cgraph* graph,ggml_tensor* shared,int layer) {
    const int old_count = ggml_graph_n_nodes(graph);
    auto** nodes = ggml_graph_nodes(graph);
    int join = -1;
    for (int i=old_count-1;i>=0;--i)
        if (deepseek_v4::device_cache_hybrid_weight(nodes[i]) &&
            nodes[i]->op_params[2]==layer && nodes[i]->op_params[11]) {join=i;break;}
    if (join<0) return;
    ggml_build_forward_expand(graph,shared);
    const int count = ggml_graph_n_nodes(graph);
    nodes = ggml_graph_nodes(graph);
    std::unordered_set<ggml_tensor*> blocked(nodes+join,nodes+old_count);
    // A shared branch may depend on the layer input, but never on routed
    // results or their views. Check before changing topological order.
    for (int i=old_count;i<count;++i) {
        for (auto* source:nodes[i]->src)
            if (source && blocked.count(source))
                throw std::runtime_error("Shared FFN depends on the routed join");
        if (nodes[i]->view_src && blocked.count(nodes[i]->view_src))
            throw std::runtime_error("Shared FFN aliases a routed join result");
    }
    auto* join_tensor=nodes[join];
    std::rotate(nodes+join,nodes+old_count,nodes+count);
    if(causal_phases_requested()&&join_tensor->src[3]) {
        auto* lookup=join_tensor->src[2];if(lookup&&lookup->op==GGML_OP_RESHAPE)lookup=lookup->src[0];
        if(!deepseek_v4::device_cache_lookup(lookup)||!causal_phase_layer_valid(layer)||lookup->op_params[9]>=46)
            throw std::runtime_error("Causal shared graph lookup/layer identity differs");
        if(ggml_nelements(lookup)==8||ggml_nelements(lookup)==16){
            int l=-1,d=-1,s=-1,j=-1;
            for(int i=0;i<count;++i){if(nodes[i]==lookup)l=i;if(nodes[i]==join_tensor->src[0])d=i;if(nodes[i]==shared)s=i;if(nodes[i]==join_tensor)j=i;}
            if(l<0||d<=l||s<=d||j<=s)throw std::runtime_error("Causal graph must order lookup, cached products, shared then join");
        }
    }
}
}
