#include "canonical_expert_graph.hpp"

#include "llama-model.h"
#include <stdexcept>
#include <string>

namespace strata::deepseek_v4 {

ggml_tensor* build_canonical_expert_down(llm_graph_context& context,
                                        const llama_model& model,
                                        ggml_tensor* activation,
                                        ggml_tensor* selected,
                                        int layer_id) {
    if (model.arch != LLM_ARCH_DEEPSEEK4 || layer_id < 0 ||
        std::size_t(layer_id) >= model.layers.size() || !activation || !selected ||
        activation->type != GGML_TYPE_F32 || selected->type != GGML_TYPE_I32 ||
        activation->ne[0] != model.hparams.n_embd || activation->ne[1] != selected->ne[1] ||
        activation->ne[2] != 1 || selected->ne[2] != 1 || activation->ne[3] != 1 ||
        selected->ne[3] != 1 || !ggml_is_contiguous(activation) || !ggml_is_contiguous(selected) ||
        selected->ne[0] < 1 || selected->ne[0] > model.hparams.n_expert_used(layer_id) ||
        context.hparams.n_expert_used(layer_id) != selected->ne[0])
        throw std::runtime_error("canonical cold expert graph input contract differs");
    const auto& hp = model.hparams;
    const auto& layer = model.layers[layer_id];
    // Router logits and weighting are dead ancestors of raw Down. Build the
    // canonical function, then keep precisely that dependency subgraph.
    auto* unused_router = ggml_new_tensor_2d(context.ctx0, GGML_TYPE_F32,
                                            hp.n_expert, activation->ne[1]);
    context.build_moe_ffn(activation, layer.ffn_gate_inp,
        layer.ffn_up_exps, layer.ffn_gate_exps, layer.ffn_down_exps, nullptr,
        hp.n_expert, selected->ne[0], LLM_FFN_SILU, hp.expert_weights_norm,
        hp.expert_weights_scale, llama_expert_gating_func_type(hp.expert_gating_func), layer_id,
        unused_router, nullptr, nullptr, nullptr, nullptr, selected);
    const auto name = "ffn_moe_down-" + std::to_string(layer_id);
    ggml_tensor* down = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(context.gf); ++i) {
        auto* node = ggml_graph_node(context.gf, i);
        if (name == ggml_get_name(node)) {
            if (down) throw std::runtime_error("canonical cold expert graph has duplicate Down output");
            down = node;
        }
    }
    if (!down || down->type != GGML_TYPE_F32 || down->ne[0] != hp.n_embd ||
        down->ne[1] != selected->ne[0] || down->ne[2] != activation->ne[1])
        throw std::runtime_error("canonical cold expert graph raw Down shape differs");
    ggml_graph_clear(context.gf);
    ggml_set_output(down);
    ggml_build_forward_expand(context.gf, down);
    return down;
}

} // namespace strata::deepseek_v4
