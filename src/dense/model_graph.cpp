#include "model_graph.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::dense {
std::unique_ptr<RequestGraph> Model::build_graph(int count, int kv, bool need_logits,
                                                bool indexed_write, int image_rows) {
    if (count < 1 || count > 8192 || kv < count || kv > capacity_ || kv > state_capacity_ || kv % 256 ||
        image_rows<0 || image_rows>count || (indexed_write && (count != 1 || !need_logits || image_rows)))
        throw std::invalid_argument("invalid dense request graph geometry");
    auto request = std::make_unique<RequestGraph>();
    request->count = count;
    request->kv = kv;
    request->axes = spec_.architecture == "qwen35" ? 4 : 1;
    request->indexed_write = indexed_write;
    request->device_greedy = need_logits && device_greedy_;
    ggml_init_params params{size_t(32)*1024*1024, nullptr, true};
    request->context = ggml_init(params);
    if (!request->context) throw std::runtime_error("dense request graph metadata allocation failed");
    request->graph = ggml_new_graph_custom(request->context, 8192, false);
    GraphContext g{request->context, request->graph, spec_, weights_, states_};
    g.pp_half = pp_half_.get();
    g.projection_pack = projection_fusion_ ? projection_pack_.get() : nullptr;
    g.gdn_norm_scale = count > 1 && norm_enabled_ && norm_fusion_ ? norm_fusion_->scale() : nullptr;
    g.start = indexed_write ? 0 : position_;
    g.tokens = count;
    g.kv_tokens = kv;
    request->tokens = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32, count);
    ggml_set_input(request->tokens);
    g.positions = request->positions = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I32,
                                                          count*request->axes);
    ggml_set_input(g.positions);
    g.mask = request->mask = ggml_new_tensor_2d(g.ctx, GGML_TYPE_F16, kv, count);
    ggml_set_input(g.mask);
    if (indexed_write) {
        g.cache_rows = request->cache_rows = ggml_new_tensor_1d(g.ctx, GGML_TYPE_I64, 1);
        ggml_set_input(g.cache_rows);
    }
    auto *hidden = ggml_get_rows(g.ctx, weights_.at(spec_.embedding), request->tokens);
    if(image_rows){
        request->image_embeddings=ggml_new_tensor_2d(g.ctx,GGML_TYPE_F32,spec_.n_embd,image_rows);
        request->image_rows=ggml_new_tensor_1d(g.ctx,GGML_TYPE_I64,image_rows);
        ggml_set_input(request->image_embeddings);ggml_set_input(request->image_rows);
        hidden=ggml_set_rows(g.ctx,hidden,request->image_embeddings,request->image_rows);
    }
    for (size_t i = 0; i < spec_.layers.size(); ++i) {
        const auto &layer = spec_.layers[i];
        auto *normalized = g.norm(hidden, layer.attn_norm);
        auto *attention = layer.recurrent ? build_recurrent(g, layer, states_[i], normalized)
                                          : build_attention(g, layer, states_[i], normalized);
        if (i+1 == spec_.layers.size() && !need_logits) break;
        hidden = ggml_add(g.ctx, hidden, attention);
        if (i+1 == spec_.layers.size() && count > 1)
            hidden = ggml_view_2d(g.ctx, hidden, spec_.n_embd, 1, hidden->nb[1],
                                  size_t(count-1)*hidden->nb[1]);
        hidden = ggml_add(g.ctx, hidden, build_ffn(g, layer, g.norm(hidden, layer.ffn_norm)));
    }
    if (need_logits) {
        request->logits = ggml_mul_mat(g.ctx, weights_.at(spec_.output),
                                       g.norm(hidden, spec_.output_norm));
        ggml_set_name(request->logits, "dense.logits");
        if (request->device_greedy) {
            // Keep the backend's max-value tie rule without perturbing logits;
            // unlike the CPU scan, it need not select the smallest tied ID.
            request->greedy_id = ggml_argmax(g.ctx, request->logits);
            request->finite_sum = ggml_sum(g.ctx, ggml_abs(g.ctx, request->logits));
            ggml_set_name(request->greedy_id, "dense.greedy_id");
            ggml_set_name(request->finite_sum, "dense.finite_sum");
            ggml_set_output(request->greedy_id);
            ggml_set_output(request->finite_sum);
            ggml_build_forward_expand(request->graph, request->greedy_id);
            ggml_build_forward_expand(request->graph, request->finite_sum);
        } else {
            ggml_set_output(request->logits);
            ggml_build_forward_expand(request->graph, request->logits);
        }
    }
    request->projection_fused_matmuls = int(g.projection_cache.size());
    request->norm_fused_nodes = g.gdn_norm_fused_nodes;
    request->host_positions.resize(size_t(count)*request->axes);
    request->host_mask.resize(size_t(kv)*count);
    return request;
}

void Model::allocate_graph(RequestGraph &request) {
    for (int i = 0; i < ggml_graph_n_nodes(request.graph); ++i) {
        const auto *node = ggml_graph_node(request.graph, i);
        if (!ggml_backend_supports_op(backend_, node))
            throw std::runtime_error(std::string("dense CUDA unsupported operation: ")+ggml_op_name(node->op));
    }
    size_t required = 0;
    ggml_gallocr_reserve_n_size(allocator_, request.graph, nullptr, nullptr, &required);
    if (memory_.weights+memory_.state+memory_.pp_half+memory_.norm_fusion+
        std::max(memory_.workspace,required)+memory_.backend_reserve > budget_)
        throw std::runtime_error("dense request exceeds complete GPU budget before allocation");
    if (!ggml_gallocr_reserve(allocator_,request.graph) ||
        !ggml_gallocr_alloc_graph(allocator_,request.graph))
        throw std::runtime_error("dense request workspace allocation failed");
    memory_.workspace = ggml_gallocr_get_buffer_size(allocator_, 0);
    if (memory_.total() > budget_) throw std::runtime_error("dense complete GPU budget exceeded");
}
} // namespace strata::dense
