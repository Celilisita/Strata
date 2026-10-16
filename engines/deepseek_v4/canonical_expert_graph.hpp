#pragma once

struct llm_graph_context;
struct llama_model;
struct ggml_tensor;

namespace strata::deepseek_v4 {

// Build only original unweighted selected-expert computation. The caller owns
// the graph context, original mmap banks, input/output buffers and execution.
// Context hparams must contain the selected route count for this layer.
ggml_tensor* build_canonical_expert_down(llm_graph_context& context,
                                        const llama_model& model,
                                        ggml_tensor* activation,
                                        ggml_tensor* selected,
                                        int layer);

} // namespace strata::deepseek_v4
