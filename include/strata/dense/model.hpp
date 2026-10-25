#pragma once
#include "strata/dense/graph.hpp"
#include "strata/dense/visual.hpp"
#include "strata/dense/compact_readback.hpp"
#include "strata/dense/norm_fusion.hpp"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include <memory>
#include <string>

namespace strata::dense {
struct RequestGraph;
struct MemoryUsage {
    size_t weights = 0, state = 0, workspace = 0, pp_half = 0, norm_fusion = 0;
    size_t backend_reserve = size_t(512)*1024*1024;
    size_t total() const { return weights + state + workspace + pp_half + norm_fusion + backend_reserve; }
};
struct AppendCost {
    double graph_ms = 0, input_ms = 0, compute_ms = 0, readback_ms = 0;
    double execute_readback_ms = 0;
    size_t input_bytes = 0, output_bytes = 0;
    int nodes = 0;
    int projection_fused_matmuls = 0;
    int compact_async_steps = 0, norm_fused_nodes = 0;
};

class Model {
public:
    Model(const std::string &path, int context, size_t budget_bytes, bool pp_half_cache = false,
          bool projection_fusion = false, bool compact_async = false, bool norm_fusion = false);
    ~Model();
    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    const Spec &spec() const { return spec_; }
    int context() const { return capacity_; }
    const MemoryUsage &memory() const { return memory_; }
    const AppendCost &last_cost() const { return cost_; }
    double pp_half_cold_ms() const { return pp_half_ ? pp_half_->cold_ms() : 0; }
    size_t pp_half_tensor_count() const { return pp_half_ ? pp_half_->tensor_count() : 0; }
    const char *pp_half_status() const { return pp_half_ ? pp_half_->selection_status() : "disabled"; }
    // Select compact greedy outputs only when the caller needs no penalties or
    // stochastic sampling. Changing output mode invalidates retained graphs.
    void set_device_greedy(bool enabled);
    bool device_greedy() const noexcept { return device_greedy_; }
    int device_greedy_token() const;
    void set_projection_fusion(bool enabled);
    bool projection_fusion() const noexcept { return projection_fusion_; }
    size_t projection_pack_group_count() const noexcept;
    size_t projection_pack_member_count() const noexcept;
    void set_cost_modes(bool compact_async, bool norm_fusion);
    bool compact_async_available() const noexcept;
    const char *compact_async_status() const noexcept;
    bool norm_fusion_enabled() const noexcept { return norm_enabled_; }
    // Every request starts from reset; append consumes only this request's KV/state.
    // Compact mode returns an empty vector; its checked ID is available through
    // device_greedy_token(). Otherwise need_logits returns the complete logits.
    void reset();
    // Cold request boundary: reserve all actual prompt/output rows, never a sliding window.
    void prepare_request(int tokens);
    int state_capacity() const { return state_capacity_; }
    bool request_kv() const { return request_kv_; }
    std::vector<float> append(const std::vector<int> &ids, bool need_logits = true);
    std::vector<float> append_input(const InputBatch &, bool need_logits = true);
    int position() const { return position_; }
private:
    GgufFile gguf_;
    Spec spec_;
    int capacity_ = 0, state_capacity_ = 0, position_ = 0, logical_position_ = 0;
    bool request_kv_ = false;
    bool device_greedy_ = false;
    bool projection_fusion_ = false;
    bool compact_async_ = false, norm_enabled_ = false;
    int device_token_ = -1;
    size_t budget_ = 0;
    MemoryUsage memory_;
    AppendCost cost_;
    std::unique_ptr<HalfCache> pp_half_;
    std::unique_ptr<ProjectionPack> projection_pack_;
    std::unique_ptr<CompactReadback> compact_readback_;
    std::unique_ptr<NormFusion> norm_fusion_;
    ggml_backend_t backend_ = nullptr;
    ggml_context *weight_ctx_ = nullptr, *state_ctx_ = nullptr;
    ggml_backend_buffer_t weight_buffer_ = nullptr, state_buffer_ = nullptr;
    ggml_gallocr_t allocator_ = nullptr;
    // Only this graph may retain pointers into the shared gallocr workspace.
    std::unique_ptr<RequestGraph> decode_graph_;
    std::unordered_map<const TensorInfo *, ggml_tensor *> weights_;
    std::vector<LayerState> states_;
    void load_weights();
    void allocate_state();
    std::unique_ptr<RequestGraph> build_graph(int count, int kv, bool need_logits,
                                              bool indexed_write, int image_rows = 0);
    void allocate_graph(RequestGraph &);
    void release() noexcept;
};
} // namespace strata::dense
