#pragma once
#include "strata/prefill/moe_mmq.hpp"
#include "strata/qwen35moe/prefill_observer.hpp"
#include "strata/qwen35moe/metrics.hpp"
#include "strata/qwen35moe/dense_profile.hpp"
#include "strata/qwen35moe/spec.hpp"
#include "strata/qwen35moe/runtime_contracts.hpp"
#include "strata/qwen35moe/nvfp4_expert_inventory.hpp"
#include "strata/qwen35moe/visual_input.hpp"
#include "strata/qwen35moe/head_pick.hpp"
#include "strata/qwen35moe/dspark_verify.hpp"
#include "strata/core/native_miss.hpp"
#include "strata/core/expert_retention.hpp"
#include "strata/kernels/sampler.hpp"
#include "strata/kernels/radix_sampler.hpp"
#include "strata/qwen35moe/device_buffer.hpp"
#include "strata/qwen35moe/expert_storage.hpp"
#include "strata/qwen35moe/expert_prefetch.hpp"
#include "strata/qwen35moe/expert_gu_copy_pipeline.hpp"
#include "strata/qwen35moe/batch_workspace.hpp"
#include "strata/qwen35moe/host_readonly_lease.hpp"
#include "strata/qwen35moe/prefill_config.hpp"
#include <cuda_runtime.h>

#include <cstddef>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::qwen35moe {
struct Nvfp4ExpertState;
class HeadReadback;
class Runtime {
    friend struct BenchmarkProjectionOptions;
public:
    explicit Runtime(const std::string& model_path, int max_context = 1024,
                     bool enable_expert_cache = false, const kernels::SamplerParams* sampling = nullptr);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;
    int run_token(int token, int position, bool need_logits, Metrics& metrics);
    void set_checkpoint_sink(CheckpointSink* sink) { checkpoint_sink_ = sink; }
    void start_decode(Metrics* metrics = nullptr);
    void finish_decode(Metrics* metrics = nullptr);
    // keep_recurrent leaves GDN state and K/V for a resumed prefix.
    void reset_session(bool keep_recurrent = false);
    // Prompt-end recurrent snapshot: decode advances the state past the prompt,
    // and a client that drops reasoning tokens can only resume at its end.
    int prefill_checkpoint() const { return prefill_checkpoint_position_; }
    void save_prefill_checkpoint(int position);
    bool restore_prefill_checkpoint();
    void set_visual_input(VisualInput input);
    int session_capacity() const;
    size_t additional_session_bytes() const;
    // Bind on the ordered runtime worker; retain the shared decode epoch/cache.
    void select_session(int id);
    void check_session_switch(bool paired = false);
    std::array<int, 2> run_session_pair(const std::array<int, 2>& tokens,
        const std::array<int, 2>& positions, Metrics& metrics);
    int max_context() const { return max_context_; }
    int reserved_context() const { return kv_capacity_; }
    void reserve_context(int tokens, Metrics& metrics, int prefill_capacity = 0);
    int batch_capacity() const { return batch_.capacity; }
    int preferred_prefill_capacity(int tokens) const {
        return prefill_config_.capacity(tokens, prefill_tc_);
    }
    const PrefillConfig& prefill_config() const { return prefill_config_; }
    void check_prefill_batch(bool bulk_projection = false, int column_tile = 8, int query_group = 1, bool shared_join = false, bool padded_cache = false, bool dense_tensor = false, bool q4_reuse = false, AttentionCheck attention_check = AttentionCheck::none, bool gdn_columns = false, bool mmq_projection = false);
    void check_precise_projection(int count);
    void check_bulk_projection(int count, bool tensor = false, int column_tile = 8, bool q4_reuse = false, bool tensor_profile = false);
    float last_logit(int token) const;
    const Spec& spec() const { return spec_; }
    size_t expert_cache_slots() const { return expert_cache_.slots; }
    const core::ExpertRetention::Cost& retention_cost() const { return expert_cache_.retention.cost(); }
    size_t pinned_expert_bytes() const { return pinned_experts_.bytes; }
    uint64_t cache_warm_bytes() const { return cache_warm_bytes_; }
    size_t cache_warm_experts() const { return cache_warm_experts_; }
    double cache_warm_ms() const { return cache_warm_ms_; }
    size_t pinned_expert_layers() const;
    size_t registered_gguf_bytes() const { return host_readonly_.bytes(); }
    const std::string& pinned_expert_error() const { return pinned_experts_.error; }
    BatchProjectionCheck check_batch_projection(int count, bool shared = false);
    PrefillProviderCheck check_prefill_provider(int count, bool scan = false);
    void check_dynamic_layer(size_t layer);
    void check_context_residency();
    void check_mmq_projection();
    uint64_t mmq_projection_calls() const { return batch_.mmq_projection_calls; }
    uint64_t gu_active_calls() const { return batch_.gu_active_calls; }
    uint64_t attention_scratch_calls() const { return batch_.attention_scratch_calls; }
    uint64_t attention_compact_calls() const { return batch_.attention_compact_calls; }
    uint64_t mtp_kv_bulk_calls() const { return batch_.mtp_kv_bulk_calls; }
    uint64_t router_tiled_calls() const { return batch_.router_tiled_calls; }
    uint64_t qkv_q4_mmq_calls() const { return batch_.qkv_q4_mmq_calls; }
    uint64_t gu_mmq_calls() const { return batch_.gu_mmq_calls; }
    uint64_t qkv_mmq_calls() const { return batch_.qkv_mmq_calls; }
    uint64_t gdn_out_mmq_calls() const { return batch_.gdn_out_mmq_calls; }
    uint64_t short_attention_calls() const { return batch_.short_attention_calls; }
    uint64_t down_mmq_calls() const { return batch_.down_mmq_calls; }
    uint64_t tensor_attention_calls() const { return batch_.tensor_attention_calls; }
    void check_mmq_pipeline(const std::string& capture_directory = "");
    void check_gdn_columns();
    void check_routed_tc(size_t layer, bool gathered = false, bool row_reuse = false);
    void check_dynamic_pair(size_t layer);
    void check_gdn_prefix();
    void check_sampling();
    void check_mtp_kv(const std::string& reference);
    bool mtp_enabled() const { return mtp_enabled_; }
    // Configure before prefill; keep resident resources/cache for a matched control.
    void set_mtp_execution(bool enabled);
    void configure_sampling(const kernels::SamplerParams& params);
    // Sample the logits from the most recent head call without changing session state.
    int sample_last_logits(const kernels::SamplerParams& params,
                           const std::vector<int>& history, Metrics& metrics);
    SpeculativeStep run_speculative(int token, int position, int remaining,
        const std::vector<int>& stops, Metrics& metrics, bool force_reject = false);
    void prepare_dspark_verify(DsparkVerifyState& state, int capacity = 16);
    DsparkVerifiedBlock verify_dspark_block(DsparkVerifyState& state,
        const std::vector<int>& tokens, int position, Metrics& metrics);
    MtpLayerCheck check_mtp_layer(const std::string& reference);
    BatchProbe batch_probe() const { return batch_probe_; }
    void prepare_batch(int capacity);
    int run_prefill(const int* tokens, int count, const std::string& reference,
                    Metrics& metrics, const PrefillObserver* observer = nullptr, int offset = 0);
    int run_prefill_long(const int* tokens, int count, Metrics& metrics,
                         int chunk_capacity = 0, const PrefillObserver* observer = nullptr,
                         int offset = 0);

private:
    void load_weights();
    LayerWeights load_layer(const LayerTensors& source);
    void initialize_mtp();
    void initialize_sessions();
    void initialize_visual();
    void clear_visual_input();
    void collect_visual_metrics(Metrics& metrics);
    const int32_t* visual_table(int session = -1) const;
    void load_prefill_embeddings(const int* tokens, int count, float* output);
    void load_embedding(int token, int position, float* output, const TensorInfo* source,
        DeviceBuffer& scratch, size_t row_bytes, int session = -1);
    void begin_session_pair(const std::array<int, 2>& positions);
    void end_session_pair();
    void batch_gdn_sessions(LayerWeights& layer, int count);
    void mtp_kv_sessions(const std::array<int, 2>& tokens, const std::array<int, 2>& positions);
    void initialize_sampling(const kernels::SamplerParams* params);
    void sampling_history(const int* tokens, int count);
    void launch_sample(Metrics& metrics, int row = 0);
    void sampling_begin_draft(int token);
    void sampling_end_draft();
    void sampling_verify_prefix();
    void sampling_reject_verify();
    void reset_sampling();
    void check_sample_pick();
    void decode_batch_layer(size_t il, int count, int first_position,
        const std::string& reference, Metrics& metrics);
    void batch_dense_layer(size_t il, int count, int first_position,
        const std::string& reference);
    std::vector<int32_t> batch_prefix_dense(size_t il, int count, int first_position,
        const std::string& reference, Metrics& metrics);
    void decode_layer(size_t il, const LayerTensors& source, LayerWeights& layer,
                      int position, Metrics& metrics, int rope_offset = 0);
    int mtp_step(int token, int cell, const float* normalized_main, Metrics& metrics);
    void mtp_prompt(const int* tokens, int count, const float* hidden, Metrics& metrics);
    void verify_pair(const int* tokens, int position, std::array<HeadPick, 2>& picks, Metrics& metrics);
    void mtp_condition(const int* tokens, int count, const float* normalized_main,
        int first_position = -1, const std::array<int, 2>* positions = nullptr);
    void mtp_kv_batch(const int* tokens, int count, const float* normalized_main, int first_cell);
    void initialize_dynamic();
    void prepare_moe_graphs();
    void invalidate_moe_graphs();
    void batch_moe_device_eager(size_t il, const LayerWeights& layer, Metrics& metrics, int count);
    void collect_dynamic_metrics(Metrics& metrics);
    void batch_moe_nvfp4_device(size_t il,const LayerWeights& layer,Metrics& metrics);
    void replay_nvfp4_routes(Metrics& metrics);
    void submit_nvfp4_admission(size_t il,Metrics& metrics);
    void wait_nvfp4_admission(size_t il);
    void flush_nvfp4_admission(Metrics& metrics);
    void batch_moe_device(size_t il, const LayerWeights& layer, Metrics& metrics, int count = 1);
    void allocate_batch(int capacity);
    void project(const GpuWeight& weight, const void* input_q8, float* output,
                 int n_in, int n_out);
    void gdn_step(LayerWeights& layer);
    bool gdn_dense_graph_requested() const;
    void prepare_gdn_dense_graphs(MoeGraphState&); bool launch_gdn_dense_graph(size_t il);
    void attention_step(LayerWeights& layer, int position, int rope_offset = 0);
    void store_attention_kv(LayerWeights& layer, float* k, const float* v, int position, int rope_offset = 0, int session = -1);
    void store_attention_kv_rows(LayerWeights& layer, float* k, const float* v,
        int first, int count, int rotary_offset);
    void moe_step(size_t il, const LayerWeights& layer, Metrics& metrics);
    void moe_with_ids(size_t il, const LayerWeights& layer, const int32_t* ids,
                      Metrics& metrics);
    void batch_gdn(LayerWeights& layer, int count, const std::string& reference);
    void batch_gdn_scan(LayerWeights& layer, int count);
    void batch_attention(LayerWeights& layer, int count, int position_base = 0);
    void batch_attention_prefill(LayerWeights& layer, int count, int position_base);
    void attend_cached(LayerWeights& layer, int position);
    void batch_moe(size_t il, const LayerWeights& layer, int count,
                   const std::string& reference, Metrics& metrics);
    void batch_moe_nvfp4(size_t il,const LayerWeights& layer,int count,const std::string& reference,Metrics& metrics);
    void batch_moe_grouped(size_t il, const LayerWeights& layer, int count,
                           const std::string& reference, Metrics& metrics);
    void batch_moe_grouped_ready(size_t il, const LayerWeights& layer, int count,
                                 const std::string& reference, Metrics& metrics,
                                 const std::vector<int32_t>& ids);
    uint64_t batch_moe_gu_mmq(int count, Metrics& metrics);
    uint64_t batch_moe_down_mmq(size_t il, int count, Metrics& metrics);
    uint64_t batch_moe_down_fused(size_t il, int count, Metrics& metrics);
    void batch_moe_grouped_compute(size_t il, const LayerWeights& layer, int count,
                                   const std::string& reference, Metrics& metrics);
    void batch_moe_resident(size_t il, const LayerWeights& layer, Metrics& metrics);
    void batch_moe_cpu(size_t il, const LayerWeights& layer, int count,
                       const std::string& reference, Metrics& metrics);
    void batch_moe_cpu_ready(size_t il, const LayerWeights& layer, int count,
                             const std::string& reference, Metrics& metrics,
                             const std::vector<int32_t>& ids);
    std::vector<int32_t> batch_route(const LayerWeights& layer, int count,
                                     const std::string& reference, Metrics& metrics,
                                     bool copy_norm);
    void batch_moe_hybrid(size_t il, const LayerWeights& layer, int count,
                          const std::string& reference, Metrics& metrics);
    void batch_moe_mixed(size_t il, const LayerWeights& layer,
                         const std::vector<int32_t>& ids, Metrics& metrics);
    void grouped_moe_token(size_t il, const LayerWeights& layer, Metrics& metrics);
    void batch_quantize_to_half(const float* input, int count, int width);
    void batch_quantize_q8(const float* input, int count, int width);
    bool batch_project_blackwell(const GpuWeight& weight, float* output,
                                 int count, int n_in, int n_out);
    bool batch_project_precise_dense(const GpuWeight& weight, float* output, int count, int n_in, int n_out);
    bool batch_project_mmq(const GpuWeight& weight, float* output, int count, int n_in, int n_out, ProjectionRole role = ProjectionRole::generic);
    void batch_project_native(const GpuWeight& weight, float* output,
                              int count, int n_in, int n_out, bool allow_tensor = true);
    double batch_stage_error(const std::string& reference, const char* name,
                             const float* device, size_t width);
    int head_token(Metrics& metrics, const GpuWeight* norm = nullptr, const GpuWeight* output = nullptr);
    struct ExpertView { const void* gate; const void* up; const void* down; float gate_scale=1,up_scale=1,down_scale=1; };
    ExpertView expert_view(size_t layer, int id, const LayerTensors& source,
                           Metrics& metrics);
    ExpertView staged_expert_view(size_t layer, int id, int entry,
                                  const LayerTensors& source, Metrics& metrics);
    void allocate_expert_cache();
    size_t expert_cache_budget(size_t reclaim, size_t additional, int kv, int prefill) const;
    void resize_expert_cache(size_t slots, Metrics& metrics);
    void prewarm_expert_cache();
    void pin_expert_weights(int count);
    void prepare_host_readonly();
    PpPrefetchCache prefetch_cache();
    void prefetch_begin();
    void prefetch_layer(size_t il,int count,const std::string& reference,Metrics& metrics);
    void prefetch_prefix(size_t il,const std::vector<int32_t>& ids,Metrics& metrics);
    void prefetch_routes(size_t il,const std::vector<int32_t>& ids);
    void prefetch_finish(Metrics& metrics);
    void prefetch_drain();
    uint64_t prepare_gu_copy(size_t il,int count,const std::string& reference,const std::vector<int32_t>& ids,Metrics& metrics);
    void finish_gu_copy(Metrics& metrics);
    uint64_t prepare_expert_copy_runs(size_t il,int count,const std::string& reference,
                                     const std::vector<int32_t>& ids,Metrics& metrics);

    struct ExpertCache {
        ExpertStorage data;
        DeviceBuffer resident_slots;
        size_t stride = 0, slots = 0, next = 0;
        uint64_t tick = 0, epoch = 1;
        std::vector<int32_t> key_to_slot, slot_to_key;
        std::vector<uint64_t> age;
        core::ExpertRetention retention;
    };

    GgufFile gguf_;
    Spec spec_;
    std::unique_ptr<Nvfp4ExpertState> nvfp4_;
    std::vector<LayerWeights> layers_;
    std::vector<LayerTensors> expert_sources_;
    std::unique_ptr<MtpState> mtp_;
    std::unique_ptr<SamplingState> sampling_;
    std::unique_ptr<kernels::RadixSampler> minicpm_sampler_;
    DeviceBuffer minicpm_sample_history_;
    std::unique_ptr<SessionSet> sessions_;
    std::unique_ptr<VisualState> visual_;
    uint64_t visual_h2d_pending_ = 0;
    SessionState* paired_session_ = nullptr;
    std::array<int, 2> paired_positions_{};
    size_t paired_layer_ = 0;
    MtpCheckpointSink* mtp_sink_ = nullptr;
    bool mtp_enabled_ = false;
    float* verify_state_ = nullptr;
    float* verify_history_ = nullptr;
    GpuWeight head_norm_, head_;
    DeviceBuffer head_partials_, head_pick_buffer_;
    DeviceBuffer attention_scratch_;
    HeadPick host_pick_{-1, -1, 0.0f};
    std::unique_ptr<HeadReadback> head_readback_;
    Workspace work_;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t diag_route_ready_ = nullptr;
    int max_context_ = 0, kv_capacity_ = 0;
    size_t embedding_row_bytes_ = 0;
    std::vector<float> host_logits_;
    CheckpointSink* checkpoint_sink_ = nullptr;
    bool enable_expert_cache_ = false;
    bool decode_phase_ = false;
    bool tg_hit_gpu_ = false;
    bool tg_dynamic_ = false;
    DeviceBuffer dynamic_args_, dynamic_pair_args_, dynamic_frame_, dynamic_ages_;
    std::unique_ptr<core::NativeMissService> miss_service_;
    core::NativeMissStats miss_accounted_;
    uint32_t decode_epoch_ = 0;
    uint64_t dynamic_layers_ = 0, dynamic_accounted_layers_ = 0;
    uint64_t dynamic_entries_ = 0, dynamic_accounted_entries_ = 0;
    const float* selected_logits_ = nullptr;
    bool gpu_head_ = false;
    bool tile_attention_ = false;
    bool prefill_tc_ = false;
    std::unique_ptr<prefill::mmq::Context> projection_mmq_;
    PrefillConfig prefill_config_;
    int prefill_budget_capacity_ = 1024;
    bool routed_tc_enabled_ = false;
    bool routed_gu_gathered_ = false;
    bool prefill_scan_ = false;
    bool batch_failed_ = false;
    bool resident_gather_ = false;
    bool tg_diag_ = false;
    int resident_layers_ = 0;
    ExpertCache expert_cache_;
    std::unique_ptr<ExpertPrefetch> expert_prefetch_;
    std::unique_ptr<ExpertGuCopyPipeline> gu_copy_;
    bool gu_copy_pending_=false;size_t gu_copy_prefix_layer_=size_t(-1);
    PpPrefetchCost gu_copy_summary_{};
    uint64_t cache_warm_bytes_ = 0;
    size_t cache_warm_experts_ = 0;
    double cache_warm_ms_ = 0;
    BatchWorkspace batch_;
    BatchProbe batch_probe_;
    std::unique_ptr<CpuExpertState> cpu_;
    std::vector<std::array<cudaEvent_t, 7>> prefill_events_;
    size_t prefill_event_offset_ = 0;
    std::vector<DeviceBuffer> checkpoint_history_, checkpoint_state_;
    int prefill_checkpoint_position_ = -1;
    DenseProfile dense_profile_;
    void prepare_prefill_events(int chunks);
    void collect_prefill_events(int chunks, Metrics& metrics);
    PinnedExpertSet pinned_experts_;
    HostReadOnlyLease host_readonly_;
    std::unique_ptr<MoeGraphState> moe_graph_;
};

std::unique_ptr<kernels::SamplerParams> sampling_from_environment();
void cuda_check(cudaError_t error, const char* what);

} // namespace strata::qwen35moe
