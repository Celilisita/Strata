#pragma once
#include "strata/qwen35/native/weights.hpp"
#include "strata/qwen35/native/token_ops.hpp"
#include <chrono>
#include <utility>
#include "strata/qwen35/native/kv_ram.hpp"
#include "strata/qwen35/native/mtp_position.hpp"
#include "strata/qwen35/native/visual.hpp"

namespace strata::qwen35::native {
struct CacheSnapshot {const void *key,*value;int capacity;};
struct ContinuationView { const void *data;size_t bytes;int layer;unsigned field;bool packed;size_t row_bytes=0;int ring_capacity=0,first_row=0; };
struct DenseLimits { int context=262144,batch=8192,ubatch=8192,max_tokens=8192; bool kv_q8=false;int kv_initial_tokens=0;int kv_window_tokens=0;int kv_ram_tokens=0;int kv_select_tokens=0;int kv_keep_queries=0; bool native32=false,qkv_planes=false,qkv_byte_planes=false,wide_grid=false,fused_pack32=false,joint_ffn=false,joint_projections=false,token256_pair=false,warp64_pair=false,operand_lookahead=false,bank_ffn=false,folded_ffn=false,folded_compact_ffn=false,folded_gray_ffn=false,folded_widewarp_ffn=false,folded_ring_ffn=false; };
// One weight inventory and shared arena for Dense PP and continuation TG.
// Split calls let a dedicated benchmark check layers without runtime callbacks.
class DenseModel {
public:
    DenseModel(const strata::qwen35::Spec &,const DenseWeights &,DenseLimits,cudaStream_t);
    void enable_whole_k_q4_pair(bool tensor_bias=false){
        if(!limits_.native32||!limits_.fused_pack32||!limits_.qkv_byte_planes||weights_.tiled_ffn())
            throw std::invalid_argument("chunk affine pair requires raw fused byte PP32");
        shared_->shared_ffn(32).enable_whole_k_q4_pair(tensor_bias);
    }
    void reserve_cache(int tokens,cudaStream_t);
    // A new complete prefill follows graph release; reset_prefill before use.
    void reserve_fresh_cache(int tokens,cudaStream_t);
    int cache_capacity()const{return cache_capacity_;}
    bool q8_cache()const{return limits_.kv_q8;}
    int kv_window()const{return limits_.kv_window_tokens;}
    int kv_resident()const{return limits_.kv_ram_tokens;}
    int kv_selected()const{return limits_.kv_select_tokens?limits_.kv_select_tokens:limits_.kv_ram_tokens;}
    int kv_reuse_horizon()const{return limits_.kv_keep_queries;}
    int cache_logical_capacity()const;
    RamKvCounters ram_counters(cudaStream_t)const;
    size_t ram_bytes()const;
    int logical_context()const{return limits_.context;}
    float *embed(const int *device_ids,int start,int tokens,cudaStream_t);
    void set_visual_input(std::unique_ptr<DenseVisualInput>,cudaStream_t);
    const int* visual_positions()const{return visual_?visual_->table:nullptr;}
    void visual_embedding(float* output,int start,int count,cudaStream_t stream)const{
        if(visual_)visual_->overlay(output,start,count,stream);
    }
    float *layer(int index,const float *hidden,int start,int tokens,cudaStream_t);
    void head(const float *last_hidden,cudaStream_t);
    void head_batch(const float *hidden,int tokens,float *normalized,int *ids,unsigned *flags,cudaStream_t);
    void use_batch_head(bool enabled){batch_head_=enabled;}
    bool batch_head_enabled()const{return batch_head_;}
    // Cold selection before eager execution or graph capture.
    void mask_tokens(const std::vector<int> &ids){token_mask_=make_token_mask(ids,int(spec_.n_vocab));}
    int masked_token_count()const{return token_mask_.count;}
    void prefill(const int *device_ids,int tokens,cudaStream_t,int start=0);
    void prefill_chunk(const int *device_ids,int start,int tokens,bool pick_last,cudaStream_t);
    void begin_prefill_progress(int tokens,int base=0);
    void suppress_prefill_progress(bool on){pp_suppress_=on;}
    // Fixed fresh PP capture: validate before replay, mark history1 afterward.
    void validate_captured_prefill(bool complete);
    // Consume the previous device-picked ID and extend existing SSM/KV state.
    void decode(int position,cudaStream_t);
    void validate_captured_decode_pair() const;
    void mtp_capture(const MtpBatchCapture *c){mtp_capture_=c;}
    void decode_capture(const DecodeCapture *c){decode_capture_=c;}
    DecodeRounding decode_rounding() const { return attention_->decode_rounding(); }
    int query_heads() const { return int(spec_.n_head); }
    std::vector<ContinuationView> continuation_views(int valid) const;
    // Reset a fresh request; active KV rows are overwritten by PP before read.
    void reset_prefill(cudaStream_t);
    // Keep host KV rows for a repeated prompt prefix; no recurrent reset.
    void resume_kv(int valid,cudaStream_t);
    // Chunk-boundary recurrent snapshot: a shorter common prefix can rewind
    // here because the linear recurrence has no per-token undo.
    int checkpoint_position()const{return checkpoint_position_;}
    void restore_checkpoint(cudaStream_t);
    const float *logits() const { return logits_; }
    const float *final_norm() const { return final_norm_; }
    const int *picked_token() const { return picked_; }
    const unsigned *invalid_logits() const { return invalid_; }
    size_t workspace_bytes() const { return storage_.bytes()+shared_->workspace_bytes(); }
    CacheSnapshot cache_snapshot(int index) const;
    int max_tokens() const { return limits_.max_tokens; }
    void use_pp_gemm(DenseGemm *service,bool ffn=true,bool projections=true){shared_->use_pp_gemm(service,ffn,projections);head_service_=service;}
    void use_pp_attention_gemm(DenseGemm *service){attention_->use_pp_attention_gemm(service);}
    void enable_tiny_batches(){shared_->enable_tiny_batches();}
    void enable_tiny_row_ffn(){shared_->enable_tiny_row_ffn();}
    void enable_tiny_meta_ffn(){shared_->enable_tiny_meta_ffn();}
    void use_pp_gdn_rows(int rows){shared_->use_pp_gdn_rows(rows);}
    void use_pp_gdn_kernel(RecurrentLayer::PpGdnKernel kernel,RecurrentLayer::PpGdnWorkspace workspace){
        shared_->use_pp_gdn_kernel(kernel,workspace);
    }
    AttentionLayer &attention_work(){return *attention_;}
    const int *positions()const{return positions_;}
    std::pair<void *,size_t> attention_scratch()const{
        const auto &ffn=shared_->shared_ffn(2);return {ffn.buffers().half_weights,ffn.half_weight_capacity()};
    }
    size_t qkv_plane_bytes()const{return qkv_plane_bytes_;}
    size_t qkv_plane_count()const{return qkv_plane_count_;}
    double qkv_plane_cold_ms()const{return qkv_plane_cold_ms_;}
private:
    friend class NativeMtp;
    friend class ServiceDecodeGraph;
    friend class MtpGraphs;
    int cache_capacity_=0;bool prefilling_=false;
    int pp_total_=0,pp_base_=0;bool pp_suppress_=false;std::chrono::steady_clock::time_point pp_started_{};
    strata::qwen35::Spec spec_;const DenseWeights &weights_;DenseLimits limits_;
    DeviceStorage storage_;
    std::unique_ptr<RecurrentLayer> shared_;
    std::unique_ptr<AttentionLayer> attention_;
    DenseGemm *head_service_=nullptr;
    struct State { float *history[2]{},*state=nullptr;unsigned current=0; };
    struct Cache { void *key=nullptr,*value=nullptr;std::unique_ptr<RamKvCache> ram; };
    std::vector<State> states_;std::vector<Cache> caches_;
    struct QkvView {const void *data=nullptr;size_t bytes=0;};
    std::vector<QkvView> qkv_views_;
    size_t qkv_plane_bytes_=0,qkv_plane_count_=0;
    double qkv_plane_cold_ms_=0;
    float *input_=nullptr,*final_norm_=nullptr,*logits_=nullptr;
    int *positions_=nullptr,*picked_=nullptr;unsigned *invalid_=nullptr;
    void *greedy_=nullptr;size_t greedy_bytes_=0;
    float *batch_logits_=nullptr;void *batch_greedy_=nullptr;bool batch_head_=false;
    TokenMask token_mask_{};
    const MtpBatchCapture *mtp_capture_=nullptr;
    const GdnPrefix *mtp_prefix_=nullptr;
    const DecodeCapture *decode_capture_=nullptr;
    std::unique_ptr<DenseVisualInput> visual_;
    void save_checkpoint(int position,cudaStream_t);
    std::vector<void*> checkpoint_state_,checkpoint_history0_,checkpoint_history1_;
    std::vector<int> checkpoint_current_;
    void *checkpoint_=nullptr;size_t checkpoint_bytes_=0;int checkpoint_position_=-1;
};
}
