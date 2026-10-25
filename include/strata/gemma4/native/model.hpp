#include "strata/runtime/capacity_storage.hpp"
#pragma once
#include "strata/gemma4/native/model_sampling.hpp"
#include "strata/gemma4/native/spec.hpp"
#include "strata/gemma4/native/decoder_ops.hpp"
#include "strata/gemma4/native/profile.hpp"
#include "strata/gemma4/native/attention_native.hpp"
#include "strata/gemma4/native/verify_model.hpp"
#include "strata/gemma4/native/q6_head_sidecar.hpp"
#include "strata/gemma4/native/q6_down.hpp"
#include "strata/gemma4/native/history.hpp"
#include <memory>
#include <vector>
namespace strata::gemma4::native {
enum class GlobalVerifyMerge {Parent174,Parallel321,SharedKv336};
enum class PpKeyTile {Keys512,Keys1024};
struct LayerWeights {
    primitive::Matrix q,k,v,out;const float *pre,*post,*qnorm,*knorm;FfnWeights ffn;
};
class Model {
    const GgufFile& file_;Spec spec_;strata::runtime::CapacityStorage weights_,memory_;
    std::unique_ptr<strata::runtime::CapacityStorage> kv_memory_;
    std::vector<LayerWeights> layers_;std::vector<CacheView> cache_;
    bool global_history_enabled_=false;
    std::vector<std::unique_ptr<HistoryCache>> history_owners_;
    std::vector<HistoryView> history_views_;
    primitive::Matrix embedding_,head_;const float *final_norm_=nullptr,*frequencies_=nullptr;
    Q6HeadSidecar head_sidecar_;
    std::unique_ptr<Ffn> ffn_;cudaStream_t stream_=nullptr;cudaGraph_t graph_=nullptr;cudaGraphExec_t executable_=nullptr;
    float *hidden_[2]{},*normalized_=nullptr,*q_=nullptr,*k_=nullptr,*v_=nullptr,*attended_=nullptr,*projected_=nullptr,*post_=nullptr,*after_=nullptr,*features_=nullptr,*logits_=nullptr,*partials_=nullptr;
    int *input_=nullptr,*position_=nullptr,*next_=nullptr;unsigned* invalid_=nullptr;
    float *verify_features_=nullptr,*verify_logits_=nullptr,*verify_partials_=nullptr,*draft_embedding_=nullptr;
    int *verify_inputs_=nullptr,*verify_picks_=nullptr;VerifyPacket* verify_packet_=nullptr;
    std::uint64_t reset_epoch_=0;
    std::unique_ptr<ModelSampling> sampling_;
    bool sampling_active() const;
    void enqueue_sampling(int tokens);
    Profile* profile_=nullptr;size_t graph_replays_=0;
    bool split_decode_=false,blas_prefill_=false,half_prefill_=false,fused_prefill_=false;
    GlobalVerifyMerge global_verify_merge_=GlobalVerifyMerge::Parent174;
    bool global_verify321_request_=false;
    PpKeyTile pp_key_tile_=PpKeyTile::Keys512;
    size_t global_verify321_enqueues_=0,global_verify336_enqueues_=0,pp_k1024_enqueues_=0;
    bool pp_three_product333_=false;size_t pp_three_product333_enqueues_=0;
    bool global_shared_long_=false,global_shared_long_request_=false;
    size_t global_shared_t2_enqueues_=0,parent_decode_attention_enqueues_=0;
    bool physical_global_pp_=false,physical_global_pp_request_=false;
    size_t physical_global_pp_enqueues_=0,parent_long_global_pp_enqueues_=0;
    bool compensated_physical_global_pp_=false,compensated_physical_global_pp_request_=false;
    size_t compensated_physical_global_pp_enqueues_=0;
    unsigned t1_paths_=0; // 1=245QKV,4=249head,8=262Q6Down,16=264SingleQKV,32=266global,64=270SWA,128=269GU,256=280long,512=291directGU(mutuallyexclusive128),1024=302Out244.
    size_t t1_qkv_enqueues_=0,t1_head_enqueues_=0;
    size_t t1_single_qkv_enqueues_=0,t1_global_merge_enqueues_=0,t1_swa270_enqueues_=0;
    size_t t1_global_shared280_enqueues_=0,t1_out302_enqueues_=0;
    std::uint64_t t1_arena_epoch_=0;
    Q6DownStorage q6_down_storage_=Q6DownStorage::Gguf;
    size_t q6_down_matrices_=0,q6_down_packed_bytes_=0,q6_down_released_bytes_=0;
    double q6_down_cold_pack_wall_s_=0;
    void mark(const char* name,int layer,int tokens){if(profile_)profile_->mark(name,layer,tokens,stream_);}
    int maximum_batch_,capacity_=0,host_position_=0;size_t weight_bytes_=0;
    primitive::Matrix matrix(const TensorInfo*);const float* scalar(const TensorInfo*);
    unsigned projection_q8_mask_=0;
    // Per token-count bits: localQKV, globalQK, localOut, globalOut, head.
    bool single_projection(int role,int tokens)const{
        return tokens>=1&&tokens<=3&&(projection_q8_mask_&(1u<<(role+5*(tokens-1))));
    }
    void validate_verify_policy(VerifyPolicy,int rows)const;
    void project_head(const float*,float*,int,bool t1_decode=false);
    void validate_global_attention_geometry()const;
    bool enqueue_parallel_global_verify(CacheView,int);
    bool enqueue_shared_global_attention(CacheView,int);
    bool enqueue_parallel_global_t1(CacheView,int);
    bool enqueue_swa_warp_t1(CacheView,int);
    bool enqueue_physical_global_pp(CacheView,int);
    float* forward_hidden(const int*,int);
    void forward(const int*,int,bool);void release_graph();
public:
    Model(const GgufFile&,int maximum_batch,cudaStream_t,Q6DownStorage=Q6DownStorage::Gguf,
        bool global_history=false);
    ~Model();
    void reset(int maximum_total);
    // Cold policy change. Caller releases controller/external Graphs first.
    // A change joins the stream, destroys the internal T1 Graph and changes
    // the lease epoch, so a stale DraftRoundController cannot replay its Graph.
    // KV, pending, features and absolute position are preserved for diagnostics.
    // Cold request policy: caller releases external graphs first. Requires P0;
    // joins the owned stream, releases T1 Graph and increments the lease epoch.
    void select_t1_q4_ffn(T1Q4FfnPolicy);
    // Cold P0 only; preserves246b FFN/PP inventory. External Graphs must be released.
    void select_t1_paths(unsigned mask);
    unsigned t1_paths()const{return t1_paths_;}
    size_t t1_out302_enqueues()const{return t1_out302_enqueues_;}
    // Cold P0 request selector; reset preserves mode and clears request counters.
    // External Graphs must already be released. Only exact1280 FFN GU changes.
    void select_short_pp_backend(ShortPpBackend);
    ShortPpBackend short_pp_backend()const{return ffn_->short_pp_backend();}
    size_t short_pp_enqueues(ShortPpBackend mode)const{return ffn_->short_pp_enqueues(mode);}

    size_t t1_qkv_enqueues()const{return t1_qkv_enqueues_;}
    size_t t1_head_enqueues()const{return t1_head_enqueues_;}
    size_t t1_single_qkv_enqueues()const{return t1_single_qkv_enqueues_;}
    size_t t1_q6_down_enqueues()const{return ffn_->t1_q6_down_enqueues();}
    bool t1_q6_down_striped_enabled()const{return ffn_->t1_q6_down_striped_enabled();}
    size_t t1_global_merge_enqueues()const{return t1_global_merge_enqueues_;}
    size_t t1_global_shared280_enqueues()const{return t1_global_shared280_enqueues_;}
    size_t t1_swa270_enqueues()const{return t1_swa270_enqueues_;}
    size_t t1_gu269_enqueues()const{return ffn_->t1_gu269_enqueues();}
    bool t1_gate_up_word_enabled()const{return ffn_->t1_gate_up_word_enabled();}
    size_t t1_gu291_enqueues()const{return ffn_->t1_gu291_enqueues();}
    bool t1_gate_up_direct_enabled()const{return ffn_->t1_gate_up_direct_enabled();}
    T1Q4FfnPolicy t1_q4_ffn_policy()const{return ffn_->t1_q4_ffn_policy();}
    size_t t1_q4_gate_up_enqueues()const{return ffn_->t1_q4_gate_up_enqueues();}
    size_t t1_q4_down_enqueues()const{return ffn_->t1_q4_down_enqueues();}
    // Cold P0 only; external Graphs released first. Reset preserves selectors,
    // clears capture/request counts. No process-global dispatch state.
    void select_global_verify_merge(GlobalVerifyMerge);
    GlobalVerifyMerge global_verify_merge()const{return global_verify_merge_;}
    size_t global_verify321_enqueues()const{return global_verify321_enqueues_;}
    size_t global_verify336_enqueues()const{return global_verify336_enqueues_;}
    bool global_verify336_request()const{return global_verify321_request_&&global_verify_merge_==GlobalVerifyMerge::SharedKv336;}
    bool global_verify321_request()const{return global_verify321_request_;}
    void select_pp_key_tile(PpKeyTile);
    PpKeyTile pp_key_tile()const{return pp_key_tile_;}
    // Logical K1024 total includes324 and333; leaf getters distinguish actual calls.
    size_t pp_k1024_enqueues()const{return pp_k1024_enqueues_;}
    size_t pp_four_product324_enqueues()const{return pp_k1024_enqueues_-pp_three_product333_enqueues_;}
    void select_pp_three_product333(bool enabled);
    bool pp_three_product333_enabled()const{return pp_three_product333_;}
    bool pp_three_product333_request()const{return pp_three_product333_&&compensated_physical_global_pp_request_&&pp_key_tile_==PpKeyTile::Keys1024;}
    size_t pp_three_product333_enqueues()const{return pp_three_product333_enqueues_;}
    size_t pp_k512_enqueues()const{return compensated_physical_global_pp_enqueues_-pp_k1024_enqueues_;}
    void select_global_shared_long(bool enabled);
    bool global_shared_long_enabled()const{return global_shared_long_;}
    bool global_shared_long_request()const{return global_shared_long_request_;}
    // Host enqueue/capture counts only; Graph replays do not increment these.
    size_t global_shared_t2_enqueues()const{return global_shared_t2_enqueues_;}
    size_t parent_decode_attention_enqueues()const{return parent_decode_attention_enqueues_;}
    // Cold opt-in after controller/external graphs are released and reset has
    // completed. Only a complete30720 request with batch8192/cap32768 qualifies.
    // reset preserves the selection but clears request identity and counters.
    void select_physical_global_pp(bool enabled);
    bool physical_global_pp_enabled()const{return physical_global_pp_;}
    bool physical_global_pp_request()const{return physical_global_pp_request_;}
    size_t physical_global_pp_enqueues()const{return physical_global_pp_enqueues_;}
    size_t parent_long_global_pp_enqueues()const{return parent_long_global_pp_enqueues_;}
    // Independent286 policy, defaultOFF. Enable only with physicalPP alreadyON.
    // Caller releases externalgraphs, thenreset/P0 beforechanging this policy.
    // Uses the same borrowedFfn/rawKV buffers; allModelowners require this ABI.
    void select_compensated_physical_global_pp(bool enabled);
    bool compensated_physical_global_pp_enabled()const{return compensated_physical_global_pp_;}
    bool compensated_physical_global_pp_request()const{return compensated_physical_global_pp_request_;}
    size_t compensated_physical_global_pp_enqueues()const{return compensated_physical_global_pp_enqueues_;}
    void set_profile(Profile* profile){profile_=profile;}
    // Cold opt-in after explicit per-token head-Q8 policy, with position0 and
    // no T1 Graph. Caller destroys controller/external Graphs before reset.
    // OFF joins and releases the sidecar; use a small reset before OFF if the
    // next long KV reservation cannot coexist with these additional bytes.
    void select_aligned_q6_head(bool enabled);
    bool aligned_q6_head_enabled()const{return head_sidecar_.enabled();}
    size_t aligned_q6_head_bytes()const{return head_sidecar_.bytes();}
    double aligned_q6_head_cold_pack_s()const{return head_sidecar_.cold_pack_wall_s();}
    void select_library_pp(bool enabled){ffn_->select_residual_pp(enabled);}
    // Cold policy only. Only the exact1280 PP branch uses the compact layout.
    void select_compact_half(bool enabled){
        if(executable_||host_position_!=0)
            throw std::invalid_argument("Gemma compact panels policy requires reset to idle");
        ffn_->select_compact_half(enabled);
    }
    // Caller releases all controller/external graphs, joins, and resets first.
    // This changes only exact1280 PP; T1/T2/T3 and long PP retain their paths.
    void select_three_panel_half(bool enabled){
        if(executable_||host_position_!=0)
            throw std::invalid_argument("Gemma three-panel policy requires reset to idle");
        ffn_->select_three_panel_half(enabled);
    }
    bool three_panel_half_enabled()const{return ffn_->three_panel_half_enabled();}
    void select_row_half(bool enabled){
        if(executable_)throw std::runtime_error("Gemma row-pair change requires reset");
        ffn_->select_row_half(enabled);
    }
    void select_projection_q8_mask(unsigned mask){
        if((t1_paths_&1024u)&&(mask&12u)!=12u)
            throw std::invalid_argument("Gemma302 requires both retained Single40 Out roles");
        if(executable_||mask>32767)throw std::invalid_argument("Gemma projection precision requires reset/15-bit policy");
        if((t1_paths_&(1u|16u))&&(mask&3u))
            throw std::invalid_argument("Gemma267 explicit QKV path conflicts with legacy Single40 mask");
        if((t1_paths_&4u)&&(mask&16u))
            throw std::invalid_argument("Gemma249 head conflicts with legacy Single40 mask");
        projection_q8_mask_=mask;
    }
    void select_periodic_row(bool enabled){
        if(executable_)throw std::runtime_error("Gemma periodic row policy requires reset");
        ffn_->select_periodic_row(enabled);
    }
    void select_fast_ffn(bool enabled){
        if(executable_)throw std::runtime_error("Gemma fast FFN policy requires reset");
        ffn_->select_fast_ffn(enabled);
    }
    void select_fused_attention(bool enabled){
        if(executable_)throw std::runtime_error("Gemma fused attention policy requires reset");
        fused_prefill_=enabled;
    }
    void select_single_ffn(bool enabled){
        if(executable_)throw std::runtime_error("Gemma FFN activation precision change requires reset");
        if(!enabled&&q6_down_storage_==Q6DownStorage::Aligned224)
            throw std::invalid_argument("Gemma native Q6 Down requires Single40 FFN");
        ffn_->select_single_ffn(enabled);
    }
    void select_native_pp(bool enabled){
        if(executable_)throw std::runtime_error("Gemma native PP policy requires reset");
        ffn_->select_native_pp(enabled);
    }
    void select_paired_pp(bool enabled){
        if(executable_)throw std::runtime_error("Gemma paired PP policy requires reset");
        ffn_->select_paired_pp(enabled);
    }
    void select_half_pp(bool enabled){
        if(executable_)throw std::runtime_error("Gemma PP precision change requires reset");
        ffn_->select_half_pp(enabled);
    }
    void select_pp_algorithm(int id){
        if(executable_)throw std::runtime_error("Gemma library algorithm change requires reset");
        ffn_->select_pp_algorithm(id);
    }
    void select_grouped(bool enabled){
        if(executable_)throw std::runtime_error("Gemma grouped vector change requires reset");
        ffn_->select_grouped(enabled);
    }
    void select_main_slice(int slice){
        if(executable_)throw std::runtime_error("Gemma PP slice change requires reset");
        ffn_->select_main_slice(slice);
    }
    void select_full_down(bool enabled){
        if(executable_)throw std::runtime_error("Gemma Down dispatch change requires reset");
        ffn_->select_full_down(enabled);
    }
    void select_all_pp(bool enabled){
        if(executable_)throw std::runtime_error("Gemma PP dispatch change requires reset");
        ffn_->select_all_pp(enabled);
    }
    void select_vector3(bool enabled){
        if(executable_)throw std::invalid_argument("Gemma vector policy requires reset");
        ffn_->select_vector3(enabled);
    }
    void select_half_attention(bool enabled){
        if(executable_)throw std::invalid_argument("Gemma PP attention precision change requires reset");
        half_prefill_=enabled;
    }
    void select_attention(bool split,bool blas){
        if(executable_)throw std::invalid_argument("Gemma attention policy requires reset before Graph capture");
        split_decode_=split;blas_prefill_=blas;
    }
    void configure_sampling(const kernels::SamplerParams&);
    int prefill(const std::vector<int>&);
    int step(bool graph_enabled=true);
    // No allocation, host inference, synchronization or D2H in enqueue calls.
    // Caller must ensure the full 2/3-row verify fits the reserved context before
    // enqueue/capture; EOS/budget clipping cannot undo an out-of-bounds KV write.
    // The same room check applies before every external graph replay. Policy
    // storage must be caller-owned outside Main/Draft workspaces for graph life.
    // draft_invalid is a caller-aggregated diagnostic: Draft.run clears its own
    // flag, so MTP2 must GPU-OR both runs before passing that aggregate here.
    void enqueue_verify(const int* draft1,const int* draft2,int draft_count,
        VerifyPolicy,const unsigned* draft_invalid=nullptr);
    // GPU ordinary T1 for the final budget/context tail; no draft or extra
    // response copy. Requires a processed prefix, positive remaining and room.
    void enqueue_base_tail(VerifyPolicy);
    // Finish exactly once after each enqueue/replay, before the next round.
    VerifyResponse finish_verify(); // One response+invalid D2H and stream join.
    const float* enqueue_scaled_embedding(const int* device_token);
    MainScratchLease draft_scratch()const;
    MainDraftLease draft_lease()const;
    MainCacheLease draft_cache(size_t layer)const;
    VerifyInspection verify_inspection()const{
        return {verify_inputs_,verify_picks_,verify_features_,verify_logits_,verify_packet_};
    }
    int position()const{return host_position_;}
    size_t graph_replays()const{return graph_replays_;}
    size_t graph_nodes()const{size_t count=0;if(graph_)primitive::check_cuda(cudaGraphGetNodes(graph_,nullptr,&count),"Gemma Graph node inventory");return count;}
    const Spec& spec()const{return spec_;}
    const float* features()const{return features_;}
    const std::vector<CacheView>& caches()const{return cache_;}
    size_t weight_bytes()const{return weight_bytes_+head_sidecar_.bytes();}
    Q6DownStorage q6_down_storage()const{return q6_down_storage_;}
    size_t q6_down_matrices()const{return q6_down_matrices_;}
    size_t q6_down_packed_bytes()const{return q6_down_packed_bytes_;}
    size_t q6_down_released_bytes()const{return q6_down_released_bytes_;}
    double q6_down_cold_pack_wall_s()const{return q6_down_cold_pack_wall_s_;}
    size_t kv_bytes()const;
    size_t history_ram_bytes()const;
    size_t history_resident_bytes()const;
    bool global_history_enabled()const{return global_history_enabled_;}
    // Stable descriptor for this request. Null means the complete request fits
    // the identity resident pool and retains the existing attention fast path.
    const HistoryView* global_history_view(size_t layer)const;
};
}
