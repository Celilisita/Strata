#pragma once
#include "strata/gemma4/native/nvfp4_ffn.hpp"
#include "strata/gemma4/native/sm120_attention.hpp"
#include "strata/gemma4/native/sm120_mtp_target.hpp"
#include <cstdint>
#include <memory>
#include <vector>
namespace strata::gemma4::native {
struct Sm120LayerWeights {
    primitive::Matrix q,k,v,out;
    const float *pre,*post,*qnorm,*knorm;
    Nvfp4FfnWeights ffn;
};
// GPU-only Gemma Dense31B math owner. The service owns sampling and token policy.
class Sm120Model {
    const GgufFile& file_;
    Spec spec_;
    cudaStream_t stream_;
    int batch_,capacity_=0,host_position_=0;
    std::uint64_t epoch_=0;
    primitive::DeviceStorage weights_,memory_;
    std::unique_ptr<primitive::DeviceStorage> kv_memory_;
    std::unique_ptr<Nvfp4ProjectionService> projections_;
    std::unique_ptr<Nvfp4Ffn> ffn_;
    std::unique_ptr<Sm120AttentionService> attention_;
    std::vector<Sm120LayerWeights> layers_;
    std::vector<CacheView> caches_;
    primitive::Matrix embedding_;
    const float *final_norm_=nullptr,*frequencies_=nullptr;
    float *hidden_[2]{},*normalized_=nullptr,*projected_=nullptr,*post_=nullptr,*after_=nullptr;
    float *q_=nullptr,*k_=nullptr,*v_=nullptr,*attended_=nullptr,*features_=nullptr,*logits_=nullptr;
    float* rope_table_=nullptr;
    int *input_=nullptr,*position_=nullptr;
    unsigned* invalid_=nullptr;
    std::vector<int> prompt_;
    cudaGraph_t graph_=nullptr;
    cudaGraphExec_t executable_=nullptr;
    const int* graph_input_=nullptr;
    bool eager_decode_seen_=false;
    int mtp_drafts_=0,target_rows_count_=0;
    float *target_features_=nullptr,*target_logits_=nullptr,*draft_embedding_=nullptr;
    detail::Sm120TargetState* target_state_=nullptr;
    const float* scalar(const TensorInfo*);
    float* forward_hidden(const int*,int);
    void project_head_rows(const float*,float*,int);
    void forward(const int*,int,bool);
    void release_graph()noexcept;
public:
    Sm120Model(const GgufFile&,int maximum_batch,cudaStream_t);
    ~Sm120Model();
    Sm120Model(const Sm120Model&)=delete;
    Sm120Model& operator=(const Sm120Model&)=delete;
    // Release request Graph/KV leases while preserving resident weights.
    void release_request();
    void reset(int maximum_total);
    // CPU validates/uploads IDs only. All model math stays on the GPU.
    void enqueue_prefill(const std::vector<int>&);
    // Borrowed token pointer is stable for Graph life and contains a valid ID.
    // First actual decode is eager; subsequent calls may capture/replay it.
    void enqueue_step(const int* device_token,bool graph_enabled=true);
    // Cold opt-in only. MTP0 never allocates these buffers. All caller-owned
    // controller/Draft graphs must be destroyed before reset, policy or release.
    void prepare_mtp(int drafts);
    int mtp_drafts()const{return mtp_drafts_;}
    Sm120MtpScratch mtp_scratch()const;
    Sm120MtpLease mtp_lease()const;
    Sm120MtpCacheLease mtp_cache(std::size_t layer)const;
    const float* enqueue_scaled_embedding(const int* device_token);
    // Verify [pending,draft1,(draft2)] without moving the authoritative cursor
    // or seed. One row is the ordinary context/budget tail. Check room before
    // every external Graph replay, not only capture. Suffix KV remains future.
    void enqueue_target_rows(const int* device_inputs,int rows);
    Sm120TargetRows target_rows()const{return {target_features_,target_logits_,target_rows_count_};}
    void enqueue_commit_prefix(const int* device_count,const int* device_next,
        const unsigned* round_invalid);
    // After the controller's one response D2H/join and invalid check. This only
    // mirrors the committed GPU cursor; no additional copy or model math.
    void finish_target_commit(int count,int position,std::uint64_t epoch);
    // Optional bounded join for direct API users; services may copy invalid_state.
    void finish();
    const float* logits()const{return logits_;}
    const unsigned* invalid_state()const{return invalid_;}
    unsigned* invalid_state(){return invalid_;}
    // Stable borrowed slot for RadixSampler::write_pick and Graph input.
    int* next_input()const{return input_;}
    const int* device_position()const{return position_;}
    cudaStream_t ordered_stream()const{return stream_;}
    const Spec& spec()const{return spec_;}
    int position()const{return host_position_;}
    int capacity()const{return capacity_;}
    std::uint64_t epoch()const{return epoch_;}
    std::size_t kv_bytes()const{return kv_memory_?kv_memory_->bytes():0;}
    std::size_t resident_bytes()const;
};
}
