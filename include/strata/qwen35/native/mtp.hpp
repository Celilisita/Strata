#pragma once
#include "strata/qwen35/native/model.hpp"
#include <array>
#include <algorithm>
#include <cstddef>
namespace strata::qwen35::native {
class MtpGraphs;
struct MtpPinned {
    struct Payload {int ids[3];unsigned flags[3];int draft_ids[2];unsigned draft_flags[2];int position,known;} *p=nullptr;
    MtpPinned(){check_cuda(cudaMallocHost(reinterpret_cast<void **>(&p),sizeof(Payload)),"MTP pinned fixed response");}
    ~MtpPinned(){if(p)cudaFreeHost(p);}
};
static_assert(offsetof(MtpPinned::Payload,draft_ids)==24&&offsetof(MtpPinned::Payload,known)==44,"MTP bulk response extent");
struct MtpCounters {
    size_t rounds=0,proposed=0,accepted=0,target_rows=0,replayed_rows=0,rejections=0;
    size_t graph_PP=0,graph_draft=0,graph_verify=0,graph_replay=0,graph_rewind=0,graph_commit=0;
    size_t prefix_written_bytes=0,prefix_restored_rows=0,graph_prefix_restore=0;
    size_t H2D_bytes=0,D2H_bytes=0,D2D_bytes=0,completion_joins=0;
};
class NativeMtp {
public:
    NativeMtp(DenseModel &,const strata::GgufFile &,cudaStream_t);
    ~NativeMtp();
    void release_graphs();
    void reserve_cache(int,cudaStream_t);
    void enable_graphs(const int *prefix,int tokens,int maximum,cudaStream_t);
    size_t graph_nodes()const;
    size_t graph_replays()const;
    double graph_cold_ms()const;
    bool graph_enabled()const{return bool(graphs_)&&!graph_suspended_;}
    void use_graphs(bool enabled){if(enabled&&!graphs_)throw std::logic_error("MTP graphs not prepared");graph_suspended_=!enabled;}
    void use_gpu_draft_chain(bool enabled){if(graphs_)throw std::logic_error("MTP chain must be selected before capture");gpu_draft_chain_=enabled;}
    void use_prefix_states(bool enabled);
    bool prefix_states_enabled()const{return retain_prefix_;}
    size_t recurrent_bytes()const;
    bool gpu_draft_chain()const{return gpu_draft_chain_;}
    void prefill(const int *ids,int tokens,cudaStream_t);
    // 'known' is already emitted. This consumes it and at most two proposals,
    // emits verified target picks and commits exactly the consumed prefix.
    std::vector<int> round(int known,int position,int proposals,cudaStream_t);
    const MtpCounters &counters()const{return counters_;}
    size_t ram_bytes()const{return ram_?ram_->host_bytes():0;}
    RamKvCounters ram_counters(cudaStream_t s)const{return ram_?ram_->counters(s):RamKvCounters{};}
    size_t bytes()const{return storage_.bytes();}
    std::array<ContinuationView,3> prefix_views(int tokens)const{
        if(tokens<2||tokens>=model_.logical_context()||(!model_.kv_window()&&tokens>=model_.cache_logical_capacity()))throw std::out_of_range("MTP prefix view extent");
        const auto &s=model_.spec_;
        const size_t bytes=size_t(std::min(tokens-1,model_.cache_logical_capacity()))*s.n_head_kv*(s.head_dim/32)*(model_.q8_cache()?34:18);
        if(ram_)return {{{ram_->view().host_key,bytes,64,2,true},{ram_->view().host_value,bytes,64,3,true},{seed_,size_t(s.n_embd)*4,64,0,false}}};
        if(model_.kv_window()){
            const int live=std::min(tokens-1,model_.kv_window());
            const size_t row=s.n_head_kv*(s.head_dim/32)*34;
            const int first=(tokens-1-live)%capacity_;
            return {{{key_,size_t(live)*row,64,2,true,row,capacity_,first},
                     {value_,size_t(live)*row,64,3,true,row,capacity_,first},{seed_,size_t(s.n_embd)*4,64,0,false}}};
        }
        return {{{key_,bytes,64,2,true},{value_,bytes,64,3,true},{seed_,size_t(s.n_embd)*4,64,0,false}}};
    }
    void check_rewind(int position,cudaStream_t);
private:
    friend class MtpGraphs;
    bool graph_suspended_=false;
    bool gpu_draft_chain_=false,retain_prefix_=false;
    std::vector<GdnPrefix> prefix_states_;
    bool graph_building_=false;int graph_offset_=0;
    MtpBatchCapture graph_batch_{};
    std::unique_ptr<MtpGraphs> graphs_;
    std::unique_ptr<MtpPinned> host_;
    int *target_ids_=nullptr;unsigned *target_flags_=nullptr;
    int *draft_ids_=nullptr;unsigned *draft_flags_=nullptr;
    struct Saved {float *state=nullptr,*history=nullptr;unsigned parity=0;};
    void snapshot(cudaStream_t);
    void restore_prefix(int consumed,cudaStream_t);
    void reject_prefix(int consumed,std::vector<int> &,int position,cudaStream_t);
    void rewind(cudaStream_t);
    void condition(const int *,const float *,int,int,cudaStream_t);
    int draft(const int *,const float *,int,cudaStream_t);
    void enqueue_draft(const int *,const float *,int,cudaStream_t);
    void catch_up(const int *,const float *,int,int,cudaStream_t);
    void enqueue_target(const int *,int,int,cudaStream_t,bool retain_prefix=false);
    void enqueue_pick(const float *,const float *,Matrix,float *,cudaStream_t);
    void enqueue_commit(int,int,cudaStream_t);
    std::vector<int> target(const int *,int,int,cudaStream_t,bool snapshot=false,bool chain_response=false);
    int read_draft(cudaStream_t);
    std::vector<int> round_gpu_chain(int known,int position,int proposals,cudaStream_t);
    DenseModel &model_;DeviceStorage storage_;AttentionWeights decoder_{};
    Matrix projection_{};const float *enorm_=nullptr,*hnorm_=nullptr,*head_norm_=nullptr;
    std::unique_ptr<AttentionLayer> attention_;std::vector<Saved> saved_;
    std::unique_ptr<RamKvCache> ram_;
    void *key_=nullptr,*value_=nullptr;int capacity_=0;
    float *embedding_=nullptr,*concat_=nullptr,*conditioned_=nullptr,*normed_=nullptr;
    float *prefix_=nullptr,*seed_=nullptr,*verify_rows_=nullptr,*catch_rows_=nullptr,*draft_seed_=nullptr;
    float *logits_=nullptr;int *device_ids_=nullptr,*picked_=nullptr;unsigned *invalid_=nullptr;
    void *greedy_=nullptr;size_t greedy_bytes_=0;MtpCounters counters_{};
};
}
