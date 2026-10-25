#pragma once
#include "strata/qwen35/native/mtp.hpp"
#include <functional>
#include "mtp_exec_policy.hpp"
#include "mtp_prefix_graph_policy.hpp"
#include "mtp_tg_graph_policy.hpp"
namespace strata::qwen35::native {
class MtpGraphOperation {
public:
    MtpGraphOperation()=default;
    ~MtpGraphOperation();
    void capture(cudaStream_t,const std::function<void()> &,size_t node_budget=8192,bool eager=true);
    void launch(cudaStream_t);
    size_t nodes()const{return nodes_;}
    size_t replays()const{return replays_;}
    void reset_replays(){replays_=0;}
private:
    void instantiate(cudaStream_t,bool lazy);
    cudaStream_t stream_=nullptr;
    cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;
    size_t nodes_=0,replays_=0;
};
class MtpGraphs {
    friend class NativeMtp;
public:
    MtpGraphs(NativeMtp &,const int *prefix,int tokens,int maximum,cudaStream_t,bool capture_prefix);
    ~MtpGraphs();
    void begin(int position,cudaStream_t);
    void prefix(cudaStream_t);
    void draft(int step,cudaStream_t);
    void target(int count,bool snapshot,cudaStream_t);
    void rewind(cudaStream_t);
    void commit(int count,cudaStream_t);
    void restore_prefix(int count,cudaStream_t);
    size_t nodes()const;
    size_t replays()const;
    double cold_ms()const{return cold_ms_;}
    cudaStream_t stream()const{return stream_;}
    bool cacheable()const{return !prefix_.nodes()&&prefix_tokens_==8192;}
    bool matches(const std::string&,int,int,cudaStream_t)const;
    void rearm(const int*,cudaStream_t);
private:
    void capture_prefix(bool lazy);
    void parity(unsigned);
    unsigned parity()const;
    DeviceStorage cursor_storage_;
    NativeMtp &m_;MtpBatchCapture batch_{};int maximum_;
    std::string request_key_;MtpSampling *sampling_owner_=nullptr;int device_=-1;
    unsigned prefix_parity_=1;const int *prefix_input_=nullptr;int prefix_tokens_=0;bool prefix_eager_=true;
    MtpGraphOperation prefix_,draft_[3],target_[2][5],rewind_[2],commit_[4];
    MtpGraphOperation prefix_restore_[2][3];
    double cold_ms_=0;cudaStream_t stream_;
};
}
