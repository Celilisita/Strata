#pragma once
#include "strata/qwen35/native/mtp.hpp"
#include <functional>
namespace strata::qwen35::native {
class MtpGraphOperation {
public:
    MtpGraphOperation()=default;
    ~MtpGraphOperation();
    void capture(cudaStream_t,const std::function<void()> &,size_t node_budget=8192);
    void launch(cudaStream_t);
    size_t nodes()const{return nodes_;}
    size_t replays()const{return replays_;}
private:
    cudaGraph_t graph_=nullptr;cudaGraphExec_t exec_=nullptr;
    size_t nodes_=0,replays_=0;
};
class MtpGraphs {
public:
    MtpGraphs(NativeMtp &,const int *prefix,int tokens,int maximum,cudaStream_t);
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
private:
    void parity(unsigned);
    unsigned parity()const;
    DeviceStorage cursor_storage_;
    NativeMtp &m_;MtpBatchCapture batch_{};int maximum_;
    unsigned prefix_parity_=1;
    MtpGraphOperation prefix_,draft_[2],target_[2][4],rewind_[2],commit_[3];
    MtpGraphOperation prefix_restore_[2][2];
    double cold_ms_=0;cudaStream_t stream_;
};
}
