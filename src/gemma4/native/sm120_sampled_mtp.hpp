#pragma once
#include "strata/gemma4/native/sm120_model.hpp"
#include "strata/kernels/radix_sampler.hpp"
#include "strata/gemma4/native/sm120_mtp_target.hpp"
#include "sm120_mtp_assistant.hpp"
#include "sm120_mtp_phase_timing.hpp"
#include <array>
#include <cstdint>
#include <vector>
namespace strata::gemma4::native::sampled_mtp {
enum class Stop:int {Continue=0,Eog=1,Budget=2,Invalid=3};
struct Response {
    int count=0,accepted=0,proposed=0,matched=0,finished=0,reason=0,position=0,ids[3]{};
    unsigned invalid=0;
};
static_assert(sizeof(Response)==44,"sampled MTP fixed response");
struct Round {
    mtp_assistant::Bounds bounds{};Response response{};
    kernels::SamplePick picks[3]{};std::uint64_t counter=0;
    int remaining=0,start=0,drafts[2]{},inputs[3]{},next=0;unsigned invalid=0;
};
struct Stats {
    std::array<std::size_t,3> captures{},replays{},nodes{};
    std::size_t rounds=0,eager_rounds=0,emitted=0,accepted=0,proposed=0;
    std::size_t history_upload_bytes=0,metadata_upload_bytes=0;
};
// No model, sampler or Draft owner: all three outlive this controller and its
// graphs. Host request history includes pending exactly once before round().
class Controller {
    Sm120Model& main_;mtp_assistant::Draft& draft_;kernels::RadixSampler& sampler_;
    primitive::DeviceStorage memory_;cudaStream_t stream_=nullptr;
    Round* state_=nullptr;int *history_=nullptr,*rows_history_=nullptr,*eog_=nullptr;
    Sm120MtpLease lease_{};Sm120MtpCacheLease local_{},global_{};
    kernels::SamplerParams params_{};int requested_=0,history_length_=0,eog_count_=0;
    std::uint64_t head_epoch_=0;bool bound_=false,poisoned_=false,eager_seen_=false,graph_enabled_=true;
    std::uint64_t* phase_ticks_=nullptr;PhaseTiming last_timing_{};
    std::array<cudaGraph_t,3> graphs_{};std::array<cudaGraphExec_t,3> exec_{};Stats stats_{};
    void validate()const;void submit(int);void capture(int);void destroy_graphs()noexcept;
public:
    Controller(Sm120Model&,mtp_assistant::Draft&,kernels::RadixSampler&);
    ~Controller();Controller(const Controller&)=delete;Controller& operator=(const Controller&)=delete;
    void bind(const kernels::SamplerParams&,int requested,const std::vector<int>& eog_ids,bool graph_enabled=true);
    Response round(const std::vector<int>& history,std::uint64_t counter,int remaining);
    void release_graphs();const Stats& stats()const{return stats_;}
    const PhaseTiming& lastTiming()const{return last_timing_;}
    bool phase_timing_enabled()const{return phase_ticks_!=nullptr;}
    std::size_t bytes()const{return memory_.bytes();}
};
void prepare_round(Round*,const int*,const unsigned*,int local_capacity,int global_capacity,int context,cudaStream_t);
void save_draft(Round*,const int*,const unsigned*,int slot,cudaStream_t);
void form_inputs(Round*,const int* pending,int drafts,cudaStream_t);
void prepare_histories(Round*,const int* history,int* rows,int length,int drafts,const unsigned* main_invalid,cudaStream_t);
void decide(Round*,int drafts,int capacity,const int* eog,int count,const unsigned* main_invalid,cudaStream_t);
void finalize(Round*,const int* position,const unsigned* main_invalid,cudaStream_t);
}
