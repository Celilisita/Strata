#pragma once
#include "strata/qwen35/native/weights.hpp"
#include "strata/kernels/radix_sampler.hpp"
#include "strata/qwen35/native/mtp_gumbel.hpp"
#include "strata/qwen35/native/mtp_prob.hpp"
#include "strata/qwen35/native/mtp_prob_policy.hpp"
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <limits>
#include <vector>

namespace strata::qwen35::native {
// Eligible stochastic requests use either probability-ratio rejection/residual
// or shared token-ID noise; both preserve the filtered target marginal.
// Causal draft rows are owned here, tagged by logical counter and epoch.
// Only accepted draft IDs compare equal; later draws never decide earlier accepts.
class MtpSampling {
    DeviceStorage storage_;
    strata::kernels::RadixSampler sampler_;
    strata::kernels::SamplerParams params_;
    uint64_t* counter_ = nullptr;
    MtpProbCursor *cursor_=nullptr;MtpProbCursor host_cursor_{};
    MtpProbRow *draft_rows_=nullptr;
    int* history_ = nullptr;
    int history_size_ = 0, capacity_ = 0;
    std::vector<int> host_history_;
    int vocabulary_=0;bool gumbel_=false,probability_=false;
public:
    explicit MtpSampling(int vocab):sampler_(vocab) {
        vocabulary_=vocab;
        cursor_=static_cast<MtpProbCursor*>(storage_.allocate(sizeof(MtpProbCursor)));
        counter_=reinterpret_cast<uint64_t*>(cursor_);
        draft_rows_=static_cast<MtpProbRow*>(storage_.allocate(3*sizeof(MtpProbRow)));
    }
    void configure(const strata::kernels::SamplerParams& params,
                   const std::vector<int>& history, cudaStream_t stream, int resident_bound=0) {
        strata::kernels::RadixSampler::validate(params);
        params_ = params;
        const bool eligible=resident_bound>0&&mtp_gumbel_params(params_)&&mtp_gumbel_supported();
        const char *mode=std::getenv("STRATA_DENSE_MTP_COUPLING");
        if(eligible&&mode&&std::strcmp(mode,"gumbel")&&std::strcmp(mode,"prob"))throw std::invalid_argument("MTP coupling mode");
        probability_=eligible&&(mode?!std::strcmp(mode,"prob"):mtp_prob_default(resident_bound));gumbel_=eligible&&!probability_;
        if(probability_)check_cuda(cudaMemsetAsync(draft_rows_,0,3*sizeof(MtpProbRow),stream),"fresh probability row epoch");
        const bool penalized = params.penalty_repeat != 1 || params.penalty_freq != 0 || params.penalty_present != 0;
        history_size_ = penalized ? params.penalty_last_n : 0;
        if (history_size_ > capacity_) {
            auto* next = static_cast<int*>(storage_.allocate((size_t(history_size_)+4)*4));
            if (history_) storage_.release(history_);
            history_ = next;
            capacity_ = history_size_;
        }
        host_history_.assign(size_t(history_size_), -1);
        begin(params.counter, history, stream);
    }
    void begin(uint64_t counter, const std::vector<int>& history, cudaStream_t stream) {
        if (counter > std::numeric_limits<uint64_t>::max()-4)
            throw std::out_of_range("MTP sampling draw index overflow");
        if(host_cursor_.epoch==std::numeric_limits<uint64_t>::max())throw std::out_of_range("MTP sampling epoch overflow");
        host_cursor_.counter=counter;++host_cursor_.epoch;
        check_cuda(cudaMemcpyAsync(cursor_, &host_cursor_, sizeof(MtpProbCursor),
            cudaMemcpyHostToDevice, stream), "MTP sampling draw cursor");
        if (history_size_) {
            const size_t take = (std::min)(history.size(), size_t(history_size_));
            std::fill(host_history_.begin(), host_history_.end(), -1);
            std::copy(history.end()-take, history.end(), host_history_.end()-take);
            check_cuda(cudaMemcpyAsync(history_, host_history_.data(), size_t(history_size_)*4,
                cudaMemcpyHostToDevice, stream), "MTP committed penalty history");
        }
    }
    bool gumbel_enabled()const{return gumbel_;}
    int coupling_mode()const{return probability_?2:(gumbel_?1:0);}
    void pick(const float* logits, unsigned offset, const int* window,
              int* id, unsigned* invalid, cudaStream_t stream, bool target=false, bool has_draft=false) {
        if (offset > 3 || !window) throw std::invalid_argument("invalid MTP sampled window");
        if(probability_&&(!target||has_draft)&&offset>=3)throw std::invalid_argument("MTP probability row index");
        const int* history = nullptr;
        if (history_size_) {
            // Row j sees the committed tail followed by actual window[0..j].
            // No later proposal is visible to its penalties or random draw.
            check_cuda(cudaMemcpyAsync(history_+history_size_, window, size_t(offset+1)*4,
                cudaMemcpyDeviceToDevice, stream), "MTP row-specific penalty prefix");
            history = history_+offset+1;
        }
        sampler_.sample_counter(logits, history, history_size_, params_, stream, counter_, offset);
        if(probability_){
            if(target)mtp_prob_target(sampler_,vocabulary_,params_,cursor_,offset,has_draft?draft_rows_+offset:nullptr,
                has_draft?window+offset+1:nullptr,id,invalid,stream,has_draft);
            else mtp_prob_draft(sampler_,vocabulary_,params_,cursor_,offset,draft_rows_+offset,id,invalid,stream);
        }else if(gumbel_)mtp_gumbel_pick(sampler_,vocabulary_,params_,counter_,offset,id,invalid,stream);
        else sampler_.write_pick(id, invalid, stream);
    }
};
}
