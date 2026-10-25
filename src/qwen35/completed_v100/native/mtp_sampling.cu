#include "sampling_positioned.hpp"
#include "strata/qwen35/completed_v100/native/mtp.hpp"
#include <algorithm>

namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void history_window(const int* history,const int* inputs,int* window,
    int length,int row,int position,const int* cursor) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=length)return;
    const int first=cursor?*cursor:position;
    const int index=first+row+1-length+i;
    window[i]=index<0?-1:(index<first?history[index]:inputs[index-first]);
}
__global__ void sample_link(const strata::kernels::SamplePick* pick,int vocabulary,
    int* output,unsigned* invalid) {
    if(threadIdx.x||blockIdx.x)return;
    const auto value=*pick;
    if(value.first_bad!=-1||value.id<0||value.id>=vocabulary)atomicOr(invalid,1u);
    *output=*invalid?0:value.id;
}
}
void MtpSampling::configure(strata::kernels::SamplerParams params,int* history,int prompt) {
    strata::kernels::RadixSampler::validate(params);
    if(!history||prompt<2)throw std::invalid_argument("MTP sampling requires its actual GPU prompt history");
    const bool penalties=params.penalty_repeat!=1||params.penalty_freq!=0||params.penalty_present!=0;
    const int length=penalties?params.penalty_last_n:0;
    if(length>capacity_) {
        if(window_)storage_.release(window_);
        window_=static_cast<int*>(storage_.allocate(size_t(length)*4));capacity_=length;
    }
    params_=params;params_.counter=0;history_=history;prompt_=prompt;history_length_=length;
}
void MtpSampling::pick(const float* logits,const int* inputs,int row,int position,
    const int* cursor,int* output,unsigned* invalid,cudaStream_t stream) {
    if(!logits||!inputs||!output||!invalid||row<0||row>2||position<prompt_)
        throw std::invalid_argument("Invalid MTP sampled target/draft extent");
    if(history_length_) {
        history_window<<<(history_length_+255)/256,256,0,stream>>>(history_,inputs,window_,
            history_length_,row,position,cursor);
        check_cuda(cudaGetLastError(),"MTP GPU penalty history");
    }
    if(cursor)sample_positioned(sampler_,logits,window_,history_length_,params_,cursor,prompt_-1,row,stream);
    else {
        auto params=params_;params.counter=uint64_t(position-prompt_+1+row);
        sampler_.sample(logits,window_,history_length_,params,stream);
    }
    sample_link<<<1,32,0,stream>>>(sampler_.result(),vocabulary_,output,invalid);
    check_cuda(cudaGetLastError(),"MTP sampled GPU ID/finite link");
}
size_t MtpSampling::commit(const int* inputs,int count,int position,cudaStream_t stream) {
    if(history_length_)check_cuda(cudaMemcpyAsync(history_+position,inputs,size_t(count)*4,
        cudaMemcpyDeviceToDevice,stream),"MTP committed GPU penalty history");
    return history_length_?size_t(count)*4:0;
}
void NativeMtp::configure_sampling(bool enabled,const strata::kernels::SamplerParams& params,
    int* history,int prompt) {
    if(graphs_)throw std::logic_error("MTP sampler must be bound before graph capture");
    if(!enabled){sampling_.reset();return;}
    if(!gpu_draft_chain_)throw std::logic_error("Sampled MTP requires the GPU draft chain");
    if(!sampling_)sampling_=std::make_unique<MtpSampling>(int(model_.spec_.n_vocab));
    sampling_->configure(params,history,prompt);
}
}
