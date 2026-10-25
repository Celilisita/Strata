#include "strata/gemma4/native/model.hpp"
#include "strata/qwen35/native/ffn.hpp"
namespace strata::gemma4::native {
ModelSampling::ModelSampling(int vocabulary, const kernels::SamplerParams& params)
    : sampler_(vocabulary) {
    kernels::RadixSampler::validate(params);
    if (params.penalty_last_n > 262144)
        throw std::invalid_argument("Gemma sampling history exceeds logical context");
    request_.params = params;
    request_.params.greedy = params.temperature <= 0;
    request_.configured = true;
    if (request_.penalized() && params.penalty_last_n)
        history_ = static_cast<int*>(storage_.allocate(size_t(params.penalty_last_n) * sizeof(int)));
}
void ModelSampling::enqueue(float* logits, int vocabulary, float cap, int* next,
                            unsigned* invalid, cudaStream_t stream) {
    const int length = int(request_.history.size());
    if (length)
        primitive::check_cuda(cudaMemcpyAsync(history_, request_.history.data(),
            size_t(length) * sizeof(int), cudaMemcpyHostToDevice, stream), "Gemma sampling history");
    prepare_sampling_logits(logits, vocabulary, cap, invalid, stream);
    sampler_.sample(logits, history_, length, request_.params, stream);
    finish_sampling_pick(sampler_.result(), next, invalid, stream);
}
void Model::configure_sampling(const kernels::SamplerParams& params) {
    release_graph();
    kernels::RadixSampler::validate(params);
    const bool penalties = params.penalty_repeat != 1 || params.penalty_freq != 0 || params.penalty_present != 0;
    sampling_.reset();
    if (params.temperature > 0 || penalties)
        sampling_ = std::make_unique<ModelSampling>(spec_.vocabulary, params);
}
bool Model::sampling_active() const { return sampling_ && sampling_->active(); }
void Model::enqueue_sampling(int tokens) {
    sampling_->enqueue(logits_, spec_.vocabulary, spec_.logit_cap, next_, invalid_, stream_);
    advance(position_, tokens, capacity_, invalid_, stream_);
}
} // namespace strata::gemma4::native
