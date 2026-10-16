#include "sampling_state.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>

namespace strata::qwen35moe {
std::unique_ptr<kernels::SamplerParams> sampling_from_environment() {
    const char* enabled = std::getenv("STRATA_QWEN35_SAMPLE");
    if (!enabled || std::string(enabled) == "0") return nullptr;
    if (std::string(enabled) != "1") throw std::invalid_argument("SAMPLE must be0 or1");
    auto p = std::make_unique<kernels::SamplerParams>();
    auto integer = [](const char* name, int& x) { if (const char* s = std::getenv(name)) {
        size_t used = 0; x = std::stoi(s, &used); if (s[used]) throw std::invalid_argument(name); } };
    auto real = [](const char* name, float& x) { if (const char* s = std::getenv(name)) {
        size_t used = 0; x = std::stof(s, &used); if (s[used]) throw std::invalid_argument(name); } };
    integer("STRATA_QWEN35_TOP_K", p->top_k); real("STRATA_QWEN35_TOP_P", p->top_p);
    real("STRATA_QWEN35_MIN_P", p->min_p); real("STRATA_QWEN35_TEMPERATURE", p->temperature);
    integer("STRATA_QWEN35_MIN_KEEP", p->min_keep);
    integer("STRATA_QWEN35_PENALTY_LAST_N", p->penalty_last_n);
    real("STRATA_QWEN35_REPEAT_PENALTY", p->penalty_repeat);
    real("STRATA_QWEN35_FREQUENCY_PENALTY", p->penalty_freq);
    real("STRATA_QWEN35_PRESENCE_PENALTY", p->penalty_present);
    if (const char* s = std::getenv("STRATA_QWEN35_SEED")) {
        size_t used = 0; p->seed = std::stoull(s, &used); if (s[used] || *s == '-') throw std::invalid_argument("SEED");
    }
    kernels::RadixSampler::validate(*p);
    return p;
}
void Runtime::initialize_sampling(const kernels::SamplerParams* p) {
    if (!p) return;
    kernels::RadixSampler::validate(*p);
    if (!gpu_head_ || p->penalty_last_n > max_context_)
        throw std::invalid_argument("sampled provider requires GPU head and bounded history");
    sampling_ = std::make_unique<SamplingState>(int(spec_.n_vocab), *p);
}
void Runtime::configure_sampling(const kernels::SamplerParams& params) {
    if (decode_phase_ || host_pick_.id != -1 || session_capacity() != 1)
        throw std::invalid_argument("sampling configuration requires a reset single request");
    kernels::RadixSampler::validate(params);
    if (params.penalty_last_n > max_context_)
        throw std::invalid_argument("sampling history exceeds context");
    if (!sampling_) { initialize_sampling(&params); return; }
    auto& s = *sampling_;
    const size_t bytes = size_t(params.penalty_last_n) * sizeof(int);
    if (s.history.bytes() < bytes) s.history.reset(bytes);
    s.tail.reserve(size_t(params.penalty_last_n)); s.saved_tail.reserve(size_t(params.penalty_last_n));
    for (auto& row : s.upload_rows) row.reserve(size_t(params.penalty_last_n));
    s.initial = s.current = params;
    s.tail.clear(); s.saved_tail.clear(); s.verify = false; s.verified = 0;
}
void Runtime::sampling_history(const int* tokens, int count) {
    if (!sampling_ || sampling_->current.penalty_last_n == 0) return;
    auto& tail = sampling_->tail;
    const int bound = sampling_->current.penalty_last_n;
    const int n = (std::min)(count, bound);
    const size_t discard = tail.size() + size_t(n) > size_t(bound) ? tail.size() + size_t(n) - size_t(bound) : 0;
    tail.erase(tail.begin(), tail.begin() + discard);
    tail.insert(tail.end(), tokens + count - n, tokens + count);
}
void Runtime::launch_sample(Metrics& metrics, int row) {
    auto& s = *sampling_;
    auto& upload = s.upload_rows.at(size_t(row)); upload = s.tail;
    if (!upload.empty()) cuda_check(cudaMemcpyAsync(s.history.data(), upload.data(),
        upload.size() * sizeof(int), cudaMemcpyHostToDevice, stream_), "sampler history");
    metrics.control_h2d += upload.size() * sizeof(int);
    s.gpu.sample(selected_logits_, s.history.as<int>(), int(s.tail.size()), s.current, stream_,
        tg_dynamic_ && decode_phase_ ? &dynamic_frame_.as<core::NativeMissFrame>()->failed : nullptr);
    ++s.current.counter;
}
void Runtime::reset_sampling() {
    if (sampling_) { sampling_->tail.clear(); sampling_->current = sampling_->initial; }
}

int Runtime::sample_last_logits(const kernels::SamplerParams& params,
                                const std::vector<int>& history, Metrics& metrics) {
    if (!selected_logits_ || sampling_)
        throw std::runtime_error("MiniCPM logits are unavailable for request sampling");
    kernels::RadixSampler::validate(params);
    if (history.size() > size_t(params.penalty_last_n))
        throw std::invalid_argument("MiniCPM sampling history exceeds its window");
    if (!minicpm_sampler_)
        minicpm_sampler_ = std::make_unique<kernels::RadixSampler>(int(spec_.n_vocab));
    const size_t bytes = history.size() * sizeof(int);
    if (minicpm_sample_history_.bytes() < bytes)
        minicpm_sample_history_.reset(bytes);
    if (bytes) {
        cuda_check(cudaMemcpyAsync(minicpm_sample_history_.data(), history.data(), bytes,
                                   cudaMemcpyHostToDevice, stream_), "MiniCPM sampler history");
        metrics.control_h2d += bytes;
    }
    minicpm_sampler_->sample(selected_logits_,
                            bytes ? minicpm_sample_history_.as<int>() : nullptr,
                            int(history.size()), params, stream_);
    kernels::SamplePick pick;
    cuda_check(cudaMemcpyAsync(&pick, minicpm_sampler_->result(), sizeof(pick),
                               cudaMemcpyDeviceToHost, stream_), "MiniCPM sampled pick");
    metrics.head_d2h += sizeof(pick);
    cuda_check(cudaStreamSynchronize(stream_), "wait for MiniCPM sampled pick");
    ++metrics.explicit_syncs;
    if (pick.first_bad <= -2)
        throw std::runtime_error("MiniCPM sampler device continuation failed");
    if (pick.first_bad >= 0 || pick.id < 0 || pick.id >= int(spec_.n_vocab))
        throw std::runtime_error("invalid MiniCPM sampled head pick");
    return pick.id;
}
} // namespace strata::qwen35moe
