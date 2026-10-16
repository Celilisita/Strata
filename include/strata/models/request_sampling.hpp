#pragma once
#include "strata/kernels/radix_sampler.hpp"
#include <algorithm>
#include <istream>
#include <string>
#include <vector>

namespace strata::models {
struct RequestSampling {
    kernels::SamplerParams params;
    std::vector<int> history;
    bool configured = false;
    bool penalized() const {
        return params.penalty_repeat != 1 || params.penalty_freq != 0 || params.penalty_present != 0;
    }
    bool enabled() const { return configured && (params.temperature > 0 || penalized()); }
    void configure(std::istream& input, int context) {
        kernels::SamplerParams p; std::string extra;
        if (!(input >> p.temperature >> p.top_k >> p.top_p >> p.min_p >> p.seed >>
              p.penalty_repeat >> p.penalty_freq >> p.penalty_present >> p.penalty_last_n >> p.min_keep) || input >> extra)
            throw std::invalid_argument("invalid sampling parameters");
        kernels::RadixSampler::validate(p);
        if (p.penalty_last_n > context) throw std::invalid_argument("sampling history exceeds context");
        params = p; history.clear(); configured = true;
    }
    void prepare(const std::vector<int>& prompt) {
        params.counter = 0; history.clear();
        if (!penalized() || !params.penalty_last_n) return;
        const size_t n = (std::min)(prompt.size(), size_t(params.penalty_last_n));
        history.assign(prompt.end() - n, prompt.end());
    }
    void remember(int token) {
        if (!penalized() || !params.penalty_last_n) return;
        if (history.size() == size_t(params.penalty_last_n)) history.erase(history.begin());
        history.push_back(token);
    }
};
} // namespace strata::models
