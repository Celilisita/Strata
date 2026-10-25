#pragma once
#include "../deepseek_v4/serve_sampling.hpp"

namespace strata::glm5_next {
// HTTP MTP shares the ordinary service's exact policy, RNG and causal history.
// Greedy proposals never call pick() or commit(); only emitted target tokens do.
class ServeMtpSampling {
    const deepseek_v4::ServeSampling& policy_;
    deepseek_v4::ServeHistory history_;
    std::mt19937_64 rng_;
    uint64_t draws_=0;
public:
    ServeMtpSampling(const deepseek_v4::ServeSampling& policy,const std::vector<int32_t>& prompt)
        :policy_(policy),history_(policy,prompt),rng_(deepseek_v4::serve_rng(policy)){}
    template<class IsEog> int32_t pick(const float* logits,int32_t count,IsEog is_eog) {
        const auto token=deepseek_v4::serve_sample(logits,count,policy_,history_,rng_,is_eog);
        ++draws_;return token;
    }
    void commit(int32_t token){history_.remember(token);}
    uint64_t draws()const{return draws_;}
};
} // namespace strata::glm5_next
