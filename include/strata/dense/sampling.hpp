#pragma once
#include <cstdint>
#include <random>
#include <vector>

namespace strata::dense {
struct Sampling {
    double temperature=0,top_p=1,repeat_penalty=1;
    int top_k=0,penalty_last_n=64;
    uint64_t seed=42;
    void validate() const;
};
class Sampler {
public:
    Sampler(Sampling p,const std::vector<int> &prompt):params_(p),history_(prompt),random_(p.seed){p.validate();}
    int next(const std::vector<float> &logits);
private:
    Sampling params_;
    std::vector<int> history_;
    std::mt19937_64 random_;
};
}
