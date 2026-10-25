#pragma once
#include "strata/qwen35/native/nvfp4.hpp"
#include <cstdint>
namespace strata::qwen35::native {
// PP-only FP4 operands, FP32 accumulate/alpha, FP16 result storage.
class Nvfp4HalfGemm {
public:
    explicit Nvfp4HalfGemm(cudaStream_t,size_t workspace=32u*1024*1024);
    ~Nvfp4HalfGemm();
    Nvfp4HalfGemm(const Nvfp4HalfGemm &)=delete;
    Nvfp4HalfGemm &operator=(const Nvfp4HalfGemm &)=delete;
    void project(Nvfp4View,Nvfp4Input,void *half_output,int tokens,cudaStream_t);
    size_t bytes()const;
    // Admission changes only after the request's graphs have been released.
    void use_measured_tactic(bool);
    uint64_t alternate_submissions()const;
private:
    struct Impl;std::unique_ptr<Impl> impl_;
};
}
