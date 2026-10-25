#pragma once
#include "strata/qwen35/native/quant.hpp"
#include <cuda_runtime.h>
#include <memory>
#include <string>

namespace strata::qwen35 {struct Spec;}
namespace strata::qwen35::native {
class DenseGemm;
class DenseWeights;
struct Q8PpSgemmStats {
    size_t cache_bytes=0,workspace_bytes=0,scratch_bytes=0,resident_bytes=0;
    size_t cold_transient_bytes=0,free_before=0,free_after=0;
    uint64_t host_submissions=0,eager_shapes=0;
    double cold_ms=0;
};
// Auxiliary owned only by the existing NVFP4 Runtime. Original raw Q8 pointers
// are borrowed from the enclosing DenseWeights, which outlives that Runtime.
// The fixed stream and one shared recurrent operand triple remain live through
// every captured graph. The service drains/destroys graphs before this owner.
class Q8PpSgemm {
public:
    Q8PpSgemm(const strata::qwen35::Spec &,const DenseWeights &,int,cudaStream_t);
    ~Q8PpSgemm();
    Q8PpSgemm(const Q8PpSgemm&)=delete;
    Q8PpSgemm& operator=(const Q8PpSgemm&)=delete;
    void project(Matrix,Matrix,const float *,float *,float *,int,cudaStream_t);
    Q8PpSgemmStats stats()const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
bool q8_pp_sgemm_requested(); // Strict cold flag: absent/0 off,1 on.
void nvfp4_configure_q8_pp(DenseGemm &,const std::string &variant,
                         const strata::qwen35::Spec &,const DenseWeights &,cudaStream_t);
// Recurrent calls this only for T>=256. False preserves the original raw pair.
bool nvfp4_q8_pp_pair(DenseGemm &,Matrix,Matrix,const float *,float *,float *,int,cudaStream_t);
Q8PpSgemmStats nvfp4_q8_pp_stats(DenseGemm &);
}
