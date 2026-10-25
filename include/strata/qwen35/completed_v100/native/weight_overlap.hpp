#pragma once
#include "strata/qwen35/completed_v100/native/quant.hpp"
#include <cuda_runtime.h>
namespace strata::completed_v100::qwen35::native {
// One transient decoded operand. The preceding main-stream fork guarantees
// the previous main consumer has finished before the preparation stream writes.
class PpWeightOverlap {
public:
    PpWeightOverlap(size_t bytes,cudaStream_t main);
    ~PpWeightOverlap();
    PpWeightOverlap(const PpWeightOverlap &)=delete;
    PpWeightOverlap &operator=(const PpWeightOverlap &)=delete;
    void begin(Matrix,cudaStream_t);
    const void *ready(cudaStream_t);
    size_t bytes()const{return bytes_;}
private:
    void release()noexcept;
    void *data_=nullptr;size_t bytes_=0;
    cudaStream_t main_=nullptr,prepare_=nullptr;
    cudaEvent_t fork_=nullptr,done_=nullptr;int device_=-1;
};
}
