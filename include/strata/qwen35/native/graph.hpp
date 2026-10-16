#pragma once
#include "strata/qwen35/native/ffn.hpp"

namespace strata::qwen35::native {
// One stable, independently owned FFN pipeline. Borrowed tensors and input
// must outlive this object; changing their contents does not change identity.
// Borrow an explicit stream that outlives the capture. The owning executor
// keeps its device current and serializes launches through this stream.
class CapturedFfn {
public:
    CapturedFfn(Ffn &, const float *input, int tokens, cudaStream_t);
    ~CapturedFfn();
    CapturedFfn(const CapturedFfn &) = delete;
    CapturedFfn &operator=(const CapturedFfn &) = delete;
    void launch();
    size_t nodes() const { return nodes_; }
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    cudaStream_t stream_ = nullptr;
    size_t nodes_ = 0;
    int device_ = -1;
};
}
