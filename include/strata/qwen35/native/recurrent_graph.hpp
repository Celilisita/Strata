#pragma once
#include "strata/qwen35/native/recurrent.hpp"

namespace strata::qwen35::native {
// Borrow the layer, all tensors and an explicit stream through the final
// replay completion. The executor maintains the current device on replay.
class CapturedRecurrent {
public:
    CapturedRecurrent(RecurrentLayer &, RecurrentInputs, RecurrentStateOutputs, int tokens, cudaStream_t);
    ~CapturedRecurrent();
    CapturedRecurrent(const CapturedRecurrent &) = delete;
    CapturedRecurrent &operator=(const CapturedRecurrent &) = delete;
    void launch();
    size_t nodes() const { return nodes_; }
private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t executable_ = nullptr;
    cudaStream_t stream_ = nullptr;
    size_t nodes_ = 0;
};
}
