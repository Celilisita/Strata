#pragma once
#include "strata/runtime/capacity_storage.hpp"
#include "strata/qwen35/completed_v100/native/attention_layer.hpp"
#include "strata/qwen35/completed_v100/spec.hpp"

namespace strata::completed_v100::qwen35::native {
// Setup ownership for one Dense inventory. Execution only borrows its pointers.
using DeviceStorage = strata::runtime::CapacityStorage;
class DenseWeights {
public:
    DenseWeights(const strata::GgufFile &,const strata::completed_v100::qwen35::Spec &,cudaStream_t,bool tiled_ffn=false);
    bool tiled_ffn() const { return tiled_ffn_; }
    Matrix embedding,head;
    const float *output_norm=nullptr;
    std::vector<RecurrentWeights> recurrent;
    std::vector<AttentionWeights> attention;
    size_t bytes() const { return storage_.bytes(); }
private:
    DeviceStorage storage_;
    bool tiled_ffn_=false;
};
}
