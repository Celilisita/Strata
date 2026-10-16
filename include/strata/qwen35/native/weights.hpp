#pragma once
#include "strata/qwen35/native/attention_layer.hpp"
#include "strata/qwen35/spec.hpp"

namespace strata::qwen35::native {
// Setup ownership for one Dense inventory. Execution only borrows its pointers.
class DeviceStorage {
public:
    ~DeviceStorage();
    static size_t reserve_bytes();
    DeviceStorage() = default;
    DeviceStorage(const DeviceStorage &) = delete;
    DeviceStorage &operator=(const DeviceStorage &) = delete;
    void *allocate(size_t);
    void release(void *);
    size_t bytes() const { return bytes_; }
private:
    std::vector<std::pair<void *,size_t>> allocations_;
    size_t bytes_=0;
};
class DenseWeights {
public:
    DenseWeights(const strata::GgufFile &,const strata::qwen35::Spec &,cudaStream_t,bool tiled_ffn=false);
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
