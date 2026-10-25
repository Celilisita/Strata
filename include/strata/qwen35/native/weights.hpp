#pragma once
#include "strata/qwen35/native/attention_layer.hpp"
#include "strata/qwen35/spec.hpp"
#include "strata/qwen35/native/nvfp4.hpp"
#include "strata/qwen35/native/cold_head_binding.hpp"
#include <array>

namespace strata::qwen35::native {
// Strict derived Q35 format gate; Spec binds the actual GGUF tensor identities.
void validate_qwen35_nvfp4_asset(const strata::qwen35::Spec &);
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
    DenseWeights(const strata::GgufFile &,const strata::qwen35::Spec &,cudaStream_t,bool tiled_ffn,ColdHeadBinding);
    bool tiled_ffn() const { return tiled_ffn_; }
    // Cold setup before make_nvfp4_runtime; no post-construction inventory append.
    void prepare_mtp_nvfp4(const strata::GgufFile &,const strata::qwen35::Spec &,cudaStream_t);
    Matrix borrowed_mtp_nvfp4(const strata::TensorInfo &,cudaStream_t) const;
    Matrix embedding,head;
    std::vector<Nvfp4View> fp4_weights;
    const float *output_norm=nullptr;
    std::vector<RecurrentWeights> recurrent;
    std::vector<AttentionWeights> attention;
    size_t bytes() const { return storage_.bytes(); }
private:
    DeviceStorage storage_;
    bool tiled_ffn_=false;
    std::array<const strata::TensorInfo *,8> mtp_fp4_tensors_{};
    std::array<Matrix,8> mtp_fp4_matrices_{};
    cudaStream_t mtp_fp4_stream_=nullptr;
    bool mtp_fp4_ready_=false,mtp_fp4_failed_=false;
};
}
