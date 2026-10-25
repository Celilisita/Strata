#pragma once
#include "strata/gemma4/native/sm120_projection_plan.hpp"
#include "strata/qwen35/native/weights.hpp"
#include <memory>
#include <cstdint>
#include "strata/prefill/moe_mmq.hpp"
#include <unordered_map>
#include <vector>
#include <initializer_list>
#include <utility>

namespace strata::gemma4::native {
// One model-owned, stream-ordered projection service. No model math or KV state.
// GgufFile and external graphs must outlive this service's queued operations.
class Nvfp4ProjectionService {
    using Matrix=qwen35::native::Matrix;
    struct Entry {Matrix matrix;qwen35::native::Nvfp4View view;};
    cudaStream_t stream_;
    int maximum_batch_;
    qwen35::native::DeviceStorage storage_;
    qwen35::native::Nvfp4Input input_{};
    qwen35::native::Nvfp4Input low_{};
    void* canonical_values_=nullptr;
    bool tiny_compensated_=false;
    GemmaWeightFormat format_=GemmaWeightFormat::Nvfp4;
    std::unique_ptr<strata::prefill::mmq::Context> original_mmq_;
    void* original_input_=nullptr;
    void initialize_original(const GgufFile&,const Nvfp4BodyPlan&);
    void project_original(Matrix,const float*,float*,int);
    std::unordered_map<const TensorInfo*,Entry> tensors_;
    std::unordered_map<const void*,const Entry*> values_;
    std::unique_ptr<qwen35::native::Nvfp4Gemm> gemm_;
    std::vector<int> prepared_;
    unsigned long long calls_=0;
    unsigned long long tiny_calls_=0,canonical_materializations_=0;
public:
    Nvfp4ProjectionService(const GgufFile&,const Spec&,int maximum_batch,cudaStream_t);
    ~Nvfp4ProjectionService();
    Nvfp4ProjectionService(const Nvfp4ProjectionService&)=delete;
    Nvfp4ProjectionService& operator=(const Nvfp4ProjectionService&)=delete;
    Matrix matrix(const TensorInfo&)const;
    // Cold PREPARE only, after old graphs are released. Include actual PP tails.
    void prepare(const std::vector<int>& token_counts);
    void project(Matrix,const float*,float*,int tokens,cudaStream_t);
    // One explicit call-local codec lease; all outputs remain materialized.
    // No new state/layout; original-quant and other extents use project().
    void project_group(const float*,std::initializer_list<std::pair<Matrix,float*>>,int tokens,cudaStream_t);
    std::size_t bytes()const;
    cudaStream_t ordered_stream()const{return stream_;}
    unsigned long long calls()const{return calls_;}
    GemmaWeightFormat format()const{return format_;}
    bool tiny_compensated()const{return tiny_compensated_;}
    unsigned long long tiny_calls()const{return tiny_calls_;}
    unsigned long long canonical_materializations()const{return canonical_materializations_;}
};
}
