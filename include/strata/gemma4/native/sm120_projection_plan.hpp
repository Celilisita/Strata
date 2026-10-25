#pragma once
#include "strata/gemma4/native/spec.hpp"
#include <cstddef>
#include <string>
#include <vector>

namespace strata::gemma4::native {
// Separate opt-in binder: original raw-GGUF bind_spec remains unchanged.
Spec bind_sm120_spec(const GgufFile&);
enum class GemmaWeightFormat { Nvfp4, OriginalQuant };
// Cold explicit format policy; absent retains the historical NVFP4 variant.
GemmaWeightFormat gemma_weight_format_requested();
const char* gemma_weight_format_name(GemmaWeightFormat);
const char* gemma_weight_variant(GemmaWeightFormat);
enum class ProjectionRole { Query, Key, Value, AttentionOutput, Gate, Up, Down };
struct Nvfp4BodyTensor {
    const TensorInfo* tensor=nullptr;
    ProjectionRole role{};
    int layer=-1;
};
struct Nvfp4BodyPlan {
    GemmaWeightFormat format=GemmaWeightFormat::Nvfp4;
    std::vector<Nvfp4BodyTensor> tensors;
    int maximum_rows=0,maximum_columns=0;
    std::size_t gguf_bytes=0;
};
// Complete 410-matrix body only. Embedding and tied head retain raw quantization.
// Global V borrows its layer's raw K projection; it has no inventory entry.
Nvfp4BodyPlan plan_nvfp4_body(const GgufFile&,const Spec&);
enum class Sm120Attention { PackedQ8Split, Float32Tiled };
// Architecture-independent Gemma cores; no Sm70 vendor attention admission.
Sm120Attention sm120_attention_choice(int tokens,int past,int reserved,int batch);
}
