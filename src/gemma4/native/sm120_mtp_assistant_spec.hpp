#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <array>
namespace strata::gemma4::native::mtp_assistant {
struct LayerSpec {
    int dimension=0,kv_heads=0;
    const TensorInfo *q=nullptr,*out=nullptr,*q_norm=nullptr;
    const TensorInfo *pre=nullptr,*post=nullptr,*gate=nullptr,*up=nullptr,*down=nullptr;
    const TensorInfo *ffn_pre=nullptr,*ffn_post=nullptr,*scale=nullptr;
};
struct Spec {
    int hidden=1024,intermediate=8192,main_hidden=5376,heads=32,vocabulary=262144;
    int context=0,window=1024;
    float epsilon=0,local_rope=0,global_rope=0;
    const TensorInfo *head=nullptr,*pre_projection=nullptr,*post_projection=nullptr;
    const TensorInfo *output_norm=nullptr,*frequencies=nullptr;
    std::array<LayerSpec,4> layers;
};
// Exact inventory for the existing 31B assistant GGUF, not a main-model binder.
// Tensor descriptors borrow the GgufFile. GPU Draft copies their payload once.
Spec bind_spec(const GgufFile&);
size_t tensor_payload_bytes(const GgufFile&,const TensorInfo&);
}
