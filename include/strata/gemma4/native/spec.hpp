#pragma once
#include "strata/artifact/gguf_reader.hpp"
#include <string>
#include <vector>
namespace strata::gemma4::native {
struct Layer {
    bool sliding=false,shared_raw_kv=false;
    int head_dim=0,kv_heads=0;
    const TensorInfo *q=nullptr,*k=nullptr,*v=nullptr,*out=nullptr;
    const TensorInfo *q_norm=nullptr,*k_norm=nullptr,*attn_norm=nullptr,*attn_post=nullptr;
    const TensorInfo *gate=nullptr,*up=nullptr,*down=nullptr,*ffn_norm=nullptr,*ffn_post=nullptr,*scale=nullptr;
};
struct Spec {
    int hidden=0,intermediate=0,vocabulary=0,heads=0,context=0,window=0;
    float epsilon=0,logit_cap=0,local_rope=0,global_rope=0;
    const TensorInfo *embedding=nullptr,*output=nullptr,*output_norm=nullptr,*rope_frequencies=nullptr;
    std::vector<Layer> layers;
};
// Only the independently owned Dense31B GGUF architecture is admitted here.
// Tensor pointers borrow the immutable GgufFile and never own CPU inference.
Spec bind_spec(const GgufFile&);
}
