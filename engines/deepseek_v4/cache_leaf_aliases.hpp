#pragma once

struct ggml_tensor;
struct llama_context;

namespace strata::deepseek_v4 {
// The original cache owns the allocation and slot directory. These headers
// retain their original buffer/view ownership, but become graph leaf weights.
void validate_cache_leaf_alias(const ggml_tensor* tensor);
void make_cache_leaf_alias(ggml_tensor* tensor) noexcept;
void configure_cache_leaf_aliases(llama_context* context);
}
