#include "cache_leaf_aliases.hpp"

#include "llama-context.h"
#include "llama-moe-cache.h"
#include <algorithm>
#include <stdexcept>

namespace strata::deepseek_v4 {
void validate_cache_leaf_alias(const ggml_tensor* tensor) {
    if (!tensor || tensor->op != GGML_OP_VIEW || !tensor->view_src ||
        tensor->src[0] != tensor->view_src || tensor->view_offs != 0 ||
        !tensor->buffer || tensor->buffer != tensor->view_src->buffer ||
        !tensor->data || tensor->data != tensor->view_src->data ||
        ggml_backend_buffer_get_usage(tensor->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
        tensor->ne[0] != tensor->view_src->ne[0] ||
        tensor->ne[1] != tensor->view_src->ne[1] ||
        tensor->ne[2] <= 0 || tensor->ne[2] >= tensor->view_src->ne[2] ||
        tensor->ne[3] != 1 || !ggml_is_contiguous(tensor))
        throw std::runtime_error("canonical cache leaf alias ownership/shape differs");
    for (int i = 1; i < GGML_MAX_SRC; ++i)
        if (tensor->src[i]) throw std::runtime_error("cache alias has extra graph dependencies");
}

void make_cache_leaf_alias(ggml_tensor* tensor) noexcept {
    // Persistent preallocated cache banks outlive every graph. Keep view_src,
    // view_offs, strides, data and buffer intact; remove only graph operations.
    tensor->op = GGML_OP_NONE;
    std::fill_n(tensor->src, GGML_MAX_SRC, nullptr);
}

void configure_cache_leaf_aliases(llama_context* context) {
    if (!context) throw std::runtime_error("cache leaf aliases need an initialized context");
    context->strata_set_cache_leaf_aliases();
}
}

void llama_context::strata_set_cache_leaf_aliases() {
    if (!moe_cache || gf_res_prev[0] || gf_res_prev[1])
        throw std::runtime_error("cache leaf aliases must be configured before the first decode");
    synchronize();
    moe_cache->strata_set_leaf_aliases();
}
