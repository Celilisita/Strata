#pragma once
#include "attention_repeat.hpp"
#include "common.cuh"
#include "fattn.cuh"
namespace strata::deepseek_v4 {
inline void attention_repeat_dispatch(ggml_backend_cuda_context& context, ggml_tensor* tensor) {
    const auto stream = reinterpret_cast<uintptr_t>(context.stream());
    if (!attention_repeat_capture(tensor, stream)) return;
    for (int sample = 1; sample <= attention_repeat_count; ++sample) {
        ggml_cuda_flash_attn_ext(context, tensor);
        attention_repeat_output(tensor, sample, stream);
    }
}
}
