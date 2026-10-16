#pragma once
#include "ggml-backend.h"
#include <array>
#include <ostream>
#include <string>
namespace strata::deepseek_v4 {
bool native_tg_operator_probe(const std::string& model_path, ggml_backend_t reference_backend,
    const std::array<ggml_tensor*,3>& real_layer_zero, const std::array<ggml_tensor*,3>& registered_source,
    const float* actual_norm, float clamp, std::ostream&);
}
