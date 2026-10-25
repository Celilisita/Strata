#pragma once
#include "strata/qwen35/completed_v100/spec.hpp"
#include <initializer_list>
namespace strata::completed_v100::qwen35::detail {
const TensorInfo *bind_tensor(const GgufFile &gguf, const std::string &name,
                              std::initializer_list<uint64_t> shape);
LayerTensors bind_layer(const GgufFile &gguf, const Spec &spec, uint64_t index,
                       bool recurrent);
} // namespace strata::completed_v100::qwen35::detail
