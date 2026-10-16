#pragma once
#include "strata/qwen35moe/spec.hpp"
#include <initializer_list>
namespace strata::qwen35moe::detail {
const TensorInfo* bind_tensor(const GgufFile&, const std::string&, std::initializer_list<uint64_t>);
LayerTensors bind_layer(const GgufFile&, const Spec&, uint64_t layer, bool recurrent);
}
