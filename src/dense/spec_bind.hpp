#pragma once

#include "strata/dense/spec.hpp"
#include <initializer_list>
#include <set>

namespace strata::dense::detail {
void validate_directory(const GgufFile &);
const TensorInfo *bind_tensor(const GgufFile &, const std::string &,
                             std::initializer_list<uint64_t>, std::set<std::string> &,
                             bool parameter = false);
LayerTensors bind_layer(const GgufFile &, const Spec &, uint64_t, bool recurrent,
                        std::set<std::string> &);
} // namespace strata::dense::detail
