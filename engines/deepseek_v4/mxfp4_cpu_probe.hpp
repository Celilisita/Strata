#pragma once
#include "ggml.h"
#include <array>
#include <ostream>
struct llama_model;
namespace strata::deepseek_v4 {
bool mxfp4_cpu_operator_probe(const llama_model&, const std::array<ggml_tensor*, 3>&,
                              const float* actual_norm, std::ostream& report);
}
