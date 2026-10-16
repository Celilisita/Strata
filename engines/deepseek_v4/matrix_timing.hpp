#pragma once
#include "ggml-backend.h"
#include <array>

namespace strata::deepseek_v4 {
// Includes the canonical activation quantizer and matrix operation. Inputs and
// weights are already on device. Three warmups; three32-replay CUDA-event windows.
// Weights may remain in L2: this is an isolated service gate, not an HBM curve.
std::array<double, 3> measure_matrix_graph(ggml_backend_t, ggml_cgraph*, bool l2_cold = false);
}
