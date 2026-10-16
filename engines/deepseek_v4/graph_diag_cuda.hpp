#pragma once
#include "ggml-cuda/common.cuh"
namespace strata::deepseek_v4 {
void graph_diag_compare(const ggml_cgraph*, const ggml_cuda_graph*);
}
