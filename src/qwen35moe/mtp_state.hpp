#pragma once
#include "strata/qwen35moe/runtime.hpp"
namespace strata::qwen35moe {
// Weight views participate in Runtime's sole expert inventory; no private expert cache.
struct MtpState {
    LayerWeights decoder;
    GpuWeight eh_proj, enorm, hnorm, output_norm, output;
    DeviceBuffer concat, main_norm, seed, embedding_row, verify_logits, picks;
    std::vector<DeviceBuffer> prefix_state, prefix_history;
    size_t embedding_row_bytes = 0;
};
}
