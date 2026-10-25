#pragma once
#include "strata/qwen35/completed_v100/native/gdn.hpp"

namespace strata::completed_v100::qwen35::native {
// Stable, caller-owned slots after the first and second verified inputs.
// The last token remains in the ordinary authoritative state/history.
struct GdnPrefix { float *state[2]{}, *history[2]{}; };
void gdn_recurrence_prefix(GdnInputs, GdnShape, float *output, float *final_state,
                           GdnPrefix, cudaStream_t);
void conv_prefix_history(const float *input, const float *history, GdnPrefix,
                         int channels, int tokens, cudaStream_t);
}
