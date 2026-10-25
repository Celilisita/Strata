#pragma once

namespace strata::prefill::mmq {
// One contract for the private DP4A host config and device templates.
inline constexpr int native_code_sum_tile_rows = 128;
inline constexpr int native_code_sum_threads = 256;
inline constexpr int native_code_sum_k = 256;
inline constexpr int native_code_sum_group_columns = 64;

// Read-only eligibility, before repacking or creating a grouped plan.
// Unsupported devices/shapes retain the caller's canonical provider.
bool native_code_sum_dense_supported(int cols, int outputs, int tokens);
bool native_code_sum_grouped_supported(int type, int cols, int outputs);
} // namespace strata::prefill::mmq
