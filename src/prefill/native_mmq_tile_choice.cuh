#pragma once
#include "strata/prefill/mmq_compat.hpp"
#include <limits>
#include <stdexcept>
namespace strata::prefill::mmq {
inline void select_native_mmq_tile(mmq_args& args, int64_t columns) {
#ifdef STRATA_DEEPSEEK_V4_GGML_API
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool fallback = ggml_cuda_mmq_needs_fallback(args.nrows_x);
    int64_t best_tiles = (std::numeric_limits<int64_t>::max)();
    int best = 0;
    for (int j = 8; j <= 128 && best_tiles > 1; j += 8) {
        const auto c = ggml_cuda_mmq_get_config(args.type_x, j, fallback, dev.cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, dev.cc) > dev.smpbo) continue;
        const int64_t tiles = (columns + c.J - 1) / c.J;
        if (tiles < best_tiles) { best_tiles = tiles; best = j; }
    }
    if (!best) throw std::runtime_error("no legal Q8 MMQ tile fits selected device");
    args.J_best = best;
#else
    set_tile_hint(args, int(columns), 0);
#endif
}
} // namespace strata::prefill::mmq
