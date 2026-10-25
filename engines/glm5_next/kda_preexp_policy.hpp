#pragma once
#include "ggml-cuda/gated_delta_net_preexp.h"
#include <cstdlib>
#include <cstring>

namespace strata::glm5_next {
// A process chooses this mode before any model graph is built. It must not
// change it while a context may reuse its existing graph. Fixture graphs set
// their own opt-in bit explicitly, without mutating this environment.
inline unsigned kda_preexp_mask() {
    static const unsigned mask = [] {
        const char * value = std::getenv("STRATA_GLM53_KDA_PREEXP_MASK");
        if (!value || !value[0]) return 0u;
        unsigned parsed = 0;
        for (int i = 0; value[i]; ++i) {
            if (i >= 2 || value[i] < '0' || value[i] > '9') return 0u;
            parsed = parsed * 10 + unsigned(value[i] - '0');
        }
        return parsed <= 15 ? parsed : 0u;
    }();
    return mask;
}

inline void apply_kda_preexp_policy(ggml_tensor * node, bool glm5_next) {
    if (!glm5_next || !node || node->op != GGML_OP_GATED_DELTA_NET) return;
    const unsigned mask = kda_preexp_mask();
    const int64_t tokens = node->src[2]->ne[2];
    // The backend independently checks the exact F32 KDA shape and K3 state
    // contract. Other operators and unsupported shapes keep the original path.
    // Retain each measured cost domain independently, without extrapolating
    // an endpoint result to other token counts in this first slice.
    const unsigned bit = tokens == 1 ? 1 : tokens == 2 ? 2 :
                         tokens == 128 ? 4 : tokens == 8192 ? 8 : 0;
    if (mask & bit) {
        ggml_cuda_gdn_set_preexp_requested(node, true);
    }
}
} // namespace strata::glm5_next
