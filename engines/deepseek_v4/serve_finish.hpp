#pragma once
#include <cstddef>
#include <cstdio>

namespace strata::deepseek_v4 {
// Resolve once when generation ends, before draining asynchronous work.
// A transport STOP after an emitted EOG/budget cannot revoke that completion.
inline const char* serve_finish_reason(bool eog,std::size_t emitted,std::size_t limit) {
    return eog?"stop":emitted>=limit?"limit":"cancel";
}
// One completion record per DONE; no prompt text or per-token diagnostics.
inline void report_serve_finish(const char* reason,std::size_t emitted,std::size_t prompt_tokens,
                                std::size_t prompt_read,double pp_ms,double tg_ms) {
    std::fprintf(stderr,"strata_serve_finish native_finish=%s emitted=%zu prompt_tokens=%zu prompt_read=%zu pp_ms=%.3f tg_ms=%.3f\n",
        reason,emitted,prompt_tokens,prompt_read,pp_ms,tg_ms);
    std::fflush(stderr);
}
} // namespace strata::deepseek_v4
