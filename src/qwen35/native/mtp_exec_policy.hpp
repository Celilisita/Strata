#pragma once
#include <cstddef>
namespace strata::qwen35::native {
// Fixed-size host diagnostics. Reset once when a request records its graphs.
// GPU replay never retains the address of these thread-local counters.
struct MtpExecStats {
    size_t recorded=0,instantiated=0,deferred=0,lazy_instantiated=0,upload_syncs=0;
    double record_ms=0,instantiate_ms=0,lazy_ms=0;
};
MtpExecStats &mtp_exec_stats();
struct MtpRewindDiagnostic {bool quiet=false;size_t checked_fields=0;};
MtpRewindDiagnostic &mtp_rewind_diagnostic();
bool defer_mtp_exec(bool gpu_chain,bool retained_prefix,int capacity);
}
