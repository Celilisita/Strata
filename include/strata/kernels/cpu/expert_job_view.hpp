#pragma once
#include "strata/kernels/cpu/pool.hpp"
namespace strata::kernels::cpu {
inline void native_gu_job(const NativeFmt& format, const ExpertJobMulti& job,
                           float* const* ff, int first, int last) {
    if (job.gate) native_gu_rows_view(format, job.gate, job.up, job.nact, job.nt, ff, first, last, job.gate_scale, job.up_scale);
    else native_gu_rows(format, job.blob, job.nact, job.nt, ff, first, last);
}
inline void native_down_job(const NativeFmt& format, const ExpertJobMulti& job,
                             const void* const* hq, int first, int last) {
    if (job.down) native_down_rows_view(format, job.down, hq, job.nt, job.out, first, last, job.down_scale);
    else native_down_rows(format, job.blob, hq, job.nt, job.out, first, last);
}
}
