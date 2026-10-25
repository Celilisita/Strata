#pragma once
#include "strata/qwen35/completed_v100/native/mtp.hpp"
namespace strata::completed_v100::qwen35::native::mtp_abort {
// Opt-in service contract: any exception irrevocably discards this request.
// Call only after Graph drain/release, before any capture. No NativeMtp ABI change.
void select(NativeMtp&,cudaStream_t,bool enabled);
bool selected(const NativeMtp&,cudaStream_t);
void guard(const NativeMtp&,cudaStream_t);
void poison(const NativeMtp&) noexcept;
void clear(const NativeMtp&) noexcept;
// Bounded test hook: raises only after a real target execution, before commit.
void arm_fault(NativeMtp&);
bool take_fault(const NativeMtp&) noexcept;
}
