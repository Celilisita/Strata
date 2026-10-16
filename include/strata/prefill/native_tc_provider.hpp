#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>

namespace strata::prefill {
struct NativeTcWork { uint64_t control_h2d = 0, launches = 0; };
// Grouped integer-code provider, fed by the caller's authoritative
// directory. No packed-weight cache. <512KiB device workspaces, four pinned
// control leases. At most four queued calls before the caller's natural drain.
class NativeTcProvider {
public:
    explicit NativeTcProvider(void* stream);
    ~NativeTcProvider();
    NativeTcProvider(const NativeTcProvider&) = delete;
    NativeTcProvider& operator=(const NativeTcProvider&) = delete;
    NativeTcWork run(int type, const unsigned long long* addresses, const int32_t* rows,
        int entries, const void* q8, float* output, int n_in, int n_out,
        bool gathered_mmvq = false, int gathered_rows = 1);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace strata::prefill
