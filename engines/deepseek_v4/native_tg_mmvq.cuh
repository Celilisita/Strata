#pragma once
// -2 is private native TG's unpublished cold slot. Copy-stream publication
// follows real weight payload completion. -1 is a CPU rank and must return
// before any matrix pointer arithmetic, weight prefetch or dot product.
static __device__ uint32_t strata_native_tg_wait_slot(const int32_t* ids, uint32_t rank, uint32_t slot) {
    if (slot != uint32_t(-2)) return slot;
    unsigned long long start, current;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    const auto* live = reinterpret_cast<const volatile int32_t*>(ids);
    while ((slot = uint32_t(live[rank])) == uint32_t(-2)) {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(current));
        if (current - start >= 1500000000ull) {
            atomicCAS(const_cast<int32_t*>(ids) + rank, -2, -3);
            return uint32_t(-3);
        }
        __nanosleep(1000);
    }
    __threadfence_system();
    return slot;
}
