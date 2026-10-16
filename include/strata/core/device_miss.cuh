#pragma once
#include "strata/core/device_miss.hpp"
#include <cuda/atomic>
#include <cuda_runtime.h>

namespace strata::core {

// Called by one warp after canonical top-k. All-hit only reads/writes device memory.
__device__ inline void publish_native_miss(const int32_t* ids, const NativeMissArgs& args) {
    const int lane = threadIdx.x;
    auto& frame = *args.frame;
    int slot = -1;
    bool bad = false;
    if (lane < args.ranks) {
        const int id = ids[lane];
        bad = id < 0 || id >= args.n_expert;
        if (!bad) slot = args.slots[id];
    }
    const unsigned invalid = __ballot_sync(0xffffffffu, bad);
    const unsigned mask = __ballot_sync(0xffffffffu, lane < args.ranks && slot < 0);
    if (lane == 0) {
        if (invalid) frame.failed = 1;
        frame.mask = mask;
    }
    __syncwarp();
    const unsigned hits = __ballot_sync(0xffffffffu, lane < args.ranks && slot >= 0);
    const uint64_t base_tick = frame.cache_tick;
    if (lane < args.ranks) {
        const auto address = !frame.failed && slot >= 0
            ? (args.cache_slot_addresses ? args.cache_slot_addresses[slot]
                : reinterpret_cast<unsigned long long>(args.cache + uint64_t(slot) * args.cache_stride)) : 0;
        args.gu_addresses[lane] = address;
        args.gu_addresses[args.ranks + lane] = address ? address + args.gu_bytes : 0;
        args.down_addresses[lane] = address ? address + 2 * args.gu_bytes : 0;
        args.gu_rows[lane] = args.gu_rows[args.ranks + lane] = lane / (args.ranks / args.tokens);
        args.down_rows[lane] = lane;
        bool last = true;
        // Two rows may route to the same slot. Only its final occurrence writes
        // the age; the clock still advances for every hit in rank order.
        for (int later = args.tokens > 1 ? lane + 1 : args.ranks; later < args.ranks; ++later)
            if (slot >= 0 && ids[later] == ids[lane]) last = false;
        if (!frame.failed && slot >= 0 && last)
            args.ages[slot] = base_tick + __popc(hits & ((1u << lane) - 1u)) + 1;
    }
    __syncwarp();
    if (lane == 0 && !frame.failed) frame.cache_tick = base_tick + __popc(hits);
    if (frame.failed || !mask) return;
    using Atomic = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
    if (lane == 0) {
        if (Atomic(args.packet->active_epoch).load(cuda::memory_order_acquire) != frame.epoch ||
            frame.seq == 0xffffffffu) frame.failed = 2;
        else ++frame.seq;
    }
    __syncwarp();
    if (frame.failed) {
        if (lane < args.ranks) {
            args.gu_addresses[lane] = args.gu_addresses[args.ranks + lane] = 0;
            args.down_addresses[lane] = 0;
        }
        return;
    }
    for (int i = lane; i < args.n_embd * args.tokens; i += 32) args.norm_host_alias[i] = args.norm[i];
    if (lane < args.ranks) args.packet->ids[lane] = ids[lane];
    if (lane == 0) {
        args.packet->layer = uint32_t(args.layer);
        args.packet->mask = mask; args.packet->ranks = uint32_t(args.ranks);
    }
    // Every publishing lane fences its own mapped writes before the leader's release.
    __threadfence_system();
    __syncwarp();
    if (lane == 0) Atomic(args.packet->request_seq).store(frame.seq, cuda::memory_order_release);
}

// Called by thread0 of the existing combine CTA; no mapped read for all-hit.
__device__ inline bool await_native_miss(const NativeMissArgs& args) {
    auto& frame = *args.frame;
    if (frame.failed) return false;
    if (!frame.mask) return true;
    using Atomic = cuda::atomic_ref<uint32_t, cuda::thread_scope_system>;
    Atomic ack(args.packet->ack_seq);
    const unsigned long long before = clock64();
    unsigned polls = 0;
    bool ready = false;
    // A poll count is not elapsed time: a descheduled CPU worker may need the
    // full deadline. Keep the wait bounded by the configured device clock.
    while (clock64() - before < args.timeout_cycles) {
        ++polls;
        if (ack.load(cuda::memory_order_acquire) == frame.seq) { ready = true; break; }
    }
    frame.ack_polls += polls;
    if (!ready) frame.failed = 3;
    else if (args.packet->status) frame.failed = 4;
    return frame.failed == 0;
}

} // namespace strata::core
