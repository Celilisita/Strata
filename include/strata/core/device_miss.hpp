#pragma once
#include <cstdint>

namespace strata::prefill::mmq { struct NativeNvfp4Matrix; struct NativeMmqTile; }

namespace strata::core {

// GPU writes all routed IDs, including hits, before the existing head sync.
// Host consumes only complete epoch/token records, never a partially published route.
struct NativeRouteRecord {
    uint32_t epoch=0, token=0, layer=0, mask=0;
    int32_t ids[8]{};
};
struct NativeRouteLedger { NativeRouteRecord layers[64]{}; };

// One ordered stream reuses this mapped packet only after its matching ACK.
// System-scope aligned atomic load/store accesses the three publication fields.
struct alignas(64) NativeMissPacket {
    alignas(4) uint32_t active_epoch = 0, request_seq = 0, ack_seq = 0;
    uint32_t layer = 0, mask = 0, ranks = 0, status = 0;
    int32_t ids[16]{};
};
struct NativeMissFrame {
    uint32_t epoch = 0, seq = 0, mask = 0, token = 0;
    int32_t failed = 0;
    uint64_t ack_polls = 0, cache_tick = 0;
};
struct NativeMissArgs {
    NativeMissPacket* packet = nullptr;
    float* norm_host_alias = nullptr;
    const volatile float* result_host_alias = nullptr;
    NativeMissFrame* frame = nullptr;
    const int32_t* slots = nullptr;
    const uint8_t* cache = nullptr;
    const uint64_t* cache_slot_addresses = nullptr;
    uint64_t* ages = nullptr;
    const float* norm = nullptr;
    unsigned long long* gu_addresses = nullptr;
    unsigned long long* down_addresses = nullptr;
    int32_t* gu_rows = nullptr;
    int32_t* down_rows = nullptr;
    uint64_t cache_stride = 0, gu_bytes = 0, timeout_cycles = 0;
    int32_t layer = 0, n_expert = 0, n_embd = 0, ranks = 0, tokens = 1;
    NativeRouteLedger* ledger = nullptr;
    // Optional typed NVFP4 controls use the existing independent expert_q8 scratch.
    prefill::mmq::NativeNvfp4Matrix *nv_gu=nullptr,*nv_down=nullptr;
    prefill::mmq::NativeMmqTile* nv_tiles=nullptr;
    int32_t *nv_sources=nullptr,*nv_outputs=nullptr;
    const float* nv_scales=nullptr; // immutable [layer,expert,gate/up/down]
};

} // namespace strata::core
