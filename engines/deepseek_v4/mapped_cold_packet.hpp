#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::deepseek_v4 {
// A single bounded job. GPU alone writes ready; CPU alone writes done. Only
// naturally aligned system-scope atomic loads/stores are used, never host RMW.
struct alignas(64) MappedColdPacket {
    alignas(64) uint32_t ready = 0;
    alignas(64) uint32_t done = 0;
    uint32_t error = 0, layer = 0, mask = 0;
    int32_t ids[6]{};
    float activation[4096]{};
    float contribution[6 * 4096]{}; // Compact ascending selected-rank order.
};
static_assert(std::is_standard_layout<MappedColdPacket>::value);
static_assert(offsetof(MappedColdPacket, ready) % 64 == 0);
static_assert(offsetof(MappedColdPacket, done) % 64 == 0);
static_assert(sizeof(MappedColdPacket) < 116 * 1024);

struct MappedColdStorage {
    MappedColdPacket* host = nullptr;
    MappedColdPacket* device = nullptr;
    uint32_t* state = nullptr; // Device-only fatal code, current mask and ticket.
    void* stream = nullptr;
    int device_id = -1;
    uint64_t timeout_ticks = 0;
};
void mapped_cold_create(ggml_backend_t, uint32_t timeout_ms, MappedColdStorage&);
void mapped_cold_destroy(MappedColdStorage&) noexcept;
void mapped_cold_synchronize(const MappedColdStorage&);
uint32_t mapped_cold_ready(const MappedColdPacket*);
void mapped_cold_done(MappedColdPacket*, uint32_t);
void mapped_cold_enqueue(const MappedColdStorage&, const ggml_tensor* norm,
                         const ggml_tensor* ids, ggml_tensor* down, uint32_t mask, uint32_t layer);
uint32_t mapped_cold_status(const MappedColdStorage&); // Diagnostic read outside timing.
}
