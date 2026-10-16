#include "strata/qwen35moe/cuda_kernels.hpp"

#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>

namespace strata::qwen35moe {
namespace {
__global__ void gather(const int32_t* ids, const int32_t* slots,
                       const uint8_t* cache, uint64_t stride, uint64_t gu_bytes,
                       unsigned long long* gu_addresses, int32_t* gu_rows,
                       unsigned long long* down_addresses, int32_t* down_rows, const uint64_t* slot_addresses) {
    const int entry = threadIdx.x;
    if (entry >= 8) return;
    const int slot = slots[ids[entry]];
    const auto* gate = slot_addresses ? reinterpret_cast<const uint8_t*>(slot_addresses[slot])
                                      : cache + uint64_t(slot) * stride;
    gu_addresses[entry] = reinterpret_cast<unsigned long long>(gate);
    gu_addresses[8 + entry] = reinterpret_cast<unsigned long long>(gate + gu_bytes);
    gu_rows[entry] = gu_rows[8 + entry] = 0;
    down_addresses[entry] = reinterpret_cast<unsigned long long>(gate + 2 * gu_bytes);
    down_rows[entry] = entry;
}
}

void gather_resident_experts(const int32_t* ids, const int32_t* slots,
                             const void* cache, uint64_t stride, uint64_t gu_bytes,
                             unsigned long long* gu_addresses, int32_t* gu_rows,
                             unsigned long long* down_addresses, int32_t* down_rows,
                             void* stream, const uint64_t* slot_addresses) {
    if (!ids || !slots || (!cache && !slot_addresses) || !gu_addresses || !gu_rows ||
        !down_addresses || !down_rows || !stream || !stride || !gu_bytes)
        throw std::invalid_argument("invalid resident Qwen35 expert gather");
    gather<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        ids, slots, static_cast<const uint8_t*>(cache), stride, gu_bytes,
        gu_addresses, gu_rows, down_addresses, down_rows, slot_addresses);
    const auto status = cudaGetLastError();
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
} // namespace strata::qwen35moe
