#include "strata/prefill/native_grouped_mmq.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "native_grouped_mmq_plan.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cstring>
#include <stdexcept>
namespace strata::prefill::mmq {
namespace {
constexpr size_t control_capacity = 256 * sizeof(NativeMmqMatrix) + 65536 * 8 + 1280 * sizeof(NativeMmqTile);
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Lease {
    uint8_t* bytes = nullptr;
    cudaEvent_t done{};
    bool used = false;
    Lease() {
        ck(cudaMallocHost(reinterpret_cast<void**>(&bytes), control_capacity));
        try { ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming)); }
        catch (...) { cudaFreeHost(bytes); throw; }
    }
    ~Lease() {
        if (used) cudaEventSynchronize(done);
        cudaEventDestroy(done); cudaFreeHost(bytes);
    }
};
}
struct NativeGroupedMmqProvider::Impl {
    cudaStream_t stream;
    std::array<Lease, 4> leases;
    size_t next = 0;
    explicit Impl(void* s) : stream(static_cast<cudaStream_t>(s)) {
        if (!stream) throw std::invalid_argument("grouped MMQ requires explicit stream");
    }
};
NativeGroupedMmqProvider::NativeGroupedMmqProvider(void* s) : impl_(std::make_unique<Impl>(s)) {}
NativeGroupedMmqProvider::~NativeGroupedMmqProvider() = default;
NativeGroupedMmqWork NativeGroupedMmqProvider::run(int type, const unsigned long long* gate, const unsigned long long* up,
    const int32_t* source_ids, int entries, int source_rows, const void* q8,
    void* controls, size_t controls_bytes, void* packed, size_t packed_bytes,
    float* output, int cols, int outputs, bool active) {
    auto& p = *impl_;
    const auto plan = native_mmq_plan(gate, up, source_ids, entries, source_rows, active);
    const size_t matrices_bytes = plan.matrices.size() * sizeof(NativeMmqMatrix);
    const size_t map_bytes = size_t(entries) * sizeof(int32_t);
    const size_t tile_bytes = plan.tiles.size() * sizeof(NativeMmqTile);
    const size_t control_bytes = matrices_bytes + 2 * map_bytes + tile_bytes;
    const size_t required = q8_bytes(entries, cols);
    if ((type != 12 && type != 14) || !controls || !packed || !q8 || !output || control_bytes > controls_bytes ||
        required > packed_bytes || cols < 512 || cols % 512 || outputs < 128 || outputs % 128)
        throw std::invalid_argument("unreserved bounded grouped MMQ workspace");
    auto& lease = p.leases[p.next++ % p.leases.size()];
    if (lease.used) {
        const auto ready = cudaEventQuery(lease.done);
        if (ready == cudaErrorNotReady) throw std::runtime_error("grouped MMQ lease requires caller drain");
        ck(ready);
    }
    std::memcpy(lease.bytes, plan.matrices.data(), matrices_bytes);
    std::memcpy(lease.bytes + matrices_bytes, plan.input_ids.data(), map_bytes);
    std::memcpy(lease.bytes + matrices_bytes + map_bytes, plan.output_ids.data(), map_bytes);
    if (tile_bytes) std::memcpy(lease.bytes + matrices_bytes + 2 * map_bytes, plan.tiles.data(), tile_bytes);
    auto* device = static_cast<uint8_t*>(controls);
    try {
        ck(cudaMemcpyAsync(device, lease.bytes, control_bytes, cudaMemcpyHostToDevice, p.stream));
        repack_native_q8_grouped(q8, packed, type, cols, source_rows, entries,
            reinterpret_cast<int32_t*>(device + matrices_bytes), p.stream);
        native_grouped_mmq(type, reinterpret_cast<NativeMmqMatrix*>(device), int(plan.matrices.size()), up ? 2 : 1,
            reinterpret_cast<int32_t*>(device + matrices_bytes + map_bytes), packed, output,
            entries, plan.max_rows, cols, outputs, p.stream,
            active ? reinterpret_cast<NativeMmqTile*>(device + matrices_bytes + 2 * map_bytes) : nullptr, int(plan.tiles.size()));
        ck(cudaEventRecord(lease.done, p.stream)); lease.used = true;
    } catch (...) {
        const auto e = cudaEventRecord(lease.done, p.stream); lease.used = e == cudaSuccess;
        if (e != cudaSuccess) cudaStreamSynchronize(p.stream);
        throw;
    }
    return {control_bytes, required, int(plan.matrices.size()), 3,
        active ? plan.active_count : ((plan.max_rows + 63) / 64) * int(plan.matrices.size()), plan.active_count}; // clear padding, repack, tiled product
}
} // namespace strata::prefill::mmq
