#include "strata/prefill/native_tc_provider.hpp"
#include "strata/kernels/native_tc_grouped.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace strata::prefill {
namespace {
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(cudaMalloc(&p, bytes)); }
    ~Device() { cudaFree(p); }
    template<class T> T* as() const { return static_cast<T*>(p); }
};
struct Lease {
    kernels::NativeTcTile* tiles = nullptr;
    cudaEvent_t done{};
    bool used = false;
    Lease() {
        ck(cudaMallocHost(reinterpret_cast<void**>(&tiles), 8192 * sizeof(*tiles)));
        try {
            ck(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        } catch (...) { cudaFreeHost(tiles); throw; }
    }
    ~Lease() {
        if (used) cudaEventSynchronize(done);
        cudaEventDestroy(done); cudaFreeHost(tiles);
    }
};
}
struct NativeTcProvider::Impl {
    cudaStream_t stream;
    Device tiles{8192 * sizeof(kernels::NativeTcTile)};
    std::array<Lease, 4> leases;
    size_t next = 0;
    explicit Impl(void* s) : stream(static_cast<cudaStream_t>(s)) {
        if (!stream) throw std::invalid_argument("TC provider needs an explicit stream");
        kernels::native_tc_configure();
    }
};
NativeTcProvider::NativeTcProvider(void* stream) : impl_(std::make_unique<Impl>(stream)) {}
NativeTcProvider::~NativeTcProvider() = default;

NativeTcWork NativeTcProvider::run(int type, const unsigned long long* addresses,
    const int32_t* rows, int entries, const void* q8, float* output, int n_in, int n_out,
    bool gathered_mmvq, int gathered_rows) {
    if ((gathered_rows != 1 && (!gathered_mmvq || type != 12 || gathered_rows != 2)) ||
        (type != 12 && type != 14) || entries <= 0 || entries > 65536 ||
        n_in <= 0 || n_in % 256 || n_out <= 0 || n_out % 8 ||
        n_in > (type == 12 ? 2048 : 1024) || !addresses || !rows || !q8 || !output)
        throw std::invalid_argument("invalid bounded native TC product");
    auto& p = *impl_;
    auto& lease = p.leases[p.next++ % p.leases.size()];
    if (lease.used) {
        const auto ready = cudaEventQuery(lease.done);
        if (ready == cudaErrorNotReady) throw std::runtime_error("TC control leases need caller drain");
        ck(ready);
    }
    const auto plan = kernels::native_tc_plan(addresses, rows, entries);
    if (plan.empty() || plan.size() > 8192 || plan.back().matrix_slot >= 512)
        throw std::invalid_argument("TC descriptor capacity exceeded");
    std::memcpy(lease.tiles, plan.data(), plan.size() * sizeof(plan[0]));
    NativeTcWork work;
    try {
        const size_t bytes = plan.size() * sizeof(plan[0]);
        ck(cudaMemcpyAsync(p.tiles.p, lease.tiles, bytes, cudaMemcpyHostToDevice, p.stream));
        if (gathered_mmvq)
            kernels::native_mmvq_gathered(type, p.tiles.as<kernels::NativeTcTile>(), int(plan.size()),
                q8, output, n_in, n_out, p.stream, gathered_rows);
        else
            kernels::native_tc_grouped(type, p.tiles.as<kernels::NativeTcTile>(), int(plan.size()),
                q8, output, n_in, n_out, p.stream);
        work.control_h2d = bytes; work.launches = 1;
        ck(cudaEventRecord(lease.done, p.stream)); lease.used = true;
    } catch (...) {
        const auto e = cudaEventRecord(lease.done, p.stream);
        lease.used = e == cudaSuccess;
        if (e != cudaSuccess) cudaStreamSynchronize(p.stream);
        throw;
    }
    return work;
}
} // namespace strata::prefill
