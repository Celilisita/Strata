#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

namespace strata::qwen35moe {
namespace {
struct HostMemory { uint64_t total = 0, available = 0; };

HostMemory host_memory() {
#ifdef _WIN32
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (GlobalMemoryStatusEx(&status))
        return {status.ullTotalPhys, status.ullAvailPhys};
    return {};
#else
    std::ifstream input("/proc/meminfo");
    std::string name, unit;
    uint64_t kib = 0;
    HostMemory memory;
    while (input >> name >> kib >> unit) {
        if (name == "MemTotal:") memory.total = kib * 1024;
        if (name == "MemAvailable:") memory.available = kib * 1024;
    }
    return memory;
#endif
}
}

size_t Runtime::pinned_expert_layers() const {
    size_t count = 0;
    for (const auto& layer : pinned_experts_.layers)
        if (layer.data) ++count;
    return count;
}

void Runtime::pin_expert_weights(int count) {
    if (count < 0 || count > int(expert_sources_.size()))
        throw std::invalid_argument("invalid pinned expert layer count");
    auto& pinned = pinned_experts_;
    pinned.layers.resize(expert_sources_.size());
    const auto memory = host_memory();
    const char* reserve_override = std::getenv("STRATA_QWEN35_HOST_RESERVE_GIB");
    const uint64_t reserve = reserve_override
        ? std::stoull(reserve_override) << 30
        : (std::max)(uint64_t(8) << 30, memory.total / 4);
    for (int il = 0; il < count; ++il) {
        const auto& source = expert_sources_[size_t(il)];
        const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
        const size_t down = kernels::native_mmvq_weight_bytes(
            int(source.down_exps->type), 512, 2048);
        const size_t stride = (2 * gu + down + 255) & ~size_t(255);
        const size_t bytes = size_t(256) * stride;
        auto& target = pinned.layers[size_t(il)];
        if (host_memory().available < bytes + reserve) {
            pinned.error = "host_memory_reserve";
            break;
        }
        void* buffer = nullptr;
        const auto status = cudaMallocHost(&buffer, bytes);
        if (status != cudaSuccess) {
            pinned.error = cudaGetErrorName(status);
            cudaGetLastError();
            break;
        }
        target.data = static_cast<uint8_t*>(buffer);
        target.stride = stride;
        const auto* gate = gguf_.tensor_data(*source.gate_exps);
        const auto* up = gguf_.tensor_data(*source.up_exps);
        const auto* down_data = gguf_.tensor_data(*source.down_exps);
        for (int id = 0; id < 256; ++id) {
            auto* slot = target.data + size_t(id) * stride;
            std::memcpy(slot, gate + size_t(id) * gu, gu);
            std::memcpy(slot + gu, up + size_t(id) * gu, gu);
            std::memcpy(slot + 2 * gu, down_data + size_t(id) * down, down);
        }
        pinned.bytes += bytes;
    }
}

} // namespace strata::qwen35moe
