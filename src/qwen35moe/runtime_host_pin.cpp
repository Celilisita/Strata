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
        const size_t gu = nvfp4_?Nvfp4ExpertInventory::matrix_bytes:kernels::native_mmvq_weight_bytes(12, 2048, 512);
        const size_t down = nvfp4_?Nvfp4ExpertInventory::matrix_bytes:kernels::native_mmvq_weight_bytes(
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

void Runtime::prepare_host_readonly() {
    const char* setting=std::getenv("STRATA_QWEN35_PP_REGISTER_MIB");
    if(!setting || !std::strcmp(setting,"0")) return;
    size_t used=0;const auto mib=std::stoull(setting,&used);
    if(setting[used] || !mib || mib>16384) throw std::invalid_argument("GGUF page lease limit0..16384MiB");
    if(pinned_experts_.bytes)
        throw std::invalid_argument("GGUF page lease conflicts with copied/per-layer pinning");
    const size_t bound=size_t(mib)<<20;
    std::vector<HostReadOnlyLease::Span> spans;
    size_t planned=0;
    const size_t experts=size_t(spec_.n_expert);
    if (!experts) throw std::invalid_argument("empty native expert inventory");
    // First full warm layers need no PP H2D, so leave their original pages pageable.
    for(size_t il=expert_cache_.slots/experts;il<expert_sources_.size();++il) {
        const auto& source=expert_sources_[il];
        const size_t gu=nvfp4_?Nvfp4ExpertInventory::matrix_bytes:kernels::native_mmvq_weight_bytes(int(source.gate_exps->type),int(spec_.n_embd),int(spec_.n_ff_exp));
        const size_t down=nvfp4_?Nvfp4ExpertInventory::matrix_bytes:kernels::native_mmvq_weight_bytes(int(source.down_exps->type),int(spec_.n_ff_exp),int(spec_.n_embd));
        const uint8_t* data[]{gguf_.tensor_data(*source.gate_exps),gguf_.tensor_data(*source.up_exps),gguf_.tensor_data(*source.down_exps)};
        const size_t sizes[]{gu*experts,gu*experts,down*experts};
        HostReadOnlyLease::Span layer[3];size_t layer_bytes=0;
        for(int i=0;i<3;++i) {
            const auto begin=reinterpret_cast<uintptr_t>(data[i]);
            layer[i]={begin&~uintptr_t(4095),(begin+sizes[i]+4095)&~uintptr_t(4095)};
            layer_bytes+=layer[i].second-layer[i].first;
        }
        if(planned+layer_bytes>bound) break;
        spans.insert(spans.end(),std::begin(layer),std::end(layer));planned+=layer_bytes;
    }
    if (spans.empty()) return; // The GPU already holds every useful source layer.
    host_readonly_.prepare(std::move(spans),bound,[] { return host_memory().available; },prefill_config_.mapped_pp);
}
} // namespace strata::qwen35moe
