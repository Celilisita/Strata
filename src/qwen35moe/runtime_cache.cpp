#include "strata/qwen35moe/runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <string>

namespace strata::qwen35moe {
namespace {
size_t positive_limit(const char* name) {
    const char* raw = std::getenv(name);
    if (!raw) return 0;
    const std::string text(raw);
    if (text.empty() || text.find_first_not_of("0123456789") != std::string::npos)
        throw std::invalid_argument(std::string("invalid ") + name);
    size_t used = 0;
    const auto value = std::stoull(text, &used);
    if (!value || used != text.size() || value > (std::numeric_limits<size_t>::max)())
        throw std::invalid_argument(std::string("invalid ") + name);
    return size_t(value);
}
}

size_t Runtime::expert_cache_budget(size_t reclaim, size_t additional, int kv, int prefill) const {
    size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "size Qwen35 expert cache");
    // Leave room for prepare_batch() and transient long-prefill buffers too.
    size_t reserve = (size_t(2) << 30) + (size_t(128) << 20);
    if (prefill > 1024) {
        // Large PP phase arena + canonical request-sized residual; no paging.
        const size_t planned = size_t(prefill) * 172032 + (size_t(32) << 20);
        reserve = (std::max)(reserve, planned + size_t(kv) * 8192 + (size_t(128) << 20));
        if (batch_.capacity > 1024) reserve -= (std::min)(reserve, batch_.owned_bytes());
    }
    // Bound total device occupancy, including existing allocations and the
    // unchanged reserve for request workspaces. Unset preserves the baseline.
    size_t available = free_bytes + reclaim;
    const size_t limit_mib = positive_limit("STRATA_QWEN35_GPU_MEMORY_MIB");
    if (limit_mib) {
        if (limit_mib > (std::numeric_limits<size_t>::max)() / (size_t(1) << 20))
            throw std::invalid_argument("GPU memory limit overflow");
        const size_t ceiling = limit_mib << 20;
        const size_t occupied = total_bytes - free_bytes;
        available = (std::min)(available, ceiling + reclaim > occupied ? ceiling + reclaim - occupied : 0);
    }
    available = available > additional ? available - additional : 0;
    const size_t budget = available > reserve ? available - reserve : 0;
    // The authoritative inventory bounds useful residency, even on large GPUs.
    const size_t inventory_slots = expert_sources_.size() * size_t(256);
    size_t slots = (std::min)(budget / expert_cache_.stride, inventory_slots);
    const size_t fixed_slots = positive_limit("STRATA_QWEN35_CACHE_SLOTS");
    if (!fixed_slots && slots >= ExpertStorage::chunk_slots)
        slots = slots / ExpertStorage::chunk_slots * ExpertStorage::chunk_slots;
    if (fixed_slots) {
        if (fixed_slots < 8 || fixed_slots > slots)
            throw std::invalid_argument("requested cache slots exceed GPU memory budget: requested=" +
                std::to_string(fixed_slots) + " available_slots=" + std::to_string(slots) +
                " workspace_reserve_mib=" + std::to_string(reserve >> 20) +
                " memory_limit_mib=" + std::to_string(limit_mib));
        slots = fixed_slots;
    }
    if (slots < 8) throw std::runtime_error("GPU budget has no room for eight experts");
    return slots;
}

void Runtime::allocate_expert_cache() {
    const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const size_t down_max = kernels::native_mmvq_weight_bytes(14, 512, 2048);
    auto& cache = expert_cache_;
    cache.stride = (2 * gu + down_max + 255) & ~size_t(255);
    size_t slots = expert_cache_budget(0, 0, kv_capacity_, prefill_budget_capacity_);
    while (slots >= 8) {
        try {
            cache.data.resize(slots, cache.stride, expert_sources_.size() * 256, stream_);
            break;
        } catch (const std::runtime_error&) {
            cudaGetLastError();
            slots /= 2;
            if (slots >= ExpertStorage::chunk_slots) slots = slots / ExpertStorage::chunk_slots * ExpertStorage::chunk_slots;
        }
    }
    if (slots < 8) throw std::runtime_error("GPU memory budget has no room for an eight-expert cache");
    cache.slots = slots;
    const char* resident = std::getenv("STRATA_QWEN35_TG_DEVICE_LAYERS");
    resident_layers_ = resident ? std::stoi(resident) : 0;
    if (resident_layers_ < 0 || resident_layers_ > int(expert_sources_.size()) ||
        size_t(resident_layers_) * 256 + 8 > slots)
        throw std::invalid_argument("invalid resident Qwen35 layer count for GPU cache");
    cache.key_to_slot.assign(expert_sources_.size() * 256, -1);
    cache.slot_to_key.assign(slots, -1);
    cache.age.assign(slots, 0);
}

void Runtime::prewarm_expert_cache() {
    if (!expert_cache_.slots) return;
    const auto start = std::chrono::steady_clock::now();
    Metrics metrics;
    const size_t count = (std::min)(expert_cache_.slots, expert_sources_.size() * size_t(256));
    for (size_t key = 0; key < count; ++key)
        expert_view(key / 256, int(key % 256), expert_sources_[key / 256], metrics);
    if (resident_layers_) {
        const size_t entries = size_t(resident_layers_) * 256;
        for (size_t key = 0; key < entries; ++key)
            if (expert_cache_.key_to_slot[key] != int(key))
                throw std::runtime_error("resident Qwen35 cache directory not stable");
        expert_cache_.resident_slots.reset(entries * sizeof(int32_t));
        cuda_check(cudaMemcpyAsync(expert_cache_.resident_slots.data(),
                                   expert_cache_.key_to_slot.data(), entries * sizeof(int32_t),
                                   cudaMemcpyHostToDevice, stream_), "mirror resident expert slots");
    }
    cuda_check(cudaStreamSynchronize(stream_), "wait Qwen35 cache prewarm");
    cache_warm_ms_ = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    cache_warm_bytes_ = metrics.expert_h2d;
    cache_warm_experts_ = count;
}

Runtime::ExpertView Runtime::expert_view(size_t layer, int id,
                                         const LayerTensors& source, Metrics& metrics) {
    if (id < 0 || id >= 256 || layer >= expert_sources_.size())
        throw std::runtime_error("invalid Qwen35 routed expert key");
    const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const int down_type = int(source.down_exps->type);
    const size_t down = kernels::native_mmvq_weight_bytes(down_type, 512, 2048);
    if (down_type != 12 && down_type != 13 && down_type != 14)
        throw std::runtime_error("unsupported Qwen35 expert down format");
    const int key = int(layer * 256 + size_t(id));
    auto& cache = expert_cache_;
    uint8_t* device = nullptr;
    if (cache.slots) {
        int slot = cache.key_to_slot[size_t(key)];
        if (slot >= 0) {
            cache.age[size_t(slot)] = ++cache.tick;
            ++metrics.expert_cache_hits;
            device = cache.data.slot(size_t(slot));
            return {device, device + gu, device + 2 * gu};
        }
        while (cache.next < cache.slots && cache.slot_to_key[cache.next] >= 0) ++cache.next;
        if (cache.next < cache.slots) slot = int(cache.next++);
        else {
            const size_t first_victim = size_t(resident_layers_) * 256;
            slot = cache.retention.leased()
                ? int(cache.retention.victim(cache.slot_to_key, cache.age, first_victim))
                : int(std::min_element(cache.age.begin() + first_victim,
                                       cache.age.end()) - cache.age.begin());
        }
        const int old_key = cache.slot_to_key[size_t(slot)];
        if (old_key >= 0) cache.key_to_slot[size_t(old_key)] = -1;
        cache.key_to_slot[size_t(key)] = slot;
        cache.slot_to_key[size_t(slot)] = key;
        cache.age[size_t(slot)] = ++cache.tick;
        device = cache.data.slot(size_t(slot));
    } else {
        device = work_.gu_gate.as<uint8_t>();
    }
    const auto* pinned = layer < pinned_experts_.layers.size() &&
                         pinned_experts_.layers[layer].data
        ? &pinned_experts_.layers[layer] : nullptr;
    const uint8_t* gate = pinned
        ? pinned->data + size_t(id) * pinned->stride
        : gguf_.tensor_data(*source.gate_exps) + size_t(id) * gu;
    const uint8_t* up = pinned
        ? gate + gu
        : gguf_.tensor_data(*source.up_exps) + size_t(id) * gu;
    const uint8_t* down_weights = pinned
        ? gate + 2 * gu
        : gguf_.tensor_data(*source.down_exps) + size_t(id) * down;
    void* device_up = cache.slots ? device + gu : work_.gu_up.data();
    void* device_down = cache.slots ? device + 2 * gu : work_.down.data();
    const auto transfer_start = std::chrono::steady_clock::now();
    if (pinned && cache.slots) {
        cuda_check(cudaMemcpyAsync(device, gate, 2 * gu + down,
                                    cudaMemcpyHostToDevice, stream_), "packed expert H2D");
    } else {
        cuda_check(cudaMemcpyAsync(device, gate, gu, cudaMemcpyHostToDevice, stream_), "expert gate H2D");
        cuda_check(cudaMemcpyAsync(device_up, up, gu, cudaMemcpyHostToDevice, stream_), "expert up H2D");
        cuda_check(cudaMemcpyAsync(device_down, down_weights, down, cudaMemcpyHostToDevice, stream_),
                   "expert down H2D");
    }
    metrics.expert_h2d_calls += pinned && cache.slots ? 1 : 3;
    metrics.expert_enqueue_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - transfer_start).count();
    metrics.expert_h2d += 2 * gu + down;
    ++metrics.expert_cache_misses;
    return {device, device_up, device_down};
}

Runtime::ExpertView Runtime::staged_expert_view(size_t layer, int id, int entry,
                                                const LayerTensors& source,
                                                Metrics& metrics) {
    if (id < 0 || id >= 256 || layer >= expert_sources_.size() || entry < 0 ||
        size_t(entry) >= batch_.expert_stage_slots || !batch_.expert_stage.data())
        throw std::runtime_error("invalid Qwen35 staged expert");
    const size_t gu = kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const int down_type = int(source.down_exps->type);
    if (down_type != 12 && down_type != 13 && down_type != 14)
        throw std::runtime_error("unsupported Qwen35 staged down format");
    const size_t down = kernels::native_mmvq_weight_bytes(down_type, 512, 2048);
    if (2 * gu + down > batch_.expert_stage_stride)
        throw std::runtime_error("Qwen35 staged expert stride too small");
    auto* device = batch_.expert_stage.as<uint8_t>() +
                   size_t(entry) * batch_.expert_stage_stride;
    const auto* pinned = layer < pinned_experts_.layers.size() &&
                         pinned_experts_.layers[layer].data
        ? &pinned_experts_.layers[layer] : nullptr;
    const uint8_t* gate = pinned
        ? pinned->data + size_t(id) * pinned->stride
        : gguf_.tensor_data(*source.gate_exps) + size_t(id) * gu;
    const auto start = std::chrono::steady_clock::now();
    if (pinned) {
        cuda_check(cudaMemcpyAsync(device, gate, 2 * gu + down,
                                    cudaMemcpyHostToDevice, stream_), "staged packed expert H2D");
    } else {
        cuda_check(cudaMemcpyAsync(device, gate, gu, cudaMemcpyHostToDevice, stream_),
                   "staged gate H2D");
        cuda_check(cudaMemcpyAsync(device + gu,
                                    gguf_.tensor_data(*source.up_exps) + size_t(id) * gu,
                                    gu, cudaMemcpyHostToDevice, stream_), "staged up H2D");
        cuda_check(cudaMemcpyAsync(device + 2 * gu,
                                    gguf_.tensor_data(*source.down_exps) + size_t(id) * down,
                                    down, cudaMemcpyHostToDevice, stream_), "staged down H2D");
    }
    metrics.expert_h2d_calls += pinned ? 1 : 3;
    metrics.expert_enqueue_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    metrics.expert_h2d += 2 * gu + down;
    ++metrics.expert_cache_misses;
    return {device, device + gu, device + 2 * gu};
}

} // namespace strata::qwen35moe
