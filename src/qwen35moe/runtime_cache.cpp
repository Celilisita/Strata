#include "strata/qwen35moe/runtime.hpp"
#include "compact_expert_prefix.hpp"
#include "tg_phase_lease.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/cache_peak_budget.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdexcept>
#include <limits>
#include <cstdio>
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
    // A leased decode workspace already exists; no future large PP arena is owed.
    // reserve_context() always requests >=1024 and restores the original policy.
    if (prefill == 16 && detail::tg_phase_lease_enabled()) reserve = size_t(128) << 20;
    const bool workset2048 = detail::pp_workset2048_enabled() && prefill == 2048;
    if (workset2048) {
        // This bound is qualified only with the one compact main-stack cache.
        // Existing GGML pool rounding must fit the reviewed provider allowance.
        if (!expert_cache_.data.compact() || mtp_enabled_ || session_capacity() != 1 ||
            resident_layers_ || batch_.capacity > 2048 || !detail::tg_phase_lease_enabled())
            throw std::invalid_argument("PP_WORKSET_2048 budget requires compact/single/MTPoff/resident0/phase lease");
        reserve = detail::pp_workset2048_reserve(kv, !batch_.gemm.stream());
        if (batch_.capacity > 1024) reserve -= (std::min)(reserve, batch_.owned_bytes());
    } else if (prefill > 1024) {
        // Large PP phase arena + canonical request-sized residual; no paging.
        const size_t planned = size_t(prefill) * 172032 + (size_t(32) << 20);
        reserve = (std::max)(reserve, planned + size_t(kv) * 8192 + (size_t(128) << 20));
        if (batch_.capacity > 1024) reserve -= (std::min)(reserve, batch_.owned_bytes());
    }
    if(max_context_>16384&&max_context_<=196608&&cache_peak_enabled()&&cache_peak_reuse_workspace()&&
       cache_peak_ready(batch_,prefill,!sessions_,mtp_enabled_,prefill_tc_,prefill_scan_,
           tile_attention_,!batch_failed_,!gpu_head_||bool(sampling_)||bool(minicpm_sampler_),stream_,bool(nvfp4_)))
        reserve=cache_peak_future_bytes(kv,expert_sources_.size()*size_t(256),max_context_);
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
    size_t slots = expert_cache_.data.layout().slots_for_bytes(budget, expert_cache_.stride, inventory_slots);
    if (expert_cache_.data.compact() && slots < expert_cache_.data.high_slots()+256)
        throw std::runtime_error("compact phase cannot retain high inventory and one low layer");
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
    if (slots < (nvfp4_ ? size_t(resident_layers_+1)*256 : 8u)) throw std::runtime_error("GPU budget has no room for protected expert working set");
    // Reuse the existing budget query; this diagnostic adds no CUDA work.
    const char* dense_events = std::getenv("STRATA_QWEN35_DENSE_EVENTS");
    if (dense_events && dense_events[0] == '1' && dense_events[1] == '\0') {
        std::fprintf(stderr, "qwen35_expert_budget total_bytes=%zu free_bytes=%zu occupied_bytes=%zu "
            "available_bytes=%zu reserve_bytes=%zu additional_bytes=%zu reclaim_bytes=%zu "
            "kv=%d prefill=%d budget_bytes=%zu slots=%zu cache_bytes=%zu batch_owned_bytes=%zu compact=%d\n",
            total_bytes, free_bytes, total_bytes-free_bytes, available, reserve,
            additional, reclaim, kv, prefill, budget, slots, expert_cache_.data.bytes(),
            batch_.owned_bytes(), int(expert_cache_.data.compact()));
    }
    return slots;
}

void Runtime::allocate_expert_cache() {
    const size_t gu = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes : kernels::native_mmvq_weight_bytes(12, 2048, 512);
    size_t down_max = kernels::native_mmvq_weight_bytes(14, 512, 2048);
    const char* raw_stride = std::getenv("STRATA_QWEN35_CACHE_STRIDE_ACTUAL");
    const std::string actual_stride = raw_stride ? raw_stride : "0";
    if (actual_stride != "0" && actual_stride != "1")
        throw std::invalid_argument("CACHE_STRIDE_ACTUAL requires 0 or 1");
    if (nvfp4_) down_max = Nvfp4ExpertInventory::matrix_bytes;
    if (!nvfp4_ && actual_stride == "1") {
        // The sole inventory includes any resident NextN decoder. The stride
        // is fixed before backing allocation and remains immutable thereafter.
        if (expert_sources_.empty()) throw std::invalid_argument("empty expert inventory");
        down_max = 0;
        for (const auto& source : expert_sources_) {
            if (!source.gate_exps || !source.up_exps || !source.down_exps ||
                int(source.gate_exps->type) != 12 || int(source.up_exps->type) != 12)
                throw std::invalid_argument("actual cache stride requires Q4_K gate/up experts");
            const int type = int(source.down_exps->type);
            if (type != 12 && type != 13 && type != 14)
                throw std::invalid_argument("unsupported expert Down format for actual cache stride");
            down_max = (std::max)(down_max,
                kernels::native_mmvq_weight_bytes(type, 512, 2048));
        }
    }
    auto& cache = expert_cache_;
    cache.stride = (2 * gu + down_max + 255) & ~size_t(255);
    size_t slots = expert_cache_budget(0, 0, kv_capacity_, prefill_budget_capacity_);
    const char* raw_compact = std::getenv("STRATA_QWEN35_COMPACT_EXPERT_SLOTS");
    const std::string compact = raw_compact ? raw_compact : "0";
    if (compact != "0" && compact != "1")
        throw std::invalid_argument("COMPACT_EXPERT_SLOTS requires 0 or 1");
    if (compact == "1" && nvfp4_)
        throw std::invalid_argument("compact Q4/Q5/Q6 slots do not accept typed NVFP4 payloads");
    if (compact == "1") {
        const char* dynamic = std::getenv("STRATA_QWEN35_TG_DYNAMIC");
        const char* resident = std::getenv("STRATA_QWEN35_TG_DEVICE_LAYERS");
        const char* warm = std::getenv("STRATA_QWEN35_PREWARM_CACHE");
        if (mtp_enabled_ || session_capacity()!=1 || expert_sources_.size()!=40 ||
            !dynamic || std::string(dynamic)!="1" || (resident && std::string(resident)!="0") ||
            std::getenv("STRATA_QWEN35_CACHE_SLOTS") || actual_stride!="1" ||
            (warm && std::string(warm)!="1"))
            throw std::invalid_argument("compact slots require main40/MTPoff/single/dynamic/resident0/actual stride, no fixed slots");
        const size_t low = 3*gu;
        size_t high_slots = 0;
        for (const auto& source : expert_sources_) {
            const size_t payload = 2*gu+kernels::native_mmvq_weight_bytes(int(source.down_exps->type),512,2048);
            if (payload!=low && payload!=cache.stride)
                throw std::invalid_argument("compact slots require exactly Q4 plus one larger Down class");
            if (payload==cache.stride) high_slots += 256;
        }
        const size_t byte_cap = slots*cache.stride;
        const bool eligible = high_slots<=expert_sources_.size()*256-256 &&
            byte_cap>=high_slots*cache.stride+256*low;
        const bool selected = eligible && cache.data.select_compact(high_slots,low,cache.stride,
            expert_sources_.size()*256,byte_cap);
        if (selected) slots = cache.data.layout().slots_for_bytes(byte_cap,cache.stride,expert_sources_.size()*256);
        std::fprintf(stderr,"qwen35_compact_slots selected=%d high_slots=%zu slots=%zu backing_bytes=%zu legacy_cap=%zu\n",
            int(selected),high_slots,slots,cache.data.layout().bytes(slots,cache.stride),byte_cap);
    }
    const size_t minimum_slots=nvfp4_?256:8;
    while (slots >= minimum_slots) {

        try {
            cache.data.resize(slots, cache.stride, expert_sources_.size() * 256, stream_);
            break;
        } catch (const std::runtime_error&) {
            cudaGetLastError();
            if (cache.data.compact()) throw; // a selected class prefix never silently changes layout
            slots /= 2;
            if (slots >= ExpertStorage::chunk_slots) slots = slots / ExpertStorage::chunk_slots * ExpertStorage::chunk_slots;
        }
    }
    if (slots < minimum_slots) throw std::runtime_error("GPU memory budget has no room for protected expert cache");
    cache.slots = slots;
    const char* resident = std::getenv("STRATA_QWEN35_TG_DEVICE_LAYERS");
    resident_layers_ = resident ? std::stoi(resident) : 0;
    if (resident_layers_ < 0 || resident_layers_ > int(expert_sources_.size()) ||
        size_t(resident_layers_) * 256 + (nvfp4_?256:8) > slots)
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
    size_t high = 0, low = 0, filled = 0;
    const size_t gu = kernels::native_mmvq_weight_bytes(12,2048,512);
    for (size_t key = 0; key < expert_sources_.size()*256 && filled<count; ++key) {
        if (expert_cache_.data.compact()) {
            const size_t payload=2*gu+kernels::native_mmvq_weight_bytes(int(expert_sources_[key/256].down_exps->type),512,2048);
            const bool large=payload==expert_cache_.stride;
            const size_t cap=large ? expert_cache_.data.high_slots() : count-expert_cache_.data.high_slots();
            if ((large ? high : low)>=cap) continue;
            ++(large ? high : low);
        }
        expert_view(key / 256, int(key % 256), expert_sources_[key / 256], metrics);
        ++filled;
    }
    if (filled!=count) throw std::logic_error("expert prewarm class coverage mismatch");
    detail::verify_compact_prefix(expert_cache_,expert_sources_);
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
    const size_t gu = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes : kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const int down_type = int(source.down_exps->type);
    const size_t down = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes : kernels::native_mmvq_weight_bytes(down_type, 512, 2048);
    if (!nvfp4_ && down_type != 12 && down_type != 13 && down_type != 14)
        throw std::runtime_error("unsupported Qwen35 expert down format");
    const auto scales=nvfp4_?nvfp4_->inventory.view(layer,id):Nvfp4ExpertSlice{};
    auto* deferred=nvfp4_?nvfp4_->tg_admission.get():nullptr;
    const bool planning=deferred&&deferred->planning();
    const int key = int(layer * 256 + size_t(id));
    auto& cache = expert_cache_;
    uint8_t* device = nullptr;
    if (cache.slots) {
        int slot = cache.key_to_slot[size_t(key)];
        if (slot >= 0) {
            if (cache.data.compact() && cache.data.capacity(size_t(slot))!=2*gu+down)
                throw std::logic_error("resident expert capacity class mismatch");
            cache.age[size_t(slot)] = ++cache.tick;
            if (!expert_prefetch_ || !expert_prefetch_->copied(layer, id))
                ++metrics.expert_cache_hits;
            device = cache.data.slot(size_t(slot));
            return {device, device + gu, device + 2 * gu,scales.gate_scale,scales.up_scale,scales.down_scale};
        }
        const auto range = cache.data.layout().range(2*gu+down,cache.slots,cache.stride);
        if (cache.next<range.first || cache.next>range.second) cache.next=range.first;
        while (cache.next < range.second && cache.slot_to_key[cache.next] >= 0) ++cache.next;
        if (cache.next < range.second) slot = int(cache.next++);
        else if (expert_prefetch_ && expert_prefetch_->active() && !decode_phase_ &&
                 (prefill_config_.expert_prefetch || prefill_config_.prefix_demand || prefill_config_.prefix_stage)) {
            uint64_t oldest = (std::numeric_limits<uint64_t>::max)();
            for (size_t candidate = (std::max)(size_t(resident_layers_) * 256,range.first); candidate < range.second; ++candidate)
                if (!expert_prefetch_->protects(cache.slot_to_key[candidate]) &&
                    (slot < 0 || cache.age[candidate] < oldest)) {
                    slot = int(candidate); oldest = cache.age[candidate];
                }
            if (slot < 0) throw std::runtime_error("no safe prefetch demand victim");
        } else {
            if (cache.data.compact() && range.first==0)
                throw std::logic_error("complete high-class inventory lost a resident key");
            const size_t first_victim = (std::max)(size_t(resident_layers_)*256,range.first);
            slot = cache.retention.leased()
                ? int(cache.retention.victim(cache.slot_to_key, cache.age, first_victim))
                : int(std::min_element(cache.age.begin() + first_victim,
                                       cache.age.begin()+range.second) - cache.age.begin());
        }
        if (size_t(slot)<range.first || size_t(slot)>=range.second ||
            (cache.data.compact() && cache.data.capacity(size_t(slot))!=2*gu+down))
            throw std::logic_error("expert destination has incompatible capacity class");
        const int old_key = cache.slot_to_key[size_t(slot)];
        if(planning)deferred->reserve(key,slot,old_key);
        if (old_key >= 0) cache.key_to_slot[size_t(old_key)] = -1;
        cache.key_to_slot[size_t(key)] = slot;
        cache.slot_to_key[size_t(slot)] = key;
        cache.age[size_t(slot)] = ++cache.tick;
        device = cache.data.slot(size_t(slot));
    } else {
        device = work_.gu_gate.as<uint8_t>();
    }
    if (expert_prefetch_ && !decode_phase_ && (prefill_config_.expert_prefetch || prefill_config_.prefix_demand || prefill_config_.prefix_stage))
        expert_prefetch_->demand(layer,id);
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
    if(planning)return {device,device_up,device_down,scales.gate_scale,scales.up_scale,scales.down_scale};
    const auto transfer_start = std::chrono::steady_clock::now();
    HostCopyCost copied;
    try {
    if (pinned && cache.slots) {
        cuda_check(cudaMemcpyAsync(device, gate, 2 * gu + down,
                                    cudaMemcpyHostToDevice, stream_), "packed expert H2D");
        copied={2*gu+down,1,1,2*gu+down,0};
    } else {
        copied+=host_readonly_.copy_to_device(device,gate,gu,stream_,"expert gate H2D",key,0);
        copied+=host_readonly_.copy_to_device(device_up,up,gu,stream_,"expert up H2D",key,1);
        copied+=host_readonly_.copy_to_device(device_down,down_weights,down,stream_,"expert down H2D",key,2);
        if(pinned){copied.registered_bytes=copied.bytes;copied.pageable_bytes=0;}
    }
    } catch (...) {
        if(cache.slots){const int bad=cache.key_to_slot[size_t(key)];
            if(bad>=0&&cache.slot_to_key[size_t(bad)]==key){cache.key_to_slot[size_t(key)]=-1;
                cache.slot_to_key[size_t(bad)]=-1;cache.age[size_t(bad)]=0;}}
        batch_failed_=true;throw;
    }
    metrics.expert_h2d_calls += copied.calls;
    metrics.expert_h2d_logical_copies += copied.logical_copies;
    metrics.expert_h2d_registered_bytes += copied.registered_bytes;
    metrics.expert_h2d_pageable_bytes += copied.pageable_bytes;
    metrics.expert_enqueue_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - transfer_start).count();
    metrics.expert_h2d += 2 * gu + down;
    ++metrics.expert_cache_misses;
    return {device, device_up, device_down,scales.gate_scale,scales.up_scale,scales.down_scale};
}

Runtime::ExpertView Runtime::staged_expert_view(size_t layer, int id, int entry,
                                                const LayerTensors& source,
                                                Metrics& metrics) {
    if (id < 0 || id >= 256 || layer >= expert_sources_.size() || entry < 0 ||
        size_t(entry) >= batch_.expert_stage_slots || !batch_.expert_stage.data())
        throw std::runtime_error("invalid Qwen35 staged expert");
    const size_t gu = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes : kernels::native_mmvq_weight_bytes(12, 2048, 512);
    const int down_type = int(source.down_exps->type);
    if (!nvfp4_ && down_type != 12 && down_type != 13 && down_type != 14)
        throw std::runtime_error("unsupported Qwen35 staged down format");
    const size_t down = nvfp4_ ? Nvfp4ExpertInventory::matrix_bytes : kernels::native_mmvq_weight_bytes(down_type, 512, 2048);
    if (2 * gu + down > batch_.expert_stage_stride)
        throw std::runtime_error("Qwen35 staged expert stride too small");
    auto* device = batch_.expert_stage.as<uint8_t>() +
                   size_t(entry) * batch_.expert_stage_stride;
    if (expert_prefetch_ && !decode_phase_ && (prefill_config_.expert_prefetch || prefill_config_.prefix_demand || prefill_config_.prefix_stage))
        expert_prefetch_->demand(layer,id);
    const auto* pinned = layer < pinned_experts_.layers.size() &&
                         pinned_experts_.layers[layer].data
        ? &pinned_experts_.layers[layer] : nullptr;
    const uint8_t* gate = pinned
        ? pinned->data + size_t(id) * pinned->stride
        : gguf_.tensor_data(*source.gate_exps) + size_t(id) * gu;
    const auto start = std::chrono::steady_clock::now();
    HostCopyCost copied;
    try {
    if (pinned) {
        cuda_check(cudaMemcpyAsync(device, gate, 2 * gu + down,
                                    cudaMemcpyHostToDevice, stream_), "staged packed expert H2D");
        copied={2*gu+down,1,1,2*gu+down,0};
    } else {
        const int64_t key=int64_t(layer)*256+id;
        copied+=host_readonly_.copy_to_device(device,gate,gu,stream_,"staged gate H2D",key,0);
        copied+=host_readonly_.copy_to_device(device+gu,
            gguf_.tensor_data(*source.up_exps)+size_t(id)*gu,gu,stream_,"staged up H2D",key,1);
        copied+=host_readonly_.copy_to_device(device+2*gu,
            gguf_.tensor_data(*source.down_exps)+size_t(id)*down,down,stream_,"staged down H2D",key,2);
    }
    } catch (...) {
        if(batch_.expert_stage_keys[size_t(id)]==entry)batch_.expert_stage_keys[size_t(id)]=-1;
        if(batch_.expert_stage_slot_keys[size_t(entry)]==id){batch_.expert_stage_slot_keys[size_t(entry)]=-1;
            batch_.expert_stage_age[size_t(entry)]=0;}
        batch_failed_=true;throw;
    }
    metrics.expert_h2d_calls += copied.calls;
    metrics.expert_h2d_logical_copies += copied.logical_copies;
    metrics.expert_h2d_registered_bytes += copied.registered_bytes;
    metrics.expert_h2d_pageable_bytes += copied.pageable_bytes;
    metrics.expert_enqueue_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    metrics.expert_h2d += 2 * gu + down;
    ++metrics.expert_cache_misses;
    const auto scales=nvfp4_?nvfp4_->inventory.view(layer,id):Nvfp4ExpertSlice{};
    return {device, device + gu, device + 2 * gu,scales.gate_scale,scales.up_scale,scales.down_scale};
}

} // namespace strata::qwen35moe
