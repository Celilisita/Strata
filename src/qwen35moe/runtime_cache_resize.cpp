#include "strata/qwen35moe/runtime.hpp"
#include <algorithm>
#include <stdexcept>

namespace strata::qwen35moe {
void Runtime::resize_expert_cache(size_t slots, Metrics& metrics) {
    auto& cache = expert_cache_;
    if (slots == cache.slots) return;
    if (slots < size_t(resident_layers_) * 256 + 8)
        throw std::invalid_argument("context budget cannot preserve resident expert layers");
    if (slots < cache.slots) {
        std::vector<int> retained;
        for (int key : cache.slot_to_key) if (key >= 0) retained.push_back(key);
        const int protected_keys = resident_layers_ * 256;
        std::sort(retained.begin(), retained.end(), [&](int a, int b) {
            if ((a < protected_keys) != (b < protected_keys)) return a < protected_keys;
            const auto age_a = cache.age[size_t(cache.key_to_slot[size_t(a)])];
            const auto age_b = cache.age[size_t(cache.key_to_slot[size_t(b)])];
            return age_a != age_b ? age_a > age_b : a < b;
        });
        retained.resize((std::min)(retained.size(), slots));
        std::vector<bool> keep(cache.key_to_slot.size(), false);
        for (int key : retained) keep[size_t(key)] = true;
        for (size_t slot = 0; slot < cache.slots; ++slot) {
            const int key = cache.slot_to_key[slot];
            if (key < 0 || keep[size_t(key)]) continue;
            cache.key_to_slot[size_t(key)] = -1;
            cache.slot_to_key[slot] = -1; cache.age[slot] = 0;
        }
        size_t hole = 0;
        for (int key : retained) {
            const size_t source = size_t(cache.key_to_slot[size_t(key)]);
            if (source < slots) continue;
            while (hole < slots && cache.slot_to_key[hole] >= 0) ++hole;
            if (hole == slots) throw std::logic_error("expert compaction has no destination");
            cuda_check(cudaMemcpyAsync(cache.data.slot(hole), cache.data.slot(source),
                cache.stride, cudaMemcpyDeviceToDevice, stream_), "compact resident expert on GPU");
            metrics.context_cache_d2d += cache.stride;
            cache.key_to_slot[size_t(key)] = int(hole);
            cache.slot_to_key[hole] = key; cache.age[hole] = cache.age[source];
            cache.slot_to_key[source] = -1; cache.age[source] = 0;
        }
        cuda_check(cudaStreamSynchronize(stream_), "drain expert compaction before tail release");
        ++metrics.explicit_syncs;
    }
    cache.data.resize(slots, cache.stride, cache.key_to_slot.size(), stream_);
    metrics.control_h2d += slots * sizeof(uint64_t);
    ++metrics.explicit_syncs;
    cache.slot_to_key.resize(slots, -1); cache.age.resize(slots, 0);
    cache.slots = slots;
    // Compaction may leave holes when the cache wasn't full; expert_view fills
    // the first free slot without changing the age clock or existing addresses.
    cache.next = 0;
    while (cache.next < slots && cache.slot_to_key[cache.next] >= 0) ++cache.next;
    if (tg_dynamic_) dynamic_ages_.reset(slots * sizeof(uint64_t));
    if (resident_layers_ && !tg_dynamic_)
        cuda_check(cudaMemcpyAsync(cache.resident_slots.data(), cache.key_to_slot.data(),
            cache.resident_slots.bytes(), cudaMemcpyHostToDevice, stream_), "update resident directory");
}
} // namespace strata::qwen35moe
