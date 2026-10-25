#pragma once
#include "strata/kernels/native_mmvq.hpp"
#include <stdexcept>
namespace strata::qwen35moe::detail {
// Descriptor assertions only; the existing cache remains the sole directory.
template<class Cache,class Sources>
void verify_compact_prefix(const Cache& cache,const Sources& sources,bool allow_empty=false) {
    if (!cache.data.compact()) return;
    if (cache.slots<cache.data.high_slots()+256 || cache.slot_to_key.size()!=cache.slots ||
        cache.key_to_slot.size()!=sources.size()*256)
        throw std::logic_error("compact prefix directory extent mismatch");
    if (allow_empty) {
        bool empty=true;
        for (int key:cache.slot_to_key) empty=empty && key<0;
        for (int slot:cache.key_to_slot) empty=empty && slot<0;
        if (empty) return; // reserve_context may resize before the first prewarm
    }
    const size_t gu=kernels::native_mmvq_weight_bytes(12,2048,512);
    size_t high=0;
    for (size_t key=0;key<cache.key_to_slot.size();++key) {
        const size_t bytes=2*gu+kernels::native_mmvq_weight_bytes(int(sources[key/256].down_exps->type),512,2048);
        const int slot=cache.key_to_slot[key];
        if (bytes==cache.stride) {
            ++high;
            if (slot<0 || size_t(slot)>=cache.data.high_slots())
                throw std::logic_error("high-class key absent from fixed prefix");
        }
        if (slot>=0 && (size_t(slot)>=cache.slots || cache.slot_to_key[size_t(slot)]!=int(key) ||
            cache.data.capacity(size_t(slot))!=bytes))
            throw std::logic_error("compact directory inverse/capacity mismatch");
    }
    if (high!=cache.data.high_slots()) throw std::logic_error("high-class inventory changed");
    for (size_t slot=0;slot<cache.data.high_slots();++slot) {
        const int key=cache.slot_to_key[slot];
        if (key<0 || size_t(key)>=cache.key_to_slot.size() || cache.key_to_slot[size_t(key)]!=int(slot))
            throw std::logic_error("high-class prefix is not fully occupied");
    }
}
} // namespace strata::qwen35moe::detail
