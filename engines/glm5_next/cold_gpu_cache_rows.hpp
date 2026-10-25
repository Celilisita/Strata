#pragma once
// Borrow existing tail rows only while the caller has retired GPU/CPU readers
// and completed pending background transactions. No weight or lifetime owner.
#include "ggml.h"
#include <array>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace strata::glm5_next {
template<class Group>
bool cold_gpu_main_group(const Group& group) {
    if (group.layers.size() != 39 || group.ref.size() != 3 || group.banks.size() != 3 ||
        group.native.slots || group.lru.n_expert != 288 || group.n_slots < 18) return false;
    size_t index = 0;
    for (int layer = 3; layer < 44; ++layer) {
        if (layer == 11 || layer == 12) continue;
        if (group.layers[index++] != layer) return false;
    }
    constexpr std::array<ggml_type,3> types{GGML_TYPE_IQ3_S,GGML_TYPE_IQ3_S,GGML_TYPE_IQ4_XS};
    constexpr std::array<size_t,3> strides{3604480,3604480,4456448};
    for (size_t role = 0; role < 3; ++role) {
        const auto* source = group.ref[role];
        const auto* bank = group.banks[role];
        const int input = role == 2 ? 2048 : 4096;
        const int output = role == 2 ? 4096 : 2048;
        if (!source || !bank || !source->data || !bank->data ||
            source->type != types[role] || bank->type != types[role] ||
            source->ne[0] != input || bank->ne[0] != input ||
            source->ne[1] != output || bank->ne[1] != output ||
            source->ne[2] != 288 || bank->ne[2] != int64_t(group.n_slots) + 1 ||
            source->ne[3] != 1 || bank->ne[3] != 1 ||
            source->nb[2] != strides[role] || bank->nb[2] != strides[role] ||
            !ggml_is_contiguous(source) || !ggml_is_contiguous(bank)) return false;
    }
    return true;
}

// Setup/reset only. Validate before changing a link, including the two tail
// nodes whose numeric indices need not be at the LRU list's head or tail.
template<class Lru>
void cold_gpu_check_lru(const Lru& lru, int count) {
    if (count < 1 || lru.n_slots != count || lru.key_of.size() != size_t(count) ||
        lru.prev.size() != size_t(count) || lru.next.size() != size_t(count))
        throw std::runtime_error("GLM cold rows LRU extent differs");
    std::vector<uint8_t> seen(size_t(count),0);
    int previous = -1, cursor = lru.head, visited = 0;
    while (cursor != -1) {
        if (cursor < 0 || cursor >= count || seen[size_t(cursor)] ||
            lru.prev[size_t(cursor)] != previous)
            throw std::runtime_error("GLM cold rows LRU chain differs");
        seen[size_t(cursor)] = 1;
        previous = cursor;
        cursor = lru.next[size_t(cursor)];
        ++visited;
    }
    if (visited != count || previous != lru.tail)
        throw std::runtime_error("GLM cold rows LRU chain is incomplete");
}

template<class Lru>
void cold_gpu_unlink(Lru& lru, int slot) {
    const int before = lru.prev[size_t(slot)], after = lru.next[size_t(slot)];
    if (before >= 0) lru.next[size_t(before)] = after; else lru.head = after;
    if (after >= 0) lru.prev[size_t(after)] = before; else lru.tail = before;
}

template<class Group>
bool reserve_cold_gpu_rows(Group& group,int maximum=2) {
    if(maximum!=2&&maximum!=4)throw std::runtime_error("GLM cold rows maximum differs");
    if(!cold_gpu_main_group(group))return false;
    const int physical=group.n_slots;auto& lru=group.lru;
    if(group.glm_cold_rows!=0&&group.glm_cold_rows!=2&&group.glm_cold_rows!=4)
        throw std::runtime_error("GLM cold rows reservation marker differs");
    cold_gpu_check_lru(lru,physical-group.glm_cold_rows);
    if(group.glm_cold_rows){
        if(group.glm_cold_rows>maximum)throw std::runtime_error("GLM cold rows must release before capacity change");
        return true;
    }
    for(int rows=maximum;rows>=2;rows-=2){
        const int resident=physical-rows;if(resident<16)continue;
        bool free=true;
        for(int slot=resident;slot<physical;++slot)free&=lru.key_of[size_t(slot)]==-1;
        for(const auto* directory:lru.slot_map)if(directory)
            for(int key=0;key<lru.n_expert;++key)if(directory[key]>=resident)free=false;
        if(!free)continue; // Complete preflight before unlink; no eviction or compaction.
        for(int slot=resident;slot<physical;++slot)cold_gpu_unlink(lru,slot);
        lru.key_of.resize(size_t(resident));lru.prev.resize(size_t(resident));lru.next.resize(size_t(resident));
        lru.n_slots=resident;group.glm_cold_rows=rows;
        return true; // Original bank P+1 views and sentinel P remain intact.
    }
    return false;
}

template<class Group>
void release_cold_gpu_rows(Group& group) {
    if(!group.glm_cold_rows)return;
    if((group.glm_cold_rows!=2&&group.glm_cold_rows!=4)||!cold_gpu_main_group(group))
        throw std::runtime_error("GLM cold rows release geometry differs");
    const int physical=group.n_slots,resident=physical-group.glm_cold_rows;auto& lru=group.lru;
    cold_gpu_check_lru(lru,resident);
    if(lru.key_of.capacity()<size_t(physical)||lru.prev.capacity()<size_t(physical)||lru.next.capacity()<size_t(physical))
        throw std::runtime_error("GLM cold rows retained LRU capacity differs");
    const int old_head=lru.head;
    lru.key_of.resize(size_t(physical),-1);lru.prev.resize(size_t(physical));lru.next.resize(size_t(physical));
    for(int slot=resident;slot<physical;++slot){
        lru.prev[size_t(slot)]=slot==resident?-1:slot-1;
        lru.next[size_t(slot)]=slot+1<physical?slot+1:old_head;}
    lru.prev[size_t(old_head)]=physical-1;lru.head=resident;lru.n_slots=physical;group.glm_cold_rows=0;
}

template<class Group>
bool query_cold_gpu_rows(const Group& group,int32_t& first,std::array<ggml_tensor*,3>& banks,int32_t* rows=nullptr) {
    first=-1;banks={};if(rows)*rows=0;
    if((group.glm_cold_rows!=2&&group.glm_cold_rows!=4)||!cold_gpu_main_group(group)||
       group.lru.n_slots!=group.n_slots-group.glm_cold_rows)return false;
    first=group.lru.n_slots;if(rows)*rows=group.glm_cold_rows;
    for(size_t role=0;role<3;++role)banks[role]=group.banks[role];return true;
}
} // namespace strata::glm5_next
