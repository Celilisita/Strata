#pragma once
#include "strata/qwen35moe/expert_storage.hpp"
#include "strata/qwen35moe/host_readonly_lease.hpp"
#include "strata/qwen35moe/prefetch_cost.hpp"
#include <array>
#include <vector>

namespace strata::qwen35moe {
// Borrow the existing authority; this view owns no weights or directory.
struct PpPrefetchCache {
    ExpertStorage* data;
    std::vector<int32_t>* keys;
    std::vector<int32_t>* slots;
    std::vector<uint64_t>* ages;
    size_t stride, resident_prefix;
    uint64_t epoch;
};
struct PpPrefetchSource {
    std::array<const uint8_t*,3> base;
    std::array<size_t,3> bytes;
};
class ExpertPrefetch {
public:
    explicit ExpertPrefetch(size_t layers);
    ~ExpertPrefetch();
    ExpertPrefetch(const ExpertPrefetch&)=delete;
    ExpertPrefetch& operator=(const ExpertPrefetch&)=delete;
    void begin_request(); // Caller has already drained previous request/readers.
    bool enqueue(PpPrefetchCache cache, size_t layer, const PpPrefetchSource& source,
                 const HostReadOnlyLease& lease, const std::vector<int32_t>* actual_ids=nullptr,
                 bool prefix_stage=false);
    void route(PpPrefetchCache cache, size_t layer, const std::vector<int32_t>& ids,
               const HostReadOnlyLease& lease, cudaStream_t compute, bool current_prefix=true);
    bool active() const { return active_; }
    bool protects(int32_t key) const;
    bool copied(size_t layer, int id) const;
    void demand(size_t layer,int id);
    void finish(); // After existing head synchronization; no new GPU drain.
    void drain(); // Required before reset/resize/TG/GGUF release.
    const PpPrefetchCost& cost() const { return cost_; }
    cudaStream_t copy_stream() const { return copy_; }
    // Typed TG borrows only this drained owner's stream/events, never PP victims.
    cudaStream_t borrow_decode_copy();
    void begin_decode_layer(size_t layer);
    void record_decode_layer(size_t layer);
    cudaEvent_t decode_ready(size_t layer) const;
    void release_decode_copy();
private:
    struct Entry { int32_t key=-1,slot=-1; bool copied=false,used=false; mutable bool looked=false; };
    static constexpr size_t none=size_t(-1);
    void validate(PpPrefetchCache cache,const HostReadOnlyLease& lease) const;
    void close_layer();
    void invalidate() noexcept;
    void release_stage() noexcept;
    size_t layers_, layer_=none, fallback_layer_=none;
    size_t stage_first_=none; // Whole64-slot physical chunk: unavailable to pending destinations.
    uint64_t request_=0, plan_request_=0, source_epoch_=0;
    PpPrefetchCache cache_{};
    const HostReadOnlyLease* lease_=nullptr;
    std::array<Entry,256> entries_{};
    std::vector<int32_t> prefix_routes_; // Exact prefix sequence, not predicted experts.
    std::array<bool,256> fallback_{};
    std::array<size_t,3> sizes_{};
    bool active_=false,joined_=false,failed_=false,actual_only_=false;
    bool decode_borrowed_=false;
    cudaStream_t copy_=nullptr;
    std::vector<std::array<cudaEvent_t,2>> events_;
    std::vector<bool> recorded_;
    PpPrefetchCost cost_{};
};
}
