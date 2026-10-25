#pragma once

#include "ggml-backend.h"

#include <map>
#include <array>
#include <memory>
#include <vector>

struct llama_model;
namespace strata::deepseek_v4 { struct BackgroundAdmission; class ExpertFrequency; struct CacheCostSnapshot; struct NativeCacheFill; struct PpCacheView; }

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
namespace strata::glm5_next {struct PhaseMapAudit;}
class llama_moe_cache {
public:
#ifdef STRATA_DEEPSEEK_EXPERT_STAGING
    size_t strata_resize_phase(int permanent,bool (*)(ggml_backend_t,ggml_tensor*,const void*,size_t,size_t,void*),void*);
    void strata_use_prefill_frequency();
    void strata_clear_glm_prefill();
    void strata_resize_glm_phase(size_t bytes);
    int strata_grow_glm_phase(size_t,bool(*)(void*),void*,strata::glm5_next::PhaseMapAudit&);
    // Caller retires GPU/CPU readers and pending background transactions first.
    bool strata_glm_reserve_cold_rows(int maximum=2);
    void strata_glm_release_cold_rows();
    bool strata_glm_cold_rows(int32_t layer, int32_t& first, std::array<ggml_tensor*,3>& banks, int32_t* rows=nullptr) const;
    size_t strata_glm_pp_budget() const;
    size_t strata_verify_glm_phase() const;
    std::array<uint64_t,4> strata_glm_prefill_coverage() const;
    size_t strata_prewarm_phase(bool (*)(ggml_backend_t,ggml_tensor*,const void*,size_t,size_t,void*),void*);
    std::vector<strata::deepseek_v4::BackgroundAdmission> strata_plan_background(const uint8_t*) const;
    size_t strata_upload_background(const std::vector<strata::deepseek_v4::BackgroundAdmission>&,bool (*)(ggml_backend_t,ggml_tensor*,const void*,size_t,size_t,void*),void*);
    void strata_commit_background(const std::vector<strata::deepseek_v4::BackgroundAdmission>&);
    size_t strata_verify_background(const std::vector<strata::deepseek_v4::BackgroundAdmission>&) const;
    using upload_callback = bool (*)(ggml_backend_t, ggml_tensor *, const void *, size_t, size_t, void *);
    size_t strata_seed_glm_phase(upload_callback,void*,uint64_t* removed_keys);
    void set_expert_upload_callback(upload_callback callback, void * user_data);
    ggml_tensor* strata_directory(int32_t layer) const;
    strata::deepseek_v4::CacheCostSnapshot strata_cost_snapshot() const;
    size_t strata_fill_selected(int32_t layer, const int32_t* ids, size_t count, upload_callback, void*, uint64_t* planning_ns = nullptr, bool frequency_observed = false, uint32_t admission_mask = 63);
    void strata_native_slots(int32_t layer, const int32_t* ids, int32_t* slots) const;
    strata::deepseek_v4::NativeCacheFill strata_fill_native(int32_t layer, const int32_t* ids, uint32_t gpu_mask,
        uint32_t epoch, int32_t* slots, upload_callback, void*, uint64_t* planning_ns);
    strata::deepseek_v4::ExpertFrequency* strata_frequency() const;
    const strata::deepseek_v4::PpCacheView* strata_pp_view(int32_t, const ggml_tensor*, const ggml_tensor*, const ggml_tensor*) const;
    using cold_callback = const int32_t * (*)(ggml_backend_t, int, const int32_t *, size_t, const ggml_tensor *, bool, ggml_cgraph *, void *);
    void set_cold_callback(cold_callback callback, void * user_data);
    void strata_set_leaf_aliases();
    using cache_map_callback = ggml_tensor* (*)(int32_t, int64_t, int64_t, ggml_tensor*, void*);
    using cache_lookup_callback = void (*)(ggml_tensor*, int32_t, void*);
    using cache_guard_callback = ggml_tensor* (*)(ggml_context*, ggml_tensor*, void*);
    void strata_set_cache_hooks(cache_map_callback, cache_lookup_callback, cache_guard_callback, void*);
    void strata_decorate_lookup(ggml_tensor*, int32_t) const;
    ggml_tensor* strata_guard_logits(ggml_context*, ggml_tensor*) const;
#endif
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache, starting at expert e and up to expert last, to the copy dst of w
    // returns the number of experts copied, 0 if expert e is not in the cache
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
