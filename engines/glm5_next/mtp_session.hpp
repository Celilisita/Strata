#pragma once
#include "llama.h"
#include "ggml.h"
#include "mtp_draft_profile.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <vector>
struct llama_context;
struct llama_memory_i;
namespace strata::glm5_next {
struct MtpProbeStep {
    llama_token token=0;
    std::array<float,4096> hidden{};
};
struct MtpProbeState {
    llama_pos next=0,attn_min=-1,attn_max=-1,index_min=-1,index_max=-1;
    uint32_t recurrent_configured_snapshots=0,recurrent_snapshots=0;
    bool recurrent_empty=false,layout_valid=false;
    uint32_t attn_stream=0,index_stream=0,attn_scan_rows=0,index_scan_rows=0;
    std::array<uint32_t,3> recurrent_tensor_counts{};
    std::vector<uint32_t> attn_layers,index_layers;
    std::vector<llama_pos> index_positions;
    std::array<int64_t,4> first_layout_difference{{-1,-1,-1,-1}};
    uint64_t memory_bytes=0,workspace_bytes=0,recurrent_storage_bytes=0,recurrent_rollback_storage_bytes=0;
    std::vector<llama_pos> positions;
    // Logical MLA rows and indexer key|gate rows, excluding unreachable pooled slots.
    std::vector<float> payload;
};
// Same metadata inspector for live-session diagnostics and zero-weight contract checks.
MtpProbeState inspect_mtp_probe_memory(llama_memory_i&,llama_pos next);
struct MtpProbeWarm {
    bool available=true;
    uint64_t calls=0,input_rows=0,workspace_peak=0,free_min=UINT64_MAX,shape_estimate=0;
    int chunk_limit=0;
};
struct MtpProbeSnapshot {
    llama_pos next=0;uint64_t limit_bytes=0,fnv1a64=0,tensor_bytes=0;
    std::vector<uint8_t> bytes;
};
// NextN uses the loaded model, backends and expert-cache authority. Only its
// trained layer45 KV/indexer state and graph scheduling buffers are private.
class MtpSession {
public:
    explicit MtpSession(llama_context& primary,ggml_type kv_type);
    ~MtpSession();
    MtpSession(const MtpSession&)=delete;
    MtpSession& operator=(const MtpSession&)=delete;
    void reset();
    void warm(const std::vector<llama_token>& ids,const std::vector<float>& primary_hidden);
    llama_token advance(llama_token token,const float* previous_hidden,llama_pos position);
    llama_token advance_pair(const llama_token* tokens,const float* previous_hidden,llama_pos position);
    void retire_graph();
    const std::vector<float>& logits() const;
    llama_pos position() const;
    MtpDraftProfile& draft_profile(); // Request-owned diagnostic; inactive outside glm_mtp_draft.
    // Diagnostic only. Normal advance/advance_pair do not export hidden or inspect KV.
    MtpProbeStep probe_step(const llama_token*,const float* previous,int count,llama_pos first);
    llama_token probe_advance(const llama_token*,const float* previous,int count,llama_pos first);
    void probe_rewind(llama_pos next);
    MtpProbeState probe_state(bool payload=false) const;
    MtpProbeWarm probe_warm(const std::vector<llama_token>&,const std::vector<float>&);
    MtpProbeSnapshot probe_snapshot() const;
    void probe_restore(const MtpProbeSnapshot&);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
bool mtp_requested();
}
