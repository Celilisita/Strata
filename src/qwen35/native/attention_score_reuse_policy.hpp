#pragma once
#include "strata/qwen35/native/attention_score_reuse.hpp"
#include <cstdint>
#include <cmath>
namespace strata::qwen35::native {
struct AttentionScoreReuseRequest {bool enabled=false,direct_resident=false;std::uint64_t submissions=0;};
inline AttentionScoreReuseRequest &attention_score_reuse_request(){
    static thread_local AttentionScoreReuseRequest state;return state;
}
// PREPARE only: release previous graphs first. The caller reads the environment
// once and supplies actual device capabilities. Graph replay never calls this.
inline void configure_attention_score_reuse_request(bool enabled,int major,int minor){
    auto &s=attention_score_reuse_request();s.enabled=enabled&&major==12&&minor==0;s.direct_resident=false;s.submissions=0;
}
inline bool attention_score_reuse_direct_resident(){return attention_score_reuse_request().direct_resident;}
// Only AttentionLayer's already-borrowed direct attention call owns this lease.
// Service wrappers cannot infer a dead arena from shape alone. Exceptions and
// nested calls restore the prior scope; no scope survives capture or replay.
class AttentionScoreReuseScope {
    bool previous_;
public:
    explicit AttentionScoreReuseScope(bool direct):previous_(attention_score_reuse_direct_resident()){
        attention_score_reuse_request().direct_resident=direct;
    }
    ~AttentionScoreReuseScope(){attention_score_reuse_request().direct_resident=previous_;}
    AttentionScoreReuseScope(const AttentionScoreReuseScope &)=delete;
    AttentionScoreReuseScope &operator=(const AttentionScoreReuseScope &)=delete;
};
// CPU eager/capture submissions, not the number of GPU graph replays.
inline std::uint64_t attention_score_reuse_submissions(){return attention_score_reuse_request().submissions;}
inline bool use_attention_score_reuse(PrefillAttentionShape s,int parts,size_t arena_bytes,bool direct_resident){
    return attention_score_reuse_request().enabled&&direct_resident&&s.q8&&s.dim==256&&s.tokens==3&&
        s.query_heads==24&&s.kv_heads==4&&parts==32&&s.capacity>=3&&s.capacity<=16384&&
        s.query_start>=0&&s.query_start<=s.capacity-3&&s.valid==s.query_start+3&&
        s.scale>0&&std::isfinite(s.scale)&&arena_bytes>=7096320;
}
}
