#pragma once
#include <cstdint>
#include <cstddef>
namespace strata::qwen35moe {
// Measured Volta PP only: F32 Q, persistent F16 KV, 16 query/2 KV heads,
// D=256, causal positions. Caller owns cache; capacity must include a64-row tail.
// Padding writes only after first+count, never any active KV row.
inline bool attention_tc_fits(int first, int count, int capacity) {
    return first >= 0 && count >= 1 && count <= 8192 && capacity >= 1 && capacity <= 196608 &&
        first <= capacity - count && ((first + count + 63) / 64) * 64 <= capacity;
}
void attention_tc_prefill(const float* q, uint16_t* k, uint16_t* v,
    int first, int count, int capacity, float* out, void* stream, bool compact = false, bool scratch_reuse = false);
struct AttentionTcResources { int registers; std::size_t shared_bytes, local_bytes; };
AttentionTcResources attention_tc_resources(bool compact, bool scratch_reuse = false);
// Explicit same-build benchmark entry, also used by the opt-in runtime dispatch.
void attention_tc_prefill_query_rows(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream,int query_rows);
struct AttentionTcQueryResources {
    int registers; std::size_t shared_bytes, local_bytes; int active_blocks_per_sm;
    std::size_t static_shared_bytes = 0, dynamic_shared_bytes = 0;
    int threads = 128;
};
AttentionTcQueryResources attention_tc_query_resources(int query_rows);
// Typed experimental entry; default FP32 APIs above are never redirected.
void attention_tc_prefill_half_pv(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream);
AttentionTcQueryResources attention_tc_half_pv_resources();
// Own GQA KV-reuse/async experiment; runtime selector defaults off.
void attention_tc_prefill_gqa_async(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream);
AttentionTcQueryResources attention_tc_gqa_async_resources();
// Optional32-key/four-head variant; canonical selector2 requires own gate.
void attention_tc_prefill_gqa32_async(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream);
AttentionTcQueryResources attention_tc_gqa32_async_resources();
// Same head/query geometry as selector2, optional64-key online tile.
void attention_tc_prefill_gqa_key64(const float* q,uint16_t* k,uint16_t* v,
    int first,int count,int capacity,float* out,void* stream);
AttentionTcQueryResources attention_tc_gqa_key64_resources();
}
