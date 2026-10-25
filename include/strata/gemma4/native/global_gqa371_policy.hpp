#pragma once
#include "strata/gemma4/native/model.hpp"
#include <type_traits>
namespace strata::gemma4::native::tg371 {
// Extend the existing fixed-int policy without redefining enum/Model or layout.
static_assert(std::is_same_v<std::underlying_type_t<GlobalVerifyMerge>,int>);
static_assert(int(GlobalVerifyMerge::Parent174)==0&&int(GlobalVerifyMerge::Parallel321)==1&&int(GlobalVerifyMerge::SharedKv336)==2);
inline constexpr GlobalVerifyMerge Gqa371=static_cast<GlobalVerifyMerge>(3);
inline bool selected(const Model& m){return m.global_verify_merge()==Gqa371;}
inline bool request(const Model& m){return selected(m)&&m.global_verify321_request();}
// Existing339 counter storage is shared ONLY between exclusive321/371 modes.
// Its old getter name is not an actual321 leaf claim when mode371 is selected.
// Reset and selector clear the storage; no request mixes the two modes.
inline size_t logical_optimized_capture_enqueues(const Model& m){return m.global_verify321_enqueues();}
inline size_t actual321_capture_enqueues(const Model& m){
 return m.global_verify_merge()==GlobalVerifyMerge::Parallel321?logical_optimized_capture_enqueues(m):0;
}
inline size_t actual371_capture_enqueues(const Model& m){return selected(m)?logical_optimized_capture_enqueues(m):0;}
inline bool supported_request_capacity(int capacity){return capacity==10240||capacity==32768;}
// Compiled availability is not permission. Future driver must bind immutable
// Root per-length actual math/state/Graph/complete-cost capabilities first.
}
