#pragma once
#include "ggml-backend.h"
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace strata::deepseek_v4 {
// Fixed host-only diagnostic payload. No model or GPU pointers are stored.
inline constexpr uint32_t pp_plan_receipt_layers = 42;
inline constexpr uint32_t pp_plan_receipt_experts = 288;
inline constexpr uint32_t pp_plan_receipt_groups = 42 * 288;
inline constexpr uint32_t pp_plan_receipt_bounds = 42 * (288 + 288);
inline constexpr size_t pp_plan_receipt_budget = 1u << 20;
enum class PpPlanSourceKind : uint32_t {
    unknown = 0, direct_cache = 1, prefix_packed = 2, prefix_planar = 3,
    source_interleaved = 4, source_planar = 5, packed_inventory = 6
};
struct PpPlanReceiptGroup {
    int32_t first_expert, experts, first_row, rows, max_rows, cache_first_slot;
    uint32_t bounds_begin;
    PpPlanSourceKind source_kind;
};
// flags: three8-bit GGML quant type IDs in bits0..23, actual state bits24..31.
struct PpPlanReceiptLayer {
    int32_t layer;
    uint32_t tokens, topk, experts, window_experts, prefix_boundary;
    uint32_t groups_begin, groups_count, bounds_begin, bounds_count, flags, clamp_bits;
    uint64_t role_expert_bytes[3], prefix_missing[3];
    char source_names[3][64];
    uint32_t route_counts[288];
    int32_t directory[288];
};
enum : uint32_t { pp_receipt_requested=1, pp_receipt_retired=2, pp_receipt_sealed=4,
    pp_receipt_incomplete=8, pp_receipt_overflow=16, pp_receipt_invalid=32 };
struct PpPlanReceiptPod {
    uint64_t request_nonce;
    uint32_t version, expected_tokens, layer_count, group_count, bound_count, flags;
    uint32_t capacity_errors, source_kind_unset;
    uint32_t full_ids_layer, full_ids_tokens, full_ids_count, full_ids_captures, reserved[2];
    PpPlanReceiptLayer layers[pp_plan_receipt_layers];
    PpPlanReceiptGroup groups[pp_plan_receipt_groups];
    int32_t bounds[pp_plan_receipt_bounds];
    int32_t layer24_route_ids[8192*8];
};
static_assert(sizeof(PpPlanReceiptGroup) == 32);
static_assert(sizeof(PpPlanReceiptLayer) == 2592);
static_assert(sizeof(PpPlanReceiptPod) == 854912);
static_assert(sizeof(PpPlanReceiptPod) < pp_plan_receipt_budget);
static_assert(std::is_trivially_copyable<PpPlanReceiptPod>::value);
static_assert(std::is_standard_layout<PpPlanReceiptPod>::value);
// Existing live CUDA Pipeline only. Both calls require a retired request.
// begin(true) starts; begin(false,sameNonce) retires without clearing.
// read borrows immutable retired data until the next explicit begin or release.
// Neither API is allowed to initialize a backend or issue a CUDA operation.
using PpPlanReceiptBegin = bool (*)(ggml_backend_t, uint64_t, bool);
using PpPlanReceiptRead = bool (*)(ggml_backend_t, const PpPlanReceiptPod**);
inline constexpr const char* pp_plan_receipt_begin_symbol = "strata_pp_plan_receipt_begin";
inline constexpr const char* pp_plan_receipt_read_symbol = "strata_pp_plan_receipt_read";
}
