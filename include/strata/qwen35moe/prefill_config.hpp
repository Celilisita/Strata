#pragma once
#include "strata/prefill/quant_format_set.hpp"
#include <algorithm>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::qwen35moe {
enum class ProjectionRole { generic, gdn_qkv, gdn_output };
enum class AttentionCheck { none, packed_query, shared_memory };
struct PrefillConfig {
    int batch_size = 1024, ubatch_size = 1024, attention_queries = 2, projection_columns = 8, attention_group = 1, projection_rows = 1;
    bool explicit_limits = false, bulk_projection = false, shared_attention_join = false, padded_attention_cache = false, dense_tensor = false;
    bool full_shared_attention = false;
    bool packed_attention_query = false;
    bool grouped_gu_active = false; // Canonical nonempty tile schedule; benchmark only.
    bool grouped_gu_mmq = false; // Measured quantized PP provider; TG stays canonical.
    bool short_attention_batch = false; // Canonical causal-row schedule; short exact gates passed.
    bool attention_tc_scratch_reuse = false; // Exact shared phases; short/actual30k gates passed.
    bool attention_tc_compact = false; // Exact-fragment PV store candidate; benchmark only.
    bool mtp_kv_bulk = false; // Bit-exact KV rows; real short/long MTP gates passed.
    bool router_tiled = false; // Bit-exact route provider; actual long gate passed.
    bool gdn_qkv_q4_mmq = false; // Q4 recurrent slice; actual short/long gate passed.
    bool gdn_out_mmq = false; // Canonical Q6 output; short/30k exact gates passed.
    bool gdn_qkv_mmq = false; // Qualified Q6 recurrent QKV PP provider.
    bool grouped_down_mmq = false; // Measured quantized PP provider; vector TG stays canonical.
    bool tensor_attention = false; // Measured Volta online WMMA prefill; vector TG stays canonical.
    bool mmq_projection = false; // Calibrated Q4 recurrent output gate.
    bool gdn_columns = false;
    bool gdn_warp_norm = false; // Canonical per-warp L2 candidate; benchmark only.
    bool q4_gu_reuse = false; // Measured gathered GU input reuse.
    bool q4_input_reuse = false; // Measured canonical Q4 two-row PP provider.
    prefill::QuantFormatSet bulk_formats;
    static PrefillConfig environment() {
        PrefillConfig c;
        auto read = [&](const char* name, int& value, bool limits = true) {
            if (const char* raw = std::getenv(name)) {
                size_t used = 0;
                value = std::stoi(raw, &used);
                if (raw[used]) throw std::invalid_argument("invalid prefill batch limit");
                if (limits) c.explicit_limits = true;
            }
        };
        if (const char* raw = std::getenv("STRATA_QWEN35_BULK_PROJECTION")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("bulk projection requires0 or1");
            c.bulk_projection = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_SHARED_JOIN")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("attention shared join requires0 or1");
            c.shared_attention_join = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_PAD_CACHE")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("attention padded cache requires0 or1");
            c.padded_attention_cache = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_DENSE_TC")) {
            if (std::string(raw) != "0")
                throw std::invalid_argument("dense Tensor Core failed route gate; benchmark only");
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_MTP_KV_BULK")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("MTP KV bulk requires 0 or 1");
            c.mtp_kv_bulk = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ROUTER_TILED")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("tiled router requires 0 or 1");
            c.router_tiled = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_QKV_Q4_MMQ")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("Q4 recurrent QKV MMQ requires 0 or 1");
            c.gdn_qkv_q4_mmq = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_GU_MMQ")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("grouped GU MMQ requires 0 or 1");
            c.grouped_gu_mmq = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_GDN_OUT_MMQ")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("GDN output MMQ requires0 or1");
            c.gdn_out_mmq = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_QKV_MMQ")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("recurrent QKV MMQ requires 0 or 1");
            c.gdn_qkv_mmq = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_DOWN_MMQ")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("grouped Down MMQ requires 0 or 1");
            c.grouped_down_mmq = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_SHORT_BATCH")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("short attention batch requires0 or1");
            c.short_attention_batch = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_SCRATCH_REUSE")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("attention scratch reuse requires0 or1");
            c.attention_tc_scratch_reuse = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_TC")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("attention Tensor Core requires 0 or 1");
            c.tensor_attention = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_MMQ_PROJECTION")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("MMQ projection requires0 or1");
            c.mmq_projection = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_GDN_COLUMNS")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("GDN columns requires0 or1");
            c.gdn_columns = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_Q4_INPUT_REUSE")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("Q4 input reuse requires0 or1");
            c.q4_input_reuse = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_Q4_GU_REUSE")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("Q4 GU reuse requires0 or1");
            c.q4_gu_reuse = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_PACKED_ATTN_QUERY")) {
            if (std::string(raw) != "0")
                throw std::invalid_argument("packed attention query failed resource/performance gate; benchmark only");
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_FULL_SHARED")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("attention shared memory requires0 or1");
            c.full_shared_attention = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_NATIVE_PROJECTION_BULK_TYPES"))
            c.bulk_formats = prefill::QuantFormatSet::parse(raw);
        read("STRATA_QWEN35_BATCH_SIZE", c.batch_size);
        read("STRATA_QWEN35_UBATCH_SIZE", c.ubatch_size);
        read("STRATA_QWEN35_ATTN_QUERY_BATCH", c.attention_queries, false);
        if (const char* raw = std::getenv("STRATA_NATIVE_PROJECTION_COLUMNS")) {
            if (std::string(raw) != "8")
                throw std::invalid_argument("projection column4 failed cost gate; requires8");
        }
        read("STRATA_NATIVE_PROJECTION_ROWS", c.projection_rows, false);
        if (c.projection_rows != 1)
            throw std::invalid_argument("projection row reuse failed performance gate; requires1");
        read("STRATA_QWEN35_ATTN_QUERY_GROUP", c.attention_group, false);
        if (c.padded_attention_cache && c.attention_group != 2)
            throw std::invalid_argument("attention padded cache requires query group2");
        if (c.attention_group != 1 && c.attention_group != 2)
            throw std::invalid_argument("attention query group4 failed performance gate; requires1 or2");
        if (c.attention_queries < 2 || c.attention_queries > 64)
            throw std::invalid_argument("attention query batch requires 2..64");
        if (c.batch_size < 2 || c.batch_size > 8192 || c.ubatch_size < 2 || c.ubatch_size > c.batch_size)
            throw std::invalid_argument("prefill requires 2 <= ubatch_size <= batch_size <= 8192");
        return c;
    }
    int capacity(int tokens, bool tensor) const {
        return explicit_limits ? ubatch_size : tensor && tokens >= 1024 ? 1024 : 16;
    }
};
} // namespace strata::qwen35moe
