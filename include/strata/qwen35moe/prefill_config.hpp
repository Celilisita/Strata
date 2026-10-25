#pragma once
#include "strata/prefill/quant_format_set.hpp"
#include "strata/qwen35moe/pp_workset2048.hpp"
#include "strata/qwen35moe/precise_dense_policy.hpp"
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
    unsigned precise_dense_types = 0; // Independent measured-format opt-in; ordinary PP only.
    bool full_shared_attention = false;
    bool packed_attention_query = false;
    bool grouped_gu_active = false; // Default-off canonical nonempty J64 PP GU tiles.
    bool grouped_gu_mmq = false; // Measured quantized PP provider; TG stays canonical.
    bool expert_copy_runs = false; // Actual-route H2D runs; independent opt-in.
    bool expert_copy_stage = false; // Borrow dead phase scratch for contiguous H2D+scatter.
    bool expert_prefetch = false; // Advisory PP copies; never alters real routes.
    bool prefix_demand = false; // Actual prefix routes overlap tail dense; default off.
    int gu_copy_wave = 0; // Default off; remaining actual demand after direct prefix.
    bool prefix_stage = false; // Actual prefix copies through a borrowed cache chunk.
    bool mapped_pp = false; // Experimental registered-source cold PP; independent gate required.
    bool down_input_fused = false; // PP backend6 canonical activation/input fusion; default off.
    bool short_attention_batch = false; // Canonical causal-row schedule; short exact gates passed.
    bool attention_tc_scratch_reuse = false; // Exact shared phases; short/actual30k gates passed.
    bool attention_tc_compact = false; // Register PV accumulation; opt-in FP32 reassociation.
    bool mtp_kv_bulk = false; // Bit-exact KV rows; real short/long MTP gates passed.
    bool router_tiled = false; // Bit-exact route provider; actual long gate passed.
    bool gdn_qkv_q4_mmq = false; // Q4 recurrent slice; actual short/long gate passed.
    bool gdn_out_mmq = false; // Canonical Q6 output; short/30k exact gates passed.
    bool gdn_qkv_mmq = false; // Qualified Q6 recurrent QKV PP provider.
    bool grouped_down_mmq = false; // Measured quantized PP provider; vector TG stays canonical.
    bool tensor_attention = false; // Measured Volta online WMMA prefill; vector TG stays canonical.
    bool mmq_projection = false; // Calibrated Q4 recurrent output gate.
    bool gdn_columns = false;
    int gdn_column_tile = 0; // 0 inherits legacy GDN_COLUMNS: false->1, true->2.
    bool gdn_warp_norm = false; // Canonical per-warp L2 candidate; benchmark only.
    bool gdn_conv_parallel = false; // Independent PP convolution windows; default off.
    bool q4_gu_reuse = false; // Measured gathered GU input reuse.
    bool q4_input_reuse = false; // Measured canonical Q4 two-row PP provider.
    prefill::QuantFormatSet bulk_formats;
    int gdn_scan_columns(int tokens, bool ordinary) const {
        if (!ordinary || tokens < 1024) return 1;
        return gdn_column_tile ? gdn_column_tile : gdn_columns ? 2 : 1;
    }
    bool gdn_parallel_convolution(int tokens, bool ordinary) const {
        return gdn_conv_parallel && ordinary && tokens >= 1024 && tokens <= 8192;
    }
    static PrefillConfig environment() {
        PrefillConfig c;
        (void) detail::pp_workset2048_enabled(); // reject malformed opt-in at initialization
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
        if (const char* raw = std::getenv("STRATA_QWEN35_PRECISE_DENSE"))
            c.precise_dense_types = strata::qwen35moe::precise_dense_types(raw);
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
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_GU_ACTIVE_J64")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("PP GU active J64 requires0 or1");
            c.grouped_gu_active = value == "1";
            if (c.grouped_gu_active && !c.grouped_gu_mmq)
                throw std::invalid_argument("PP GU active J64 requires GU_MMQ1");
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_GDN_CONV_PARALLEL")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("parallel GDN convolution requires0 or1");
            c.gdn_conv_parallel = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_DOWN_INPUT_FUSED")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("fused Down input requires0 or1");
            c.down_input_fused = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_COPY_RUNS")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("PP copy runs requires0 or1");
            c.expert_copy_runs = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_COPY_STAGE")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("copy stage requires0 or1");
            c.expert_copy_stage = value == "1";
        }
        if(c.expert_copy_stage&&c.expert_copy_runs)
            throw std::invalid_argument("copy stage and copy runs are exclusive");
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_PREFETCH")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("PP prefetch requires0 or1");
            c.expert_prefetch = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_PREFIX_DEMAND")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("prefix demand requires0 or1");
            c.prefix_demand = value == "1";
        }
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_PREFIX_STAGE")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("prefix stage requires0 or1");
            c.prefix_stage = value == "1";
        }
        if(c.prefix_stage&&(c.prefix_demand||c.expert_copy_stage||c.expert_copy_runs||c.expert_prefetch))
            throw std::invalid_argument("prefix stage requires prefix demand/copy stage/copy runs/prefetch disabled");
        if (const char* raw = std::getenv("STRATA_QWEN35_GU_COPY_PIPELINE")) {
            const std::string value(raw);
            if(value!="0"&&value!="32")
                throw std::invalid_argument("GU copy pipeline requires0 or32");
            c.gu_copy_wave=std::stoi(value);
        }
        if(c.gu_copy_wave&&(!c.prefix_demand||c.prefix_stage||c.expert_copy_stage||c.expert_copy_runs||c.expert_prefetch))
            throw std::invalid_argument("GU copy pipeline requires direct prefix only");
        if (const char* raw = std::getenv("STRATA_QWEN35_PP_MAPPED")) {
            const std::string value(raw);
            if (value != "0" && value != "1") throw std::invalid_argument("mapped PP requires0 or1");
            c.mapped_pp = value == "1";
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
        if (const char* raw = std::getenv("STRATA_QWEN35_ATTN_TC_COMPACT")) {
            const std::string value(raw);
            if (value != "0" && value != "1")
                throw std::invalid_argument("attention register accumulation requires0 or1");
            c.attention_tc_compact = value == "1";
        }
        if (c.attention_tc_compact && c.attention_tc_scratch_reuse)
            throw std::invalid_argument("attention register accumulation and scratch reuse are exclusive");
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
        if (const char* raw = std::getenv("STRATA_QWEN35_GDN_COLUMN_TILE")) {
            const std::string value(raw);
            if (value != "1" && value != "2" && value != "4" && value != "8")
                throw std::invalid_argument("GDN column tile requires1,2,4 or8");
            c.gdn_column_tile = std::stoi(value); // Explicit tile overrides legacy bool.
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
        const int original = explicit_limits ? ubatch_size : tensor && tokens >= 1024 ? 1024 : 16;
        return detail::pp_workset2048_capacity(original, tokens);
    }
};
} // namespace strata::qwen35moe
