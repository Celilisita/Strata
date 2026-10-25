#include "gdn_long_chunk_policy.hpp"
#pragma once
#include "strata/qwen35/native/model.hpp"
#include "strata/qwen35/native/dense_gemm.hpp"
#include "strata/qwen35/native/gdn_vectorized.hpp"
#include "strata/qwen35/native/attention_gather.hpp"
#include "strata/qwen35/native/attention_keys64.hpp"
#include "strata/qwen35/native/attention_verify_grouped.hpp"
#include "strata/qwen35/native/attention_verify_padded.hpp"
#include "ffn_shape_policy.hpp"
#include "compact_attention.hpp"
#include "attention_pipeline_policy.hpp"
#include <cstdlib>
#include <cstdio>
#include <string>
#include "gdn_float_value_service.hpp"
#include "gdn_weighted_qk_policy.hpp"
#include "pv_reuse_policy.hpp"
#include "row_inverse_policy.hpp"
#include "q6_warp_policy.hpp"
#include "fp16pv_policy.hpp"
#include "resident_half_request.hpp"
#include "rescale_skip_policy.hpp"
#include "attention_score_reuse_policy.hpp"
#include "inputnorm_policy.hpp"
#include "inputnorm_long_policy.hpp"
#include "ram_score_policy.hpp"
#include "page_row_policy.hpp"
#include "ram_batch_select_policy.hpp"
#include "ram40_triple_wave_policy.hpp"
#include "ram40_prefix512_policy.hpp"
#include "regp_policy.hpp"
#include "fp8_pp_policy.hpp"

namespace strata::qwen35::native {
inline DenseLimits service_limits(int context, bool ram) {
    DenseLimits limits;
    limits.context = context;
    int device = 0;
    check_cuda(cudaGetDevice(&device), "Dense profile device");
    const auto caps = capabilities(device);
    const bool volta = caps.major == 7 && caps.minor == 0;
    if (!volta && caps.major < 8)
        throw std::invalid_argument("Dense service requires SM70 or the SM80+ staged provider");
    // Volta's prepared PP32, joint and folded layouts keep their original admission.
    // Other devices use the existing raw staged inventory with native tiny/vector
    // operations and the architecture-selected cuBLAS/CUTLASS long-PP service.
    limits.native32 = volta; limits.joint_ffn = volta;
    limits.joint_projections = volta; limits.folded_ffn = volta;
    std::fprintf(stderr, "strata_dense_profile sm=%d.%d native32=%d long_pp=%s\n",
                 caps.major, caps.minor, int(volta), volta ? "volta-folded" : "staged-cublas-cutlass");
    limits.kv_q8 = true; limits.kv_initial_tokens = (std::min)(8192, context);
    const char *gpu_rows=std::getenv("STRATA_DENSE_RAM_GPU_ROWS");
    const std::string rows=gpu_rows?gpu_rows:"32768";
    if(rows!="32768"&&rows!="40960")throw std::invalid_argument("RAM GPU rows must be32768 or40960");
    if(rows=="40960"&&(!ram||caps.major!=12||caps.minor!=0))
        throw std::invalid_argument("RAM40960 requires admitted SM120 RAM KV profile");
    configure_phase_kv(ram&&rows=="40960"&&caps.major==12&&caps.minor==0);
    if (ram) {
        limits.kv_ram_tokens = rows=="40960"?40960:32768;
        limits.kv_select_tokens = 8192;
        limits.kv_keep_queries = 3;
    }
    return limits;
}
inline void service_verify(const float* q, const void* k, const void* v, float* out,
                           void* scratch, size_t bytes, PrefillAttentionShape s,
                           int partitions, cudaStream_t stream) {
    auto kernel = s.tokens == 2 || s.capacity >= 16384
        ? verify_attention_grouped_q8 : verify_attention_padded_q8;
    if(kernel==verify_attention_grouped_q8&&
       use_attention_score_reuse(s,partitions,bytes,attention_score_reuse_direct_resident()))
        kernel=verify_attention_grouped_score_q8;
    kernel(q, k, v, out, scratch, bytes, s, partitions, stream);
}
inline void service_prefill_compact(const float* q,const void* k,const void* v,float* out,
                                    void* scratch,size_t bytes,PrefillAttentionShape s,
                                    int rows,cudaStream_t stream) {
    if(use_fp16pv(s)){
        if(try_fp8_pp_attention(q,k,v,out,scratch,bytes,s,use_rescale_skip(s),stream))return;
        ++fp16pv_submissions();
        if(use_rescale_skip(s)){++rescale_skip_submissions();rescale_skip_attention(q,k,v,out,scratch,bytes,s,stream);}
        else fp16pv_attention(q,k,v,out,scratch,bytes,s,stream);
        return;
    }
    if(use_row_inverse(s)){row_inverse_attention(q,k,v,out,scratch,bytes,s,stream);return;}
    if(use_pv_reuse(s)){pv_reuse_attention(q,k,v,out,scratch,bytes,s,stream);return;}
    const int mode=attention_pipeline_mode(s.query_start);
    if(mode)pipelined_attention(q,k,v,out,scratch,bytes,s,mode,stream);
    else compact_attention(q,k,v,out,scratch,bytes,s,rows,true,stream);
}
inline bool use_compact_prefill(int prompt) {
    if(prompt<1280)return false;
    const char* control=std::getenv("STRATA_DENSE_SM120_COMPACT_ATTENTION");
    if(control&&control[0]=='0')return false;
    int device=0;check_cuda(cudaGetDevice(&device),"compact prefill admission device");
    const auto caps=capabilities(device);
    return caps.major==12&&caps.minor==0;
}
inline void apply_service_profile(DenseModel& model, DenseGemm& shape, int prompt, int proposals, bool qwen36_profile=false) {
    configure_gdn_long_profile(qwen36_profile);
    const char *score=std::getenv("STRATA_DENSE_ATTENTION_SCORE_REUSE");
    if(score&&std::string(score)!="0"&&std::string(score)!="1"&&std::string(score)!="auto")
        throw std::invalid_argument("attention score reuse mode");
    int device=0;check_cuda(cudaGetDevice(&device),"score reuse admission device");
    const auto caps=capabilities(device);
    configure_attention_score_reuse_request(!score||score[0]!='0',caps.major,caps.minor);
    if(attention_score_reuse_request().enabled){
        const auto resource=verify_grouped_score_resources();
        if(resource.local_bytes||resource.blocks_per_SM<1)throw std::runtime_error("score reuse resource budget");
    }
    nvfp4_configure_half_request(shape,model.attention_work().ram_request_bound()!=0,prompt);
    configure_rescale_skip_request(model.attention_work().ram_request_bound()!=0,prompt);
    configure_q6_warp_head();
    configure_fp16pv_request(model.attention_work().ram_request_bound()!=0);
    model.attention_work().use_verify_split(true);
    model.attention_work().use_verify_grid(prompt >= 8192);
    model.use_batch_head(true); model.enable_tiny_batches();
    if (proposals == 1) model.enable_tiny_row_ffn(); else model.enable_tiny_meta_ffn();
    model.use_pp_gemm(&shape); model.use_pp_gdn_rows(4);
    const bool f32_value=use_gdn_float_value(prompt);
    configure_gdn_delta_request(f32_value&&gdn_register_requested());
    configure_gdn_qk_group_request(f32_value&&gdn_register_requested()&&gdn_delta_active());
    configure_gdn_kreg_request(f32_value&&gdn_register_requested()&&gdn_delta_active()&&gdn_qk_group_active());
    void*weighted_arena=nullptr;size_t weighted_capacity=0;
    const bool weighted_kreg=f32_value&&use_gdn_kreg({128,128,16,48,8192});
    if(gdn_weighted_qk_requested()&&weighted_kreg&&gdn_weighted_request(prompt)&&!model.visual_positions()&&caps.major==12&&caps.minor==0){
        const auto lease=model.attention_scratch();weighted_arena=lease.first;weighted_capacity=lease.second;
    }
    configure_gdn_weighted_qk_request(weighted_kreg,prompt,!model.visual_positions(),caps.major,caps.minor,weighted_arena,weighted_capacity);
    configure_inputnorm_request(inputnorm_request_admit(prompt));
    configure_ram_score_request(true);
    configure_page_row_request(true);
    configure_ram_batch_select_request(prompt>8192&&proposals==2);
    configure_ram40_triple_wave_request(prompt>8192&&proposals==2);
    configure_ram40_prefix512_request(prompt>8192&&proposals==2);
    if(f32_value){gdn_float_value_prepare_configure();gdn_float_value_scan_configure();
        if(gdn_register_requested())gdn_register_prepare_configure();}
    model.use_pp_gdn_kernel(f32_value ? service_gdn_float_value : gdn_recurrence_vectorized,
                            f32_value ? service_gdn_float_workspace : gdn_vectorized_workspace_bytes);
    configure_prefill_gather(48);
    const bool compact=use_compact_prefill(prompt);
    configure_regp_request(compact&&fp16pv_request_active()&&prompt==8192);
    configure_fp8_pp_request(compact&&fp16pv_request_active()&&!model.visual_positions()&&!regp_stats().admitted,prompt);
    if(compact){
        configure_compact_attention(64,true);
        if(fp16pv_request_active()){configure_fp16pv_attention();prepare_rescale_skip();}
        if(pv_reuse_enabled()){configure_pv_reuse_attention();
            if(row_inverse_enabled())configure_row_inverse_attention();}
        for(int start:{0,8192})if(const int mode=attention_pipeline_mode(start))configure_pipelined_attention(mode);
    }
    else if(prompt>=1280)configure_prefill_keys64(48);
    model.attention_work().use_pp_attention_kernel(compact ? service_prefill_compact :
        (prompt>=1280 ? prefill_attention_keys64 : prefill_attention_gather),compact ? 64 : 48);
    const char *q8_direct=std::getenv("STRATA_DENSE_Q8_ATTENTION");
    if(q8_direct&&q8_direct[0]=='1'&&prompt>=1280&&!compact){
        configure_prefill_keys64_q8(48);
        model.attention_work().use_pp_attention_kernel_q8(prefill_attention_keys64_q8,48);
    }
    model.attention_work().use_verify_attention_kernel(service_verify);
}
} // namespace strata::qwen35::native
