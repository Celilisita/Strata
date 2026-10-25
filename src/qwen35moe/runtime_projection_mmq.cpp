#include "strata/qwen35moe/runtime.hpp"
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include "strata/prefill/dense_f16_gemm.hpp"
#include <cstdlib>
#include <stdexcept>
#include <cstdlib>
#include <cstring>
#include "strata/prefill/native_code_sum_support.hpp"
#include "strata/kernels/elementwise.hpp"

namespace strata::qwen35moe {
namespace {
bool ordinary_mma_enabled() {
    const char* flag=std::getenv("STRATA_QWEN35_PP_ORDINARY_MMA");
    if(!flag || std::strcmp(flag,"0")==0) return false;
    if(std::strcmp(flag,"1")!=0) throw std::invalid_argument("PP_ORDINARY_MMA must be0 or1");
    return true;
}
bool q5_qkv_enabled() {
    const char* flag=std::getenv("STRATA_QWEN35_PP_GDN_Q5_QKV_MMQ");
    if(!flag || std::strcmp(flag,"0")==0) return false;
    if(std::strcmp(flag,"1")!=0) throw std::invalid_argument("PP_GDN_Q5_QKV_MMQ must be0 or1");
    return true;
}
bool ordinary_sm75_supported(int type,int outputs) {
    int device=0,major=0,minor=0;
    cuda_check(cudaGetDevice(&device),"ordinary MMQ active device");
    cuda_check(cudaDeviceGetAttribute(&major,cudaDevAttrComputeCapabilityMajor,device),"ordinary MMQ major");
    cuda_check(cudaDeviceGetAttribute(&minor,cudaDevAttrComputeCapabilityMinor,device),"ordinary MMQ minor");
    return major==7 && minor==5 && prefill::mmq::supported(type) && prefill::mmq::fits(type,outputs);
}
}
bool Runtime::batch_project_mmq(const GpuWeight& weight, float* output,
                                int count, int n_in, int n_out, ProjectionRole role) {
#ifdef STRATA_BLACKWELL_MMQ
    static const unsigned formats=prefill::dense_f16_gemm_policy(std::getenv("STRATA_QWEN35_DENSE_F16_GEMM"));
    if(formats&&!decode_phase_&&!checkpoint_sink_&&!paired_session_&&!sessions_&&!mtp_enabled_&&
       !verify_state_&&!verify_history_&&prefill_tc_&&prefill_scan_&&
       prefill::dense_f16_gemm_domain(weight.type,n_in,count,n_out)&&(weight.type!=12||count>=2048)&&
       (formats&(1u<<weight.type))) {
        auto& b=batch_;
        if(dense_profile_.active())throw std::invalid_argument("Dense F16 GEMM device launch profile is uncalibrated");
        if(!b.raw_input||b.raw_input_width!=n_in||b.raw_input_count!=count||b.gemm.stream()!=stream_)
            throw std::invalid_argument("Dense F16 GEMM originalF32 source epoch/shape");
        // RawF16 must never carry the existing reconstructedQ8 half-cache tag.
        b.half_width=0;b.half_count=0;
        prefill::run_dense_f16_gemm(b.gemm,weight.type,weight.ptr(),weight.data.bytes(),
            b.raw_input,size_t(count)*n_in*4,b.input_half.as<uint16_t>(),b.input_half.bytes(),
            output,size_t(count)*n_out*4,n_in,count,n_out);
        return true;
    }
    if (batch_project_precise_dense(weight, output, count, n_in, n_out)) return true;
    if (batch_project_blackwell(weight, output, count, n_in, n_out)) return true;
#endif
    if (decode_phase_ || count < 1024) return false;
    // Independently measured roles. Other projections retain their provider.
    const bool output_gate = prefill_config_.mmq_projection && weight.type == 12 &&
        n_in == 2048 && n_out == 4096;
    const bool recurrent_out_q4 = prefill_config_.mmq_projection && weight.type == 12 &&
        role == ProjectionRole::gdn_output && n_in == 4096 && n_out == 2048;
    bool attention_q4 = false;
    if (prefill_config_.mmq_projection && weight.type == 12 && role == ProjectionRole::generic &&
        ((n_in == 2048 && (n_out == 512 || n_out == 8192)) || (n_in == 4096 && n_out == 2048))) {
        // Generic QKV fallback has the same shape as attention query. Bind the
        // actual main-stack weight object, preserving the independent QKV flag.
        for (size_t il = 0; il < layers_.size() && il < spec_.layers.size(); ++il) {
            const auto& layer = layers_[il];
            if (!spec_.layers[il].recurrent &&
                (&weight == &layer.q || &weight == &layer.k || &weight == &layer.v || &weight == &layer.out)) {
                attention_q4 = true;
                break;
            }
        }
    }
    const bool q4_projection = output_gate || recurrent_out_q4 || attention_q4;
    const bool recurrent_qkv = role == ProjectionRole::gdn_qkv && prefill_config_.gdn_qkv_mmq && weight.type == 14 &&
        n_in == 2048 && n_out == 8192;
    const bool recurrent_qkv_q4 = role == ProjectionRole::gdn_qkv && prefill_config_.gdn_qkv_q4_mmq && weight.type == 12 &&
        n_in == 2048 && n_out == 8192;
    const bool recurrent_qkv_q5 = role == ProjectionRole::gdn_qkv && weight.type == 13 &&
        n_in == 2048 && n_out == 8192 && q5_qkv_enabled();
    const bool recurrent_out = role == ProjectionRole::gdn_output && prefill_config_.gdn_out_mmq &&
        weight.type == 14 && n_in == 4096 && n_out == 2048;
    if (!q4_projection && !recurrent_qkv && !recurrent_qkv_q4 && !recurrent_qkv_q5 && !recurrent_out) return false;
    bool ordinary = false;
    if (ordinary_mma_enabled() && count <= 8192 && (q4_projection || recurrent_qkv_q4 || recurrent_qkv_q5 || recurrent_out)) {
        bool bound_role = attention_q4;
        for (size_t il=0; !bound_role && il<layers_.size() && il<spec_.layers.size(); ++il) {
            if (!spec_.layers[il].recurrent) continue;
            const auto& layer=layers_[il];
            bound_role = (output_gate && role==ProjectionRole::generic && &weight==&layer.gate) ||
                (recurrent_qkv_q4 && &weight==&layer.qkv) ||
                (recurrent_qkv_q5 && il<size_t(spec_.n_layers) && &weight==&layer.qkv) ||
                ((recurrent_out_q4 || recurrent_out) && &weight==&layer.ssm_out);
        }
        ordinary = bound_role && ordinary_sm75_supported(weight.type,n_out);
    }
    // Qualification is decided before repack or scratch-tag mutation. Unsupported
    // requests retain the existing provider, including the private code-sum gate.
    if (recurrent_qkv_q5 && (!ordinary || !projection_mmq_)) return false;
    if (weight.type == 12 && !ordinary && !prefill::mmq::native_code_sum_dense_supported(n_in, n_out, count)) return false;
    bool canonical_qkv = false;
#ifdef STRATA_BLACKWELL_MMQ
    canonical_qkv = recurrent_qkv;
#endif
    const size_t bytes = prefill::mmq::q8_bytes(count, n_in);
    if (!projection_mmq_ || bytes > batch_.input_half.bytes())
        throw std::runtime_error("unprepared bounded GDN MMQ workspace");
    // This conversion view is dead until the next tensor projection. Its tag must
    // be invalidated so a later shared-expert GEMM rebuilds its F16 activations.
    batch_.half_width = 0; batch_.half_count = 0;
    if (ordinary)
        prefill::mmq::repack_native_q8(batch_.input_q8.data(), batch_.input_half.data(), weight.type, n_in, count, stream_);
    else if (weight.type == 12)
        prefill::mmq::repack_native_q8_code_sum(batch_.input_q8.data(), batch_.input_half.data(), n_in, count, stream_);
    else if (recurrent_out || canonical_qkv)
        prefill::mmq::repack_q6_canonical(batch_.input_q8.data(), batch_.input_half.data(), n_in, count, stream_);
    else
        prefill::mmq::repack_native_q8(batch_.input_q8.data(), batch_.input_half.data(), weight.type, n_in, count, stream_);
    prefill::mmq::Product product;
    product.w = weight.ptr(); product.type = weight.type;
    product.w_rows = n_out; product.w_cols = n_in; product.n = 1;
    product.expert_bytes = prefill::mmq::matrix_bytes(weight.type, n_out, n_in);
    product.xq = batch_.input_half.data(); product.dst = output;
    product.total_rows = count; product.max_rows = count; product.ld_dst = n_out;
    uint64_t out_launches = 0;
    if (ordinary) {
        projection_mmq_->run(product, stream_);
        if (q4_projection) ++batch_.mmq_projection_calls;
        else if (recurrent_qkv_q4) ++batch_.qkv_q4_mmq_calls;
        else if (recurrent_qkv_q5) ++batch_.qkv_mmq_calls;
        else ++batch_.gdn_out_mmq_calls;
    }
    else if (q4_projection) { projection_mmq_->run_code_sum(product, stream_); ++batch_.mmq_projection_calls; }
    else if (recurrent_qkv_q4) { projection_mmq_->run_code_sum(product, stream_); ++batch_.qkv_q4_mmq_calls; }
    else {
        if (recurrent_out || canonical_qkv) out_launches = prefill::mmq::run_q6_canonical(product, stream_);
        else projection_mmq_->run(product, stream_);
        if (recurrent_out) ++batch_.gdn_out_mmq_calls;
        else ++batch_.qkv_mmq_calls;
    }
    dense_profile_.launches((recurrent_out || canonical_qkv) && !ordinary ? 1+out_launches : 2);
    return true;
}
} // namespace strata::qwen35moe
