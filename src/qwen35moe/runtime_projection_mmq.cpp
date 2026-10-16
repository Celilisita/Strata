#include "strata/qwen35moe/runtime.hpp"
#include "strata/prefill/native_q6_accum_mmq.hpp"
#include <stdexcept>
#include "strata/kernels/elementwise.hpp"

namespace strata::qwen35moe {
bool Runtime::batch_project_mmq(const GpuWeight& weight, float* output,
                                int count, int n_in, int n_out, ProjectionRole role) {
    // Independently measured roles. Other projections retain their provider.
    const bool output_gate = prefill_config_.mmq_projection && weight.type == 12 &&
        n_in == 2048 && n_out == 4096;
    const bool recurrent_qkv = role == ProjectionRole::gdn_qkv && prefill_config_.gdn_qkv_mmq && weight.type == 14 &&
        n_in == 2048 && n_out == 8192;
    const bool recurrent_qkv_q4 = role == ProjectionRole::gdn_qkv && prefill_config_.gdn_qkv_q4_mmq && weight.type == 12 &&
        n_in == 2048 && n_out == 8192;
    const bool recurrent_out = role == ProjectionRole::gdn_output && prefill_config_.gdn_out_mmq &&
        weight.type == 14 && n_in == 4096 && n_out == 2048;
    if (decode_phase_ || count < 1024 || (!output_gate && !recurrent_qkv && !recurrent_qkv_q4 && !recurrent_out)) return false;
    const size_t bytes = prefill::mmq::q8_bytes(count, n_in);
    if (!projection_mmq_ || bytes > batch_.input_half.bytes())
        throw std::runtime_error("unprepared bounded GDN MMQ workspace");
    // This conversion view is dead until the next tensor projection. Its tag must
    // be invalidated so a later shared-expert GEMM rebuilds its F16 activations.
    batch_.half_width = 0; batch_.half_count = 0;
    if (weight.type == 12)
        prefill::mmq::repack_native_q8_code_sum(batch_.input_q8.data(), batch_.input_half.data(), n_in, count, stream_);
    else if (recurrent_out)
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
    if (output_gate) { projection_mmq_->run_code_sum(product, stream_); ++batch_.mmq_projection_calls; }
    else if (recurrent_qkv_q4) { projection_mmq_->run_code_sum(product, stream_); ++batch_.qkv_q4_mmq_calls; }
    else {
        if (recurrent_out) out_launches = prefill::mmq::run_q6_canonical(product, stream_);
        else projection_mmq_->run(product, stream_);
        if (recurrent_out) ++batch_.gdn_out_mmq_calls;
        else ++batch_.qkv_mmq_calls;
    }
    dense_profile_.launches(recurrent_out ? 1+out_launches : 2);
    return true;
}
} // namespace strata::qwen35moe
