#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dequant_bf16.hpp"

#include <cublas_v2.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace strata::qwen35moe {

void Runtime::batch_gdn(LayerWeights& layer, int count, const std::string& reference) {
    auto& b = batch_;
    auto& w = work_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    if (dense_profile_.active() && (!prefill_scan_ || !reference.empty() || paired_session_))
        throw std::invalid_argument("dense profile requires canonical GDN scan");
    // Explicit recurrent role: full Attention query has the same matrix shape.
    if (!batch_project_mmq(layer.qkv, f(b.qkv), count, 2048, 8192, ProjectionRole::gdn_qkv))
        batch_project_native(layer.qkv, f(b.qkv), count, 2048, 8192, false);
    dense_profile_.mark(DenseStage::Query, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->gdn_detail("qkv", b.qkv.data(), size_t(count) * 8192 * 4, false, stream_);
    if (!reference.empty())
        batch_probe_.qkv = batch_stage_error(reference, "linear_attn_qkv_mixed-0", f(b.qkv), 8192);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("input_q8", b.input_q8.data(),
            kernels::native_q8_1_bytes(2048) * count, true, stream_);
    batch_project_native(layer.gate, f(b.z), count, 2048, 4096, false);
    dense_profile_.mark(DenseStage::Key, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("z", b.z.data(), size_t(count) * 4096 * 4, false, stream_);
    if (!reference.empty())
        batch_probe_.z = batch_stage_error(reference, "z-0", f(b.z), 4096);
    batch_project_native(layer.alpha, f(b.alpha), count, 2048, 32);
    batch_project_native(layer.beta, f(b.beta), count, 2048, 32);
    dense_profile_.mark(DenseStage::Value, stream_);
    if (checkpoint_sink_) {
        checkpoint_sink_->gdn_detail("alpha", b.alpha.data(), size_t(count) * 32 * 4, false, stream_);
        checkpoint_sink_->gdn_detail("beta", b.beta.data(), size_t(count) * 32 * 4, false, stream_);
    }
    if (paired_session_) batch_gdn_sessions(layer, count);
    else if (prefill_scan_ && reference.empty()) batch_gdn_scan(layer, count);
    else for (int t = 0; t < count; ++t) {
        float* qkv = f(b.qkv) + size_t(t) * 8192;
        float* z = f(b.z) + size_t(t) * 4096;
        float* alpha = f(b.alpha) + size_t(t) * 32;
        float* beta = f(b.beta) + size_t(t) * 32;
        kernels::native_gdn_conv_silu(f(layer.history), qkv, layer.conv.f32(),
                                       f(w.raw), f(w.activated), 8192, 4, stream_);
        gdn_l2_norm(f(w.activated), 32, 128, float(spec_.rms_epsilon), stream_, 1, 0, prefill_config_.gdn_warp_norm);
        kernels::native_gdn_beta_gate(beta, 32, stream_);
        kernels::native_gdn_gate(alpha, layer.dt.f32(), layer.a.f32(),
                                 f(w.gate), 32, stream_);
        kernels::native_gdn_step(f(layer.state), f(w.activated), f(w.activated) + 2048,
                                 f(w.activated) + 4096, f(w.gate), beta, f(w.core),
                                 {128, 16, 32}, stream_);
        gdn_post_silu(f(w.core), z, layer.ssm_norm.f32(),
                      f(b.post) + size_t(t) * 4096, 32,
                      float(spec_.rms_epsilon), stream_);
    }
    if (!reference.empty())
        batch_probe_.gdn_final = batch_stage_error(reference, "final_output-0", f(b.post), 4096);
    if (!reference.empty()) {
        kernels::f32_to_f16_bulk(f(b.post), b.input_half.as<uint16_t>(),
                                 int64_t(count) * 4096, stream_);
        b.gemm.native(b.input_half.as<uint16_t>(), layer.ssm_out.type, layer.ssm_out.ptr(),
                      f(b.mixer), count, 2048, 4096);
        batch_probe_.mixer = batch_stage_error(reference, "linear_attn_out-0", f(b.mixer), 2048);
        kernels::native_quantize_q8_1(f(b.post), w.q8_wide.data(), 4096, 1, stream_);
        project(layer.ssm_out, w.q8_wide.data(), f(w.mixer), 4096, 2048);
        batch_probe_.native_mixer = batch_stage_error(reference, "linear_attn_out-0", f(w.mixer), 2048);
        DeviceBuffer f32_weight(size_t(2048) * 4096 * sizeof(float));
        kernels::dequant_f32(layer.ssm_out.type, layer.ssm_out.ptr(), 0, 2048, 4096,
                             f32_weight.as<float>(), stream_);
        cublasHandle_t handle = nullptr;
        if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch diagnostic cublasCreate failed");
        if (cublasSetStream(handle, stream_) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch diagnostic cublasSetStream failed");
        const float one = 1.0f, zero = 0.0f;
        const auto status = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2048, 1, 4096,
                                        &one, f32_weight.as<float>(), 4096, f(b.post), 4096,
                                        &zero, f(w.mixer), 2048);
        if (status != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch diagnostic cublasSgemm failed");
        batch_probe_.f32_mixer = batch_stage_error(reference, "linear_attn_out-0", f(w.mixer), 2048);
        batch_quantize_to_half(f(b.post), count, 4096);
        b.gemm.native(b.input_half.as<uint16_t>(), layer.ssm_out.type,
                      layer.ssm_out.ptr(), f(b.mixer), count, 2048, 4096);
        batch_probe_.q8_gemm_mixer = batch_stage_error(reference, "linear_attn_out-0",
                                                      f(b.mixer), 2048);

        std::vector<float> oracle_post(4096);
        const auto path = std::filesystem::path(reference) / "final_output-0.bin";
        std::ifstream input(path, std::ios::binary);
        if (!input.read(reinterpret_cast<char*>(oracle_post.data()),
                        std::streamsize(oracle_post.size() * sizeof(float))))
            throw std::runtime_error("missing oracle GDN output: " + path.string());
        cuda_check(cudaMemcpyAsync(w.post.data(), oracle_post.data(),
                                    oracle_post.size() * sizeof(float),
                                    cudaMemcpyHostToDevice, stream_), "inject oracle GDN output");
        kernels::native_quantize_q8_1(f(w.post), w.q8_wide.data(), 4096, 1, stream_);
        project(layer.ssm_out, w.q8_wide.data(), f(w.mixer), 4096, 2048);
        batch_probe_.oracle_native_mixer = batch_stage_error(reference, "linear_attn_out-0",
                                                             f(w.mixer), 2048);
        kernels::f32_to_f16_bulk(f(w.post), b.input_half.as<uint16_t>(), 4096, stream_);
        b.gemm.native(b.input_half.as<uint16_t>(), layer.ssm_out.type,
                      layer.ssm_out.ptr(), f(w.mixer), 1, 2048, 4096);
        batch_probe_.oracle_f16_mixer = batch_stage_error(reference, "linear_attn_out-0",
                                                          f(w.mixer), 2048);
        batch_quantize_to_half(f(w.post), 1, 4096);
        b.gemm.native(b.input_half.as<uint16_t>(), layer.ssm_out.type,
                      layer.ssm_out.ptr(), f(w.mixer), 1, 2048, 4096);
        batch_probe_.oracle_q8_gemm_mixer = batch_stage_error(reference, "linear_attn_out-0",
                                                              f(w.mixer), 2048);
        if (cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, 2048, 1, 4096,
                        &one, f32_weight.as<float>(), 4096, f(w.post), 4096,
                        &zero, f(w.mixer), 2048) != CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("batch diagnostic oracle cublasSgemm failed");
        batch_probe_.oracle_f32_mixer = batch_stage_error(reference, "linear_attn_out-0",
                                                          f(w.mixer), 2048);
        cublasDestroy(handle);
    }
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("gdn_post", b.post.data(), size_t(count) * 4096 * 4, false, stream_);
    batch_quantize_q8(f(b.post), count, 4096);
    dense_profile_.mark(DenseStage::OutputQuant, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("post_q8", b.input_q8.data(),
            kernels::native_q8_1_bytes(4096) * count, true, stream_);
    if (!batch_project_mmq(layer.ssm_out, f(b.mixer), count, 4096, 2048, ProjectionRole::gdn_output))
        batch_project_native(layer.ssm_out, f(b.mixer), count, 4096, 2048, false);
    dense_profile_.mark(DenseStage::OutputProjection, stream_);
    if (checkpoint_sink_)
        checkpoint_sink_->batch_stage("ssm_mixer", b.mixer.data(), size_t(count) * 2048 * 4, false, stream_);
}

} // namespace strata::qwen35moe
