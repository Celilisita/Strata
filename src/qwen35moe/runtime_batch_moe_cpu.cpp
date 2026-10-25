#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/nvfp4_expert_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen35moe {

void Runtime::batch_moe_cpu(size_t il, const LayerWeights& layer, int count,
                            const std::string& reference, Metrics& metrics) {
    const auto ids = batch_route(layer, count, reference, metrics, true);
    batch_moe_cpu_ready(il, layer, count, reference, metrics, ids);
}

void Runtime::batch_moe_cpu_ready(size_t il, const LayerWeights& layer, int count,
                                  const std::string& reference, Metrics& metrics,
                                  const std::vector<int32_t>& ids) {
    std::chrono::steady_clock::time_point prepare_start;
    if (decode_phase_ && tg_diag_) prepare_start = std::chrono::steady_clock::now();
    auto& b = batch_;
    auto& c = *cpu_;
    const auto& source = expert_sources_[il];
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    if (ids.size() != size_t(count) * 8 || c.norm.size() != size_t(count) * 2048)
        throw std::invalid_argument("CPU expert route count mismatch");

    std::string error;
    kernels::cpu::NativeFmt fmt;
    if (!kernels::cpu::native_fmt(int(source.gate_exps->type), int(source.down_exps->type), 2048, 512, fmt, error))
        throw std::runtime_error(error);
    c.activations.resize(size_t(count) * fmt.act_bytes);
    c.parts.resize(size_t(count) * 8 * 2048);
    const auto* gate = gguf_.tensor_data(*source.gate_exps);
    const auto* up = gguf_.tensor_data(*source.up_exps);
    const auto* down = gguf_.tensor_data(*source.down_exps);
    const size_t down_bytes = fmt.bytes - fmt.down_off;
    if (decode_phase_ && tg_diag_)
        metrics.diag_cpu_prepare_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prepare_start).count();

    // Shared-expert GPU work runs while the host pool evaluates routed experts.
    batch_quantize_q8(f(b.norm), count, 2048);
    batch_project_native(layer.shared_gate, f(b.shared_gate), count, 2048, 512);
    batch_project_native(layer.shared_up, f(b.shared_up), count, 2048, 512);
    swiglu(f(b.shared_gate), f(b.shared_up), f(b.shared_hidden), count * 512, stream_);
    const size_t row_bytes = kernels::native_q8_1_bytes(512);
    for (int first = 0; first < count; first += 8) {
        const int ncols = count - first < 8 ? count - first : 8;
        kernels::native_quantize_q8_1(f(b.shared_hidden) + size_t(first) * 512,
                                       b.shared_q8.as<uint8_t>() + size_t(first) * row_bytes,
                                       512, ncols, stream_);
        kernels::native_mmvq(layer.shared_down.type, layer.shared_down.ptr(),
                             b.shared_q8.as<uint8_t>() + size_t(first) * row_bytes,
                             f(b.shared_output) + size_t(first) * 2048,
                             512, 2048, ncols, stream_);
    }
    gate_shared_batch(f(b.shared_output), f(b.logits), count, stream_);

    if (decode_phase_ && tg_diag_) prepare_start = std::chrono::steady_clock::now();
    for (int t = 0; t < count; ++t) {
        auto* act = c.activations.data() + size_t(t) * fmt.act_bytes;
        kernels::cpu::native_quant_act(fmt, c.norm.data() + size_t(t) * 2048, act);
        const auto pack_start = std::chrono::steady_clock::now();
        for (int rank = 0; rank < 8; ++rank) {
            const int id = ids[size_t(t) * 8 + rank];
            if (id < 0 || id >= 256) throw std::runtime_error("invalid CPU expert ID");
            auto& job = c.jobs[size_t(rank)];
            job = {};
            job.gate = gate + size_t(id) * fmt.up_off;
            job.up = up + size_t(id) * fmt.up_off;
            job.down = down + size_t(id) * down_bytes;
            if(nvfp4_){const auto scales=nvfp4_->inventory.view(il,id);
                job.gate_scale=scales.gate_scale;job.up_scale=scales.up_scale;job.down_scale=scales.down_scale;}
            job.nt = 1;
            job.nact[0] = act;
            job.out[0] = c.parts.data() + (size_t(t) * 8 + rank) * 2048;
        }
        const auto pool_start = std::chrono::steady_clock::now();
        metrics.cpu_pack_ms += std::chrono::duration<double, std::milli>(pool_start - pack_start).count();
        if (decode_phase_ && tg_diag_)
            metrics.diag_cpu_prepare_ms += std::chrono::duration<double, std::milli>(
                pool_start - prepare_start).count();
        c.pool.run_split_multi_native(fmt, c.jobs.data(), 8);
        metrics.cpu_pool_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pool_start).count();
        metrics.cpu_expert_read += size_t(8) * fmt.bytes;
        if (decode_phase_ && tg_diag_) prepare_start = std::chrono::steady_clock::now();
    }
    std::chrono::steady_clock::time_point result_start;
    if (decode_phase_ && tg_diag_) result_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpyAsync(b.expert_parts.data(), c.parts.data(),
                                c.parts.size() * sizeof(float), cudaMemcpyHostToDevice, stream_),
               "CPU expert results");
    if (decode_phase_ && tg_diag_)
        metrics.diag_result_h2d_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - result_start).count();
    metrics.cpu_result_h2d += c.parts.size() * sizeof(float);
    std::chrono::steady_clock::time_point join_start;
    if (decode_phase_ && tg_diag_) join_start = std::chrono::steady_clock::now();
    moe_combine_batch(f(b.expert_parts), f(b.weights), f(b.shared_output),
                      f(b.mixer), f(b.hidden), count, stream_);
    if (decode_phase_ && tg_diag_)
        metrics.diag_join_submit_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - join_start).count();
    metrics.expert_launches += 3 + size_t((count + 7) / 8) * 4;
    if (!reference.empty()) {
        batch_probe_.shared = batch_stage_error(reference, "ffn_shexp_gated-0",
                                                 f(b.shared_output), 2048);
        batch_probe_.moe_out = batch_stage_error(reference, "l_out-0", f(b.hidden), 2048);
    }
}

} // namespace strata::qwen35moe
