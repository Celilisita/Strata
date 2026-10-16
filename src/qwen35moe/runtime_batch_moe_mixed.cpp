#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <array>
#include <chrono>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::qwen35moe {

void Runtime::batch_moe_mixed(size_t il, const LayerWeights& layer,
                              const std::vector<int32_t>& ids, Metrics& metrics) {
    if (ids.size() != 8 || !cpu_ || cpu_->norm.size() != 2048)
        throw std::invalid_argument("invalid mixed Qwen35 expert workset");
    auto& b = batch_;
    auto& c = *cpu_;
    const auto& source = expert_sources_[il];
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    std::array<int, 8> gpu_rank{}, cpu_rank{};
    int gpu_count = 0, cpu_count = 0;
    for (int rank = 0; rank < 8; ++rank) {
        const int id = ids[size_t(rank)];
        if (id < 0 || id >= 256) throw std::runtime_error("invalid mixed expert ID");
        if (expert_cache_.key_to_slot[il * 256 + size_t(id)] >= 0)
            gpu_rank[size_t(gpu_count++)] = rank;
        else cpu_rank[size_t(cpu_count++)] = rank;
    }
    if (!gpu_count || !cpu_count)
        throw std::logic_error("mixed Qwen35 workset requires both providers");

    for (int entry = 0; entry < gpu_count; ++entry) {
        const int id = ids[size_t(gpu_rank[size_t(entry)])];
        const auto view = expert_view(il, id, source, metrics);
        b.host_expert_addresses[size_t(entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.gate));
        b.host_expert_addresses[size_t(gpu_count + entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.up));
        b.host_expert_rows[size_t(entry)] = 0;
        b.host_expert_rows[size_t(gpu_count + entry)] = 0;
        b.host_down_addresses[size_t(entry)] =
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(view.down));
        b.host_down_rows[size_t(entry)] = entry;
    }
    cuda_check(cudaMemcpyAsync(b.expert_addresses.data(), b.host_expert_addresses.data(),
                               size_t(gpu_count * 2) * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, stream_), "mixed gate/up addresses");
    cuda_check(cudaMemcpyAsync(b.expert_rows.data(), b.host_expert_rows.data(),
                               size_t(gpu_count * 2) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_), "mixed gate/up rows");
    cuda_check(cudaMemcpyAsync(b.down_addresses.data(), b.host_down_addresses.data(),
                               size_t(gpu_count) * sizeof(unsigned long long),
                               cudaMemcpyHostToDevice, stream_), "mixed down addresses");
    cuda_check(cudaMemcpyAsync(b.down_rows.data(), b.host_down_rows.data(),
                               size_t(gpu_count) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream_), "mixed down rows");
    batch_quantize_q8(f(b.norm), 1, 2048);
    kernels::native_mmvq_grouped(12, b.expert_addresses.as<unsigned long long>(),
                                 b.expert_rows.as<int32_t>(), b.input_q8.data(),
                                 f(b.expert_gu), 2048, 512, gpu_count * 2, stream_);
    swiglu(f(b.expert_gu), f(b.expert_gu) + size_t(gpu_count) * 512,
           f(b.expert_hidden), gpu_count * 512, stream_);
    kernels::native_quantize_q8_1(f(b.expert_hidden), b.expert_q8.data(),
                                   512, gpu_count, stream_);
    const int down_type = int(source.down_exps->type);
    kernels::native_mmvq_grouped(down_type, b.down_addresses.as<unsigned long long>(),
                                 b.down_rows.as<int32_t>(), b.expert_q8.data(),
                                 f(b.expert_mixed_parts), 512, 2048, gpu_count, stream_);
    for (int entry = 0; entry < gpu_count; ++entry)
        cuda_check(cudaMemcpyAsync(f(b.expert_parts) + size_t(gpu_rank[size_t(entry)]) * 2048,
                                   f(b.expert_mixed_parts) + size_t(entry) * 2048,
                                   2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
                   "scatter mixed GPU expert");

    batch_project_native(layer.shared_gate, f(b.shared_gate), 1, 2048, 512);
    batch_project_native(layer.shared_up, f(b.shared_up), 1, 2048, 512);
    swiglu(f(b.shared_gate), f(b.shared_up), f(b.shared_hidden), 512, stream_);
    kernels::native_quantize_q8_1(f(b.shared_hidden), b.shared_q8.data(),
                                   512, 1, stream_);
    kernels::native_mmvq(layer.shared_down.type, layer.shared_down.ptr(),
                         b.shared_q8.data(), f(b.shared_output), 512, 2048, 1, stream_);
    gate_shared_batch(f(b.shared_output), f(b.logits), 1, stream_);

    std::chrono::steady_clock::time_point prepare_start;
    if (decode_phase_ && tg_diag_) prepare_start = std::chrono::steady_clock::now();
    std::string error;
    kernels::cpu::NativeFmt fmt;
    if (!kernels::cpu::native_fmt(12, down_type, 2048, 512, fmt, error))
        throw std::runtime_error(error);
    c.activations.resize(fmt.act_bytes);
    c.parts.resize(8 * 2048);
    kernels::cpu::native_quant_act(fmt, c.norm.data(), c.activations.data());
    const auto* gate = gguf_.tensor_data(*source.gate_exps);
    const auto* up = gguf_.tensor_data(*source.up_exps);
    const auto* down = gguf_.tensor_data(*source.down_exps);
    const size_t down_bytes = fmt.bytes - fmt.down_off;
    const auto pack_start = std::chrono::steady_clock::now();
    for (int entry = 0; entry < cpu_count; ++entry) {
        const int rank = cpu_rank[size_t(entry)];
        const int id = ids[size_t(rank)];
        auto& job = c.jobs[size_t(entry)];
        job = {};
        job.gate = gate + size_t(id) * fmt.up_off;
        job.up = up + size_t(id) * fmt.up_off;
        job.down = down + size_t(id) * down_bytes;
        job.nt = 1;
        job.nact[0] = c.activations.data();
        job.out[0] = c.parts.data() + size_t(rank) * 2048;
    }
    const auto pool_start = std::chrono::steady_clock::now();
    if (decode_phase_ && tg_diag_)
        metrics.diag_cpu_prepare_ms += std::chrono::duration<double, std::milli>(
            pool_start - prepare_start).count();
    metrics.cpu_pack_ms += std::chrono::duration<double, std::milli>(
        pool_start - pack_start).count();
    c.pool.run_split_multi_native(fmt, c.jobs.data(), cpu_count);
    metrics.cpu_pool_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pool_start).count();
    metrics.cpu_expert_read += size_t(cpu_count) * fmt.bytes;
    std::chrono::steady_clock::time_point result_start;
    if (decode_phase_ && tg_diag_) result_start = std::chrono::steady_clock::now();
    for (int entry = 0; entry < cpu_count; ++entry) {
        const size_t offset = size_t(cpu_rank[size_t(entry)]) * 2048;
        cuda_check(cudaMemcpyAsync(f(b.expert_parts) + offset, c.parts.data() + offset,
                                   2048 * sizeof(float), cudaMemcpyHostToDevice, stream_),
                   "scatter mixed CPU expert");
        metrics.cpu_result_h2d += 2048 * sizeof(float);
    }
    if (decode_phase_ && tg_diag_)
        metrics.diag_result_h2d_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - result_start).count();
    std::chrono::steady_clock::time_point join_start;
    if (decode_phase_ && tg_diag_) join_start = std::chrono::steady_clock::now();
    moe_combine_batch(f(b.expert_parts), f(b.weights), f(b.shared_output),
                      f(b.mixer), f(b.hidden), 1, stream_);
    if (decode_phase_ && tg_diag_)
        metrics.diag_join_submit_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - join_start).count();
    metrics.expert_launches += 5 + 4;
}

} // namespace strata::qwen35moe
