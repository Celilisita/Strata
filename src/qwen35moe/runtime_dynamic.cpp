#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace strata::qwen35moe {

void Runtime::initialize_dynamic() {
    const char* flag = std::getenv("STRATA_QWEN35_TG_DYNAMIC");
    tg_dynamic_ = false;
    if (flag && std::string(flag) != "0" && std::string(flag) != "1")
        throw std::invalid_argument("TG_DYNAMIC must be0 or1");
    if (flag && std::string(flag) == "0") return;
    int device = -1;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "dynamic device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "dynamic capabilities");
    if (!cpu_ || cpu_->from_layer != 0 || !tg_hit_gpu_ || !gpu_head_ ||
        !expert_cache_.slots || batch_.expert_stage.data() || resident_gather_ ||
        !properties.canMapHostMemory || !properties.unifiedAddressing) {
        if (flag) throw std::invalid_argument("dynamic continuation requires shared cache, native CPU pool, GPU head and mapped UVA");
        return;
    }
    tg_dynamic_ = true;
    std::vector<core::NativeMissSource> sources;
    for (const auto& layer : expert_sources_) {
        core::NativeMissSource source;
        source.gate = gguf_.tensor_data(*layer.gate_exps);
        source.up = gguf_.tensor_data(*layer.up_exps);
        source.down = gguf_.tensor_data(*layer.down_exps);
        source.n_expert = int(spec_.n_expert);
        std::string error;
        if (!kernels::cpu::native_fmt(int(layer.gate_exps->type), int(layer.down_exps->type),
                                       int(spec_.n_embd), int(spec_.n_ff_exp), source.format, error))
            throw std::runtime_error(error);
        sources.push_back(source);
    }
    // Only a delivery adapter: the pool, weight views and cache remain authoritative.
    miss_service_ = std::make_unique<core::NativeMissService>(
        cpu_->pool, std::move(sources), int(spec_.n_embd), int(spec_.n_expert_used), (mtp_enabled_ || sessions_) ? 2 : 1);
    expert_cache_.resident_slots.reset(expert_cache_.key_to_slot.size() * sizeof(int32_t));
    dynamic_args_.reset(expert_sources_.size() * sizeof(core::NativeMissArgs));
    if (mtp_enabled_ || sessions_) dynamic_pair_args_.reset(dynamic_args_.bytes());
    dynamic_frame_.reset(sizeof(core::NativeMissFrame));
    dynamic_ages_.reset(expert_cache_.age.size() * sizeof(uint64_t));
}

void Runtime::start_decode(Metrics* metrics) {
    if (metrics) collect_visual_metrics(*metrics);
    if (!tg_dynamic_) { decode_phase_ = true; return; }
    cuda_check(cudaStreamSynchronize(stream_), "drain before decode epoch");
    miss_service_->stop();
    if (decode_epoch_ == (std::numeric_limits<uint32_t>::max)())
        throw std::runtime_error("decode epoch exhausted; recreate runtime");
    core::NativeMissFrame frame{};
    frame.epoch = ++decode_epoch_;
    frame.cache_tick = expert_cache_.tick;
    std::vector<core::NativeMissArgs> args(expert_sources_.size());
    int device = -1;
    cudaDeviceProp properties{};
    cuda_check(cudaGetDevice(&device), "decode device");
    cuda_check(cudaGetDeviceProperties(&properties, device), "decode clock");
    unsigned timeout_ms = 50;
    if (const char* raw = std::getenv("STRATA_QWEN35_MISS_TIMEOUT_MS")) {
        const std::string value(raw);
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::invalid_argument("invalid dynamic miss timeout");
        const auto parsed = std::stoul(value);
        if (parsed < 1 || parsed > 1000)
            throw std::invalid_argument("dynamic miss timeout must be 1..1000 ms");
        timeout_ms = unsigned(parsed);
    }
    for (size_t il = 0; il < args.size(); ++il) {
        auto& a = args[il];
        a = miss_service_->device_args();
        a.frame = dynamic_frame_.as<core::NativeMissFrame>();
        a.slots = expert_cache_.resident_slots.as<int32_t>() + il * spec_.n_expert;
        a.cache_slot_addresses = expert_cache_.data.device_addresses();
        a.cache_stride = expert_cache_.stride;
        a.ages = dynamic_ages_.as<uint64_t>();
        a.gu_bytes = kernels::native_mmvq_weight_bytes(12, int(spec_.n_embd), int(spec_.n_ff_exp));
        a.norm = batch_.norm.as<float>();
        a.gu_addresses = batch_.expert_addresses.as<unsigned long long>();
        a.down_addresses = batch_.down_addresses.as<unsigned long long>();
        a.gu_rows = batch_.expert_rows.as<int32_t>();
        a.down_rows = batch_.down_rows.as<int32_t>();
        a.layer = int(il); a.n_expert = int(spec_.n_expert);
        a.timeout_cycles = uint64_t(properties.clockRate) * timeout_ms; // kHz * ms
    }
    std::vector<core::NativeMissArgs> pair_args;
    if (dynamic_pair_args_.bytes()) {
        pair_args = args;
        for (auto& a : pair_args) { a.tokens = 2; a.ranks *= 2; }
    }
    cuda_check(cudaMemcpyAsync(dynamic_frame_.data(), &frame, sizeof(frame),
                               cudaMemcpyHostToDevice, stream_), "decode frame");
    cuda_check(cudaMemcpyAsync(expert_cache_.resident_slots.data(), expert_cache_.key_to_slot.data(),
                               expert_cache_.resident_slots.bytes(), cudaMemcpyHostToDevice, stream_),
               "decode directory snapshot");
    cuda_check(cudaMemcpyAsync(dynamic_ages_.data(), expert_cache_.age.data(), dynamic_ages_.bytes(),
                               cudaMemcpyHostToDevice, stream_), "decode cache ages");
    cuda_check(cudaMemcpyAsync(dynamic_args_.data(), args.data(), dynamic_args_.bytes(),
                               cudaMemcpyHostToDevice, stream_), "decode argument views");
    if (dynamic_pair_args_.bytes()) {
        cuda_check(cudaMemcpyAsync(dynamic_pair_args_.data(), pair_args.data(), dynamic_pair_args_.bytes(),
            cudaMemcpyHostToDevice, stream_), "decode pair arguments");
    }
    // Local vectors may be reused only after their upload finishes.
    cuda_check(cudaStreamSynchronize(stream_), "finish decode argument upload");
    miss_service_->start(frame.epoch);
    miss_accounted_ = {}; dynamic_layers_ = dynamic_accounted_layers_ = 0;
    dynamic_entries_ = dynamic_accounted_entries_ = 0;
    if (metrics) {
        metrics->control_h2d += dynamic_frame_.bytes() + dynamic_args_.bytes() +
                                expert_cache_.resident_slots.bytes() + dynamic_ages_.bytes() + dynamic_pair_args_.bytes();
        metrics->explicit_syncs += 2;
    }
    decode_phase_ = true;
}

// Transfer only the cache policy at the TG boundary. Weights/directory never change in TG.
void Runtime::finish_decode(Metrics* metrics) {
    if (metrics) collect_visual_metrics(*metrics);
    if (!tg_dynamic_ || !decode_phase_) return;
    core::NativeMissFrame frame{};
    cuda_check(cudaMemcpyAsync(&frame, dynamic_frame_.data(), sizeof(frame),
                               cudaMemcpyDeviceToHost, stream_), "finish decode frame");
    cuda_check(cudaMemcpyAsync(expert_cache_.age.data(), dynamic_ages_.data(), dynamic_ages_.bytes(),
                               cudaMemcpyDeviceToHost, stream_), "finish decode cache ages");
    cuda_check(cudaStreamSynchronize(stream_), "finish decode policy merge");
    miss_service_->stop();
    if (frame.failed) throw std::runtime_error("failed device decode epoch");
    expert_cache_.tick = frame.cache_tick;
    if (metrics) {
        collect_dynamic_metrics(*metrics);
        metrics->control_d2h += sizeof(frame) + dynamic_ages_.bytes();
        ++metrics->explicit_syncs;
    }
    decode_phase_ = false;
}

void Runtime::batch_moe_device_eager(size_t il, const LayerWeights& layer, Metrics& metrics, int count) {
    if (count < 1 || count > 2 || (count == 2 && !dynamic_pair_args_.bytes()))
        throw std::invalid_argument("unprepared dynamic pair");
    auto& b = batch_;
    route_project(layer.router.f32(), b.norm.as<float>(), b.logits.as<float>(),
                   int(spec_.n_embd), int(spec_.n_expert) + 1, stream_, count);
    auto& args = count == 2 ? dynamic_pair_args_ : dynamic_args_;
    route_top8_continue(b.logits.as<float>(), b.ids.as<int32_t>(), b.weights.as<float>(),
                        args.as<core::NativeMissArgs>() + il, stream_, count);
    batch_moe_grouped_compute(il, layer, count, "", metrics);
    ++dynamic_layers_; dynamic_entries_ += size_t(count) * spec_.n_expert_used;
}

// Called after the existing head synchronization. No device readback is added.
void Runtime::collect_dynamic_metrics(Metrics& metrics) {
    const auto current = miss_service_->snapshot();
    const auto requests = current.requests - miss_accounted_.requests;
    const auto experts = current.experts - miss_accounted_.experts;
    const auto layers = dynamic_layers_ - dynamic_accounted_layers_;
    metrics.route_d2h += current.id_bytes - miss_accounted_.id_bytes;
    metrics.cpu_norm_d2h += current.norm_bytes - miss_accounted_.norm_bytes;
    metrics.cpu_result_h2d += current.result_bytes - miss_accounted_.result_bytes;
    metrics.cpu_expert_read += current.expert_bytes - miss_accounted_.expert_bytes;
    metrics.cpu_pool_ms += current.pool_ms - miss_accounted_.pool_ms;
    metrics.cpu_gu_ms += current.gu_ms - miss_accounted_.gu_ms;
    metrics.cpu_quant_ms += current.quant_ms - miss_accounted_.quant_ms;
    metrics.cpu_down_ms += current.down_ms - miss_accounted_.down_ms;
    metrics.cpu_jobs += current.jobs - miss_accounted_.jobs;
    metrics.cpu_paired_jobs += current.paired_jobs - miss_accounted_.paired_jobs;
    metrics.host_continuations += requests;
    metrics.tg_cpu_layers += requests;
    metrics.tg_gpu_hit_layers += layers - requests;
    metrics.tg_device_layers += layers - requests;
    metrics.expert_cache_hits += dynamic_entries_ - dynamic_accounted_entries_ - experts;
    miss_accounted_ = current; dynamic_accounted_layers_ = dynamic_layers_;
    dynamic_accounted_entries_ = dynamic_entries_;
}

} // namespace strata::qwen35moe
