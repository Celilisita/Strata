#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_mmvq_bulk.hpp"
#include "strata/kernels/native_quantize_rows.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/prefill/precise.hpp"

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <cstdlib>
#include <utility>
#include <vector>

namespace strata::qwen35moe {

void Runtime::prepare_batch(int capacity) {
    if (batch_failed_) throw std::runtime_error("batch workspace failed; recreate runtime");
    if (capacity < 2 || capacity > 8192)
        throw std::invalid_argument("Qwen35 batched prefill supports 2..8192 tokens");
    if ((prefill_config_.grouped_down_mmq || prefill_config_.grouped_gu_mmq) && !batch_.down_mmq)
        batch_.down_mmq = std::make_unique<prefill::mmq::NativeGroupedMmqProvider>(stream_);
    if ((prefill_config_.mmq_projection || prefill_config_.gdn_qkv_mmq || prefill_config_.gdn_qkv_q4_mmq || prefill_config_.gdn_out_mmq) && !projection_mmq_)
        projection_mmq_ = std::make_unique<prefill::mmq::Context>();
    if (batch_.capacity) {
        if (capacity > batch_.capacity) {
            cuda_check(cudaStreamSynchronize(stream_), "drain before batch workspace growth");
            invalidate_moe_graphs();
            batch_failed_ = true;
            batch_.expert_stage.reset();
            allocate_batch(capacity);
            prepare_moe_graphs();
            batch_failed_ = false;
        }
        return;
    }
    allocate_batch(capacity);
    auto& b = batch_;
    std::string error;
    if (!b.gemm.init(stream_, int64_t(8192) * 2048, error)) throw std::runtime_error(error);
    const char* cpu_first = std::getenv("STRATA_QWEN35_CPU_FROM_LAYER");
    const int first = cpu_first ? std::stoi(cpu_first) : 0;
    if (first < 0 || first > int(spec_.layers.size()))
        throw std::invalid_argument("invalid CPU expert first layer");
    const char* hits = std::getenv("STRATA_QWEN35_TG_HIT_GPU");
    tg_hit_gpu_ = !hits || std::string(hits) != "0";
    const char* gather = std::getenv("STRATA_QWEN35_TG_DEVICE_GATHER");
    resident_gather_ = gather && std::string(gather) == "1";
    const char* diag = std::getenv("STRATA_QWEN35_TG_DIAG");
    tg_diag_ = diag && std::string(diag) == "1";
    if (tg_diag_)
        cuda_check(cudaEventCreateWithFlags(&diag_route_ready_, cudaEventDisableTiming),
                   "create Qwen35 TG diagnostic event");
    const char* pin_count = std::getenv("STRATA_QWEN35_PIN_MODEL_LAYERS");
    pin_expert_weights(pin_count ? std::stoi(pin_count) : int(expert_sources_.size()));
    if (const char* warm = std::getenv("STRATA_QWEN35_PREWARM_CACHE");
        resident_layers_ || !warm || std::string(warm) != "0")
        prewarm_expert_cache();
    if (first < int(spec_.layers.size()) && kernels::cpu::native_experts_available())
        cpu_ = std::make_unique<CpuExpertState>(first);
    if (std::getenv("STRATA_QWEN35_CUDA_EVENTS")) {
        prefill_events_.resize(expert_sources_.size());
        for (auto& group : prefill_events_)
            for (auto& event : group)
                cuda_check(cudaEventCreate(&event), "create Qwen35 prefill event");
    }
    if (const char* dense = std::getenv("STRATA_QWEN35_DENSE_EVENTS")) {
        if (std::string(dense) != "1") throw std::invalid_argument("DENSE_EVENTS must be1 or unset");
        dense_profile_.enable();
        dense_profile_.prepare(expert_sources_.size());
    }
    initialize_dynamic();
    b.capacity = capacity;
    prepare_moe_graphs();
}

void Runtime::batch_quantize_to_half(const float* input, int count, int width) {
    batch_quantize_q8(input, count, width);
    q8_1_to_f16(batch_.input_q8.data(), batch_.input_half.as<uint16_t>(),
                 count * width, stream_, count >= 1024 ? batch_.input_low.as<uint16_t>() : nullptr);
    dense_profile_.launches(1);
    batch_.half_width = width; batch_.half_count = count;
}

void Runtime::batch_quantize_q8(const float* input, int count, int width) {
    if (count < 1 || count > batch_.capacity || (width != 512 && width != 2048 && width != 4096))
        throw std::invalid_argument("invalid Qwen35 batch quantization shape");
    batch_.half_width = 0; batch_.half_count = 0;
    const char* setting = std::getenv("STRATA_QWEN35_BULK_QUANT");
    const bool bulk = !decode_phase_ && setting && std::string(setting) == "1";
    const auto launches = kernels::native_quantize_rows(input, batch_.input_q8.data(), width, count, bulk, stream_);
    dense_profile_.launches(launches);
}

void Runtime::batch_project_native(const GpuWeight& weight, float* output,
                                   int count, int n_in, int n_out, bool allow_tensor) {
    if (!allow_tensor && batch_project_mmq(weight, output, count, n_in, n_out)) return;
    const bool dense = !allow_tensor && prefill_config_.dense_tensor &&
        prefill::precise_dense_candidate(weight.type);
    if ((allow_tensor || dense) && prefill_tc_ && !decode_phase_ && count >= 1024 && n_out >= 512) {
        if (batch_.half_width != n_in || batch_.half_count != count) {
            q8_1_to_f16(batch_.input_q8.data(), batch_.input_half.as<uint16_t>(),
                         count * n_in, stream_, batch_.input_low.as<uint16_t>());
            batch_.half_width = n_in; batch_.half_count = count;
        }
        if (dense_profile_.active()) throw std::invalid_argument("dense profile TC projection not calibrated");
        auto* weight_f32 = batch_.weight_f32.as<float>();
        auto* weight_low = batch_.weight_low.as<uint16_t>();
        int64_t elements = int64_t(batch_.weight_f32.bytes() / sizeof(float));
        if (dense) {
            // This activation view is dead during projections, including the
            // post-scan output projection. Reuse it on the same ordered stream.
            elements = (std::min)(int64_t(batch_.scan_activated.bytes() / (sizeof(float) + sizeof(uint16_t))), batch_.gemm.scratch_elems());
            elements = elements / n_in * n_in;
            if (elements < n_in) throw std::runtime_error("unreserved dense Tensor Core workspace");
            weight_f32 = batch_.scan_activated.as<float>();
            weight_low = reinterpret_cast<uint16_t*>(weight_f32 + elements);
        }
        prefill::precise_native(batch_.gemm, batch_.input_half.as<uint16_t>(),
            batch_.input_low.as<uint16_t>(), weight.type, weight.ptr(), output,
            count, n_out, n_in, weight_f32, weight_low, elements);
        return;
    }
    if (!decode_phase_ && prefill_config_.bulk_projection && count >= 16 &&
        prefill_config_.bulk_formats.contains(weight.type) &&
        kernels::native_mmvq_bulk_supported(weight.type)) {
        const auto launches = kernels::native_mmvq_bulk(weight.type, weight.ptr(), batch_.input_q8.data(),
            output, n_in, n_out, count, stream_, weight.type == 14 ? prefill_config_.projection_columns : 8, prefill_config_.projection_rows);
        dense_profile_.launches(launches);
        return;
    }
    if (!decode_phase_ && prefill_config_.q4_input_reuse && weight.type == 12 && count >= 16) {
        const auto launches = kernels::native_mmvq_bulk(weight.type, weight.ptr(), batch_.input_q8.data(), output,
            n_in, n_out, count, stream_, 8, 2);
        dense_profile_.launches(launches);
        return;
    }
    const size_t row_bytes = kernels::native_q8_1_bytes(n_in);
    for (int first = 0; first < count; first += 8) {
        const int ncols = count - first < 8 ? count - first : 8;
        const auto* input = static_cast<const uint8_t*>(batch_.input_q8.data()) +
                            size_t(first) * row_bytes;
        kernels::native_mmvq(weight.type, weight.ptr(), input,
                             output + size_t(first) * n_out,
                             n_in, n_out, ncols, stream_);
        dense_profile_.launches(1);
    }
}

int Runtime::run_prefill(const int* tokens, int count, const std::string& reference,
                         Metrics& metrics, const PrefillObserver* observer) {
    finish_decode(&metrics);
    decode_phase_ = false;
    if (batch_failed_ || !tokens || count < 2 || count > batch_.capacity || count > kv_capacity_)
        throw std::invalid_argument("unprepared Qwen35 batched prefill");
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    batch_probe_ = BatchProbe{};
    prepare_prefill_events(1);
    const auto embedding_start = std::chrono::steady_clock::now();
    load_prefill_embeddings(tokens, count, f(b.hidden));
    metrics.pp_embedding_submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embedding_start).count();
    if (!reference.empty())
        batch_probe_.embed = batch_stage_error(reference, "model.input_embed", f(b.hidden), 2048);
    for (size_t il = 0; il < layers_.size(); ++il) {
        if (!prefill_events_.empty())
            cuda_check(cudaEventRecord(prefill_events_[il][0], stream_), "prefill layer begin");
        decode_batch_layer(il, count, 0, reference, metrics);
        // Current route drain proves every preceding layer has completed.
        if (observer && il) observer->notify(int(il), int(layers_.size()));
    }
    cuda_check(cudaMemcpyAsync(work_.hidden.data(), f(b.hidden) + size_t(count - 1) * 2048,
                                2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
               "prefill final hidden");
    sampling_history(tokens, count);
    const int next = head_token(metrics);
    if (mtp_enabled_) {
        const auto mtp_start = std::chrono::steady_clock::now();
        mtp_prompt(tokens, count, b.hidden.as<float>(), metrics);
        metrics.pp_mtp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mtp_start).count();
    }
    collect_prefill_events(1, metrics);
    if (observer) observer->notify(int(layers_.size()), int(layers_.size()));
    return next;
}

int Runtime::run_prefill_long(const int* tokens, int count, Metrics& metrics,
                              int chunk_capacity, const PrefillObserver* observer) {
    finish_decode(&metrics);
    decode_phase_ = false;
    const int chunk = chunk_capacity ? chunk_capacity : batch_.capacity;
    if (batch_failed_ || !tokens || chunk < 2 || chunk > batch_.capacity || count <= chunk ||
        count > kv_capacity_)
        throw std::invalid_argument("invalid Qwen35 long prefill shape");
    for (int t = 0; t < count; ++t)
        if (tokens[t] < 0 || tokens[t] >= int(spec_.n_vocab))
            throw std::out_of_range("Qwen35 long prefill token");
    const int chunks = (count + chunk - 1) / chunk;
    prepare_prefill_events(chunks);
    const size_t bytes = size_t(count) * 2048 * sizeof(float);
    // A chunk is read into batch.hidden before any output is written. Past
    // attention inputs live in K/V, and GDN carries its own recurrent state;
    // later chunks only read their still untouched rows. Keep layer-major PP
    // while reusing the one full-length activation allocation.
    const char* workspace = std::getenv("STRATA_QWEN35_LONG_WORKSPACE");
    const bool control = workspace && std::string(workspace) == "control";
    if (workspace && !control && std::string(workspace) != "reuse")
        throw std::invalid_argument("LONG_WORKSPACE must be reuse or control");
    DeviceBuffer current(bytes), next(control ? bytes : 0);
    size_t free_bytes = 0, total_bytes = 0;
    cuda_check(cudaMemGetInfo(&free_bytes, &total_bytes), "long prefill GPU budget");
    metrics.gpu_min_free_bytes = (std::min)(metrics.gpu_min_free_bytes, free_bytes);
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    const auto embedding_start = std::chrono::steady_clock::now();
    load_prefill_embeddings(tokens, count, current.as<float>());
    metrics.pp_embedding_submit_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - embedding_start).count();
    for (size_t il = 0; il < layers_.size(); ++il) {
        for (int first = 0; first < count; first += chunk) {
            const int n = (std::min)(chunk, count - first);
            const size_t chunk_bytes = size_t(n) * 2048 * sizeof(float);
            cuda_check(cudaMemcpyAsync(b.hidden.data(),
                                        current.as<float>() + size_t(first) * 2048,
                                        chunk_bytes, cudaMemcpyDeviceToDevice, stream_),
                       "long prefill layer input");
            prefill_event_offset_ = size_t(first / chunk) * layers_.size();
            if (!prefill_events_.empty())
                cuda_check(cudaEventRecord(prefill_events_[prefill_event_offset_ + il][0], stream_),
                           "long prefill layer begin");
            decode_batch_layer(il, n, first, "", metrics);
            // First current-layer route drain proves all previous-layer chunks
            // complete on this stream. Do not wait for the current MoE here.
            if (observer && il && first == 0)
                observer->notify(int(il), int(layers_.size()));
            cuda_check(cudaMemcpyAsync((control ? next : current).as<float>() + size_t(first) * 2048,
                                        b.hidden.data(), chunk_bytes,
                                        cudaMemcpyDeviceToDevice, stream_),
                       "long prefill layer output");
        }
        if (control) std::swap(current, next);
    }
    cuda_check(cudaMemcpyAsync(work_.hidden.data(),
                                current.as<float>() + size_t(count - 1) * 2048,
                                2048 * sizeof(float), cudaMemcpyDeviceToDevice, stream_),
               "long prefill final hidden");
    sampling_history(tokens, count);
    const int result = head_token(metrics);
    collect_prefill_events(chunks, metrics);
    prefill_event_offset_ = 0;
    if (mtp_enabled_) {
        const auto mtp_start = std::chrono::steady_clock::now();
        mtp_prompt(tokens, count, current.as<float>(), metrics);
        metrics.pp_mtp_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mtp_start).count();
    }
    if (observer) observer->notify(int(layers_.size()), int(layers_.size()));
    return result;
}

} // namespace strata::qwen35moe
