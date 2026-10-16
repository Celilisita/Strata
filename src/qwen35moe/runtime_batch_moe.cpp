#include "strata/qwen35moe/runtime.hpp"
#include "strata/qwen35moe/cpu_state.hpp"
#include "strata/qwen35moe/cuda_kernels.hpp"
#include "strata/qwen35moe/router_tiled.hpp"

#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <vector>

namespace strata::qwen35moe {

void Runtime::batch_moe(size_t il, const LayerWeights& layer, int count,
                        const std::string& reference, Metrics& metrics) {
    if (tg_dynamic_ && decode_phase_ && (count == 1 || (count == 2 && dynamic_pair_args_.bytes()))) {
        batch_moe_device(il, layer, metrics, count);
        return;
    }
    if (resident_gather_ && decode_phase_ && count == 1 && int(il) < resident_layers_ &&
        expert_cache_.resident_slots.data()) {
        batch_moe_resident(il, layer, metrics);
        return;
    }
    // A full cache keeps every routed entry resident through grouped execution.
    // Smaller caches use one bounded staging slot per entry for the same kernels.
    if (decode_phase_ && cpu_ && int(il) >= cpu_->from_layer) {
        if (tg_hit_gpu_ && count == 1 && expert_cache_.slots && !batch_.expert_stage.data())
            batch_moe_hybrid(il, layer, count, reference, metrics);
        else {
            ++metrics.tg_cpu_layers;
            batch_moe_cpu(il, layer, count, reference, metrics);
        }
    }
    else batch_moe_grouped(il, layer, count, reference, metrics);
}

std::vector<int32_t> Runtime::batch_route(const LayerWeights& layer, int count,
                                          const std::string& reference, Metrics& metrics,
                                          bool copy_norm) {
    std::chrono::steady_clock::time_point route_start;
    if (decode_phase_ && tg_diag_) route_start = std::chrono::steady_clock::now();
    if (count < 1 || count > batch_.capacity)
        throw std::invalid_argument("invalid Qwen35 routed batch size");
    auto& b = batch_;
    auto f = [](DeviceBuffer& buffer) { return buffer.as<float>(); };
    const char* setting = std::getenv("STRATA_QWEN35_BULK_ROUTE");
    if (!decode_phase_ && setting && std::string(setting) == "1") {
        if (prefill_config_.router_tiled && count >= 4) {
            route_project_tiled(layer.router.f32(), f(b.norm), f(b.logits), 2048, 257, count, stream_);
            ++b.router_tiled_calls;
        } else route_project(layer.router.f32(), f(b.norm), f(b.logits), 2048, 257, stream_, count);
        route_top8_batch(f(b.logits), b.ids.as<int32_t>(), f(b.weights), count, stream_);
    } else for (int t = 0; t < count; ++t) {
        route_project(layer.router.f32(), f(b.norm) + size_t(t) * 2048,
                      f(b.logits) + size_t(t) * 257, 2048, 257, stream_);
        route_top8(f(b.logits) + size_t(t) * 257,
                   b.ids.as<int32_t>() + size_t(t) * 8,
                   f(b.weights) + size_t(t) * 8, stream_);
    }
    if (checkpoint_sink_) {
        checkpoint_sink_->batch_stage("router_logits", b.logits.data(), size_t(count) * 257 * 4, false, stream_);
        checkpoint_sink_->batch_stage("route_weights", b.weights.data(), size_t(count) * 8 * 4, false, stream_);
    }
    if (decode_phase_ && tg_diag_) {
        cuda_check(cudaEventRecord(diag_route_ready_, stream_), "record TG route completion");
        const auto wait_start = std::chrono::steady_clock::now();
        cuda_check(cudaEventSynchronize(diag_route_ready_), "wait TG route GPU work");
        metrics.diag_route_gpu_wait_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - wait_start).count();
    }
    std::vector<int32_t> ids(size_t(count) * 8);
    std::chrono::steady_clock::time_point copy_start;
    if (decode_phase_ && tg_diag_) copy_start = std::chrono::steady_clock::now();
    cuda_check(cudaMemcpyAsync(ids.data(), b.ids.data(), ids.size() * sizeof(int32_t),
                               cudaMemcpyDeviceToHost, stream_), "read batch routed experts");
    if (decode_phase_ && tg_diag_)
        metrics.diag_route_copy_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - copy_start).count();
    metrics.route_d2h += ids.size() * sizeof(int32_t);
    if (copy_norm) {
        if (!cpu_) throw std::runtime_error("missing Qwen35 CPU expert state");
        cpu_->norm.resize(size_t(count) * 2048);
        cuda_check(cudaMemcpyAsync(cpu_->norm.data(), b.norm.data(),
                                   cpu_->norm.size() * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream_), "read CPU expert norm");
    }
    const auto start = std::chrono::steady_clock::now();
    cuda_check(cudaStreamSynchronize(stream_), "wait batch routed experts");
    metrics.route_wait_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    ++metrics.host_continuations;
    ++metrics.explicit_syncs;
    if (!reference.empty()) {
        for (int rank = 0; rank < 8; ++rank) batch_probe_.ids[size_t(rank)] = ids[size_t(rank)];
        cuda_check(cudaMemcpyAsync(batch_probe_.weights.data(), b.weights.data(),
                                   8 * sizeof(float), cudaMemcpyDeviceToHost, stream_),
                   "read batch router weights");
        cuda_check(cudaStreamSynchronize(stream_), "wait batch router weights");
        ++metrics.explicit_syncs;
    }
    if (decode_phase_ && tg_diag_)
        metrics.diag_route_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - route_start).count();
    return ids;
}

void Runtime::batch_moe_hybrid(size_t il, const LayerWeights& layer, int count,
                               const std::string& reference, Metrics& metrics) {
    const double before_gpu_wait = metrics.diag_route_gpu_wait_ms;
    const double before_copy = metrics.diag_route_copy_ms;
    const double before_sync = metrics.route_wait_ms;
    const auto ids = batch_route(layer, count, reference, metrics, false);
    std::chrono::steady_clock::time_point classify_start;
    if (tg_diag_) classify_start = std::chrono::steady_clock::now();
    int hits = 0;
    for (int id : ids) {
        if (id < 0 || id >= 256) throw std::runtime_error("invalid Qwen35 routed expert ID");
        hits += expert_cache_.key_to_slot[il * 256 + size_t(id)] >= 0;
    }
    if (tg_diag_)
        metrics.diag_classify_wall_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - classify_start).count();
    if (hits == int(ids.size())) {
        if (tg_diag_) {
            metrics.diag_hit_gpu_wait_ms += metrics.diag_route_gpu_wait_ms - before_gpu_wait;
            metrics.diag_hit_copy_ms += metrics.diag_route_copy_ms - before_copy;
            metrics.diag_hit_sync_ms += metrics.route_wait_ms - before_sync;
            metrics.diag_hit_classify_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - classify_start).count();
        }
        ++metrics.tg_gpu_hit_layers;
        std::chrono::steady_clock::time_point branch_start;
        if (tg_diag_) branch_start = std::chrono::steady_clock::now();
        batch_moe_grouped_ready(il, layer, count, reference, metrics, ids);
        if (tg_diag_)
            metrics.diag_hit_branch_wall_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - branch_start).count();
    } else {
        std::chrono::steady_clock::time_point branch_start;
        if (tg_diag_) branch_start = std::chrono::steady_clock::now();
        auto& c = *cpu_;
        c.norm.resize(size_t(count) * 2048);
        std::chrono::steady_clock::time_point norm_copy_start;
        if (tg_diag_) norm_copy_start = std::chrono::steady_clock::now();
        cuda_check(cudaMemcpyAsync(c.norm.data(), batch_.norm.data(),
                                   c.norm.size() * sizeof(float), cudaMemcpyDeviceToHost,
                                   stream_), "read missed expert norm");
        if (tg_diag_)
            metrics.diag_miss_norm_copy_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - norm_copy_start).count();
        std::chrono::steady_clock::time_point norm_wait_start;
        if (tg_diag_) norm_wait_start = std::chrono::steady_clock::now();
        cuda_check(cudaStreamSynchronize(stream_), "wait missed expert norm");
        if (tg_diag_)
            metrics.diag_miss_norm_wait_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - norm_wait_start).count();
        ++metrics.explicit_syncs;
        ++metrics.tg_cpu_layers;
        if (hits == 0) batch_moe_cpu_ready(il, layer, count, reference, metrics, ids);
        else batch_moe_mixed(il, layer, ids, metrics);
        if (tg_diag_)
            metrics.diag_miss_branch_wall_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - branch_start).count();
    }
}

} // namespace strata::qwen35moe
