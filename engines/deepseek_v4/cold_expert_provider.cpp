#include "cold_expert_provider.hpp"

#include "llama-model.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t activation_bytes = 4096 * sizeof(float);
constexpr std::size_t contribution_bytes = activation_bytes * 6;
constexpr uint64_t expert_weight_bytes = 13369344;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
}

ColdExpertProvider::ColdExpertProvider(const llama_model& model, ggml_backend_t gpu, int threads, bool admit_free,
                                     ggml_backend_sched_eval_callback observer, void* data) :
    gpu_(gpu), admit_free_(admit_free), service_(model, threads), observer_(observer), observer_data_(data) {
    if (!gpu_) throw std::runtime_error("cold expert provider needs the selected CUDA backend");
    const auto device = ggml_backend_get_device(gpu_);
    const auto type = ggml_backend_dev_host_buffer_type(device);
    if (!type || std::strcmp(ggml_backend_buft_name(type), "CUDA_Host") != 0)
        throw std::runtime_error("cold expert contributions require canonical CUDA_Host memory");
    try {
        device_event_ = ggml_backend_event_new(device);
        if (!device_event_) throw std::runtime_error("cold expert device event allocation failed");
        pinned_ = ggml_backend_buft_alloc_buffer(type, activation_bytes + contribution_bytes);
        if (!pinned_ || ggml_backend_buffer_get_type(pinned_) != type ||
            !ggml_backend_buffer_is_host(pinned_) ||
            ggml_backend_buffer_get_size(pinned_) < activation_bytes + contribution_bytes)
            throw std::runtime_error("cold expert pinned allocation fell back or is undersized");
        activation_ = static_cast<float*>(ggml_backend_buffer_get_base(pinned_));
        if (!activation_) throw std::runtime_error("cold expert pinned address unavailable");
        output_ = activation_ + 4096;
        worker_ = std::thread(&ColdExpertProvider::worker_loop, this);
    } catch (...) { release(); throw; }
}

ColdExpertProvider::~ColdExpertProvider() { release(); }

void ColdExpertProvider::release() noexcept {
    { std::lock_guard<std::mutex> lock(mutex_); stop_ = true; }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
    // Contributions and request-local slot maps are async copy sources.
    if (gpu_) ggml_backend_synchronize(gpu_);
    if (pinned_) { ggml_backend_buffer_free(pinned_); pinned_ = nullptr; }
    if (device_event_) { ggml_backend_event_free(device_event_); device_event_ = nullptr; }
}

const int32_t* ColdExpertProvider::prepare(ggml_backend_t gpu, int layer, const int32_t* ids,
                                         std::size_t count, const ggml_tensor* map,
                                         bool has_free_slot, ggml_cgraph* graph, void* data) {
    auto& self = *static_cast<ColdExpertProvider*>(data);
    if (count != 6) return nullptr; // PP retains the canonical grouped upload path.
    if (gpu != self.gpu_) throw std::runtime_error("cold expert GPU backend differs");
    self.metrics_.route_control_d2h_bytes += 6 * sizeof(int32_t);
    ++self.metrics_.host_continuations;
    // Reuse the original LRU admission/fill path while a real slot is free.
    // Once full, CPU misses preserve the resident working set across PP/TG.
    if (self.admit_free_ && has_free_slot) return nullptr;
    return self.submit(layer, ids, map, graph);
}

const int32_t* ColdExpertProvider::submit(int layer, const int32_t* ids,
                                        const ggml_tensor* map, ggml_cgraph* graph) {
    if (layer_ != -1 || !ids || !map || !map->data || map->type != GGML_TYPE_I32 ||
        ggml_nelements(map) != 256 || !ggml_is_contiguous(map))
        throw std::runtime_error("cold expert request or cache map contract differs");
    const auto* resident = static_cast<const int32_t*>(map->data);
    std::copy_n(resident, 256, upload_map_.begin());
    count_ = 0;
    for (uint32_t rank = 0; rank < 6; ++rank) {
        const int32_t id = ids[rank];
        if (id < 0 || id >= 256 || std::find(ids, ids + rank, id) != ids + rank)
            throw std::runtime_error("cold expert original router ids invalid or duplicate");
        if (resident[id] >= 0) continue;
        ids_[count_] = id;
        ranks_[count_++] = rank;
        // Slot zero is a valid initialized canonical lane; its result is
        // overwritten before original route weighting. It owns no cold expert.
        upload_map_[id] = 0;
    }
    if (!count_) return resident;
    const auto gate_name = "ffn_moe_gate-" + std::to_string(layer);
    const ggml_tensor* activation = nullptr;
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
        const auto* node = ggml_graph_node(graph, i);
        if (gate_name != ggml_get_name(node)) continue;
        if (activation || node->op != GGML_OP_MUL_MAT_ID)
            throw std::runtime_error("cold expert canonical Gate seam differs");
        activation = node->src[1];
    }
    if (!activation || activation->type != GGML_TYPE_F32 || !ggml_is_contiguous(activation) ||
        activation->ne[0] != 4096 || ggml_nelements(activation) != 4096)
        throw std::runtime_error("cold expert normalized activation seam unavailable");
    ggml_backend_tensor_get_async(gpu_, activation, activation_, 0, activation_bytes);
    ggml_backend_synchronize(gpu_);
    metrics_.activation_d2h_bytes += activation_bytes;
    metrics_.cold_experts += count_;
    metrics_.placeholder_gpu_experts += count_;
    metrics_.avoided_weight_bytes += count_ * expert_weight_bytes;
    ++metrics_.jobs;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        layer_ = layer;
        down_name_ = "ffn_moe_down-" + std::to_string(layer);
        worker_error_.clear();
        done_ = false;
        ready_ = true;
    }
    condition_.notify_one();
    return upload_map_.data();
}

void ColdExpertProvider::worker_loop() noexcept {
    for (;;) {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return ready_ || stop_; });
        if (stop_) return;
        ready_ = false;
        lock.unlock();
        const auto start = Clock::now();
        std::string error;
        try { service_.compute(layer_, activation_, ids_.data(), count_, output_); }
        catch (const std::exception& failure) { error = failure.what(); }
        catch (...) { error = "unknown canonical cold expert compute failure"; }
        const auto work = elapsed(start);
        lock.lock();
        worker_error_ = std::move(error);
        work_ms_ = work;
        done_ = true;
        lock.unlock();
        condition_.notify_one();
    }
}

void ColdExpertProvider::join(ggml_tensor* tensor) {
    if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 4096 || tensor->ne[1] != 6 ||
        tensor->ne[2] != 1 || tensor->ne[3] != 1 || !ggml_is_contiguous(tensor))
        throw std::runtime_error("cold expert canonical Down join shape differs");
    const auto waiting = Clock::now();
    {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] { return done_; });
        if (!worker_error_.empty()) throw std::runtime_error(worker_error_);
        metrics_.cpu_service_ms += work_ms_;
    }
    metrics_.join_wait_ms += elapsed(waiting);
    for (std::size_t first = 0; first < count_;) {
        std::size_t end = first + 1;
        while (end < count_ && ranks_[end] == ranks_[end - 1] + 1) ++end;
        const auto bytes = (end - first) * activation_bytes;
        ggml_backend_tensor_set_async(gpu_, tensor, output_ + first * 4096,
                                      ranks_[first] * activation_bytes, bytes);
        metrics_.contribution_h2d_bytes += bytes;
        first = end;
    }
    ++metrics_.joins;
    layer_ = -1;
    down_name_.clear();
}

bool ColdExpertProvider::evaluate(ggml_tensor* tensor, bool ask, void* data) {
    auto& self = *static_cast<ColdExpertProvider*>(data);
    const bool join = self.layer_ >= 0 && self.down_name_ == ggml_get_name(tensor);
    if (ask) return join || (self.observer_ && self.observer_(tensor, true, self.observer_data_));
    if (join) self.join(tensor);
    if (!self.observer_ || !self.observer_(tensor, true, self.observer_data_)) return true;
    if (join) ggml_backend_synchronize(self.gpu_);
    return self.observer_(tensor, false, self.observer_data_);
}

} // namespace strata::deepseek_v4
