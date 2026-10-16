#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include "strata/qwen35moe/dense_cost.hpp"
#include <array>
#include <chrono>
#include <vector>

namespace strata::qwen35moe {
// Boundary names denote the work just completed. GPU values are ordered-stream
// intervals (including exposed host gaps), CPU values are submission wall time.
void cuda_check(cudaError_t error, const char* what);
// Opt-in fixed checkpoints, collected only after the existing PP completion.
// No new kernels, tensor workspace, weight ownership or stream drains.
class DenseProfile {
    using Clock = std::chrono::steady_clock;
    struct Row {
        std::array<cudaEvent_t, dense_stages + 1> events{};
        DenseCost cost;
        size_t next = 0;
    };
    std::vector<Row> rows_;
    Row* active_ = nullptr;
    Clock::time_point last_;
    bool enabled_ = false;
public:
    ~DenseProfile() { clear(); }
    void clear() {
        active_ = nullptr;
        for (auto& row : rows_) for (auto event : row.events)
            if (event) cudaEventDestroy(event);
        rows_.clear();
    }
    bool active() const { return active_ != nullptr; }
    void enable() { enabled_ = true; }
    size_t event_count() const { return rows_.size() * (dense_stages + 1); }
    void prepare(size_t count) {
        if (!enabled_) return;
        if (count > 192 * 256) throw std::invalid_argument("dense profile capacity");
        active_ = nullptr;
        const size_t old = rows_.size();
        rows_.resize((std::max)(old, count));
        for (size_t i = old; i < rows_.size(); ++i) for (auto& event : rows_[i].events)
            cuda_check(cudaEventCreate(&event), "dense profile event create");
    }
    void begin(size_t index, bool prefill, cudaStream_t stream) {
        active_ = nullptr;
        if (!enabled_ || !prefill) return;
        if (index >= rows_.size()) throw std::out_of_range("dense profile layer");
        active_ = &rows_[index]; active_->next = 0; active_->cost = {};
        cuda_check(cudaEventRecord(active_->events[0], stream), "dense profile begin");
        last_ = Clock::now();
    }
    void launches(uint64_t count) {
        if (active_ && active_->next < dense_stages)
            active_->cost.launches[active_->next] += count;
    }
    void copies(uint64_t calls, uint64_t h2d, uint64_t d2d) {
        if (!active_ || active_->next >= dense_stages) return;
        const auto i = active_->next;
        active_->cost.copies[i] += calls;
        active_->cost.h2d_bytes[i] += h2d; active_->cost.d2d_bytes[i] += d2d;
    }
    void mark(DenseStage stage, cudaStream_t stream) {
        if (!active_) return;
        const size_t i = size_t(stage);
        if (i != active_->next) throw std::logic_error("dense profile boundary order");
        const auto now = Clock::now();
        active_->cost.submit_ms[i] = std::chrono::duration<double, std::milli>(now-last_).count();
        cuda_check(cudaEventRecord(active_->events[i+1], stream), "dense profile boundary");
        last_ = Clock::now(); ++active_->next;
        if (active_->next == dense_stages) active_ = nullptr;
    }
    void collect(size_t index, DenseCost& sum) {
        if (!enabled_) return;
        auto& row = rows_.at(index);
        if (row.next != dense_stages) throw std::logic_error("incomplete dense profile");
        for (size_t i = 0; i < dense_stages; ++i) {
            float ms = 0;
            cuda_check(cudaEventElapsedTime(&ms, row.events[i], row.events[i+1]), "dense profile elapsed");
            sum.gpu_ms[i] += ms; sum.submit_ms[i] += row.cost.submit_ms[i];
            sum.launches[i] += row.cost.launches[i];
            sum.copies[i] += row.cost.copies[i];
            sum.h2d_bytes[i] += row.cost.h2d_bytes[i]; sum.d2d_bytes[i] += row.cost.d2d_bytes[i];
        }
        ++sum.layers;
    }
};
} // namespace strata::qwen35moe
