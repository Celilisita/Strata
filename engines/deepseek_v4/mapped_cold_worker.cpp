#include "mapped_cold_worker.hpp"
#include <array>
#include <chrono>
#include <stdexcept>

namespace strata::deepseek_v4 {
MappedColdWorker::MappedColdWorker(CpuExpertService& service, const MappedColdStorage& storage) :
    service_(service), storage_(storage) {
    if (!storage_.host) throw std::runtime_error("mapped cold worker needs its persistent packet");
    thread_ = std::thread([this] { run(); });
}
MappedColdWorker::~MappedColdWorker() {
    // Let any GPU waiter finish while its worker is still available. A timed
    // out GPU request may leave one CPU job; joining drains it before free.
    mapped_cold_synchronize(storage_);
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
}
MappedColdMetrics MappedColdWorker::metrics() const noexcept {
    return {jobs_.load(), experts_.load(), service_ns_.load()};
}
void MappedColdWorker::run() noexcept {
    uint32_t completed = 0;
    auto* packet = storage_.host;
    while (!stop_.load(std::memory_order_acquire)) {
        const uint32_t ready = mapped_cold_ready(packet);
        if (ready == completed) { std::this_thread::yield(); continue; }
        const auto start = std::chrono::steady_clock::now();
        uint32_t count = 0;
        packet->error = 0;
        try {
            if (packet->layer >= 43 || !packet->mask || packet->mask > 63)
                throw std::runtime_error("mapped cold job header differs");
            std::array<int32_t, 6> ids{};
            for (uint32_t rank = 0; rank < 6; ++rank)
                if (packet->mask & (1u << rank)) ids[count++] = packet->ids[rank];
            service_.compute(int(packet->layer), packet->activation, ids.data(), count, packet->contribution);
        } catch (...) { packet->error = 1; }
        const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start).count();
        service_ns_.fetch_add(elapsed > 0 ? uint64_t(elapsed) : 0);
        jobs_.fetch_add(1); experts_.fetch_add(count);
        mapped_cold_done(packet, ready);
        completed = ready;
    }
}
}
