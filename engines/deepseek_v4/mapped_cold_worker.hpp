#pragma once
#include "cpu_expert_service.hpp"
#include "mapped_cold_packet.hpp"
#include <atomic>
#include <thread>

namespace strata::deepseek_v4 {
struct MappedColdMetrics { uint64_t jobs = 0, experts = 0, service_ns = 0; };
// The original CPU expert service owns the only CPU math graphs. This worker
// owns just submission/completion; the model and mapped storage outlive it.
class MappedColdWorker {
public:
    MappedColdWorker(CpuExpertService&, const MappedColdStorage&);
    ~MappedColdWorker();
    MappedColdWorker(const MappedColdWorker&) = delete;
    MappedColdWorker& operator=(const MappedColdWorker&) = delete;
    MappedColdMetrics metrics() const noexcept;
private:
    void run() noexcept;
    CpuExpertService& service_;
    const MappedColdStorage& storage_;
    std::atomic<bool> stop_{false};
    std::atomic<uint64_t> jobs_{0}, experts_{0}, service_ns_{0};
    std::thread thread_;
};
}
