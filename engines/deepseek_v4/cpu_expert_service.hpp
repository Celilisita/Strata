#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct llama_model;

namespace strata::deepseek_v4 {

// One canonical CPU compute resource, borrowing original mmap expert banks.
// Calls are serial; the hybrid provider owns its persistent submission worker.
class CpuExpertService {
public:
    CpuExpertService(const llama_model& model, int threads);
    ~CpuExpertService();
    CpuExpertService(const CpuExpertService&) = delete;
    CpuExpertService& operator=(const CpuExpertService&) = delete;
    void compute(int layer, const float* activation, const int32_t* ids,
                 std::size_t count, float* output);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace strata::deepseek_v4
