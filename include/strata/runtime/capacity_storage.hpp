#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
namespace strata::runtime {
// One owner's live bytes; allocation obeys actual cudaMalloc capacity.
// There is no global free-memory reserve, retry, mirror, or inference fallback.
class CapacityStorage {
public:
    CapacityStorage() = default;
    ~CapacityStorage();
    CapacityStorage(const CapacityStorage&) = delete;
    CapacityStorage& operator=(const CapacityStorage&) = delete;
    static size_t reserve_bytes();
    void* allocate(size_t);
    void release(void*);
    size_t bytes() const { return bytes_; }
private:
    std::vector<std::pair<void*, size_t>> allocations_;
    size_t bytes_ = 0;
};
} // namespace strata::runtime
