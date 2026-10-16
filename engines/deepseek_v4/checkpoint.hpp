#pragma once

#include "ggml-backend.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace strata::deepseek_v4 {

inline constexpr uint64_t checkpoint_tensor_cap = 64ull << 20;
inline constexpr uint64_t checkpoint_total_cap = 128ull << 20;

struct Checkpoint {
    std::string name, dtype, file, buffer_type;
    int type = -1;
    bool buffer_is_host = false;
    int buffer_usage = -1;
    std::array<int64_t, 4> dimensions{};
    std::array<uint64_t, 4> source_strides{};
    uint64_t bytes = 0, checksum = 0;
};

// Only layer-zero activations are observed. We never request expert weights.
// The scheduler must propagate a false observation result as GGML_STATUS_ABORTED.
class CheckpointCollector {
public:
    explicit CheckpointCollector(std::filesystem::path directory, bool fusion_seam = false);
    static bool callback(ggml_tensor* tensor, bool ask, void* user) noexcept;
    bool complete() const { return complete_; }
    bool failed() const { return !error_.empty(); }
    const std::string& error() const { return error_; }
    const std::vector<Checkpoint>& checkpoints() const { return checkpoints_; }
    void write_manifest() const;
    void pause(bool value) noexcept { paused_ = value; }

private:
    bool selected(const ggml_tensor* tensor) const;
    bool observe(ggml_tensor* tensor);
    std::filesystem::path directory_;
    std::vector<Checkpoint> checkpoints_;
    std::string error_;
    uint64_t total_bytes_ = 0;
    bool complete_ = false;
    bool fusion_seam_ = false;
    bool paused_ = false;
};

struct CheckpointComparison {
    bool equal = false;
    std::string first_difference, error;
    uint64_t compared_tensors = 0, first_difference_byte = 0;
    double max_absolute_error = 0, max_relative_error = 0;
    bool has_nonfinite = false;
};

struct CheckpointData {
    Checkpoint metadata;
    std::vector<uint8_t> data;
};
CheckpointData read_checkpoint_data(const std::filesystem::path&, const std::string& name);
CheckpointData read_checkpoint_occurrence(const std::filesystem::path&, const std::string& name, std::size_t occurrence);

CheckpointComparison compare_checkpoints(const std::filesystem::path& reference,
                                         const std::filesystem::path& candidate);

} // namespace strata::deepseek_v4
