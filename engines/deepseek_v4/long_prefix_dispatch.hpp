#pragma once
#include "layer_snapshots.hpp"
#include "long_prefix_compare.hpp"
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strata::deepseek_v4 {
enum class PrefixSnapshotScope { Boundary, Attention37, Ffn37, Full };
class PrefixDispatchSnapshots {
public:
    explicit PrefixDispatchSnapshots(Runtime& runtime) {
        auto reg = ggml_backend_dev_backend_reg(runtime.device());
        proc_ = reinterpret_cast<LayerSnapshotProc>(ggml_backend_reg_get_proc_address(reg, "strata_layer_snapshots"));
        if (!proc_) throw std::runtime_error("dispatch snapshots unavailable");
        command(LayerSnapshotAction::Initialize);
    }
    // Runtime/graphs must be destroyed before this object (including exceptions).
    ~PrefixDispatchSnapshots() { if (packet_.handle) { packet_.action = LayerSnapshotAction::Release; proc_(&packet_); } }
    void set_scope(PrefixSnapshotScope scope) {
        packet_.mask[0] = packet_.mask[1] = 0;
        for (size_t p = 0; p < layer_snapshot_layers; ++p)
            if (p < 47 || scope == PrefixSnapshotScope::Full ||
                (scope == PrefixSnapshotScope::Attention37 && p >= 47 && p <= 55) ||
                (scope == PrefixSnapshotScope::Ffn37 && p >= 56 && p <= 67))
                packet_.mask[p/64] |= uint64_t(1)<<(p%64);
    }
    void begin() { frozen_.clear(); command(LayerSnapshotAction::Begin); }
    void end(Runtime& runtime) {
        // Nonblocking CUDA streams must finish before the diagnostic D2H read.
        llama_synchronize(runtime.context());
        command(LayerSnapshotAction::End);
        frozen_ = read_device(); // Freeze PP before subsequent CUDA graph replays.
    }
    const std::vector<float>& read() const {
        if (frozen_.empty()) throw std::runtime_error("dispatch snapshot not frozen");
        return frozen_;
    }
    void save(const std::filesystem::path& path) const {
        std::ofstream data(path, std::ios::binary);
        data.write(reinterpret_cast<const char*>(read().data()), read().size()*sizeof(float));
        if (!data) throw std::runtime_error("dispatch snapshot write failed");
    }
    const auto& types() const { return types_; }
    const auto& row_values() const { return row_values_; }
private:
    std::vector<float> read_device() {
        std::vector<float> values(layer_snapshot_layers * layer_snapshot_values);
        packet_.rows = values.data(); packet_.observed = observed_.data(); packet_.values = values.size();
        packet_.types = types_.data(); packet_.row_values = row_values_.data();
        command(LayerSnapshotAction::Read);
        packet_.rows = nullptr; packet_.observed = nullptr; packet_.types = nullptr; packet_.row_values = nullptr;
        for (size_t layer = 0; layer < observed_.size(); ++layer)
            if ((packet_.mask[layer/64] & (uint64_t(1)<<(layer%64))) && !observed_[layer]) throw std::runtime_error("missing dispatch boundary layer" + layer_snapshot_name(layer));
        for (size_t point = 0; point < layer_snapshot_layers; ++point)
            if (types_[point] == GGML_TYPE_F32) for (size_t k = 0; k < row_values_[point]; ++k)
                if (!std::isfinite(values[point*layer_snapshot_values+k])) throw std::runtime_error("nonfinite dispatch snapshot");
        return values;
    }
    void command(LayerSnapshotAction action) {
        packet_.action = action;
        if (!proc_(&packet_)) throw std::runtime_error("dispatch snapshot action failed: " + std::to_string(packet_.error));
    }
    std::vector<float> frozen_;
    LayerSnapshotProc proc_ = nullptr;
    LayerSnapshotPacket packet_{LayerSnapshotAction::Initialize};
    std::array<uint32_t, layer_snapshot_layers> observed_{};
    std::array<int32_t, layer_snapshot_layers> types_{};
    std::array<size_t, layer_snapshot_layers> row_values_{};
};
inline void prefix_write_dispatch_difference(const std::vector<float>& a, const std::vector<float>& b,
                                             const std::filesystem::path& output, const PrefixDispatchSnapshots& snapshots) {
    std::ofstream report(output / "dispatch-layer-attribution.json");
    report << "{\"scheduler_splits_unchanged\":true,\"canonical_fusion_unchanged\":true,\"diagnostic_adds_cuda_copy_edges\":true,\"snapshot_read_at\":\"prefill_end_before_tg\",\"snapshot_gpu_bytes\":"
           << layer_snapshot_layers * layer_snapshot_values * sizeof(float) << ",\"boundaries\":[";
    int first = -1;

    bool comma = false;
    for (size_t layer = 0; layer < layer_snapshot_layers; ++layer) {
        if (!snapshots.row_values()[layer]) continue;
        PrefixDifference difference;
        double squares = 0;
        const size_t offset = layer * layer_snapshot_values;
        for (size_t k = 0; k < snapshots.row_values()[layer]; ++k) if (std::memcmp(&a[offset+k], &b[offset+k], sizeof(float))) {
            ++difference.differing;
            int32_t ai = 0, bi = 0;
            std::memcpy(&ai, &a[offset+k], sizeof(ai)); std::memcpy(&bi, &b[offset+k], sizeof(bi));
            const double delta = snapshots.types()[layer] == GGML_TYPE_I32 ? double(ai)-bi : double(a[offset+k])-b[offset+k];
            difference.max_abs = std::max(difference.max_abs, std::abs(delta)); squares += delta * delta;
        }
        if (difference.differing) { difference.step = 0; if (first < 0) first = int(layer); }
        difference.rms = std::sqrt(squares / snapshots.row_values()[layer]);
        if (comma) report << ','; comma = true;
        report << "{\"index\":" << layer << ",\"name\":"
               << session_json(layer_snapshot_name(layer))
               << ",\"type\":" << snapshots.types()[layer] << ",\"values\":" << snapshots.row_values()[layer]
               << ",\"difference\":";
        prefix_write_difference(report, difference); report << '}';
    }
    report << "],\"first_difference_index\":" << first << '}';
    for (int copy = 0; copy < 2; ++copy) {
        const auto& values = copy ? b : a;
        std::ofstream data(output / (copy ? "dispatch-boundaries-candidate.f32" : "dispatch-boundaries-control.f32"), std::ios::binary);
        data.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
        if (!data) throw std::runtime_error("dispatch sample write failed");
    }
    if (!report) throw std::runtime_error("dispatch report write failed");
}
}
