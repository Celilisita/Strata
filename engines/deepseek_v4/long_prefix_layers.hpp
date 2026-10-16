#pragma once
#include "long_prefix_compare.hpp"
#include "layer_snapshots.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
namespace strata::deepseek_v4 {
// Diagnostic only: observe complete final-token mHC rows at one decode step.
// Earlier steps retain whole scheduler splits; observed steps are not timed.
struct PrefixLayerProbe {
    struct Row { std::string name; std::vector<float> values; };
    std::vector<Row> rows;
    bool armed = false;
    std::string error;
    static bool callback(ggml_tensor* tensor, bool ask, void* data) noexcept {
        auto& self = *static_cast<PrefixLayerProbe*>(data);
        const char* name = ggml_get_name(tensor);
        if (ask) { const int layer = layer_snapshot_index(name); return self.armed && layer >= 0 && layer < 43; }
        try {
            const int layer = layer_snapshot_index(name);
            if (layer < 0 || layer >= 43) throw std::runtime_error("prefix boundary layer outside0..42");
            const std::string row_name = "l_last-" + std::to_string(layer);
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] * tensor->ne[1] != 16384 ||
                tensor->nb[0] != sizeof(float))
                throw std::runtime_error("prefix layer row contract differs: " + row_name + " type=" +
                    std::to_string(tensor->type) + " ne0=" + std::to_string(tensor->ne[0]) +
                    " ne1=" + std::to_string(tensor->ne[1]) + " nb0=" + std::to_string(tensor->nb[0]));
            size_t offset = 0;
            for (int dim = 2; dim < 4; ++dim) offset += (tensor->ne[dim] - 1) * tensor->nb[dim];
            Row row{row_name, std::vector<float>(16384)};
            for (int64_t ih = 0; ih < tensor->ne[1]; ++ih)
                ggml_backend_tensor_get(tensor, row.values.data() + ih * tensor->ne[0], offset + ih * tensor->nb[1],
                                        size_t(tensor->ne[0]) * sizeof(float));
            for (float value : row.values) if (!std::isfinite(value)) throw std::runtime_error("nonfinite layer row");
            auto previous = std::find_if(self.rows.begin(), self.rows.end(), [&](const Row& value) { return value.name == row_name; });
            if (previous != self.rows.end()) *previous = std::move(row);
            else {
                if (self.rows.size() >= 43) throw std::runtime_error("more than43unique layer boundaries");
                self.rows.push_back(std::move(row));
            }
            return true;
        } catch (const std::exception& failure) { self.error = failure.what(); return false; }
    }
};
inline void prefix_layer_case(Runtime& runtime, const PrefixReference& reference, PrefixLayerProbe& probe, int step) {
    if (step < 0 || step > 205) throw std::runtime_error("layer attribution step outside bounded prefix");
    runtime.reset();
    probe.armed = step == 0;
    std::string error;
    if (!runtime.evaluate(reference.input, error)) throw std::runtime_error(error + probe.error);
    for (int index = 1; index <= step; ++index) {
        probe.armed = index == step;
        if (!runtime.evaluate({reference.output[index - 1]}, error)) throw std::runtime_error(error + probe.error);
    }
    llama_synchronize(runtime.context());
    verify_device_expert_cache(runtime.context());
    if (probe.rows.size() != 43) throw std::runtime_error("layer attribution did not observe43boundaries");
}
inline void prefix_compare_layers(Runtime& runtime, const Options& options, const PrefixReference& reference,
                                  const std::filesystem::path& output, int step) {
    PrefixLayerProbe control, candidate;
    std::string error;
    session_set_device_cache(false); session_set_frequency(false); session_set_source_pin(0);
    if (!runtime.reconfigure_context(options, error, PrefixLayerProbe::callback, &control)) throw std::runtime_error(error);
    prefix_layer_case(runtime, reference, control, step);
    session_set_device_cache(true); session_set_frequency(true); session_set_source_pin(131072);
    if (!runtime.reconfigure_context(options, error, PrefixLayerProbe::callback, &candidate)) throw std::runtime_error(error);
    prefix_layer_case(runtime, reference, candidate, step);
    std::ofstream report(output / "layer-attribution.json");
    report << "{\"decode_step\":" << step << ",\"observation_changes_scheduler_splits\":true,\"compared\":43,\"first_difference\":";
    bool found = false;
    for (size_t index = 0; index < control.rows.size(); ++index) {
        const auto& a = control.rows[index]; const auto& b = candidate.rows[index];
        if (a.name != b.name) throw std::runtime_error("layer attribution names differ");
        PrefixDifference difference;
        double squares = 0;
        for (size_t k = 0; k < a.values.size(); ++k) if (std::memcmp(&a.values[k], &b.values[k], sizeof(float))) {
            ++difference.differing;
            const double delta = double(a.values[k]) - b.values[k];
            difference.max_abs = std::max(difference.max_abs, std::abs(delta)); squares += delta * delta;
        }
        if (!difference.differing) continue;
        difference.step = step; difference.rms = std::sqrt(squares / a.values.size());
        report << "{\"name\":" << session_json(a.name) << ",\"row_values\":16384,\"difference\":";
        prefix_write_difference(report, difference); report << '}';
        for (int copy = 0; copy < 2; ++copy) {
            const auto& row = copy ? b : a;
            std::ofstream data(output / (copy ? "layer-candidate.f32" : "layer-control.f32"), std::ios::binary);
            data.write(reinterpret_cast<const char*>(row.values.data()), row.values.size() * sizeof(float));
            if (!data) throw std::runtime_error("layer row write failed");
        }
        found = true; break;
    }
    if (!found) report << "null";
    report << '}';
    if (!report) throw std::runtime_error("layer attribution report failed");
}
}
