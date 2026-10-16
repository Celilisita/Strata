#include "session_gate.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <cstring>
#include <stdexcept>

namespace strata::deepseek_v4 {
void session_set_frequency(bool enabled) {
#ifdef _WIN32
    if (_putenv_s("STRATA_DSV4_EXPERT_FREQUENCY", enabled ? "1" : "0")) throw std::runtime_error("cannot set frequency mode");
#else
    if (setenv("STRATA_DSV4_EXPERT_FREQUENCY", enabled ? "1" : "0", 1)) throw std::runtime_error("cannot set frequency mode");
#endif
}
void session_set_device_cache(bool enabled) {
#ifdef _WIN32
    const int code = _putenv_s("STRATA_DSV4_DEVICE_CACHE", enabled ? "1" : "0");
#else
    const int code = setenv("STRATA_DSV4_DEVICE_CACHE", enabled ? "1" : "0", 1);
#endif
    if (code) throw std::runtime_error("cannot select suite cache mode");
}
namespace {
struct FiniteProbe {
    std::string first_bad, error;
    size_t sampled_rows = 0;
    static bool selected(const ggml_tensor* tensor) {
        const char* name = ggml_get_name(tensor);
        return std::strncmp(name, "l_last-", 7) == 0 || std::strcmp(name, "result_norm") == 0 ||
               std::strcmp(name, "result_output") == 0;
    }
    static bool callback(ggml_tensor* tensor, bool ask, void* data) noexcept {
        auto& self = *static_cast<FiniteProbe*>(data);
        if (ask) return selected(tensor);
        try {
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] < 1 || tensor->ne[0] > 129280 ||
                tensor->nb[0] != sizeof(float))
                throw std::runtime_error("finite probe shape differs");
            std::vector<float> row(size_t(tensor->ne[0]));
            // Last logical row: covers both PP/TG mHC layout variants.
            // This is sampled attribution, not a whole-tensor finite assertion.
            size_t offset = 0;
            for (int dim = 1; dim < 4; ++dim) offset += (tensor->ne[dim] - 1) * tensor->nb[dim];
            ggml_backend_tensor_get(tensor, row.data(), offset, row.size() * sizeof(float));
            ++self.sampled_rows;
            for (float value : row) if (!std::isfinite(value)) {
                self.first_bad = ggml_get_name(tensor);
                return false;
            }
            return true;
        } catch (const std::exception& failure) { self.error = failure.what(); return false; }
    }
};
}
void session_diagnose(Runtime& runtime, const Options& options, const std::vector<llama_token>& input,
                      const std::filesystem::path& output) {
    FiniteProbe probe;
    std::string error;
    int status = -1;
    std::vector<llama_token> generated;
    try {
        // Keep the loaded model. Fresh canonical context, same active provider,
        // observe only layer boundaries; retain the original failure verdict.
        if (!runtime.reconfigure_context(options, error, FiniteProbe::callback, &probe))
            throw std::runtime_error(error);
        runtime.reset();
        status = runtime.decode_tokens(input.data(), input.size(), 0, true, error);
        for (int i = 0; status == 0 && i < 32; ++i) {
            generated.push_back(runtime.greedy(true));
            if (i != 31) status = runtime.decode_tokens(&generated.back(), 1, runtime.position(), true, error);
        }
        llama_synchronize(runtime.context());
        verify_device_expert_cache(runtime.context());
    } catch (const std::exception& failure) { error += failure.what(); }
    std::ofstream report(output / "finite-diagnosis.json");
    report << "{\"model_reloaded\":false,\"sample\":\"last logical row per layer boundary\",\"first_nonfinite_tensor\":"
           << session_json(probe.first_bad) << ",\"sampled_rows\":" << probe.sampled_rows
           << ",\"decode_status\":" << status << ",\"error\":" << session_json(error + probe.error)
           << ",\"output_ids\":[";
    for (size_t i = 0; i < generated.size(); ++i) { if (i) report << ','; report << generated[i]; }
    report << "]}\n";
}
}
