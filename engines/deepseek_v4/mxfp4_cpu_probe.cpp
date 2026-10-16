#include "mxfp4_cpu_probe.hpp"
#include "mxfp4_cpu_service.hpp"
#include "cpu_expert_service.hpp"
#include "llama-model.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <vector>
namespace strata::deepseek_v4 {
bool mxfp4_cpu_operator_probe(const llama_model& model, const std::array<ggml_tensor*, 3>& banks,
                              const float* norm, std::ostream& report) {
    CpuExpertService reference(model, 4);
    Mxfp4CpuService native(4);
    const Mxfp4CpuBank view{banks[0], banks[1], banks[2], model.hparams.swiglu_clamp_exp[0]};
    std::array<int32_t, 6> ids{0, 43, 86, 129, 172, 215};
    std::array<float, 6 * 4096> a{}, b{};
    auto time = [](auto call) {
        const auto begin = std::chrono::steady_clock::now(); call();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
    };
    bool passed = true;
    report << "{\"threads\":4,\"input_origin\":\"actual layer-zero PP5 norm row0\",\"scope\":\"operator only; exact original MXFP4; same CPU quantization/math; sample routes rotate, original and pinned-copy source addresses differ\",\"cases\":[";
    for (int count : {1, 3, 6}) {
        reference.compute(0, norm, ids.data(), count, a.data());
        native.compute(view, norm, ids.data(), count, b.data());
        std::array<double, 3> ra{}, rb{};
        for (int i = 0; i < 3; ++i) {
            for (int r = 0; r < 6; ++r) ids[r] = (i * 17 + count * 29 + r * 43) % 256;
            ra[i] = time([&] { reference.compute(0, norm, ids.data(), count, a.data()); });
            rb[i] = time([&] { native.compute(view, norm, ids.data(), count, b.data()); });
        }
        double max_abs = 0, square_error = 0, square_ref = 0;
        uint64_t nonfinite = 0;
        for (int i = 0; i < count * 4096; ++i) {
            if (!std::isfinite(a[i]) || !std::isfinite(b[i])) { ++nonfinite; continue; }
            const double e = double(a[i]) - b[i]; max_abs = std::max(max_abs, std::abs(e));
            square_error += e * e; square_ref += double(a[i]) * a[i];
        }
        const double relative_rms = std::sqrt(square_error / std::max(square_ref, 1e-30));
        const bool valid = !nonfinite && relative_rms <= 1e-4 && max_abs <= 0.01;
        passed &= valid;
        auto sa = ra, sb = rb; std::sort(sa.begin(), sa.end()); std::sort(sb.begin(), sb.end());
        if (count != 1) report << ',';
        report << std::setprecision(12) << "{\"experts\":" << count << ",\"source_payload_bytes\":" << count * 13369344ull
            << ",\"contribution_bytes\":" << count * 4096 * sizeof(float)
            << ",\"canonical_ms\":[" << ra[0] << ',' << ra[1] << ',' << ra[2]
            << "],\"native_ms\":[" << rb[0] << ',' << rb[1] << ',' << rb[2]
            << "],\"speedup\":" << sa[1] / sb[1] << ",\"max_abs\":" << max_abs
            << ",\"relative_rms\":" << relative_rms << ",\"nonfinite\":" << nonfinite
            << ",\"math_passed\":" << (valid ? "true" : "false") << '}';
    }
    report << "]}";
    return passed;
}
}
