#pragma once

#include <cstring>
#include <optional>
#include <string_view>

#ifdef STRATA_DEEPSEEK_V4
int strata_deepseek_v4_serve(int argc, char** argv);
#endif

#ifdef STRATA_QWEN35_MOE
int strata_qwen35_moe_serve(int argc, char** argv);
#endif
#ifdef STRATA_QWEN35_DENSE
int strata_qwen35_dense_serve(int argc, char** argv);
#endif

namespace strata::models {

using ModelEntry = int (*)(int, char**);
struct ModelProvider {
    std::string_view family;
    ModelEntry entry;
};

// Qwen remains the existing main-program path. New families register one
// provider here; the public executable and protocol stay the same.
inline std::optional<int> dispatch_model(int argc, char** argv) {
    std::string_view requested;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--family") == 0) {
            requested = argv[i + 1];
            break;
        }
    }
    if (requested.empty() || requested == "qwen4exp") return std::nullopt;
    constexpr ModelProvider providers[] = {{"", nullptr},
#ifdef STRATA_DEEPSEEK_V4
        {"deepseek_v4", strata_deepseek_v4_serve},
#endif
#ifdef STRATA_QWEN35_MOE
        {"qwen35_moe", strata_qwen35_moe_serve},
#endif
#ifdef STRATA_QWEN35_DENSE
        {"qwen35_dense", strata_qwen35_dense_serve},
#endif
    };
    for (const auto& provider : providers) {
        if (provider.entry && provider.family == requested) return provider.entry(argc, argv);
    }
    return 2;
}

}  // namespace strata::models
