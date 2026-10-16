#pragma once
#include "ggml.h"
#include <array>
namespace strata::deepseek_v4 {
// A borrowed projection of the authoritative cache, not a residency table.
// PP runs after the TG service is drained/deactivated. It reads the live
// directory and bank payload without changing admission or ownership.
struct PpCacheView {
    std::array<const ggml_tensor*,3> source{}, bank{};
    const ggml_tensor* directory = nullptr;
    uint32_t* prefill_counts=nullptr;int* prefill_tokens=nullptr;
};
}
