#pragma once
#include <array>
#include <cstdint>
namespace strata::glm5_next {
// One serial worker's bounded transient activations/results. No weights or
// residency authority; allocate once outside service and reuse until drained.
struct HybridFourScratch {
    std::array<int32_t,32> selected{},ranks{};
    std::array<uint8_t,32> positions{};
    std::array<float,4*4096> activation{};
    std::array<float,32*4096> output{};
};
}
