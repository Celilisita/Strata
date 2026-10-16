#pragma once
#include "session_gate.hpp"
#include <vector>
namespace strata::deepseek_v4 {
class PrefixDispatchSnapshots;
struct PrefixReference {
    std::vector<llama_token> input, output;
    std::vector<std::vector<float>> logits;
};
struct PrefixDifference {
    int step = -1, token_step = -1;
    uint64_t differing = 0;
    double max_abs = 0, rms = 0;
    llama_token expected = -1, actual = -1;
};
PrefixReference prefix_read_reference(const std::filesystem::path&);
PrefixDifference prefix_case(Runtime&, PrefixReference&, const std::filesystem::path&,
    const std::string& name, bool collect, int count, bool capture_candidate = false,
    PrefixDispatchSnapshots* snapshots = nullptr, bool follow_actual_ids = false);
void prefix_write_difference(std::ostream&, const PrefixDifference&);
}
