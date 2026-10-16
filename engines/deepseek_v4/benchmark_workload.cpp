#include "benchmark_workload.hpp"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace strata::deepseek_v4 {
std::vector<llama_token> benchmark_input(Runtime& runtime, const Options& options, const DriverOptions& driver) {
    auto input = runtime.tokenize(options.prompt);
    if (input.empty()) throw std::runtime_error("benchmark prompt has no tokens");
    if (driver.prompt_tokens) {
        const auto target = std::size_t(driver.prompt_tokens);
        if (input.size() < target)
            throw std::runtime_error("prompt supplies " + std::to_string(input.size()) +
                                     " tokens; exact benchmark target is " + std::to_string(target));
        // Keep the start and the final instructions/chat markers. This is
        // explicit benchmark input preparation, before any model evaluation.
        const auto tail = std::min(std::size_t(256), target / 2);
        const auto first = target - tail;
        if (input.size() > target)
            std::copy(input.end() - tail, input.end(), input.begin() + first);
        input.resize(target);
    }
    if (driver.mode != "layer" && input.size() > 16384)
        throw std::runtime_error("PP/TG prompt exceeds 16384 tokens");
    return input;
}

void benchmark_progress(const std::filesystem::path& directory, const char* event, uint32_t request_index,
                        const Runtime& runtime, double request_ms, std::size_t input_tokens, int generated_tokens) {
    const auto perf = runtime.performance();
    std::ofstream output(directory / "progress.tsv", std::ios::app);
    output << std::setprecision(12) << event << '\t' << request_index << '\t' << request_ms << '\t'
           << input_tokens << '\t' << generated_tokens << '\t' << perf.n_p_eval << '\t'
           << perf.t_p_eval_ms << '\t' << perf.n_eval << '\t' << perf.t_eval_ms << '\n';
    if (!output) throw std::runtime_error("cannot write bounded benchmark progress");
}
} // namespace strata::deepseek_v4
