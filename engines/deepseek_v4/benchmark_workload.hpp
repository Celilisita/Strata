#pragma once
#include "benchmark_args.hpp"
#include "runtime.hpp"

namespace strata::deepseek_v4 {
std::vector<llama_token> benchmark_input(Runtime&, const Options&, const DriverOptions&);
void benchmark_progress(const std::filesystem::path&, const char* event, uint32_t request_index,
                        const Runtime&, double request_ms, std::size_t input_tokens, int generated_tokens);
}
