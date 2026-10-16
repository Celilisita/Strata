#pragma once

#include "cold_expert_provider.hpp"
struct llama_context;

namespace strata::deepseek_v4 {
void configure_cold_experts(llama_context* context, int threads, bool admit_free);
ColdExpertMetrics cold_expert_metrics(const llama_context* context);
}
