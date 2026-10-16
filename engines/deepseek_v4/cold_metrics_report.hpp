#pragma once
#include "cold_expert_provider.hpp"
#include <iosfwd>

namespace strata::deepseek_v4 {
ColdExpertMetrics cold_metrics_delta(const ColdExpertMetrics& after, const ColdExpertMetrics& before);
ColdExpertMetrics cold_metrics_sum(const ColdExpertMetrics& total, const ColdExpertMetrics& delta);
void write_cold_metrics(std::ostream& output, const ColdExpertMetrics& metrics, bool available);
}
