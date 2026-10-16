#pragma once

#include "expert_staging.hpp"
#include "llama.h"

namespace strata::deepseek_v4 {

void configure_expert_staging(llama_context* context, bool enabled,
                             std::shared_ptr<ExpertSourcePin>* retained = nullptr,
                             std::shared_ptr<PackedExpertStore>* packed = nullptr);
ExpertStagingMetrics expert_staging_metrics(const llama_context* context);
SourcePinMetrics expert_source_pin_metrics(const llama_context* context);

} // namespace strata::deepseek_v4
