#pragma once

#include <cstdint>

namespace strata::deepseek_v4 {

// Admission still belongs to the canonical cache. This only permits its bank
// views for PP when the same bank can hold the layer's entire expert set.
bool pp_cache_bank_eligible(int64_t tokens, int64_t experts, int64_t slots);

}
