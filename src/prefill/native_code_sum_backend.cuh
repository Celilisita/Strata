#pragma once
#include "common.cuh"
// DS.high stores an integer code sum. Only the DP4A loader understands it.
// Scope this selection to the custom translation unit; ordinary MMQ keeps MMA.
#pragma push_macro("ggml_cuda_mmq_config")
#pragma push_macro("TURING_MMA_AVAILABLE")
#pragma push_macro("BLACKWELL_MMA_AVAILABLE")
#pragma push_macro("turing_mma_available")
#pragma push_macro("blackwell_mma_available")
#undef ggml_cuda_mmq_config
#define ggml_cuda_mmq_config strata_code_sum_mmq_config
#undef TURING_MMA_AVAILABLE
#undef BLACKWELL_MMA_AVAILABLE
#define turing_mma_available(cc) false
#define blackwell_mma_available(cc) false
#include "mmq.cuh"
#pragma pop_macro("blackwell_mma_available")
#pragma pop_macro("turing_mma_available")
#pragma pop_macro("BLACKWELL_MMA_AVAILABLE")
#pragma pop_macro("TURING_MMA_AVAILABLE")
#pragma pop_macro("ggml_cuda_mmq_config")
