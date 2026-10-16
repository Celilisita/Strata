#pragma once
// Frozen DeepSeek4 GGML makes the already-Q8 source precision explicit.
// Older GGML has the same Q8 input contract without this template parameter.
#ifdef STRATA_DEEPSEEK_V4_GGML_API
#define STRATA_NATIVE_MMQ_Q8 , GGML_PREC_Q8
#else
#define STRATA_NATIVE_MMQ_Q8
#endif
