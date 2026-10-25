#pragma once
#include "strata/qwen35/completed_v100/native/attention_prefill.hpp"
#include "strata/qwen35/completed_v100/native/dense_gemm.hpp"

namespace strata::completed_v100::qwen35::native {
struct AttentionGemmLayout {
    size_t q_elements,kv_elements,score_elements,bytes;
    void *query=nullptr,*key=nullptr,*value=nullptr,*scores=nullptr,*probability_hi=nullptr,*probability_lo=nullptr;
};
AttentionGemmLayout attention_gemm_layout(PrefillAttentionShape,void *workspace);
// Stable causal softmax, FP32 value accumulation; two scaled half probability
// components retain FP32 probabilities without a slow scalar FP32 GEMM.
// Capacity failure returns false before issuing CUDA work, allowing native fallback.
bool gemm_prefill_attention(DenseGemm &,const float *,const void *,const void *,float *,float *head_output,
                            void *,size_t,PrefillAttentionShape,cudaStream_t);
}
