#pragma once
#include <strata/qwen35/native/attention_online.hpp>
namespace strata::qwen35::native::fp8_dual_cta_unit {
constexpr int TraceRows=20;
struct Trace {
    unsigned char *q_codes=nullptr,*k_codes=nullptr,*v_codes=nullptr,*p_codes=nullptr;
    float *q_scales=nullptr,*k_scales=nullptr,*v_scales=nullptr,*p_scales=nullptr,*scores=nullptr,*maximum=nullptr,*adjust=nullptr,*denominator=nullptr;
    unsigned short *p_half=nullptr;unsigned long long *flags=nullptr;
};
struct Resources {int registers,blocks,binary_version,kernel_count;size_t shared_bytes,local_bytes;int normal_blocks,diagnostic_blocks,prepare_blocks,normal_registers,diagnostic_registers,prepare_registers;};
void configure();
Resources resources();
size_t workspace_bytes(PrefillAttentionShape);
void attention(const float*,const void*,const void*,float*,void*,size_t,PrefillAttentionShape,bool rescale_skip,cudaStream_t,Trace trace={});
// Trace rows are rows{0,31,32,63,tokens-1} x heads{0,5,6,23}, head-major.
// q_codes=20*256; q_scales=20. K/V codes=[valid,4,256], Kscale=[valid,4].
// Vscale=[valid,4]. Pscale/statistics=[20,ceil(valid/64)]; P/score/P16=[20,valid].
// P8 encodes causal HalfP*Vscale/Pscale; PV restores only Pscale (Vscale is folded into P).
// Future causal tiles are initialized score=-Inf/P=0/statistics=0 (-Inf maximum).
// flags[4]={nonfinite_operands,finite_clip,invalid_scale,reserved}; optional trace is all-or-none.
}
