#pragma once
#include "strata/qwen35/completed_v100/native/small_prefill.hpp"

namespace strata::completed_v100::qwen35::native {
// Distinct prepared view: cannot be passed to any raw/byte-code Matrix operator.
// Code plane: [row_tile64][K256][chunk128][row64][half128].
// Metadata follows all codes: [row_tile64][K256][row64][32 bytes].
struct HalfCodeMatrix {
    const void *data=nullptr;
    int rows=0,columns=0;
    size_t bytes=0;
};
constexpr size_t HalfCodeBlockBytes=544;
inline size_t half_code_matrix_bytes(HalfCodeMatrix weight){
    if(!weight.data||weight.rows<=0||weight.rows%64||weight.columns<=0||weight.columns%256||
       weight.bytes!=size_t(weight.rows)*size_t(weight.columns/256)*HalfCodeBlockBytes)
        throw std::invalid_argument("invalid tiled half-code view or declared extent");
    return weight.bytes;
}
void small_prefill_q6_register_linear(Matrix,const void *,float *,cudaStream_t,bool prefetch=false);
size_t prepared_q6_half_codes_bytes(Matrix raw);
HalfCodeMatrix prepare_q6_half_codes(Matrix raw,void *output,cudaStream_t stream);
size_t small_prefill_half_codes_split_bytes(HalfCodeMatrix prepared,int tokens);
void small_prefill_half_codes_split_linear(HalfCodeMatrix prepared,const void *input,float *output,int tokens,
                                         void *scratch,size_t bytes,cudaStream_t stream,int processors);
void small_prefill_half_codes_compact_linear(HalfCodeMatrix prepared,const void *input,float *output,int tokens,
                                            void *scratch,size_t bytes,cudaStream_t stream,int processors);
void small_prefill_half_codes_complete_linear(HalfCodeMatrix prepared,const void *input,float *output,int tokens,
                                             cudaStream_t stream,int processors);
// One stream owns this scratch lease. Tickets are initialized once at setup,
// then the final producer resets them before the ordered kernel completes.
struct HalfCodeWorkspace {
    void *data=nullptr;
    size_t bytes=0;
    int rows=0;
    cudaStream_t stream=nullptr;
    bool ready=false;
};
size_t half_code_workspace_bytes(HalfCodeMatrix prepared);
HalfCodeWorkspace initialize_half_code_workspace(HalfCodeMatrix prepared,void *scratch,
                                                size_t bytes,cudaStream_t stream);
void small_prefill_half_codes_fused_linear(HalfCodeMatrix prepared,const void *input,float *output,int tokens,
                                         HalfCodeWorkspace workspace,cudaStream_t stream,int processors);
}
