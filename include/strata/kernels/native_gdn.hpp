#pragma once
#include "strata/kernels/gdn.hpp"
#include <cstddef>

namespace strata::kernels {

// Configure before creating/capturing a session. Existing captures retain their
// selected kernel, so changing the flag does not change an existing graph.
void native_gdn_set_enabled(bool enabled);
bool native_gdn_enabled();

// Pinned CUDA gated_delta_net recurrence for one token, S=128 and h_v % h_k=0.
// q/k are normalized F32 inputs; q MUST NOT be pre-scaled by 1/sqrt(S).
// Scaling is applied to the completed readout, as in the pinned fused operator.
// State retains Strata's layout: state[(i*h_v+h)*S+j]. The caller owns it and
// output; both must be disjoint from each other and from every read-only input.
// All spans must be aligned to four bytes. A nonnull ordered stream is required.
// No allocation, wait, or change to normalization/gate preprocessing occurs.
void native_gdn_step(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shapes, void* stream);

// Same per-token arithmetic/reduction; keep state in F32 registers across T.
// Q/K/V are strided token rows; output is contiguous [T,h_v,128].
void native_gdn_scan(float* state, const float* q, const float* k, const float* v,
                     const float* gate, const float* beta, float* output,
                     const GdnShapes& shapes, int tokens, int input_stride, void* stream, float* first_token_state = nullptr, int column_tile = 1);

// Opt-in ordinary PP algebra: metadata plus one persistent two-token scan.
// No allocation/synchronization. Metadata uses ceil(T/2)*h_v*32 bytes; caller
// supplies dead phase scratch disjoint from every live input/state/output.
bool native_gdn_pair2_enabled();
size_t native_gdn_pair2_workspace(const GdnShapes& shape,int tokens);
void native_gdn_scan_pair2(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,int tokens,
    int input_stride,void* scratch,size_t scratch_bytes,void* stream);

// Four-token variant: ceil(T/4)*h_v*96 bytes, same lifetime contract.
bool native_gdn_chunk4_enabled();
size_t native_gdn_chunk4_workspace(const GdnShapes& shape,int tokens);
void native_gdn_scan_chunk4(float* state,const float* q,const float* k,const float* v,
    const float* gate,const float* beta,float* output,const GdnShapes& shape,int tokens,
    int input_stride,void* scratch,size_t scratch_bytes,void* stream);

// Ordinary PP explicit alpha input; legacy APIs continue accepting log gate.
void native_gdn_scan_alpha(float* state,const float* q,const float* k,const float* v,
    const float* alpha,const float* beta,float* output,const GdnShapes& shape,int tokens,int input_stride,void* stream);
} // namespace strata::kernels
