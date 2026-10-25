#pragma once
#include "strata/gemma4/native/ffn.hpp"

namespace strata::gemma4::native {
// Project writes [token][2*rows]: Gate rows followed by Up rows. This is not
// the parent FfnBuffers layout with Up at gate+maximum_tokens*rows.
struct HalfRowPairOutput {
    float* interleaved=nullptr;
    float* activation=nullptr;
    size_t interleaved_values=0,activation_values=0;
    int rows=0,tokens=0;
};
struct HalfRowPairLease {
    const float* input=nullptr;
    void *weight_arena=nullptr,*input_arena=nullptr,*workspace=nullptr;
    size_t weight_bytes=0,input_bytes=0,workspace_bytes=0;
    HalfRowPairOutput output;
    cublasHandle_t handle=nullptr;
    cudaStream_t stream=nullptr;
};
// K896 is fixed for this candidate. Project leaves the interleaved Gate/Up
// available for numerical inspection; callers invoke GELU separately.
void half_row_pair_project(primitive::Matrix,primitive::Matrix,const HalfRowPairLease&);
void half_row_pair_gelu(const HalfRowPairOutput&,cudaStream_t);
// Optional diagnostic deinterleave, excluded from the production FFN path.
void half_row_pair_unpack(const HalfRowPairOutput&,float* gate,float* up,
    size_t branch_values,cudaStream_t);
}
