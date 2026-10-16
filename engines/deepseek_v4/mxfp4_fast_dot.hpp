#pragma once
namespace strata::deepseek_v4 {
bool mxfp4_fast_dot_supported();
// One or two Q8_0 rows, MXFP4 weights. n=2048/4096; outputs have one/two floats.
void mxfp4_fast_dot(int n,const void* weight,const void* first,const void* second,float* output);
}
