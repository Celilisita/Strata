#include "strata/qwen35/completed_v100/native/token_ops.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "strata/qwen35/completed_v100/native/original_formats.hpp"
#include "codec.cuh"
#include "q8_raw.cuh"
#include <math_constants.h>

namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void embedding(Matrix m,const int *ids,float *out,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const int token=ids[i/m.columns],column=i%m.columns;
    if(token<0 || token>=m.rows){out[i]=CUDART_NAN_F;return;}
    if(m.quant==Quant::Q8){out[i]=q8_raw_value(m.data,m.columns,token,column);return;}
    const size_t block=size_t(token)*(m.columns/256)+column/256;
    out[i]=block_element(static_cast<const unsigned char *>(m.data)+block*(m.quant==Quant::Q4K?144:210),m.quant,column%256,m.layout);
}
}
void gather_embedding(Matrix m,const int *ids,float *out,int tokens,cudaStream_t s){
    original_matrix_bytes(m);
    if(m.quant!=Quant::Q8&&m.quant!=Quant::Q4K&&m.quant!=Quant::Q6K)throw std::invalid_argument("Embedding format needs its registered GPU gather");
    if(!ids || !out || tokens<1 || tokens>8192)throw std::invalid_argument("invalid own embedding invocation");
    const size_t n=size_t(tokens)*m.columns;embedding<<<unsigned((n+255)/256),256,0,s>>>(m,ids,out,n);
    check_cuda(cudaGetLastError(),"own device embedding gather");
}
}
