#include "strata/gemma4/native/decoder_ops.hpp"
#include "codec.cuh"
#include <cmath>
namespace strata::gemma4::native {
namespace {
__device__ float sum(float x){for(int d=16;d;d/=2)x=__fadd_rn(x,__shfl_down_sync(0xffffffff,x,d));return x;}
__global__ void matvec(primitive::Matrix m,const float* x,float* y){
    __shared__ float partial[4];const int row=blockIdx.x,lane=threadIdx.x%32,warp=threadIdx.x/32;
    const int stride=m.quant==primitive::Quant::Q4K?144:210;float v=0;
    const auto* start=static_cast<const unsigned char*>(m.data)+size_t(row)*(m.columns/256)*stride;
    for(int k=threadIdx.x;k<m.columns;k+=128)v=__fmaf_rn(primitive::block_element(start+(k/256)*stride,m.quant,k%256),x[k],v);
    v=sum(v);if(!lane)partial[warp]=v;__syncthreads();
    if(warp==0){v=lane<4?partial[lane]:0.f;v=sum(v);if(!lane)y[row]=v;}
}
__global__ void embedding(primitive::Matrix m,const int* ids,float* out,int tokens,float scale,unsigned* invalid){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=size_t(tokens)*m.columns)return;
    const int token=ids[i/m.columns],k=i%m.columns;
    if(token<0||token>=m.rows){atomicExch(invalid,1u);out[i]=0.f;return;}
    const int stride=m.quant==primitive::Quant::Q4K?144:210;
    const auto* start=static_cast<const unsigned char*>(m.data)+(size_t(token)*(m.columns/256)+k/256)*stride;
    out[i]=primitive::block_element(start,m.quant,k%256)*scale;
}
}
void project_vector(primitive::Matrix m,const float* x,float* y,cudaStream_t stream){
    if(!m.data||!x||!y||x==y||m.rows<1||m.columns%256||m.layout!=primitive::BlockLayout::Gguf||
       (m.quant!=primitive::Quant::Q4K&&m.quant!=primitive::Quant::Q6K))throw std::invalid_argument("Gemma raw FP32 projection geometry");
    matvec<<<m.rows,128,0,stream>>>(m,x,y);primitive::check_cuda(cudaGetLastError(),"Gemma raw quantized FP32 matvec");
}
void embed(primitive::Matrix m,const int* ids,float* out,int tokens,float scale,unsigned* invalid,cudaStream_t stream){
    if(!ids||!out||!invalid||tokens<1||m.columns%256||m.rows<1||!std::isfinite(scale))throw std::invalid_argument("Gemma embedding geometry");
    embedding<<<unsigned((size_t(tokens)*m.columns+255)/256),256,0,stream>>>(m,ids,out,tokens,scale,invalid);
    primitive::check_cuda(cudaGetLastError(),"Gemma scaled raw embedding");
}
}
