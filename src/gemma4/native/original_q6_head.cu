#include "original_q6_head.hpp"
#include <cuda_fp16.h>
#include <cstdlib>
#include <stdexcept>
#include <string>
namespace strata::gemma4::native {
namespace {
// Four independent rows/CTA, one warp/row. A lane loads the low/high
// bytes once for eight columns; signed subgroup scales are broadcast by half-warp.
__global__ void q6_head_four_rows(const unsigned char* raw,const float* x,float* out){
    const int lane=threadIdx.x%32,row=int(blockIdx.x)*4+int(threadIdx.x)/32;
    const auto* base=raw+std::size_t(row)*21*210;float value=0;
    for(int block=0;block<21;++block){
        const auto* p=base+block*210;
        const float delta=__shfl_sync(0xffffffff,lane==0?__half2float(*reinterpret_cast<const half*>(p+208)):0.f,0);
        const unsigned low0=p[lane],low1=p[32+lane],low2=p[64+lane],low3=p[96+lane];
        const unsigned high0=p[128+lane],high1=p[160+lane];
#pragma unroll
        for(int group=0;group<8;++group){
            const float own=(lane%16==0)?__fmul_rn(delta,float(reinterpret_cast<const signed char*>(p+192)[2*group+lane/16])):0.f;
            const float coefficient=__shfl_sync(0xffffffff,own,lane&16);
            const unsigned low=group<4?(group&1?low1:low0):(group&1?low3:low2);
            const unsigned high=group<4?high0:high1;
            const int code=int(((low>>((group%4)/2*4))&15)|(((high>>((group%4)*2))&3)<<4))-32;
            const float weight=__fmul_rn(coefficient,float(code));
            value=__fmaf_rn(weight,x[block*256+group*32+lane],value);
        }
    }
    for(int offset=16;offset;offset/=2)value=__fadd_rn(value,__shfl_down_sync(0xffffffff,value,offset));
    if(!lane)out[row]=value;
}
template<int Rows> __global__ void q6_head_multi_rows(const unsigned char* raw,const float* x,float* out){
    static_assert(Rows==2||Rows==3,"Tied Q6 head multirow extent");
    const int lane=threadIdx.x%32,row=int(blockIdx.x)*4+int(threadIdx.x)/32;
    const auto* base=raw+std::size_t(row)*21*210;float values[Rows]{};
    for(int block=0;block<21;++block){
        const auto* p=base+block*210;
        const float delta=__shfl_sync(0xffffffff,lane==0?__half2float(*reinterpret_cast<const half*>(p+208)):0.f,0);
        const unsigned low0=p[lane],low1=p[32+lane],low2=p[64+lane],low3=p[96+lane];
        const unsigned high0=p[128+lane],high1=p[160+lane];
#pragma unroll
        for(int group=0;group<8;++group){
            const float own=(lane%16==0)?__fmul_rn(delta,float(reinterpret_cast<const signed char*>(p+192)[2*group+lane/16])):0.f;
            const float coefficient=__shfl_sync(0xffffffff,own,lane&16);
            const unsigned low=group<4?(group&1?low1:low0):(group&1?low3:low2);
            const unsigned high=group<4?high0:high1;
            const int code=int(((low>>((group%4)/2*4))&15)|(((high>>((group%4)*2))&3)<<4))-32;
            const float weight=__fmul_rn(coefficient,float(code));
#pragma unroll
            for(int r=0;r<Rows;++r)
                values[r]=__fmaf_rn(weight,x[std::size_t(r)*5376+block*256+group*32+lane],values[r]);
        }
    }
#pragma unroll
    for(int r=0;r<Rows;++r){
        for(int offset=16;offset;offset/=2)values[r]=__fadd_rn(values[r],__shfl_down_sync(0xffffffff,values[r],offset));
        if(!lane)out[std::size_t(r)*262144+row]=values[r];
    }
}
}
bool original_q6_head_requested(){
    static const bool enabled=[](){const char* p=std::getenv("STRATA_GEMMA_ORIGINAL_Q6_HEAD");
        if(!p||std::string(p)=="0")return false;if(std::string(p)=="1")return true;
        throw std::invalid_argument("STRATA_GEMMA_ORIGINAL_Q6_HEAD requires0/1");}();return enabled;
}
void original_q6_head(primitive::Matrix m,const float* input,float* output,cudaStream_t stream){
    if(!m.data||!input||!output||input==output||m.quant!=primitive::Quant::Q6K||
        m.layout!=primitive::BlockLayout::Gguf||m.rows!=262144||m.columns!=5376)
        throw std::invalid_argument("Original tied-Q6 head requires raw262144x5376 F32 input");
    q6_head_four_rows<<<65536,128,0,stream>>>(static_cast<const unsigned char*>(m.data),input,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma fixed raw-Q6 F32 head");
}
void original_q6_head_rows(primitive::Matrix m,const float* input,float* output,int rows,cudaStream_t stream){
    if(!m.data||!input||!output||input==output||m.quant!=primitive::Quant::Q6K||
        m.layout!=primitive::BlockLayout::Gguf||m.rows!=262144||m.columns!=5376||(rows!=2&&rows!=3))
        throw std::invalid_argument("Original tied-Q6 multirow head requires2/3 raw262144x5376 F32 inputs");
    if(rows==2)q6_head_multi_rows<2><<<65536,128,0,stream>>>(static_cast<const unsigned char*>(m.data),input,output);
    else q6_head_multi_rows<3><<<65536,128,0,stream>>>(static_cast<const unsigned char*>(m.data),input,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma fixed raw-Q6 F32 multirow head");
}
}
