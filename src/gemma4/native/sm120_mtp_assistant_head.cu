#include "sm120_mtp_assistant_head.hpp"
#include <cuda_fp16.h>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::gemma4::native::mtp_assistant::head399 {
namespace {
constexpr int Threads=128,LanesPerRow=8,RowsPerCta=Threads/LanesPerRow;
__global__ void direct_head(const unsigned char* weights,const float* input,float* output){
    __shared__ float x[InputValues];
    const int tid=threadIdx.x,lane=tid%LanesPerRow;
    const int row=int(blockIdx.x)*RowsPerCta+tid/LanesPerRow;
    for(int i=tid;i<InputValues;i+=Threads)x[i]=input[i];
    __syncthreads();
    const auto* first=weights+std::size_t(row)*(InputValues/32)*18;
    float value=0;
    #pragma unroll 1
    for(int b=0;b<InputValues/32;++b){
        const auto* block=first+b*18;
        const float d=__half2float(*reinterpret_cast<const __half*>(block));
        const unsigned q=*reinterpret_cast<const unsigned short*>(block+2+2*lane);
        const int j=b*32+2*lane;
        const float w0=__fmul_rn(d,float(int(q&15u)-8));
        const float w1=__fmul_rn(d,float(int((q>>8)&15u)-8));
        const float w2=__fmul_rn(d,float(int((q>>4)&15u)-8));
        const float w3=__fmul_rn(d,float(int((q>>12)&15u)-8));
        value=__fmaf_rn(w0,x[j],value);
        value=__fmaf_rn(w1,x[j+1],value);
        value=__fmaf_rn(w2,x[j+16],value);
        value=__fmaf_rn(w3,x[j+17],value);
    }
    #pragma unroll
    for(int delta=4;delta;delta/=2)
        value=__fadd_rn(value,__shfl_xor_sync(0xffffffffu,value,delta,LanesPerRow));
    if(!lane)output[row]=value;
}
bool span(const void* p,std::size_t bytes,std::size_t alignment){
    const auto address=reinterpret_cast<std::uintptr_t>(p);
    return p&&address%alignment==0&&bytes<=(std::numeric_limits<std::uintptr_t>::max)()-address;
}
bool overlap(const void* a,std::size_t na,const void* b,std::size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
}
void project(const float* input,float* output,Lease w,Binding live){
    constexpr std::size_t input_bytes=InputValues*sizeof(float),output_bytes=OutputValues*sizeof(float);
    if(!live.owner||!live.stream||w.binding.owner!=live.owner||w.binding.epoch!=live.epoch||
       w.binding.stream!=live.stream||w.matrix.rows!=OutputValues||w.matrix.columns!=InputValues||
       w.weight_bytes!=WeightBytes||!span(w.matrix.data,WeightBytes,2)||
       !span(input,input_bytes,alignof(float))||!span(output,output_bytes,alignof(float)))
        throw std::invalid_argument("399 Assistant Q4_0 head binding/shape/span");
    if(overlap(w.matrix.data,WeightBytes,input,input_bytes)||
       overlap(w.matrix.data,WeightBytes,output,output_bytes)||overlap(input,input_bytes,output,output_bytes))
        throw std::invalid_argument("399 Assistant head immutable/input/output alias");
    static_assert(OutputValues%RowsPerCta==0,"399 exact head rows");
    direct_head<<<OutputValues/RowsPerCta,Threads,0,live.stream>>>(
        static_cast<const unsigned char*>(w.matrix.data),input,output);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)throw std::runtime_error(std::string("399 Assistant direct head launch: ")+cudaGetErrorString(status));
}
}
