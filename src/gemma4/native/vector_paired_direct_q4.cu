#include "strata/gemma4/native/vector_paired_direct_q4.hpp"
#include <cuda_fp16.h>
#include <limits>
namespace strata::gemma4::native {
namespace {
constexpr int Threads=128,Rows=4,Stripes=4,Columns=5376,Outputs=21504;
constexpr size_t RawBytes=size_t(Outputs)*(Columns/256)*144,InputBytes=size_t(Columns/32)*40;
struct Span {const void* p;size_t bytes;};
bool valid(Span s){
    const auto p=reinterpret_cast<std::uintptr_t>(s.p);
    return s.p&&s.bytes&&p%4==0&&s.bytes<=(std::numeric_limits<std::uintptr_t>::max)()-p;
}
bool overlap(Span a,Span b){
    const auto x=reinterpret_cast<std::uintptr_t>(a.p),y=reinterpret_cast<std::uintptr_t>(b.p);
    return x<=y?y-x<a.bytes:x-y<b.bytes;
}
__device__ unsigned metadata(const unsigned char* raw,int group){
    const unsigned shift=unsigned(group&3)*8;
    const unsigned scale=(*reinterpret_cast<const unsigned*>(raw+4)>>shift)&255u;
    const unsigned minimum=(*reinterpret_cast<const unsigned*>(raw+8)>>shift)&255u;
    if(group<4)return (scale&63u)|((minimum&63u)<<6);
    const unsigned high=(*reinterpret_cast<const unsigned*>(raw+12)>>shift)&255u;
    return ((high&15u)|((scale>>6)<<4))|(((high>>4)|((minimum>>6)<<4))<<6);
}
__device__ float finish_term(const unsigned char* raw,const SingleQ8Block& x,
    int group,int dot,float previous){
    const unsigned descriptor=metadata(raw,group),delta=*reinterpret_cast<const unsigned*>(raw);
    const float scale=__half2float(__ushort_as_half(static_cast<unsigned short>(delta)))*float(descriptor&63);
    const float minimum=__half2float(__ushort_as_half(static_cast<unsigned short>(delta>>16)))*float(descriptor>>6);
    const float product=__fmaf_rn(x.d,float(dot),0.f),correction=__fmaf_rn(x.d,float(x.sum),0.f);
    return __fadd_rn(previous,__fmaf_rn(scale,product,-__fmul_rn(minimum,correction)));
}
__device__ void pair_term(const unsigned char* gate,const unsigned char* up,const SingleQ8Block& x,
    int group,float& a,float& b){
    int g0=0,g1=0,u0=0,u1=0;
    const unsigned shift=unsigned(group&1)*4;
    // Consume one16B weight chunk and two8B code chunks before the next half.
    // Each integer dot keeps the parent's four-word accumulators and final sum.
    #pragma unroll
    for(int half=0;half<2;++half){
        const int offset=16+(group/2)*32+half*16;
        const uint4 gw=*reinterpret_cast<const uint4*>(gate+offset);
        const uint4 uw=*reinterpret_cast<const uint4*>(up+offset);
        int ga=0,ua=0;
        {
            const uint2 code=*reinterpret_cast<const uint2*>(x.code+half*16);
            ga=__dp4a(int((gw.x>>shift)&0x0f0f0f0fu),int(code.x),ga);
            ua=__dp4a(int((uw.x>>shift)&0x0f0f0f0fu),int(code.x),ua);
            ga=__dp4a(int((gw.y>>shift)&0x0f0f0f0fu),int(code.y),ga);
            ua=__dp4a(int((uw.y>>shift)&0x0f0f0f0fu),int(code.y),ua);
        }
        {
            const uint2 code=*reinterpret_cast<const uint2*>(x.code+half*16+8);
            ga=__dp4a(int((gw.z>>shift)&0x0f0f0f0fu),int(code.x),ga);
            ua=__dp4a(int((uw.z>>shift)&0x0f0f0f0fu),int(code.x),ua);
            ga=__dp4a(int((gw.w>>shift)&0x0f0f0f0fu),int(code.y),ga);
            ua=__dp4a(int((uw.w>>shift)&0x0f0f0f0fu),int(code.y),ua);
        }
        if(!half){g0=ga;u0=ua;}else{g1=ga;u1=ua;}
    }
    a=finish_term(gate,x,group,g0+g1,a);b=finish_term(up,x,group,u0+u1,b);
}
__device__ float sum_stripes(float value,unsigned active){
    #pragma unroll
    for(int distance=4;distance;distance/=2)value=__fadd_rn(value,__shfl_xor_sync(active,value,distance));
    value=__fadd_rn(value,__shfl_xor_sync(active,value,16));
    return __fadd_rn(value,__shfl_xor_sync(active,value,8));
}
__global__ void multiply(primitive::Matrix gate,primitive::Matrix up,const SingleQ8Block* input,
    float* first,float* second,float* gelu){
    const int row=int(blockIdx.x)*Rows+threadIdx.x/32,lane=threadIdx.x%32;
    const int group=lane%8,stripe=lane/8;const unsigned active=__activemask();
    float a=0.f,b=0.f;
    for(int k=stripe;k<Columns/256;k+=Stripes){
        const size_t offset=(size_t(row)*(Columns/256)+k)*144;
        pair_term(static_cast<const unsigned char*>(gate.data)+offset,
            static_cast<const unsigned char*>(up.data)+offset,input[k*8+group],group,a,b);
    }
    const float value=sum_stripes(a,active),other=sum_stripes(b,active);
    if(!lane){
        first[row]=value;second[row]=other;
        const float z=0.7978845608028654f*(value+0.044715f*value*value*value);
        gelu[row]=(0.5f*value*(1.f+tanhf(z)))*other;
    }
}
}
PairedQ4DirectResources paired_q4_direct_resources(){
    cudaFuncAttributes a{};primitive::check_cuda(cudaFuncGetAttributes(&a,multiply),"Gemma291 coldresources");
    return {a.numRegs,int(a.sharedSizeBytes),int(a.localSizeBytes),a.maxThreadsPerBlock};
}
void pair_single_q4_direct_t1(PairedQ4WordLease lease,float* gate,float* up,float* gelu,PairedQ4WordBinding live){
    const auto b=lease.binding;const auto x=lease.input;
    if(!live.owner||!live.epoch||!live.stream||b.owner!=live.owner||b.epoch!=live.epoch||b.stream!=live.stream)
        throw std::invalid_argument("Gemma291 paired-word owner/epoch/stream");
    for(auto w:{lease.gate,lease.up})if(w.quant!=primitive::Quant::Q4K||w.layout!=primitive::BlockLayout::Gguf||
        w.rows!=Outputs||w.columns!=Columns)throw std::invalid_argument("Gemma291 exactnativeQ4 GateUp");
    if(x.columns!=Columns||x.tokens!=1||x.bytes!=InputBytes)
        throw std::invalid_argument("Gemma291 typed Single40 T1");
    if(reinterpret_cast<std::uintptr_t>(lease.gate.data)%16||reinterpret_cast<std::uintptr_t>(lease.up.data)%16||
        reinterpret_cast<std::uintptr_t>(x.data)%8)throw std::invalid_argument("Gemma291 weight16/input8 alignment");
    const Span spans[]{{lease.gate.data,RawBytes},{lease.up.data,RawBytes},{lease.normalized,size_t(Columns)*4},
        {x.data,x.bytes},{gate,size_t(Outputs)*4},{up,size_t(Outputs)*4},{gelu,size_t(Outputs)*4}};
    for(const auto s:spans)if(!valid(s))throw std::invalid_argument("Gemma291 aligned span/overflow");
    for(int i=0;i<7;++i)for(int j=0;j<i;++j)
        if(overlap(spans[i],spans[j]))throw std::invalid_argument("Gemma291 source/weight/output alias");
    multiply<<<Outputs/Rows,Threads,0,live.stream>>>(lease.gate,lease.up,x.data,gate,up,gelu);
    primitive::check_cuda(cudaGetLastError(),"Gemma291 paired-word Q4 GateUp");
}
}
