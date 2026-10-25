#include "strata/gemma4/native/vector_two_plane.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "small_prefill_values.cuh"
#include <cuda_fp16.h>
#include <limits>

namespace strata::gemma4::native {
namespace {
constexpr int Hidden=5376,Planes=2,Groups=Hidden/32;
bool span(const void* p,size_t bytes,size_t alignment){
    const auto at=reinterpret_cast<std::uintptr_t>(p);
    return p&&bytes&&at%alignment==0&&bytes<=(std::numeric_limits<std::uintptr_t>::max)()-at;
}
bool overlaps(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<std::uintptr_t>(a),y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
bool binding_valid(TwoPlaneQ8Binding b){return b.owner&&b.epoch&&b.stream;}
bool same(TwoPlaneQ8Binding a,TwoPlaneQ8Binding b){
    return a.owner==b.owner&&a.epoch==b.epoch&&a.stream==b.stream;
}
__global__ void encode(const float* input,TwoPlaneQ8Block* output){
    const int group=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(group>=Groups)return;
    float residual=input[group*32+lane];auto& block=output[group];
    // Exactly179's first two rounds, including finite handling and F32 residual
    // FMA. No second-plane shortcut, half scale or requantized weight is used.
    #pragma unroll
    for(int r=0;r<Planes;++r){
        const unsigned invalid=__ballot_sync(0xffffffff,!isfinite(residual));
        float maximum=fabsf(residual);
        for(int d=16;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
        const float scale=invalid?nanf(""):maximum/127.f;
        const int code=invalid||!(scale>0)?0:max(-127,min(127,__float2int_rn(residual/scale)));
        block.code[r][lane]=static_cast<signed char>(code);int total=code;
        for(int d=16;d;d/=2)total+=__shfl_xor_sync(0xffffffff,total,d);
        if(!lane){block.d[r]=scale;block.sum[r]=total;}
        residual=__fmaf_rn(-scale,float(code),residual);
    }
}
template<primitive::Quant Type>
__device__ float term(const unsigned char* raw,const TwoPlaneQ8Block& x,int group,float previous){
    int weight[8];
    #pragma unroll
    for(int w=0;w<8;++w)weight[w]=primitive::detail::weight_codes<Type>(raw,group,w*4);
    float product=0.f,correction=0.f;
    #pragma unroll
    for(int r=0;r<Planes;++r){
        int first=0,second=0;
        #pragma unroll
        for(int w=0;w<8;++w){
            const int code=reinterpret_cast<const int*>(x.code[r])[w];
            if(w<4)first=__dp4a(weight[w],code,first);
            else second=__dp4a(weight[w],code,second);
        }
        if constexpr(Type==primitive::Quant::Q4K){
            product=__fmaf_rn(x.d[r],float(first+second),product);
            correction=__fmaf_rn(x.d[r],float(x.sum[r]),correction);
        }else{
            const auto* scales=reinterpret_cast<const signed char*>(raw+192)+group*2;
            product=__fmaf_rn(x.d[r],float(int(scales[0])*first+int(scales[1])*second),product);
        }
    }
    if constexpr(Type==primitive::Quant::Q4K){
        const unsigned meta=primitive::detail::metadata(raw,group,false);
        const float scale=__half2float(*reinterpret_cast<const half*>(raw))*float(meta&63);
        const float minimum=__half2float(*reinterpret_cast<const half*>(raw+2))*float(meta>>6);
        return __fadd_rn(previous,__fmaf_rn(scale,product,-__fmul_rn(minimum,correction)));
    }else return __fmaf_rn(__half2float(*reinterpret_cast<const half*>(raw+208)),product,previous);
}
__device__ float sum8(float value,unsigned active){
    for(int d=4;d;d/=2)value=__fadd_rn(value,__shfl_xor_sync(active,value,d));
    return value;
}
template<primitive::Quant Type>
__global__ void multiply(primitive::Matrix matrix,const TwoPlaneQ8Block* input,float* output){
    const int row=blockIdx.x*16+threadIdx.x/8,group=threadIdx.x%8;
    if(row>=matrix.rows)return;
    const unsigned active=__activemask();
    constexpr int Bytes=Type==primitive::Quant::Q4K?144:210;
    float value=0.f;
    for(int k=0;k<Hidden/256;++k){
        const auto* raw=static_cast<const unsigned char*>(matrix.data)+(size_t(row)*(Hidden/256)+k)*Bytes;
        value=term<Type>(raw,input[k*8+group],group,value);
    }
    value=sum8(value,active);if(!group)output[row]=value;
}
void validate_lease(const TwoPlaneQ8Lease& lease,TwoPlaneQ8Binding live){
    const size_t bytes=size_t(Groups)*sizeof(TwoPlaneQ8Block);
    if(!binding_valid(live)||!same(lease.binding,live)||lease.columns!=Hidden||lease.tokens!=1||
        lease.bytes!=bytes||lease.data!=lease.arena||lease.arena_bytes<bytes||
        !span(lease.data,lease.arena_bytes,alignof(TwoPlaneQ8Block))||
        !span(lease.source,size_t(Hidden)*sizeof(float),alignof(float))||
        overlaps(lease.source,size_t(Hidden)*sizeof(float),lease.arena,lease.arena_bytes))
        throw std::invalid_argument("Gemma two-plane T1 stale owner/epoch/source/arena/stream");
}
}
size_t two_plane_q8_t1_bytes(int columns){
    if(columns!=Hidden)throw std::invalid_argument("Gemma two-plane QKV/head hidden5376 only");
    return size_t(Groups)*sizeof(TwoPlaneQ8Block);
}
TwoPlaneQ8Lease prepare_two_plane_q8_t1(const float* input,void* arena,size_t arena_bytes,
    int columns,TwoPlaneQ8Binding binding){
    const size_t bytes=two_plane_q8_t1_bytes(columns),source_bytes=size_t(columns)*sizeof(float);
    if(!binding_valid(binding)||arena_bytes<bytes||!span(input,source_bytes,alignof(float))||
        !span(arena,arena_bytes,alignof(TwoPlaneQ8Block))||overlaps(input,source_bytes,arena,arena_bytes))
        throw std::invalid_argument("Gemma two-plane T1 producer owner/source/arena extent");
    auto* output=static_cast<TwoPlaneQ8Block*>(arena);
    encode<<<(Groups+3)/4,128,0,binding.stream>>>(input,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma two-plane T1 F32 residual encoding");
    return {output,bytes,columns,1,input,arena,arena_bytes,binding};
}
void linear_two_plane_q8_t1(primitive::Matrix matrix,const TwoPlaneQ8Lease& input,
    float* output,TwoPlaneQ8Binding live){
    validate_lease(input,live);
    const bool rows=matrix.rows==2048||matrix.rows==4096||matrix.rows==8192||
                    matrix.rows==16384||matrix.rows==262144;
    if(!rows||matrix.columns!=Hidden||matrix.layout!=primitive::BlockLayout::Gguf||
        (matrix.quant!=primitive::Quant::Q4K&&matrix.quant!=primitive::Quant::Q6K))
        throw std::invalid_argument("Gemma two-plane T1 raw Q4/Q6 QKV/head extent");
    const size_t raw=size_t(matrix.rows)*(Hidden/256)*(matrix.quant==primitive::Quant::Q4K?144:210);
    const size_t result=size_t(matrix.rows)*sizeof(float);
    if(!span(matrix.data,raw,2)||!span(output,result,alignof(float))||
        overlaps(output,result,input.arena,input.arena_bytes)||
        overlaps(output,result,input.source,size_t(Hidden)*sizeof(float))||
        overlaps(output,result,matrix.data,raw)||overlaps(matrix.data,raw,input.arena,input.arena_bytes))
        throw std::invalid_argument("Gemma two-plane T1 weight/source/output lifetime");
    const unsigned blocks=(unsigned(matrix.rows)+15)/16;
    if(matrix.quant==primitive::Quant::Q4K)
        multiply<primitive::Quant::Q4K><<<blocks,128,0,live.stream>>>(matrix,input.data,output);
    else multiply<primitive::Quant::Q6K><<<blocks,128,0,live.stream>>>(matrix,input.data,output);
    primitive::check_cuda(cudaGetLastError(),"Gemma two-plane T1 grouped raw projection");
}
}
