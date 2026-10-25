#include "strata/qwen35/native/nvfp4_packed.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include <math_constants.h>
#include <stdexcept>
namespace strata::qwen35::native {
namespace {
struct alignas(16) PackedInput {int4 code[3];float scale[3];unsigned invalid;};
static_assert(sizeof(PackedInput)==64,"Packed input per16 matches original F32 input bytes");
__device__ float group_max(float value) {
    for(int d=8;d;d/=2)value=fmaxf(value,__shfl_xor_sync(0xffffffff,value,d,16));
    return value;
}
__global__ void prepare(const float* input,PackedInput* output,int groups) {
    const int group=(blockIdx.x*blockDim.x+threadIdx.x)/16,lane=threadIdx.x&15;
    if(group>=groups)return;
    float residual=input[size_t(blockIdx.y)*groups*16+group*16+lane];
    const unsigned invalid=__any_sync(0xffffffff,!isfinite(residual));
    PackedInput* item=output+size_t(blockIdx.y)*groups+group;
    #pragma unroll
    for(int term=0;term<3;++term) {
        const float maximum=group_max(fabsf(residual));
        const float scale=maximum>0.f&&maximum/127.f>0.f?maximum/127.f:1.f;
        const int quant=__float2int_rn(residual/scale);
        const unsigned b0=unsigned(quant)&255u;
        const unsigned b1=unsigned(__shfl_down_sync(0xffffffff,quant,1,16))&255u;
        const unsigned b2=unsigned(__shfl_down_sync(0xffffffff,quant,2,16))&255u;
        const unsigned b3=unsigned(__shfl_down_sync(0xffffffff,quant,3,16))&255u;
        if((lane&3)==0)reinterpret_cast<int*>(&item->code[term])[lane/4]=int(b0|(b1<<8)|(b2<<16)|(b3<<24));
        if(!lane)item->scale[term]=invalid?CUDART_NAN_F:scale;
        residual=fmaf(-float(quant),scale,residual);
    }
    if(!lane)item->invalid=invalid;
}
__device__ __forceinline__ int unpack(unsigned word,bool high) {
    const unsigned code=(word>>(high?4:0))&0x0f0f0f0f;
    const unsigned mag=code&0x07070707;
    const unsigned selector=(mag&7)|((mag>>4)&0x70)|((mag>>8)&0x700)|((mag>>12)&0x7000);
    const unsigned value=__byte_perm(0x03020100,0x0c080604,selector);
    const unsigned sign=((code&0x08080808)>>3)*255u;
    return int(__vsub4(value^sign,sign));
}
__device__ __forceinline__ int dot(int a,int b,int c,int d,int4 x) {
    int result=__dp4a(a,x.x,0);result=__dp4a(b,x.y,result);
    result=__dp4a(c,x.z,result);return __dp4a(d,x.w,result);
}
template<int Tokens,bool Cached> __global__ void project(const unsigned char* blocks,const float* global,
    const PackedInput* input,float* output,int rows,int columns) {
    const int row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=rows)return;
    const int groups=columns/16;float sum[Tokens]{};
    const auto* weights=blocks+size_t(row)*(columns/64)*36;
    for(int group=lane;group<groups;group+=32) {
        const auto* block=weights+(group/4)*36;
        const auto* packed=reinterpret_cast<const unsigned*>(block+4+(group%4)*8);
        const unsigned first=packed[0],last=packed[1];
        const int a=unpack(first,false),b=unpack(last,false),c=unpack(first,true),d=unpack(last,true);
        const float weight_scale=nvfp4::block_scale(block[group%4]);
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const PackedInput* item=input+size_t(token)*groups+group;PackedInput value;
            if constexpr(Cached) {
                // Force GPU read-only cached vector loads. Each activation group
                // is reused by many matrix rows; the old plain global loads
                // repeatedly route four strided vectors through L2.
                #pragma unroll
                for(int term=0;term<3;++term)value.code[term]=__ldg(&item->code[term]);
                const float4 scale=__ldg(reinterpret_cast<const float4*>(item->scale));
                value.scale[0]=scale.x;value.scale[1]=scale.y;value.scale[2]=scale.z;
            }else value=*item;
            float total=0;
            #pragma unroll
            for(int term=0;term<3;++term)total=fmaf(float(dot(a,b,c,d,value.code[term])),value.scale[term],total);
            sum[token]=fmaf(total,weight_scale,sum[token]);
        }
    }
    #pragma unroll
    for(int token=0;token<Tokens;++token) {
        for(int d=16;d;d/=2)sum[token]+=__shfl_down_sync(0xffffffff,sum[token],d);
        if(!lane)output[size_t(token)*rows+row]=sum[token]*(*global);
    }
}
}
size_t nvfp4_packed_input_bytes(int width,int tokens) {
    if(width<64||width%64||tokens<1||tokens>3)throw std::invalid_argument("NVFP4 packed input extent");
    return size_t(width)*tokens*4;
}
template<bool Cached> void packed_projection(const void* blocks,const float* global,const float* input,float* output,
    int rows,int columns,int tokens,void* workspace,cudaStream_t stream) {
    nvfp4_packed_input_bytes(columns,tokens);
    if(!blocks||!global||!input||!output||!workspace||rows<1||input==output)
        throw std::invalid_argument("NVFP4 packed GPU owner boundary");
    prepare<<<dim3(unsigned((columns/16+7)/8),unsigned(tokens)),128,0,stream>>>(input,static_cast<PackedInput*>(workspace),columns/16);
    check_cuda(cudaGetLastError(),"NVFP4 three-residual activation pack");
    const dim3 grid(unsigned((rows+3)/4));
    if(tokens==1)project<1,Cached><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(blocks),global,static_cast<const PackedInput*>(workspace),output,rows,columns);
    else if(tokens==2)project<2,Cached><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(blocks),global,static_cast<const PackedInput*>(workspace),output,rows,columns);
    else project<3,Cached><<<grid,128,0,stream>>>(static_cast<const unsigned char*>(blocks),global,static_cast<const PackedInput*>(workspace),output,rows,columns);
    check_cuda(cudaGetLastError(),"NVFP4 original packed DP4A projection");
}
void nvfp4_packed_projection(const void* blocks,const float* global,const float* input,float* output,
    int rows,int columns,int tokens,void* workspace,cudaStream_t stream) {
    packed_projection<false>(blocks,global,input,output,rows,columns,tokens,workspace,stream);
}
void nvfp4_packed_cached_projection(const void* blocks,const float* global,const float* input,float* output,
    int rows,int columns,int tokens,void* workspace,cudaStream_t stream) {
    packed_projection<true>(blocks,global,input,output,rows,columns,tokens,workspace,stream);
}

}
