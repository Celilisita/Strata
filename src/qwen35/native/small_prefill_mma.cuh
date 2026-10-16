#pragma once
#include <cuda_fp16.h>

namespace strata::qwen35::native::detail {
// Each SM70 operand row holds128 half codes plus four padding words.
// A68-word pitch keeps every row and every16-value dot16-byte aligned.
constexpr int HalfOperandPitch=68;
// Two-word alignment suffices for split LDS.64 input reads. This pitch puts
// the two8-row token quadrants on different banks without address XORs.
constexpr int CompactHalfOperandPitch=66;
// ISA-defined register ownership; no opaque WMMA fragment layout assumptions.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
using IntegerDot=int;
__device__ inline int result_row(int warp,int lane,int i){
    return (warp/2)*16+lane/4+((i%4)>=2?8:0);
}
__device__ inline int result_token(int warp,int lane,int i){
    return (warp%2)*16+(i/4)*8+2*(lane%4)+(i&1);
}
template<int Rows,bool CompactInput=false>
__device__ inline void code_dot16(const unsigned *weight,const unsigned *input,int group,int part,
                                 int warp,int lane,IntegerDot (&sum)[8]){
    const int row=(warp/2)*16+lane/4,word=group*8+part*4+lane%4;
    const unsigned a0=weight[word*Rows+row],a1=weight[word*Rows+row+8];
    #pragma unroll
    for(int n=0;n<2;++n){
        const int token=(warp%2)*16+n*8+lane/4;
        const unsigned b=input[word*32+token];
        auto *c=sum+n*4;
        asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 "
                     "{%0,%1,%2,%3},{%4,%5},{%6},{%0,%1,%2,%3};"
                     : "+r"(c[0]),"+r"(c[1]),"+r"(c[2]),"+r"(c[3])
                     : "r"(a0),"r"(a1),"r"(b));
    }
}
#else
using IntegerDot=float;
__device__ inline int result_row(int warp,int lane,int i){
    const int matrix=(lane%16)/4;
    return (warp/2)*16+(matrix/2)*8+(lane&1)+(i&2)+(lane>=16?4:0);
}
__device__ inline int result_token(int warp,int lane,int i){
    const int matrix=(lane%16)/4;
    return (warp%2)*16+(matrix%2)*8+(i&4)+(lane&2)+(i&1);
}
__device__ inline uint4 half_operand4(const unsigned *base,int row,int word){
    const unsigned address=unsigned(__cvta_generic_to_shared(base+row*HalfOperandPitch+word));
    uint4 value;
    asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];"
                 : "=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w)
                 : "r"(address) : "memory");
    return value;
}
template<bool CompactInput>
__device__ inline uint4 input_operand4(const unsigned *base,int row,int word){
    if constexpr(!CompactInput)return half_operand4(base,row,word);
    else{
        const unsigned address=unsigned(__cvta_generic_to_shared(base+row*CompactHalfOperandPitch+word));
        uint4 value;
        asm volatile("ld.shared.v2.u32 {%0,%1}, [%4]; ld.shared.v2.u32 {%2,%3}, [%4+8];"
                     : "=r"(value.x),"=r"(value.y),"=r"(value.z),"=r"(value.w)
                     : "r"(address) : "memory");
        return value;
    }
}
template<int Rows,bool CompactInput=false>
__device__ inline void code_dot16(const unsigned *weight,const unsigned *input,int group,int part,
                                 int warp,int lane,IntegerDot (&sum)[8]){
    const int matrix=(lane%16)/4,local=(lane%4)+(lane>=16?4:0);
    const int row=(warp/2)*16+(matrix/2)*8+local;
    const int token=(warp%2)*16+(matrix%2)*8+local;
    const int word=group*16+part*8;
    const uint4 a0=half_operand4(weight,row,word),a1=half_operand4(weight,row,word+4);
    const uint4 b0=input_operand4<CompactInput>(input,token,word),b1=input_operand4<CompactInput>(input,token,word+4);
    #pragma unroll
    for(int step=0;step<4;++step){
        const unsigned ax=step==0?a0.x:step==1?a0.z:step==2?a1.x:a1.z;
        const unsigned ay=step==0?a0.y:step==1?a0.w:step==2?a1.y:a1.w;
        const unsigned bx=step==0?b0.x:step==1?b0.z:step==2?b1.x:b1.z;
        const unsigned by=step==0?b0.y:step==1?b0.w:step==2?b1.y:b1.w;
        asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
                     "{%0,%1,%2,%3,%4,%5,%6,%7},{%8,%9},{%10,%11},{%0,%1,%2,%3,%4,%5,%6,%7};"
                     : "+f"(sum[0]),"+f"(sum[1]),"+f"(sum[2]),"+f"(sum[3]),
                       "+f"(sum[4]),"+f"(sum[5]),"+f"(sum[6]),"+f"(sum[7])
                     : "r"(ax),"r"(ay),"r"(bx),"r"(by));
    }
}
#endif
// The code dot and scale-weighted sum are bounded below2^24. SM70
// F32 integer-valued MMA results can therefore stay F32 without round trips.
__device__ inline float weighted_q6(IntegerDot one,IntegerDot two,int a,int b){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return float(a*one+b*two);
#else
    return __fmaf_rn(float(a),one,__fmul_rn(float(b),two));
#endif
}
__device__ inline float weighted_q4(IntegerDot one,IntegerDot two,int scale){
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    return float(scale*(one+two));
#else
    return __fmul_rn(float(scale),__fadd_rn(one,two));
#endif
}
__device__ inline unsigned half_codes(unsigned packed,bool signed_codes,int first){
    // Half1024 has integer ULP: its low byte carries exactly an unsigned8.
    // XOR rebiases signed codes, then one packed half subtraction removes
    // 1024 (unsigned) or1152 (signed), with exact integer results in both lanes.
    const unsigned bytes=packed^(signed_codes?0x80808080u:0u);
    const unsigned bits=__byte_perm(bytes,0x64646464u,first==0?0x4140u:0x4342u);
    const unsigned bias=signed_codes?0x64806480u:0x64006400u;
    const half2 value=__hsub2(__halves2half2(__ushort_as_half(bits&65535u),__ushort_as_half(bits>>16)),
                             __halves2half2(__ushort_as_half(bias&65535u),__ushort_as_half(bias>>16)));
    return unsigned(__half_as_ushort(__low2half(value)))|
           (unsigned(__half_as_ushort(__high2half(value)))<<16);
}
}
