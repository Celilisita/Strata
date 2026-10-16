#include "mxfp4_fast_dot.hpp"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstring>
#if defined(_MSC_VER) && defined(_M_X64)
#include <immintrin.h>
#include <intrin.h>
namespace strata::deepseek_v4 {
bool mxfp4_fast_dot_supported(){
 int cpu[4];__cpuid(cpu,0);if(cpu[0]<7)return false;__cpuid(cpu,1);
 if((cpu[2]&(1<<27))==0||(_xgetbv(0)&0xe6)!=0xe6)return false;
 __cpuidex(cpu,7,0);constexpr unsigned required=(1u<<16)|(1u<<17)|(1u<<30)|(1u<<31);
 return (unsigned(cpu[1])&required)==required&&(cpu[2]&(1<<11))!=0;
}
static float exponent(uint8_t e){uint32_t bits=e<2?(0x00200000u<<e):(uint32_t(e-1)<<23);float x;std::memcpy(&x,&bits,4);return x;}
static float scale(const uint8_t* q){uint16_t d;std::memcpy(&d,q,2);return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(d)));}
static __m256i decode(const uint8_t* row){const auto lut=_mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12),mask=_mm_set1_epi8(15);
 const auto bits=_mm_loadu_si128(reinterpret_cast<const __m128i*>(row+1));const auto lo=_mm_shuffle_epi8(lut,_mm_and_si128(bits,mask));
 const auto hi=_mm_shuffle_epi8(lut,_mm_and_si128(_mm_srli_epi16(bits,4),mask));return _mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1);}
template<int tokens> static void dot(int n,const uint8_t* x,const uint8_t* a,const uint8_t* b,float* out){
 __m512 accum[tokens][2];for(auto& token:accum)for(auto& lane:token)lane=_mm512_setzero_ps();
 for(int block=0;block<n/32;block+=4)for(int lane=0;lane<2;++lane){const int k=block+lane*2;const auto w0=decode(x+k*17),w1=decode(x+(k+1)*17);
  const auto w=_mm512_inserti64x4(_mm512_castsi256_si512(w0),w1,1);const auto magnitude=_mm512_abs_epi8(w);const auto negative=_mm512_movepi8_mask(w);
  const float s0=exponent(x[k*17]),s1=exponent(x[(k+1)*17]);const uint8_t* inputs[]{a,b};
  for(int token=0;token<tokens;++token){const auto* q0=inputs[token]+k*34;const auto* q1=q0+34;
   auto q=_mm512_inserti64x4(_mm512_castsi256_si512(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q0+2))),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(q1+2)),1);
   q=_mm512_mask_sub_epi8(q,negative,_mm512_setzero_si512(),q);const auto product=_mm512_dpbusd_epi32(_mm512_setzero_si512(),magnitude,q);
   const auto factor=_mm512_mask_mov_ps(_mm512_set1_ps(s0*scale(q0)),0xff00,_mm512_set1_ps(s1*scale(q1)));
   accum[token][lane]=_mm512_fmadd_ps(factor,_mm512_cvtepi32_ps(product),accum[token][lane]);}}
 for(int token=0;token<tokens;++token)out[token]=_mm512_reduce_add_ps(_mm512_add_ps(accum[token][0],accum[token][1]));
}
void mxfp4_fast_dot(int n,const void* x,const void* a,const void* b,float* out){if(b)dot<2>(n,static_cast<const uint8_t*>(x),static_cast<const uint8_t*>(a),static_cast<const uint8_t*>(b),out);else dot<1>(n,static_cast<const uint8_t*>(x),static_cast<const uint8_t*>(a),nullptr,out);}
}
#else
namespace strata::deepseek_v4 {
bool mxfp4_fast_dot_supported(){return false;}
void mxfp4_fast_dot(int,const void*,const void*,const void*,float*){}
}
#endif
