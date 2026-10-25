#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif

namespace strata::kernels::cpu {
#if defined(_M_X64) || defined(__x86_64__)
namespace {
inline float scale(uint8_t code) {
    if (code > 126) return std::numeric_limits<float>::quiet_NaN();
    if (code < 8) return float(code) * (1.f / 512.f);
    const uint32_t bits = (uint32_t(code) + 960u) << 20;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}
inline int dot16(const uint8_t* packed, const int8_t* input) {
    const __m128i table = _mm_setr_epi8(0,1,2,3,4,6,8,12,0,-1,-2,-3,-4,-6,-8,-12);
    const __m128i mask = _mm_set1_epi8(15);
    const auto bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(packed));
    const auto low = _mm_and_si128(bytes, mask);
    const auto high = _mm_and_si128(_mm_srli_epi16(bytes, 4), mask);
    const auto weights = _mm_shuffle_epi8(table, _mm_unpacklo_epi64(low, high));
    // Signed int16 multiplication handles -128 activations without sign/abs overflow.
    const auto products = _mm256_madd_epi16(_mm256_cvtepi8_epi16(weights),
        _mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(input))));
    auto sum = _mm_add_epi32(_mm256_castsi256_si128(products), _mm256_extracti128_si256(products, 1));
    sum = _mm_hadd_epi32(sum, sum);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(sum, sum));
}
}
float nvfp4_avx2_r4_dot(const uint8_t* weight, const void* input, int columns) {
    const auto* activation = static_cast<const uint8_t*>(input);
    float sum = 0.f;
    for (int block = 0; block < columns / 64; ++block) {
        const auto* w = weight + size_t(block) * 36;
        const auto* a0 = activation + size_t(block) * 68;
        const auto* a1 = a0 + 34;
        uint16_t d0, d1;
        std::memcpy(&d0, a0, 2);
        std::memcpy(&d1, a1, 2);
        const auto half = _mm_cvtph_ps(_mm_cvtsi32_si128(int(uint32_t(d0) | (uint32_t(d1) << 16))));
        const float delta[2] = {_mm_cvtss_f32(half),
            _mm_cvtss_f32(_mm_shuffle_ps(half, half, _MM_SHUFFLE(1,1,1,1)))};
        for (int group = 0; group < 4; ++group) {
            const auto* a = group < 2 ? a0 : a1;
            const int product = dot16(w + 4 + group * 8,
                reinterpret_cast<const int8_t*>(a + 2) + (group % 2) * 16);
            sum += (delta[group / 2] * scale(w[group]) * .5f) * float(product);
        }
    }
    return sum;
}
namespace {
inline __m256i paired_quads(const uint8_t* packed,const int8_t* input) {
    // (E2M1*2)+12 is unsigned0..24. Every maddubs pair is within[-6144,6096].
    const auto table=_mm256_broadcastsi128_si256(_mm_setr_epi8(
        12,13,14,15,16,18,20,24,12,11,10,9,8,6,4,0));
    const auto mask=_mm_set1_epi8(15);
    const auto raw=_mm_loadu_si128(reinterpret_cast<const __m128i*>(packed));
    const auto lo=_mm_and_si128(raw,mask),hi=_mm_and_si128(_mm_srli_epi16(raw,4),mask);
    const auto indices=_mm256_set_m128i(_mm_unpackhi_epi64(lo,hi),_mm_unpacklo_epi64(lo,hi));
    const auto weight=_mm256_shuffle_epi8(table,indices);
    const auto q=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(input));
    const auto biased=_mm256_maddubs_epi16(weight,q);
    const auto correction=_mm256_maddubs_epi16(_mm256_set1_epi8(12),q);
    // Correlated subtraction restores signed products; each corrected pair<=3072.
    return _mm256_madd_epi16(_mm256_sub_epi16(biased,correction),_mm256_set1_epi16(1));
}
inline __m128i products4(const uint8_t* w,const uint8_t* a0,const uint8_t* a1) {
    const auto first=paired_quads(w+4,reinterpret_cast<const int8_t*>(a0+2));
    const auto second=paired_quads(w+20,reinterpret_cast<const int8_t*>(a1+2));
    const auto pair=_mm256_hadd_epi32(first,second);
    const auto totals=_mm256_hadd_epi32(pair,pair);
    // Lower128=[g0,g2,g0,g2], upper128=[g1,g3,g1,g3].
    return _mm_unpacklo_epi32(_mm256_castsi256_si128(totals),_mm256_extracti128_si256(totals,1));
}
inline __m128 scales4(const uint8_t* w) {
    uint32_t raw;std::memcpy(&raw,w,4);
    const auto codes=_mm_cvtepu8_epi32(_mm_cvtsi32_si128(int(raw)));
    const auto normal=_mm_castsi128_ps(_mm_slli_epi32(_mm_add_epi32(codes,_mm_set1_epi32(960)),20));
    const auto small=_mm_mul_ps(_mm_cvtepi32_ps(codes),_mm_set1_ps(1.f/512.f));
    const auto values=_mm_blendv_ps(normal,small,_mm_castsi128_ps(_mm_cmplt_epi32(codes,_mm_set1_epi32(8))));
    return _mm_blendv_ps(values,_mm_castsi128_ps(_mm_set1_epi32(0x7fc00000)),
        _mm_castsi128_ps(_mm_cmpgt_epi32(codes,_mm_set1_epi32(126))));
}
}
float nvfp4_avx2_v2_dot(const uint8_t* weight,const void* input,int columns) {
    const auto* activation=static_cast<const uint8_t*>(input);float sum=0.f;
    for(int block=0;block<columns/64;++block) {
        const auto* w=weight+size_t(block)*36;const auto* a0=activation+size_t(block)*68;const auto* a1=a0+34;
        uint16_t d0,d1;std::memcpy(&d0,a0,2);std::memcpy(&d1,a1,2);
        const auto half=_mm_cvtph_ps(_mm_cvtsi32_si128(int(uint32_t(d0)|(uint32_t(d1)<<16))));
        const auto deltas=_mm_unpacklo_ps(half,half);
        const auto factor=_mm_mul_ps(_mm_mul_ps(deltas,scales4(w)),_mm_set1_ps(.5f));
        int32_t product[4];float factors[4];
        _mm_storeu_si128(reinterpret_cast<__m128i*>(product),products4(w,a0,a1));
        _mm_storeu_ps(factors,factor);
        // Preserve ascending four-group FP32 accumulation; no new tree reduction.
        for(int g=0;g<4;++g)sum+=factors[g]*float(product[g]);
    }
    return sum;
}
#else
float nvfp4_avx2_r4_dot(const uint8_t* weight, const void* input, int columns) {
    throw std::invalid_argument("NVFP4 AVX2 entry requires an x86 AVX2/F16C caller guard");
}
#endif

#if !defined(_M_X64) && !defined(__x86_64__)
float nvfp4_avx2_v2_dot(const uint8_t*,const void*,int) {
    throw std::invalid_argument("NVFP4 AVX2 entry requires an x86 AVX2/F16C caller guard");
}
#endif
#ifndef STRATA_NVFP4_CPU_ENTRY
#define STRATA_NVFP4_CPU_ENTRY nvfp4_avx2_dot
#endif
float STRATA_NVFP4_CPU_ENTRY(const uint8_t* weight,const void* input,int columns) {
    static const bool candidate=[] {
        const char* value=std::getenv("STRATA_NVFP4_CPU_AVX2_V2");
        if(!value||!std::strcmp(value,"0"))return false;
        if(!std::strcmp(value,"1"))return true;
        throw std::invalid_argument("NVFP4_CPU_AVX2_V2 requires0 or1");
    }();
    return candidate?nvfp4_avx2_v2_dot(weight,input,columns):nvfp4_avx2_r4_dot(weight,input,columns);
}
}
