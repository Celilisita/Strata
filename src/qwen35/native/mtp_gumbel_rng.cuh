#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#if defined(__CUDACC__)
#define STRATA_MTP_GUMBEL_HD __host__ __device__
#else
#define STRATA_MTP_GUMBEL_HD
#endif
namespace strata::qwen35::native::mtp_gumbel_detail {
// Same ten native Philox rounds, with an injective token-ID seed domain.
// Counter+token domains remain separate; no product can overflow the position.
STRATA_MTP_GUMBEL_HD inline double uniform(uint64_t seed,uint64_t counter,int token){
    const uint64_t domain=seed^(uint64_t(unsigned(token)+1)*0x9e3779b97f4a7c15ull);
    uint32_t c0=uint32_t(counter),c1=uint32_t(counter>>32),c2=uint32_t(domain),c3=uint32_t(domain>>32);
    for(uint32_t round=0;round<10;++round){
        #if defined(__CUDA_ARCH__)
        const uint32_t high2=__umulhi(0xBB67AE85u,c2),high0=__umulhi(0x9E3779B9u,c0);
        #else
        const uint32_t high2=uint32_t((uint64_t(0xBB67AE85u)*c2)>>32),high0=uint32_t((uint64_t(0x9E3779B9u)*c0)>>32);
        #endif
        const uint32_t a=high2^c1^round,b=0xBB67AE85u*c2,c=high0^c3,d=0x9E3779B9u*c0;
        c0=a;c1=b;c2=c;c3=d;
    }
    // 52-bit midpoints are exactly representable and strictly inside(0,1).
    const uint64_t bits=(uint64_t(c0)<<32)|c1;
    return (double(bits>>12)+.5)*(1.0/4503599627370496.0);
}
}

#undef STRATA_MTP_GUMBEL_HD
