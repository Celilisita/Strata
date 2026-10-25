#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace strata::completed_v100::qwen35::native::q8wide {
// Original Q8 blocks have34byte stride; each16-code address is0or2 mod4.
// A carry read is needed only for shift16. For columns%64==0, the final
// 16-code field is4-byte aligned, so its carry must not read beyond payload.
__device__ __forceinline__ int4 codes16(const unsigned char* bytes) {
    const auto address=reinterpret_cast<uintptr_t>(bytes);
    const auto* p=reinterpret_cast<const unsigned*>(address&~uintptr_t(3));
    const unsigned shift=unsigned(address&2)*8;
    const unsigned a=__ldg(p),b=__ldg(p+1),c=__ldg(p+2),d=__ldg(p+3);
    const unsigned e=shift?__ldg(p+4):0;
    return make_int4(int(__funnelshift_r(a,b,shift)),int(__funnelshift_r(b,c,shift)),
        int(__funnelshift_r(c,d,shift)),int(__funnelshift_r(d,e,shift)));
}
}
