#pragma once
#include <cutlass/arch/mma_sm80.h>
namespace cutlass::arch {
struct OpPartial16 {};
template<>
struct Mma<gemm::GemmShape<16,8,16>,32,half_t,layout::RowMajor,
           half_t,layout::ColumnMajor,float,layout::RowMajor,OpPartial16> {
    using Shape=gemm::GemmShape<16,8,16>;
    using ElementA=half_t;using LayoutA=layout::RowMajor;using FragmentA=Array<half_t,8>;
    using ElementB=half_t;using LayoutB=layout::ColumnMajor;using FragmentB=Array<half_t,4>;
    using ElementC=float;using LayoutC=layout::RowMajor;using FragmentC=Array<float,4>;
    using Operator=OpPartial16;using ArchTag=Sm80;
    CUTLASS_HOST_DEVICE
    void operator()(FragmentC &d,const FragmentA &a,const FragmentB &b,const FragmentC &c)const{
#if defined(CUTLASS_ARCH_MMA_SM80_ENABLED)
        // The half accumulator always starts at zero and spans exactlyK16.
        // The long K reduction stays in FP32; no priorFP32 sum is rounded down.
        const auto *av=reinterpret_cast<const unsigned *>(&a),*bv=reinterpret_cast<const unsigned *>(&b);
        Array<half_t,4> partial;auto *pv=reinterpret_cast<unsigned *>(&partial);
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
            "{%0,%1},{%2,%3,%4,%5},{%6,%7},{%8,%8};"
            :"=r"(pv[0]),"=r"(pv[1])
            :"r"(av[0]),"r"(av[1]),"r"(av[2]),"r"(av[3]),"r"(bv[0]),"r"(bv[1]),"r"(0));
        CUTLASS_PRAGMA_UNROLL
        for(int i=0;i<4;++i)d[i]=__fadd_rn(c[i],float(partial[i]));
#else
        CUTLASS_UNUSED(d);CUTLASS_UNUSED(a);CUTLASS_UNUSED(b);CUTLASS_UNUSED(c);
        CUTLASS_NOT_IMPLEMENTED();
#endif
    }
};
}
