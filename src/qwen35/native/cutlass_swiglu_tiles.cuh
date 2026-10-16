#pragma once
#include "cutlass_swiglu_kernel.cuh"
#include "strata/qwen35/native/dense_gemm.hpp"

namespace strata::qwen35::native::cutlass_detail {
// Cold, explicit shapes. K32 baseline remains unchanged. Reviewed candidates
// trade input reuse and accumulator registers against CTA occupancy.
template<class Arch,class Operation>
void dispatch_tile(int tile,Operation &op){
    switch(tile){
    case 0:op.template apply<FusedKernel<Arch,32>>();break;
    case 1:op.template apply<FusedKernel<Arch,64>>();break;
    case 2:op.template apply<FusedKernel<Arch,32,128,128,32,64>>();break;
    case 3:op.template apply<FusedKernel<Arch,32,128,64,32,64>>();break;
    case 4:op.template apply<FusedKernel<Arch,32,64,128,32,64>>();break;
    case 5:op.template apply<FusedKernel<Arch,32,128,128,64,64,false>>();break;
    case 6:op.template apply<FusedKernel<Arch,32,128,256,64,64>>();break;
    default:throw std::invalid_argument("CUTLASS tile outside reviewed candidates");
    }
}
struct ConfigureTile {
    CutlassTileInfo &info;
    template<class K>void apply(){
        using Kernel=typename K::Kernel;
        K::configure();cudaFuncAttributes attributes{};
        check_cuda(cudaFuncGetAttributes(&attributes,cutlass::Kernel<Kernel>),"CUTLASS tile attributes");
        info.threads=Kernel::kThreadCount;
        info.registers=attributes.numRegs;
        info.shared_bytes=sizeof(typename Kernel::SharedStorage);
    }
};
struct LaunchTile {
    const void *w,*x;void *y;float *logical,*gate,*up;
    int r,k,t;cudaStream_t stream;
    template<class K>void apply(){K::launch(w,x,y,logical,gate,up,r,k,t,stream);}
};
}
