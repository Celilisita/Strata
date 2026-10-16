#pragma once
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "cutlass_swiglu_iterator.cuh"
#include "strata/qwen35/native/ffn.hpp"

namespace strata::qwen35::native::cutlass_detail {
template<class Arch,int Depth,int M=128,int N=128,int WM=64,int WN=64,bool Debug=true>
struct FusedKernel {
    using Shape=cutlass::gemm::GemmShape<M,N,Depth>;
    using Warp=cutlass::gemm::GemmShape<WM,WN,Depth>;
    using Instruction=typename cutlass::platform::conditional<Arch::kMinComputeCapability==70,
        cutlass::gemm::GemmShape<8,8,4>,cutlass::gemm::GemmShape<16,8,16>>::type;
    using OutputOp=cutlass::epilogue::thread::LinearCombination<float,4,float,float,
        cutlass::epilogue::thread::ScaleType::Nothing>;
    using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
    using Default=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,cutlass::layout::RowMajor,8,
        cutlass::half_t,cutlass::layout::ColumnMajor,8,float,cutlass::layout::RowMajor,float,
        cutlass::arch::OpClassTensorOp,Arch,Shape,Warp,Instruction,OutputOp,Swizzle,
        Arch::kMinComputeCapability==70?2:3,false,cutlass::arch::OpMultiplyAdd>::GemmKernel;
    using BaseEpilogue=typename Default::Epilogue;
    using Iterator=SwiGluIterator<typename BaseEpilogue::OutputTileIterator,Debug>;
    using Epilogue=cutlass::epilogue::threadblock::Epilogue<typename BaseEpilogue::Shape,
        typename BaseEpilogue::WarpMmaOperator,BaseEpilogue::kPartitionsK,Iterator,
        typename BaseEpilogue::AccumulatorFragmentIterator,typename BaseEpilogue::WarpTileIterator,
        typename BaseEpilogue::SharedLoadIterator,OutputOp,typename BaseEpilogue::Padding>;
    using Kernel=cutlass::gemm::kernel::Gemm<typename Default::Mma,Epilogue,Swizzle,false>;
    static void configure(){
        const int shared=int(sizeof(typename Kernel::SharedStorage));
        check_cuda(cudaFuncSetAttribute(cutlass::Kernel<Kernel>,cudaFuncAttributeMaxDynamicSharedMemorySize,shared),
                   "CUTLASS fused shared-memory lease");
    }
    static void launch(const void *weights,const void *input,void *out,float *logical,float *gate,float *up,
                       int r,int k,int t,cudaStream_t stream){
        const cutlass::gemm::GemmCoord problem(t,2*r,k);
        const auto tiled=Swizzle().get_tiled_shape(problem,
            cutlass::gemm::GemmCoord(Shape::kM,Shape::kN,Shape::kK),1);
        using A=typename Kernel::Mma::IteratorA::TensorRef;
        using B=typename Kernel::Mma::IteratorB::TensorRef;
        using C=typename Iterator::TensorRef;
        const A a(static_cast<cutlass::half_t *>(const_cast<void *>(input)),cutlass::layout::RowMajor(k));
        const B b(static_cast<cutlass::half_t *>(const_cast<void *>(weights)),cutlass::layout::ColumnMajor(k));
        const C c(nullptr,cutlass::layout::RowMajor(2*r)),d(logical,cutlass::layout::RowMajor(2*r));
        if(Kernel::can_implement(problem,a,b,c,d)!=cutlass::Status::kSuccess)
            throw std::invalid_argument("CUTLASS fused operands violate prepared geometry");
        typename Kernel::Params parameters(problem,tiled,a,b,c,d,typename OutputOp::Params(1.f,0.f));
        parameters.params_D.half_output=static_cast<half *>(out);
        parameters.params_D.gate=gate;parameters.params_D.up=up;
        const dim3 grid=Swizzle().get_grid_shape(tiled);
        cutlass::Kernel<Kernel><<<grid,Kernel::kThreadCount,sizeof(typename Kernel::SharedStorage),stream>>>(parameters);
        check_cuda(cudaGetLastError(),"Strata CUTLASS fused Gate/Up SwiGLU");
    }
};
}
