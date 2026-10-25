#pragma once
#include "original_fused_iterator.cuh"
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <stdexcept>
#include <cutlass/gemm/threadblock/mma_singlestage.h>
#include <type_traits>

namespace strata::qwen35::native::original_fused {
template<bool Q8,int M,int N,int WM,int WN,bool Tiled=false,int Stages=2>
struct FusedKernel {
    using Row=cutlass::layout::RowMajor;
    using Column=cutlass::layout::ColumnMajor;
    using OutputOp=cutlass::epilogue::thread::LinearCombination<float,4,float,float,
        cutlass::epilogue::thread::ScaleType::Nothing>;
    using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
    using Default=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,Row,8,
        cutlass::half_t,Column,8,float,Row,float,cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm70,cutlass::gemm::GemmShape<M,N,32>,
        cutlass::gemm::GemmShape<WM,WN,32>,cutlass::gemm::GemmShape<8,8,4>,
        OutputOp,Swizzle,2,false,cutlass::arch::OpMultiplyAdd>::GemmKernel;
    using BaseMma=typename Default::Mma;
    using B=QuantizedB<typename BaseMma::IteratorB,Q8,Tiled>;
    using Single=cutlass::gemm::threadblock::MmaSingleStage<typename BaseMma::Shape,
        typename BaseMma::IteratorA,typename BaseMma::SmemIteratorA,B,
        typename BaseMma::SmemIteratorB,float,Row,typename BaseMma::Policy>;
    using Pipelined=cutlass::gemm::threadblock::MmaPipelined<typename BaseMma::Shape,
        typename BaseMma::IteratorA,typename BaseMma::SmemIteratorA,B,
        typename BaseMma::SmemIteratorB,float,Row,typename BaseMma::Policy>;
    using Mma=std::conditional_t<Stages==1,Single,Pipelined>;
    using BaseEpilogue=typename Default::Epilogue;
    using D=PartialOutput<typename BaseEpilogue::OutputTileIterator>;
    using Epilogue=cutlass::epilogue::threadblock::Epilogue<typename BaseEpilogue::Shape,
        typename BaseEpilogue::WarpMmaOperator,BaseEpilogue::kPartitionsK,D,
        typename BaseEpilogue::AccumulatorFragmentIterator,typename BaseEpilogue::WarpTileIterator,
        typename BaseEpilogue::SharedLoadIterator,OutputOp,typename BaseEpilogue::Padding>;
    using Kernel=cutlass::gemm::kernel::Gemm<Mma,Epilogue,Swizzle,false>;
    static_assert(sizeof(typename Kernel::SharedStorage)<=48*1024,"fused shared memory needs a cold larger lease");
    static void run(const void* raw,const half* input,float* partial,int rows,int columns,
        int tokens,int splits,cudaStream_t stream) {
        const cutlass::gemm::GemmCoord problem(2*tokens,Q8?2*rows:rows,columns);
        const auto tiled=Swizzle().get_tiled_shape(problem,cutlass::gemm::GemmCoord(M,N,32),splits);
        using A=typename Mma::IteratorA::TensorRef;
        const A a(reinterpret_cast<cutlass::half_t*>(const_cast<half*>(input)),Row(columns));
        const typename B::TensorRef b(reinterpret_cast<cutlass::half_t*>(const_cast<void*>(raw)),Column(columns));
        const typename D::TensorRef c(partial,Row(problem.n()));
        if(Kernel::can_implement(problem,a,b,c,c)!=cutlass::Status::kSuccess)
            throw std::invalid_argument("Fused CUTLASS original-format alignment");
        typename Kernel::Params parameters(problem,tiled,a,b,c,c,typename OutputOp::Params(1.f,0.f));
        parameters.params_C.partition_stride=parameters.params_D.partition_stride=
            static_cast<long long>(problem.m())*problem.n();
        cutlass::Kernel<Kernel><<<Swizzle().get_grid_shape(tiled),Kernel::kThreadCount,
            sizeof(typename Kernel::SharedStorage),stream>>>(parameters);
    }
};
}
