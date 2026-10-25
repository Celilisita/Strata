#include "strata/qwen35/completed_v100/native/original_cutlass.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include <stdexcept>
#include <string>

namespace strata::completed_v100::qwen35::native {
namespace {
using Row=cutlass::layout::RowMajor;
using Column=cutlass::layout::ColumnMajor;
using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using Epilogue=cutlass::epilogue::thread::LinearCombination<float,1,float,float>;
using TensorEpilogue=cutlass::epilogue::thread::LinearCombination<float,4,float,float>;
// This is NVIDIA's complete GEMM kernel, not a private MMA implementation.
// The existing dependency supplies kernel interfaces, without device::Gemm.
using TensorGemm=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,Row,8,
    cutlass::half_t,Column,8,float,Row,float,cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,cutlass::gemm::GemmShape<32,128,32>,
    cutlass::gemm::GemmShape<32,64,32>,cutlass::gemm::GemmShape<8,8,4>,
    TensorEpilogue,Swizzle,2,false,cutlass::arch::OpMultiplyAdd>::GemmKernel;
using PrefillGemm=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,Row,8,
    cutlass::half_t,Column,8,float,Row,float,cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<8,8,4>,
    TensorEpilogue,Swizzle,2,false,cutlass::arch::OpMultiplyAdd>::GemmKernel;
using FloatGemm=typename cutlass::gemm::kernel::DefaultGemm<float,Row,1,float,Column,1,
    float,Row,float,cutlass::arch::OpClassSimt,cutlass::arch::Sm70,
    cutlass::gemm::GemmShape<32,128,8>,cutlass::gemm::GemmShape<32,32,8>,
    cutlass::gemm::GemmShape<1,1,1>,Epilogue,Swizzle,2,false,
    cutlass::arch::OpMultiplyAdd>::GemmKernel;
template<class Kernel,class Element>
void launch(const Element* weights,const Element* input,float* output,
    int rows,int columns,int tokens,int stride,const float* alpha,const float* beta,cudaStream_t stream) {
    if(!weights||!input||!output||!alpha||!beta||!stream||rows<1||columns<1||tokens<1||stride<rows)
        throw std::invalid_argument("Original CUTLASS GPU tile boundary");
    using Shape=typename Kernel::Mma::Shape;
    using A=typename Kernel::Mma::IteratorA::TensorRef;
    using B=typename Kernel::Mma::IteratorB::TensorRef;
    using C=typename Kernel::Epilogue::OutputTileIterator::TensorRef;
    const cutlass::gemm::GemmCoord problem(tokens,rows,columns);
    const auto tiled=Swizzle().get_tiled_shape(problem,
        cutlass::gemm::GemmCoord(Shape::kM,Shape::kN,Shape::kK),1);
    const A a(const_cast<Element*>(input),Row(columns));
    const B b(const_cast<Element*>(weights),Column(columns));
    const C c(output,Row(stride)),d(output,Row(stride));
    if(Kernel::can_implement(problem,a,b,c,d)!=cutlass::Status::kSuccess)
        throw std::invalid_argument("Original CUTLASS operand admission");
    // No split-K, semaphores, dynamic allocation or private reduction.
    using OutputOp=typename Kernel::Epilogue::OutputOp;
    typename Kernel::Params parameters(problem,tiled,a,b,c,d,typename OutputOp::Params(alpha,beta));
    cutlass::Kernel<Kernel><<<Swizzle().get_grid_shape(tiled),Kernel::kThreadCount,
        sizeof(typename Kernel::SharedStorage),stream>>>(parameters);
    const auto result=cudaGetLastError();
    if(result!=cudaSuccess)throw std::runtime_error(std::string("Original CUTLASS launch: ")+cudaGetErrorString(result));
}
}
void original_cutlass_half(const void* weights,const void* input,float* output,
    int rows,int columns,int tokens,int stride,const float* alpha,const float* beta,cudaStream_t stream) {
    if(tokens>=128) {
        launch<PrefillGemm>(static_cast<const cutlass::half_t*>(weights),
            static_cast<const cutlass::half_t*>(input),output,rows,columns,tokens,stride,alpha,beta,stream);return;
    }
    launch<TensorGemm>(static_cast<const cutlass::half_t*>(weights),
        static_cast<const cutlass::half_t*>(input),output,rows,columns,tokens,stride,alpha,beta,stream);
}
void original_cutlass_float(const float* weights,const float* input,float* output,
    int rows,int columns,int tokens,int stride,const float* alpha,const float* beta,cudaStream_t stream) {
    launch<FloatGemm>(weights,input,output,rows,columns,tokens,stride,alpha,beta,stream);
}
}
