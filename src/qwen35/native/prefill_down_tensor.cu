#include "prefill_down_tensor.hpp"
#include <cutlass/cutlass.h>
#include <cutlass/device_kernel.h>
#include <cutlass/gemm/kernel/default_gemm.h>
#include <cutlass/epilogue/thread/linear_combination.h>
#include "strata/qwen35/native/ffn.hpp"
#include <cstdint>
namespace strata::qwen35::native {
namespace n=strata::qwen35::native;
namespace {
using Row=cutlass::layout::RowMajor;
using Column=cutlass::layout::ColumnMajor;
using Epilogue=cutlass::epilogue::thread::LinearCombination<float,4,float,float>;
template<int Width>using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<Width>;
template<int Width>using Kernel=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,Row,8,
    cutlass::half_t,Column,8,float,Row,float,cutlass::arch::OpClassTensorOp,
    cutlass::arch::Sm70,cutlass::gemm::GemmShape<128,128,32>,
    cutlass::gemm::GemmShape<64,64,32>,cutlass::gemm::GemmShape<8,8,4>,
    Epilogue,Swizzle<Width>,2,false,cutlass::arch::OpMultiplyAdd>::GemmKernel;
template<int Width>void run(const void* weights,const void* encoded,float* output,int rows,int columns,
    int tokens,const float* alpha,const float* beta,cudaStream_t stream){
    using K=Kernel<Width>;
    using A=typename K::Mma::IteratorA::TensorRef;
    using B=typename K::Mma::IteratorB::TensorRef;
    using C=typename K::Epilogue::OutputTileIterator::TensorRef;
    const cutlass::gemm::GemmCoord problem(tokens,rows,columns);
    const auto tiled=Swizzle<Width>().get_tiled_shape(problem,{128,128,32},1);
    const A a(const_cast<cutlass::half_t*>(static_cast<const cutlass::half_t*>(encoded)),Row(columns));
    const B b(const_cast<cutlass::half_t*>(static_cast<const cutlass::half_t*>(weights)),Column(columns));
    const C c(output,Row(rows));
    if(K::can_implement(problem,a,b,c,c)!=cutlass::Status::kSuccess)throw std::invalid_argument("Raster operands");
    typename K::Params parameters(problem,tiled,a,b,c,c,typename Epilogue::Params(alpha,beta));
    cutlass::Kernel<K><<<Swizzle<Width>().get_grid_shape(tiled),K::kThreadCount,sizeof(typename K::SharedStorage),stream>>>(parameters);
    n::check_cuda(cudaGetLastError(),"Raster CUTLASS GEMM");
}
}
void prefill_down_tensor(const void* weights,const void* encoded,float* output,int rows,int columns,
    int tokens,const float* alpha,const float* beta,cudaStream_t stream){
    if(!weights||!encoded||!output||!alpha||!beta||!stream||weights==encoded||weights==output||encoded==output||
        rows<128||rows%128||columns<64||columns%64||tokens!=8192||
        reinterpret_cast<uintptr_t>(weights)%16||reinterpret_cast<uintptr_t>(encoded)%16||reinterpret_cast<uintptr_t>(output)%16)
        throw std::invalid_argument("Raster real half span/alignment/lease");
    run<8>(weights,encoded,output,rows,columns,tokens,alpha,beta,stream);
}
}
