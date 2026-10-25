#include "partial_gemm.hpp"
#include "partial_swiglu.cuh"
namespace strata::qwen35::native {
namespace {
using Shape=cutlass::gemm::GemmShape<128,256,32>;
using Warp=cutlass::gemm::GemmShape<64,64,32>;
using Instruction=cutlass::gemm::GemmShape<16,8,16>;
using Output=cutlass::epilogue::thread::LinearCombination<float,4,float,float,
    cutlass::epilogue::thread::ScaleType::Nothing>;
using Swizzle=cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<1>;
using Linear=typename cutlass::gemm::kernel::DefaultGemm<cutlass::half_t,cutlass::layout::RowMajor,8,
    cutlass::half_t,cutlass::layout::ColumnMajor,8,float,cutlass::layout::RowMajor,float,
    cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,Shape,Warp,Instruction,Output,Swizzle,3,false,
    cutlass::arch::OpPartial16>::GemmKernel;
using Fused=partial_detail::PartialFusedKernel<32>;
class PartialGemm final:public DenseGemm {
    DenseGemm &fallback_;cudaStream_t stream_;unsigned long long count_=0;
public:
    PartialGemm(DenseGemm &fallback,cudaStream_t stream):fallback_(fallback),stream_(stream){
        const auto workspace=fallback_.borrowed_workspace();
        int device;check_cuda(cudaGetDevice(&device),"partial16 device");const auto caps=capabilities(device);
        if(caps.major!=12||caps.minor!=0||!stream||workspace.stream!=stream)
            throw std::invalid_argument("partial16 SM120 stream lease");
        check_cuda(cudaFuncSetAttribute(cutlass::Kernel<Linear>,cudaFuncAttributeMaxDynamicSharedMemorySize,
            int(sizeof(typename Linear::SharedStorage))),"partial16 linear shared lease");
        Fused::configure();
    }
    ~PartialGemm()override{cudaStreamSynchronize(stream_);}
    bool can_fuse_swiglu(int r,int k,int t)const override{return r>0&&r%64==0&&k>0&&k%32==0&&t>=128&&t%128==0;}
    void project_swiglu(const void *w,const void *x,void *y,float *logical,float *g,float *u,
                        int r,int k,int t,cudaStream_t s)override{
        if(!w||!x||!y||!logical||bool(g)!=bool(u)||s!=stream_||!can_fuse_swiglu(r,k,t))
            throw std::invalid_argument("partial16 fused operand lease");
        Fused::launch(w,x,y,logical,g,u,r,k,t,s);++count_;
    }
    void project(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{
        if(!w||!x||!y||s!=stream_||r<8||r%8||k<16||k%16||t<128||t%128)
            throw std::invalid_argument("partial16 linear operand lease");
        const cutlass::gemm::GemmCoord problem(t,r,k);
        const auto tiled=Swizzle().get_tiled_shape(problem,cutlass::gemm::GemmCoord(128,256,32),1);
        using A=typename Linear::Mma::IteratorA::TensorRef;using B=typename Linear::Mma::IteratorB::TensorRef;
        using C=typename Linear::Epilogue::OutputTileIterator::TensorRef;
        const A a(static_cast<cutlass::half_t *>(const_cast<void *>(x)),cutlass::layout::RowMajor(k));
        const B b(static_cast<cutlass::half_t *>(const_cast<void *>(w)),cutlass::layout::ColumnMajor(k));
        const C c(nullptr,cutlass::layout::RowMajor(r)),d(y,cutlass::layout::RowMajor(r));
        if(Linear::can_implement(problem,a,b,c,d)!=cutlass::Status::kSuccess)
            throw std::invalid_argument("partial16 linear prepared geometry");
        typename Linear::Params params(problem,tiled,a,b,c,d,typename Output::Params(1.f,0.f));
        cutlass::Kernel<Linear><<<Swizzle().get_grid_shape(tiled),Linear::kThreadCount,
            sizeof(typename Linear::SharedStorage),s>>>(params);
        check_cuda(cudaGetLastError(),"partial16 linear launch");++count_;
    }
    bool project_pair(const void *w,const void *x,float *y,int r,int k,int t,cudaStream_t s)override{
        project(w,x,y,2*r,k,t,s);return true;
    }
    void batched_fp16(const void *a,const void *b,float *c,GemmBatch p,cudaStream_t s)override{fallback_.batched_fp16(a,b,c,p,s);}
    size_t workspace_bytes()const override{return fallback_.workspace_bytes();}
    unsigned long long calls()const override{return count_+fallback_.calls();}
    GemmWorkspace borrowed_workspace()const override{return fallback_.borrowed_workspace();}
};
}
std::unique_ptr<DenseGemm> make_partial16_gemm(DenseGemm &fallback,cudaStream_t stream){
    return std::make_unique<PartialGemm>(fallback,stream);
}
}
