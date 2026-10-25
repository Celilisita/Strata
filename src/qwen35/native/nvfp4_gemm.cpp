#include "strata/qwen35/native/nvfp4.hpp"
#include "nvfp4_output_lease.hpp"
#include <cublasLt.h>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>

namespace strata::qwen35::native {
namespace {
void checked(cublasStatus_t s,const char *op){if(s!=CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(op)+": cuBLASLt status "+std::to_string(int(s)));}
void checked(cudaError_t s,const char *op){if(s!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));}
struct Plan {
    int r,k,t;
    cublasLtMatmulDesc_t op=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulAlgo_t algo{};
    ~Plan(){if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);
        if(c)cublasLtMatrixLayoutDestroy(c);if(op)cublasLtMatmulDescDestroy(op);}
};
}
struct Nvfp4Gemm::Impl {
    cudaStream_t stream;
    cublasLtHandle_t handle=nullptr;
    void *workspace=nullptr;
    float *tiny_output=nullptr;
    size_t workspace_size;
    int max_rows;
    std::vector<std::unique_ptr<Plan>> plans;
    Impl(cudaStream_t s,size_t bytes,int rows):stream(s),workspace_size(bytes),max_rows(rows){
        int device;cudaDeviceProp prop{};checked(cudaGetDevice(&device),"NVFP4 device");
        checked(cudaGetDeviceProperties(&prop,device),"NVFP4 capabilities");
        if(prop.major!=12||prop.minor!=0||rows<1||bytes<16384)
            throw std::invalid_argument("NVFP4 provider requires measured SM120 and bounded workspace");
        try{checked(cublasLtCreate(&handle),"NVFP4 handle");
            checked(cudaMalloc(&workspace,bytes),"NVFP4 workspace");
            checked(cudaMalloc(reinterpret_cast<void **>(&tiny_output),size_t(rows)*16*4),"NVFP4 tiny output");
        }catch(...){if(handle)cublasLtDestroy(handle);cudaFree(workspace);cudaFree(tiny_output);throw;}
    }
    ~Impl(){cudaStreamSynchronize(stream);plans.clear();if(handle)cublasLtDestroy(handle);cudaFree(workspace);cudaFree(tiny_output);}
    Plan &plan(int r,int k,int t,const void *weight_scales,const void *input_scales){
        for(auto &p:plans)if(p->r==r&&p->k==k&&p->t==t)return *p;
        cudaStreamCaptureStatus capture{};checked(cudaStreamIsCapturing(stream,&capture),"NVFP4 plan capture status");
        if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("NVFP4 shape must be prepared before graph capture");
        auto p=std::make_unique<Plan>();p->r=r;p->k=k;p->t=t;
        checked(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"NVFP4 operation");
        const auto transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)),"NVFP4 TN A");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)),"NVFP4 TN B");
        const auto mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)),"NVFP4 A block scale");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)),"NVFP4 B block scale");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
            &weight_scales,sizeof(weight_scales)),"NVFP4 heuristic weight scales");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
            &input_scales,sizeof(input_scales)),"NVFP4 heuristic input scales");
        checked(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_4F_E2M1,k,r,k),"NVFP4 weights layout");
        checked(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_4F_E2M1,k,t,k),"NVFP4 input layout");
        checked(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_32F,r,t,r),"NVFP4 float output layout");
        cublasLtMatmulPreference_t pref=nullptr;
        checked(cublasLtMatmulPreferenceCreate(&pref),"NVFP4 preference");
        cublasLtMatmulHeuristicResult_t result{};int count=0;
        try{checked(cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &workspace_size,sizeof(workspace_size)),"NVFP4 workspace bound");
            checked(cublasLtMatmulAlgoGetHeuristic(handle,p->op,p->a,p->b,p->c,p->c,pref,1,&result,&count),"NVFP4 heuristic");
        }catch(...){cublasLtMatmulPreferenceDestroy(pref);throw;}
        cublasLtMatmulPreferenceDestroy(pref);
        if(count!=1||result.state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("no native NVFP4 float-output tactic for shape");
        p->algo=result.algo;plans.push_back(std::move(p));return *plans.back();
    }
};
Nvfp4Gemm::Nvfp4Gemm(cudaStream_t s,size_t bytes,int rows):impl_(std::make_unique<Impl>(s,bytes,rows)){}
Nvfp4Gemm::~Nvfp4Gemm()=default;
size_t Nvfp4Gemm::bytes()const{return impl_->workspace_size+size_t(impl_->max_rows)*16*4;}
void Nvfp4Gemm::prepare(Nvfp4View w,Nvfp4Input x,int tokens){
    if(!w.scales||!x.scales||tokens<1)throw std::invalid_argument("NVFP4 cold-plan operands");
    impl_->plan(w.rows,w.columns,tokens<16?16:tokens,w.scales,x.scales);
}
void Nvfp4Gemm::project(Nvfp4View w,Nvfp4Input x,float *out,int tokens,cudaStream_t stream){
    project_impl(w,x,nullptr,out,tokens,stream);
}
void Nvfp4Gemm::project_add(Nvfp4View w,Nvfp4Input x,const float *residual,float *out,int tokens,cudaStream_t stream){
    validate_nvfp4_add_output(w,x,residual,out,tokens);
    project_impl(w,x,residual,out,tokens,stream);
}
void Nvfp4Gemm::project_impl(Nvfp4View w,Nvfp4Input x,const float *residual,float *out,int tokens,cudaStream_t stream){
    auto &i=*impl_;const int padded=tokens<16?16:tokens;
    if(!w.values||!w.scales||!x.values||!x.scales||!out||w.rows<1||w.rows>i.max_rows||w.columns%64||tokens<1||
       padded>x.capacity_rows||w.columns>x.capacity_columns||stream!=i.stream||
       !(w.weight_scale>0)||!(w.input_scale>0)||!std::isfinite(w.weight_scale*w.input_scale))
        throw std::invalid_argument("NVFP4 GEMM operand/stream/scale mismatch");
    auto &p=i.plan(w.rows,w.columns,padded,w.scales,x.scales);
    checked(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&w.scales,sizeof(w.scales)),"NVFP4 weight scale pointer");
    checked(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&x.scales,sizeof(x.scales)),"NVFP4 input scale pointer");
    const float alpha=w.weight_scale*w.input_scale,beta=residual?1.f:0.f;
    auto *dst=tokens<16?i.tiny_output:out;
    const float *accum=residual?residual:dst;
    checked(cublasLtMatmul(i.handle,p.op,&alpha,w.values,p.a,x.values,p.b,&beta,accum,p.c,dst,p.c,
        &p.algo,i.workspace,i.workspace_size,stream),"NVFP4 native matmul");
    if(dst!=out)checked(cudaMemcpyAsync(out,dst,size_t(tokens)*w.rows*4,cudaMemcpyDeviceToDevice,stream),"NVFP4 tiny output span");
}
}
