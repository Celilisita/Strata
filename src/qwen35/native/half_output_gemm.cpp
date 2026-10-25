#include "half_output_gemm.hpp"
#include "half_tactic_policy.hpp"
#include <array>
#include <cublasLt.h>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
namespace strata::qwen35::native {
namespace {
void checked(cublasStatus_t s,const char *op){if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(op)+": cuBLASLt "+std::to_string(int(s)));}
void checked(cudaError_t s,const char *op){if(s!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(s));}
struct Plan {
    int rows,columns,tokens;cublasLtMatmulDesc_t op=nullptr;cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulAlgo_t algorithm{},alternate{};bool alternate_ready=false;
    ~Plan(){if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);if(op)cublasLtMatmulDescDestroy(op);}
};
}
struct Nvfp4HalfGemm::Impl {
    cudaStream_t stream;cublasLtHandle_t handle=nullptr;void *workspace=nullptr;size_t bytes;
    std::vector<std::unique_ptr<Plan>> plans;bool measured=false;uint64_t alternate_calls=0;
    Impl(cudaStream_t s,size_t n):stream(s),bytes(n){
        int device;cudaDeviceProp property{};checked(cudaGetDevice(&device),"half result device");checked(cudaGetDeviceProperties(&property,device),"half result caps");
        if(property.major!=12||property.minor!=0||n<16384)throw std::invalid_argument("half result SM120/workspace");
        try{checked(cublasLtCreate(&handle),"half result handle");checked(cudaMalloc(&workspace,bytes),"half result workspace");}
        catch(...){if(handle)cublasLtDestroy(handle);cudaFree(workspace);throw;}
    }
    ~Impl(){cudaStreamSynchronize(stream);plans.clear();if(handle)cublasLtDestroy(handle);cudaFree(workspace);}
    Plan &plan(Nvfp4View w,Nvfp4Input x,int tokens){
        for(auto &p:plans)if(p->rows==w.rows&&p->columns==w.columns&&p->tokens==tokens)return *p;
        cudaStreamCaptureStatus capture;checked(cudaStreamIsCapturing(stream,&capture),"half result capture status");
        if(capture!=cudaStreamCaptureStatusNone)throw std::logic_error("half result plan must be cold");
        auto p=std::make_unique<Plan>();p->rows=w.rows;p->columns=w.columns;p->tokens=tokens;
        checked(cublasLtMatmulDescCreate(&p->op,CUBLAS_COMPUTE_32F,CUDA_R_32F),"half result operation");
        const auto transpose=CUBLAS_OP_T,normal=CUBLAS_OP_N;const auto scaling=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)),"half result TN A");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_TRANSB,&normal,sizeof(normal)),"half result TN B");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&scaling,sizeof(scaling)),"half result A mode");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&scaling,sizeof(scaling)),"half result B mode");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&w.scales,sizeof(w.scales)),"half result A SF");
        checked(cublasLtMatmulDescSetAttribute(p->op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&x.scales,sizeof(x.scales)),"half result B SF");
        checked(cublasLtMatrixLayoutCreate(&p->a,CUDA_R_4F_E2M1,w.columns,w.rows,w.columns),"half result A layout");
        checked(cublasLtMatrixLayoutCreate(&p->b,CUDA_R_4F_E2M1,w.columns,tokens,w.columns),"half result B layout");
        checked(cublasLtMatrixLayoutCreate(&p->c,CUDA_R_16F,w.rows,tokens,w.rows),"half result C layout");
        cublasLtMatmulPreference_t preference=nullptr;checked(cublasLtMatmulPreferenceCreate(&preference),"half result preference");
        std::array<cublasLtMatmulHeuristicResult_t,4> selected{};int count=0;
        const bool admitted=half_tactic_shape(w.rows,w.columns,tokens);
        try{checked(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&bytes,sizeof(bytes)),"half result workspace bound");
            checked(cublasLtMatmulAlgoGetHeuristic(handle,p->op,p->a,p->b,p->c,p->c,preference,admitted?4:1,selected.data(),&count),"half result heuristic");}
        catch(...){cublasLtMatmulPreferenceDestroy(preference);throw;}
        cublasLtMatmulPreferenceDestroy(preference);
        if(count<1||selected[0].state!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("native FP4 half result tactic unavailable");
        p->algorithm=selected[0].algo;
        p->alternate_ready=admitted&&count>2&&selected[2].state==CUBLAS_STATUS_SUCCESS&&selected[2].workspaceSize<=bytes&&
            half_tactic_matches(selected[0].algo,false)&&half_tactic_matches(selected[2].algo,true);
        if(p->alternate_ready)p->alternate=selected[2].algo;
        plans.push_back(std::move(p));return *plans.back();
    }
};
Nvfp4HalfGemm::Nvfp4HalfGemm(cudaStream_t s,size_t bytes):impl_(std::make_unique<Impl>(s,bytes)){}
Nvfp4HalfGemm::~Nvfp4HalfGemm()=default;
size_t Nvfp4HalfGemm::bytes()const{return impl_->bytes;}
void Nvfp4HalfGemm::use_measured_tactic(bool active){impl_->measured=active;}
uint64_t Nvfp4HalfGemm::alternate_submissions()const{return impl_->alternate_calls;}
void Nvfp4HalfGemm::project(Nvfp4View w,Nvfp4Input x,void *output,int tokens,cudaStream_t stream){
    auto &i=*impl_;
    if(stream!=i.stream||!w.values||!w.scales||!x.values||!x.scales||!output||tokens<128||tokens>x.capacity_rows||w.columns<64||w.columns%64||w.rows<1||w.columns>x.capacity_columns||
       !(w.weight_scale>0)||!(w.input_scale>0)||!std::isfinite(w.weight_scale*w.input_scale))throw std::invalid_argument("half result operands/geometry");
    auto &p=i.plan(w,x,tokens);
    checked(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&w.scales,sizeof(w.scales)),"half result weight SF");
    checked(cublasLtMatmulDescSetAttribute(p.op,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&x.scales,sizeof(x.scales)),"half result input SF");
    const bool alternate=i.measured&&p.alternate_ready;
    const auto &algorithm=alternate?p.alternate:p.algorithm;if(alternate)++i.alternate_calls;
    const float alpha=w.weight_scale*w.input_scale,beta=0;
    checked(cublasLtMatmul(i.handle,p.op,&alpha,w.values,p.a,x.values,p.b,&beta,output,p.c,output,p.c,&algorithm,i.workspace,i.bytes,stream),"native FP4 half result");
}
}
