#include "strata/qwen35/native/nvfp4_prefill_lt.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cublasLt.h>
#include <array>
#include <string>
namespace strata::qwen35::native {
namespace {
constexpr size_t Budget=32u*1024*1024;
void check(cublasStatus_t status,const char* operation){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(operation)+": "+std::to_string(int(status)));
}
struct Plan {
    cublasLtHandle_t handle=nullptr;
    cublasLtMatmulDesc_t desc=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;
    cublasLtMatmulAlgo_t algo{};Nvfp4LtInfo information{};
    Plan(int rows,int columns,int tokens){
        information.rows=rows;information.columns=columns;information.tokens=tokens;
        information.library=cublasLtGetVersion();
        try {
            int device=0;cudaDeviceProp props{};
            check_cuda(cudaGetDevice(&device),"PP Lt device");check_cuda(cudaGetDeviceProperties(&props,device),"PP Lt GPU");
            information.sm=10*props.major+props.minor;
            check(cublasLtCreate(&handle),"PP Lt handle");
            check(cublasLtMatmulDescCreate(&desc,CUBLAS_COMPUTE_32F,CUDA_R_32F),"PP Lt descriptor");
            const cublasOperation_t transpose=CUBLAS_OP_T;
            check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)),"PP Lt TN");
            const cublasLtPointerMode_t mode=CUBLASLT_POINTER_MODE_DEVICE;
            check(cublasLtMatmulDescSetAttribute(desc,CUBLASLT_MATMUL_DESC_POINTER_MODE,&mode,sizeof(mode)),"PP Lt GPU coefficients");
            check(cublasLtMatrixLayoutCreate(&a,CUDA_R_16F,columns,rows,columns),"PP Lt weights layout");
            check(cublasLtMatrixLayoutCreate(&b,CUDA_R_16F,columns,tokens,columns),"PP Lt input layout");
            check(cublasLtMatrixLayoutCreate(&c,CUDA_R_32F,rows,tokens,rows),"PP Lt output layout");
            check(cublasLtMatmulPreferenceCreate(&preference),"PP Lt preference");
            check(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&Budget,sizeof(Budget)),"PP Lt budget");
            std::array<cublasLtMatmulHeuristicResult_t,16> found{};int count=0;
            const auto status=cublasLtMatmulAlgoGetHeuristic(handle,desc,a,b,c,c,preference,int(found.size()),found.data(),&count);
            if(status==CUBLAS_STATUS_NOT_SUPPORTED)return;
            check(status,"PP Lt heuristic");
            // Explicit experimental choice from the real8192 Down/fullFFN A/B.
            // Record the concrete algorithm, not only this heuristic ordinal.
            if(count<=3||found[3].state!=CUBLAS_STATUS_SUCCESS||found[3].workspaceSize>Budget)return;
            algo=found[3].algo;information.workspace=found[3].workspaceSize;
            auto attribute=[&](cublasLtMatmulAlgoConfigAttributes_t name){
                int value=-1;size_t written=0;
                if(cublasLtMatmulAlgoConfigGetAttribute(&algo,name,&value,sizeof(value),&written)!=CUBLAS_STATUS_SUCCESS)return -1;
                return value;
            };
            information.algorithm=attribute(CUBLASLT_ALGO_CONFIG_ID);
            information.tile=attribute(CUBLASLT_ALGO_CONFIG_TILE_ID);
            information.stages=attribute(CUBLASLT_ALGO_CONFIG_STAGES_ID);
            information.splitk=attribute(CUBLASLT_ALGO_CONFIG_SPLITK_NUM);
            information.reduction=attribute(CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME);
            information.swizzle=attribute(CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING);
            information.custom=attribute(CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION);
            information.ready=true;
        }catch(...){release();throw;}
    }
    void release(){
        if(preference)cublasLtMatmulPreferenceDestroy(preference);
        if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);if(c)cublasLtMatrixLayoutDestroy(c);
        if(desc)cublasLtMatmulDescDestroy(desc);if(handle)cublasLtDestroy(handle);
    }
    ~Plan(){release();}
};
}
Nvfp4PrefillLt::Nvfp4PrefillLt(int rows,int columns,int tokens){
    if(rows<1||columns<1||tokens<1)throw std::invalid_argument("PP Lt dimensions");
    if(tokens==8192)implementation_=new Plan(rows,columns,tokens);
}
Nvfp4PrefillLt::~Nvfp4PrefillLt(){delete static_cast<Plan*>(implementation_);}
Nvfp4LtInfo Nvfp4PrefillLt::info()const{return implementation_?static_cast<Plan*>(implementation_)->information:Nvfp4LtInfo{};}
bool Nvfp4PrefillLt::supports(int rows,int columns,int tokens)const{
    const auto i=info();return i.ready&&rows==i.rows&&columns==i.columns&&tokens==i.tokens;
}
void Nvfp4PrefillLt::project(const void* weights,const void* input,float* output,const float* alpha,const float* zero,
    void* workspace,size_t bytes,cudaStream_t stream){
    auto* p=static_cast<Plan*>(implementation_);
    if(!p||!p->information.ready||!weights||!input||!output||!alpha||!zero||!workspace||bytes<Budget||
       reinterpret_cast<uintptr_t>(workspace)%256)
        throw std::invalid_argument("PP Lt plan/buffer lease");
    check(cublasLtMatmul(p->handle,p->desc,alpha,weights,p->a,input,p->b,zero,output,p->c,output,p->c,
        &p->algo,workspace,bytes,stream),"PP Lt Down projection");
}
}
