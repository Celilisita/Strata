#include "p2_vector_encode.hpp"
#include "strata/qwen35/native/nvfp4_prefill_gdn.hpp"
#include "strata/qwen35/native/nvfp4_gemm.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_decode8.hpp"
#include "q8_power2_prefill.cuh"
#include <string>
namespace strata::qwen35::native {
namespace {
struct Span {int end;float* output;const float* scale;};
struct Group {Span spans[4];int rows,tokens;};
__global__ void unpack(const float* packed,Group g){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(g.rows)*g.tokens)return;
    const int row=i%g.rows,token=i/g.rows;int first=0;
    #pragma unroll
    for(int j=0;j<4;++j){
        if(row>=first&&row<g.spans[j].end){
            const int width=g.spans[j].end-first;
            g.spans[j].output[size_t(token)*width+row-first]=packed[i]*(*g.spans[j].scale);
        }
        first=g.spans[j].end;
    }
}
void checked(cublasStatus_t status,const char* op){
    if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string(op)+": "+std::to_string(int(status)));
}
}
void OriginalNvfp4Projection::run_prefill_gdn(const GdnPrefillProjection* p,const float* x,float* packed,
    size_t capacity,int tokens,cudaStream_t stream){
    if(!p||!x||!packed||packed==x||!q8_pp_p2_||!q8_pp_decode8_||
       !q8_pp_shape(tokens)||tokens>maximum_tokens_)
        throw std::invalid_argument("NVFP4 GDN prefill input/policy lease");
    const int columns=p[0].matrix.columns;int total=0;
    if(columns>input_width_)throw std::invalid_argument("NVFP4 GDN input width lease");
    for(int i=0;i<4;++i){const auto m=p[i].matrix;original_matrix_bytes(m);
        if(m.quant!=Quant::NVFP4||m.layout!=OriginalNvfp4Scale||m.columns!=columns||!p[i].output||p[i].output==packed||p[i].output==x)
            throw std::invalid_argument("NVFP4 GDN projection lease");
        total+=m.rows;
    }
    if(capacity<size_t(total)*tokens||size_t(total)*columns>2*weight_values_)
        throw std::invalid_argument("NVFP4 GDN transient arena capacity");
    // The caller lends the idle FFN Up arena. It is consumed before conv/GDN
    // and FFN reuse it; no persistent decoded weights or new allocation.
    if(q8_pp_vector_)q8p2_encode_vector(x,high_,size_t(tokens)*columns,stream);
    else q8p2_encode(x,high_,size_t(tokens)*columns,stream);
    checked(cublasSetStream(handle_,stream),"NVFP4 GDN cuBLAS stream");
    checked(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"NVFP4 GDN cuBLAS workspace");
    checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),"NVFP4 GDN device coefficients");
    const int width=tokens==1280?2:4;
    for(int first=0;first<4;first+=width){
        Group g{};g.tokens=tokens;size_t offset=0;
        for(int j=0;j<width;++j){const auto m=p[first+j].matrix;
            nvfp4_decode_raw_half8(static_cast<const unsigned char*>(m.data)+OriginalProviderHeaderBytes,
                static_cast<unsigned char*>(weights_)+2*offset,m.rows,m.columns,stream);
            offset+=size_t(m.rows)*columns;g.rows+=m.rows;
            g.spans[j]={g.rows,p[first+j].output,static_cast<const float*>(m.data)};
        }
        for(int j=width;j<4;++j)g.spans[j].end=g.rows;
        checked(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,g.rows,tokens,columns,constants_,
            weights_,CUDA_R_16F,columns,high_,CUDA_R_16F,columns,constants_+3,packed,CUDA_R_32F,g.rows,
            CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"NVFP4 GDN combined projection");
        unpack<<<unsigned((size_t(g.rows)*tokens+255)/256),256,0,stream>>>(packed,g);
        check_cuda(cudaGetLastError(),"NVFP4 GDN scaled unpack");
    }
}
bool OriginalNvfp4Gemm::prefill_gdn(const GdnPrefillProjection* p,const float* x,float* packed,size_t capacity,int tokens,cudaStream_t stream){
    if(!q8_pp_gdn_||!q8_pp_shape(tokens))return false;
    if(!p)throw std::invalid_argument("NVFP4 GDN descriptors");
    for(int i=0;i<4;++i)if(p[i].matrix.quant!=Quant::NVFP4||p[i].matrix.layout!=OriginalNvfp4Scale)return false;
    projection_.run_prefill_gdn(p,x,packed,capacity,tokens,stream);calls_+=4;return true;
}
bool try_nvfp4_gdn_prefill(DenseGemm* service,const GdnPrefillProjection* p,const float* x,float* packed,
    size_t capacity,int tokens,cudaStream_t stream){
    if(tokens!=1280&&tokens!=8192&&tokens!=6144)return false;
    auto* provider=dynamic_cast<OriginalNvfp4Gemm*>(service);
    return provider&&provider->prefill_gdn(p,x,packed,capacity,tokens,stream);
}
}
