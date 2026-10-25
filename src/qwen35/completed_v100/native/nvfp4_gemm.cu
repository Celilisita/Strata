#include "strata/qwen35/completed_v100/native/nvfp4_gemm.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
namespace strata::completed_v100::qwen35::native {
namespace {
__global__ void initialize(float* header) {
    if(!threadIdx.x){header[1]=header[0]/4096.f;header[2]=0.f;header[3]=1.f;}
}
}
void initialize_nvfp4_header(void* header,cudaStream_t stream) {
    if(!header)throw std::invalid_argument("Missing original NVFP4 scale owner");
    initialize<<<1,1,0,stream>>>(static_cast<float*>(header));
    check_cuda(cudaGetLastError(),"NVFP4 original scale descriptor preparation");
}
void Nvfp4Gemm::project_original(Matrix matrix,const float* input,float* output,int tokens,cudaStream_t stream) {
    original_matrix_bytes(matrix);++calls_;
    if(matrix.quant==Quant::Q8&&matrix.layout==BlockLayout::Gguf) {
        projection_.run_q8(matrix.data,input,output,matrix.rows,matrix.columns,tokens,stream);return;
    }
    if(matrix.quant!=Quant::NVFP4||matrix.layout!=BlockLayout::Nvfp4Scale)
        throw std::invalid_argument("Original GPU provider received an unregistered quantized layout");
    const auto* raw=static_cast<const unsigned char*>(matrix.data)+Nvfp4HeaderBytes;
    if(q8_pp_lt_&&q8_pp_p2()&&q8_pp_decode8()&&pp_lt_.supports(matrix.rows,matrix.columns,tokens)){
        projection_.run_prefill_lt(pp_lt_,raw,static_cast<const float*>(matrix.data),input,output,matrix.rows,matrix.columns,tokens,stream);
        ++pp_lt_calls_;return;
    }
    projection_.run_device(raw,static_cast<const float*>(matrix.data),input,output,matrix.rows,matrix.columns,tokens,stream);
}
void project_original(DenseGemm* service,Matrix matrix,const float* input,float* output,int tokens,cudaStream_t stream) {
    auto* provider=dynamic_cast<Nvfp4Gemm*>(service);
    if(!provider)throw std::invalid_argument("Original quantized projection has no owned GPU provider");
    provider->project_original(matrix,input,output,tokens,stream);
}
}
