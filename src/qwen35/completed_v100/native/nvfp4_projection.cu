#include "strata/qwen35/completed_v100/native/nvfp4_projection.hpp"
#include "strata/qwen35/completed_v100/native/nvfp4.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include "nvfp4_codec.cuh"
#include "nvfp4_q8_activation.cuh"
#include "nvfp4_q8_fast.cuh"
#include "nvfp4_q8_soa.cuh"
#include "nvfp4_q8_fastsoa.cuh"
#include "q8_power2_prefill.cuh"
#include "raw_q8_wide.cuh"
#include "q8_raw.cuh"
#include "strata/qwen35/completed_v100/native/nvfp4_packed.hpp"
#include "strata/qwen35/completed_v100/native/raw_q8_projection.hpp"
#include "strata/qwen35/completed_v100/native/original_cutlass.hpp"
#include "strata/qwen35/completed_v100/native/original_fused.hpp"
#include <cmath>
#include <stdexcept>
#include <string>

namespace strata::completed_v100::qwen35::native {
namespace {
// tile039 full-role Graph costs; preserve packed T1, Q8 head and cuBLAS PP.
// Return -1 for every shape without a positive original-format A/B.
int selective_nvfp4(int rows,int columns,int tokens) {
    if(tokens==2)return rows==1024&&columns==5120?0:rows==5120&&columns==17408?2:-1;
    if(tokens!=3)return -1;
    if(rows==5120&&columns==17408)return 2;
    if(columns==5120&&(rows==48||rows==6144||rows==12288))return 2;
    if((columns==5120&&(rows==1024||rows==10240||rows==17408))||
       (rows==5120&&columns==6144))return 0;
    return -1;
}
// 048/050 real-role forward/reverse costs. RawGGUF, no weightlayout change.
int mixed_nvfp4(int rows,int columns,int tokens) {
    if(tokens>=2&&tokens<=3&&columns==5120) {
        if(rows==6144||rows==17408||rows==12288)return 5;
        if(rows==10240)return 4;
        if(rows==1024&&tokens==2)return 5;
    }
    if(rows==5120&&columns==6144&&tokens==3)return 4;
    return selective_nvfp4(rows,columns,tokens);
}
void checked(cublasStatus_t result,const char* operation) {
    if(result!=CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(operation)+": cuBLAS "+std::to_string(int(result)));
}
__global__ void split_input(const float* input,half* high,half* low,size_t count) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count){const float value=input[i];const half h=__float2half_rn(value);
        high[i]=h;low[i]=__float2half_rn((value-__half2float(h))*4096.f);}
}
__global__ void direct(const void* blocks,float scale,const float* device_scale,const float* input,float* output,
    int rows,int columns,int tokens) {
    const int row=blockIdx.x;if(row>=rows)return;
    float sum[3]{0.f,0.f,0.f};__shared__ float shared[3][4];
    for(int c=threadIdx.x;c<columns;c+=blockDim.x) {
        const float weight=nvfp4::matrix_value(blocks,columns,row,c);
        #pragma unroll
        for(int t=0;t<3;++t)if(t<tokens)sum[t]=fmaf(weight,input[size_t(t)*columns+c],sum[t]);
    }
    #pragma unroll
    for(int t=0;t<3;++t)if(t<tokens) {
        for(int offset=16;offset;offset/=2)sum[t]+=__shfl_down_sync(0xffffffff,sum[t],offset);
        if((threadIdx.x&31)==0)shared[t][threadIdx.x/32]=sum[t];
    }
    __syncthreads();
    if(threadIdx.x<32)for(int t=0;t<tokens;++t) {
        float total=threadIdx.x<4?shared[t][threadIdx.x]:0.f;
        for(int offset=16;offset;offset/=2)total+=__shfl_down_sync(0xffffffff,total,offset);
        if(threadIdx.x==0)output[size_t(t)*rows+row]=total*(device_scale?*device_scale:scale);
    }
}
__global__ void decode_q8_f32(const void* raw,float* output,int rows,int columns) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<size_t(rows)*columns)output[i]=q8_raw_value(raw,columns,int(i/columns),int(i%columns));
}
__global__ void split_q8(const void* raw,half* high,half* low,int rows,int columns) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*columns)return;
    const float value=q8_raw_value(raw,columns,int(i/columns),int(i%columns));
    const half h=__float2half_rn(value);high[i]=h;low[i]=__float2half_rn((value-__half2float(h))*4096.f);
}
}
Nvfp4Projection::Nvfp4Projection(size_t weights,int width,int tokens):
    weight_values_(weights),input_width_(width),maximum_tokens_(tokens) {
    if(!weights||weights>size_t(1)<<30||width<64||width%64||tokens<1||tokens>8192)
        throw std::invalid_argument("NVFP4 native workspace geometry");
    try {
        int device=0;cudaDeviceProp caps{};
        check_cuda(cudaGetDevice(&device),"Original projection device");
        check_cuda(cudaGetDeviceProperties(&caps,device),"Original projection SM inventory");sm_count_=caps.multiProcessorCount;volta_=caps.major==7&&caps.minor==0;
        check_cuda(cudaMalloc(&weights_,weights*4),"NVFP4 transient GPU weight arena");
        check_cuda(cudaMalloc(&high_,size_t(width)*tokens*4),"Original contiguous high/low input arena");
        low_=static_cast<half*>(high_)+size_t(width)*tokens; // Non-owning view, never freed independently.
        check_cuda(cudaMalloc(reinterpret_cast<void**>(&constants_),20),"Quantized GEMM device constants");
        const float constants[5]{1.f,1.f/4096.f,1.f/(4096.f*4096.f),0.f,1.f};
        check_cuda(cudaMemcpy(constants_,constants,20,cudaMemcpyHostToDevice),"Quantized GEMM constants upload");
        check_cuda(cudaMalloc(&vendor_workspace_,VendorWorkspaceBytes),"Original GEMM owned cuBLAS workspace");
        check_cuda(cudaMalloc(&packed_input_,size_t(width)*(tokens<3?tokens:3)*4),"NVFP4 bounded packed activation workspace");
        checked(cublasCreate(&handle_),"NVFP4 vendor handle");
        checked(cublasSetMathMode(handle_,CUBLAS_DEFAULT_MATH),"NVFP4 accumulation policy");
    }catch(...){if(handle_)cublasDestroy(handle_);cudaFree(packed_input_);cudaFree(vendor_workspace_);cudaFree(constants_);cudaFree(high_);cudaFree(weights_);throw;}
}
Nvfp4Projection::~Nvfp4Projection(){if(handle_)cublasDestroy(handle_);cudaFree(packed_input_);cudaFree(vendor_workspace_);cudaFree(constants_);cudaFree(high_);cudaFree(weights_);}
void Nvfp4Projection::run(const void* blocks,float scale,const float* input,float* output,
    int rows,int columns,int tokens,cudaStream_t stream) {
    if(!blocks||!input||!output||input==output||rows<1||columns<64||columns%64||
       columns>input_width_||tokens<1||tokens>maximum_tokens_||
       !(scale>0)||!std::isfinite(scale))
        throw std::invalid_argument("NVFP4 projection owner/scale/extent mismatch");
    if(tokens<=3) {
        direct<<<rows,128,0,stream>>>(blocks,scale,nullptr,input,output,rows,columns,tokens);
        check_cuda(cudaGetLastError(),"NVFP4 direct F32 projection");return;
    }
    if(size_t(rows)*columns>weight_values_)throw std::invalid_argument("NVFP4 PP decoded weight budget exceeded");
    nvfp4_decode_raw_half(blocks,weights_,rows,columns,stream);
    const size_t count=size_t(tokens)*columns;
    split_input<<<unsigned((count+255)/256),256,0,stream>>>(input,static_cast<half*>(high_),static_cast<half*>(low_),count);
    check_cuda(cudaGetLastError(),"NVFP4 high/low FP32 input decomposition");
    checked(cublasSetStream(handle_,stream),"NVFP4 vendor stream");
    checked(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"Original GEMM stable workspace after stream bind");
    checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_HOST),"NVFP4 diagnostic host coefficients");
    constexpr float zero=0.f,one=1.f;
    for(int term=0;term<2;++term) {
        const float alpha=term?scale/4096.f:scale;
        checked(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,rows,tokens,columns,&alpha,
            weights_,CUDA_R_16F,columns,term?low_:high_,CUDA_R_16F,columns,term?&one:&zero,
            output,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "NVFP4 high/low Tensor Core projection");
    }
}
void Nvfp4Projection::run_device(const void* blocks,const float* scale_header,const float* input,float* output,
    int rows,int columns,int tokens,cudaStream_t stream) {
    if(!blocks||!scale_header||!input||!output||input==output||rows<1||columns<64||columns%64||
       columns>input_width_||tokens<1||tokens>maximum_tokens_)
        throw std::invalid_argument("NVFP4 device-scale projection owner/extent mismatch");
    if(tokens==3&&q8_activation_&&q8_fastsoa_) {
        nvfp4_q8fs_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8fs_project(blocks,scale_header,nullptr,nullptr,packed_input_,output,nullptr,nullptr,rows,columns,tokens,stream);return;
    }
    if(tokens==3&&q8_activation_&&q8_soa_) {
        nvfp4_q8s_prepare(input,packed_input_,columns,tokens,stream);
        nvfp4_q8s_project(blocks,scale_header,nullptr,nullptr,packed_input_,output,nullptr,nullptr,rows,columns,tokens,stream);return;
    }
    if(tokens<=3&&q8_activation_) {
        if(q8_fast_&&(tokens==1||(tokens==2&&((rows==48&&columns==5120)||(rows==5120&&(columns==6144||columns==17408)))))) {
            nvfp4_q8f_prepare(input,packed_input_,columns,tokens,stream);
            nvfp4_q8f_project(blocks,scale_header,nullptr,nullptr,packed_input_,output,nullptr,nullptr,rows,columns,tokens,stream);
        }else{
            nvfp4_q8a_prepare(input,packed_input_,columns,tokens,stream);
            nvfp4_q8a_project(blocks,scale_header,nullptr,nullptr,packed_input_,output,nullptr,nullptr,rows,columns,tokens,stream);
        }return;
    }
    const int tile=mixed_decode_?mixed_nvfp4(rows,columns,tokens):fused_selective_?selective_nvfp4(rows,columns,tokens):fused_tile_;
    if((tokens<=3?fused_decode_:fused_prefill_)&&tile>=0) {
        ++fused_calls_;
        original_fused_projection(false,blocks,scale_header,input,output,rows,columns,tokens,
            high_,size_t(input_width_)*maximum_tokens_*2,static_cast<float*>(weights_),weight_values_,sm_count_,tile,stream);return;
    }
    if(tokens<=3&&!library_decode_) {
        if(packed_decode_) {
            if(packed_cached_)nvfp4_packed_cached_projection(blocks,scale_header,input,output,rows,columns,tokens,packed_input_,stream);
            else nvfp4_packed_projection(blocks,scale_header,input,output,rows,columns,tokens,packed_input_,stream);
            return;
        }
        direct<<<rows,128,0,stream>>>(blocks,1.f,scale_header,input,output,rows,columns,tokens);
        check_cuda(cudaGetLastError(),"NVFP4 device-scale direct projection");return;
    }
    if(q8_pp_p2_&&tokens>=128) {
        Q8P2Workspace workspace{weights_,high_,handle_,vendor_workspace_,VendorWorkspaceBytes,
            constants_+3,weight_values_,size_t(input_width_)*maximum_tokens_,q8_pp_decode8_,q8_pp_vector_};
        q8p2_nvfp4_prefill(blocks,scale_header,input,output,rows,columns,tokens,workspace,stream);return;
    }
    if(combined_pp_&&tokens>=128) {
        run_combined_prefill(blocks,scale_header,input,output,rows,columns,tokens,stream);return;
    }
    if(size_t(rows)*columns>weight_values_)throw std::invalid_argument("NVFP4 PP decoded weight budget exceeded");
    nvfp4_decode_raw_half(blocks,weights_,rows,columns,stream);
    const size_t count=size_t(tokens)*columns;
    split_input<<<unsigned((count+255)/256),256,0,stream>>>(input,static_cast<half*>(high_),static_cast<half*>(low_),count);
    check_cuda(cudaGetLastError(),"NVFP4 scaled input residual decomposition");
    checked(cublasSetStream(handle_,stream),"NVFP4 device-scale stream");
    checked(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"Original GEMM stable workspace after stream bind");
    checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),"NVFP4 device-scale coefficients");
    for(int term=0;term<2;++term) {
        if(cutlass_) {
            original_cutlass_half(weights_,term?low_:high_,output,rows,columns,tokens,rows,
                scale_header+term,constants_+(term?4:3),stream);continue;
        }
        checked(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,rows,tokens,columns,scale_header+term,
            weights_,CUDA_R_16F,columns,term?low_:high_,CUDA_R_16F,columns,constants_+(term?4:3),
            output,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "NVFP4 original-scale high/low GEMM");
    }
}
void Nvfp4Projection::run_q8(const void* blocks,const float* input,float* output,
    int rows,int columns,int tokens,cudaStream_t stream) {
    if(!blocks||!input||!output||input==output||rows<1||columns<32||columns%32||
       columns>input_width_||tokens<1||tokens>maximum_tokens_)
        throw std::invalid_argument("Original Q8 projection owner/extent mismatch");
    // 073 full-head cost admits T1/T2 only; T3 keeps the faster063SoA path.
    if(tokens<=2&&q8_activation_==2&&q8_head_wide_&&rows==248320&&columns==5120) {
        nvfp4_q8a_prepare(input,packed_input_,columns,tokens,stream);
        raw_q8_wide_project(blocks,packed_input_,output,rows,columns,tokens,stream);return;
    }
    if(tokens==3&&q8_activation_==2&&q8_soa_) {
        nvfp4_q8s_prepare(input,packed_input_,columns,tokens,stream);
        raw_q8s_project(blocks,packed_input_,output,rows,columns,tokens,stream);return;
    }
    if(tokens<=3&&q8_activation_==2) {
        nvfp4_q8a_prepare(input,packed_input_,columns,tokens,stream);
        raw_q8a_project(blocks,packed_input_,output,rows,columns,tokens,stream);return;
    }
    // 043/044d real-role ordered A/B: small K/V favor warp4, other Q8
    // roles favor group16 for T1/T2 and group32 for T3. Default remains off.
    if(tokens<=3&&q8_optimized_) {
        Matrix matrix{blocks,Quant::Q8,rows,columns};
        if(rows==1024&&columns==5120)raw_q8_projection_warp(matrix,input,output,tokens,4,stream);
        else if(q8_blocked_&&(tokens==2||(tokens==1&&rows!=248320)||
            (tokens==3&&rows==5120&&(columns==6144||columns==17408||columns==10240))))
            raw_q8_blocked_projection(matrix,input,output,tokens,16,packed_input_,stream);
        else raw_q8_packed_projection(matrix,input,output,tokens,tokens==3?32:16,packed_input_,stream);
        return;
    }
    if(tokens<=3&&q8_group_) {
        raw_q8_packed_projection({blocks,Quant::Q8,rows,columns},input,output,tokens,q8_group_,packed_input_,stream);return;
    }
    if(tokens<=3&&q8_warps_) {
        raw_q8_projection_warp({blocks,Quant::Q8,rows,columns},input,output,tokens,q8_warps_,stream);return;
    }
    if((tokens<=3?fused_decode_:fused_prefill_)&&!fused_selective_) {
        ++fused_calls_;
        original_fused_projection(true,blocks,nullptr,input,output,rows,columns,tokens,
            high_,size_t(input_width_)*maximum_tokens_*2,static_cast<float*>(weights_),weight_values_,sm_count_,fused_tile_,stream);return;
    }
    if(tokens<=3&&library_decode_){run_q8_library(blocks,input,output,rows,columns,tokens,stream);return;}
    if(tokens<=3){raw_q8_projection({blocks,Quant::Q8,rows,columns},input,output,tokens,stream);return;}
    if(size_t(rows)*columns>weight_values_)throw std::invalid_argument("Q8 PP decoded weight budget exceeded");
    const size_t count=size_t(tokens)*columns,weight_count=size_t(rows)*columns;
    half* low_weights=static_cast<half*>(weights_)+weight_values_;
    split_q8<<<unsigned((weight_count+255)/256),256,0,stream>>>(blocks,static_cast<half*>(weights_),low_weights,rows,columns);
    split_input<<<unsigned((count+255)/256),256,0,stream>>>(input,static_cast<half*>(high_),static_cast<half*>(low_),count);
    check_cuda(cudaGetLastError(),"Q8 scaled weight/input residual decomposition");
    checked(cublasSetStream(handle_,stream),"Q8 vendor stream");
    checked(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"Original GEMM stable workspace after stream bind");
    checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),"Q8 vendor device coefficients");
    for(int w=0;w<2;++w)for(int x=0;x<2;++x) {
        if(cutlass_) {
            original_cutlass_half(w?low_weights:weights_,x?low_:high_,output,rows,columns,tokens,rows,
                constants_+w+x,constants_+((w||x)?4:3),stream);continue;
        }
        checked(cublasGemmEx(handle_,CUBLAS_OP_T,CUBLAS_OP_N,rows,tokens,columns,constants_+w+x,
            w?low_weights:weights_,CUDA_R_16F,columns,x?low_:high_,CUDA_R_16F,columns,
            constants_+((w||x)?4:3),output,CUDA_R_32F,rows,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),
            "Original Q8 four-term high/low GEMM");
    }
}

void Nvfp4Projection::run_q8_library(const void* blocks,const float* input,float* output,
    int rows,int columns,int tokens,cudaStream_t stream) {
    if(!blocks||!input||!output||rows<1||columns<32||columns%32||tokens<1||tokens>3||
       columns>input_width_||weight_values_<size_t(columns))
        throw std::invalid_argument("Original Q8 library matrix owner/extent");
    checked(cublasSetStream(handle_,stream),"Original Q8 cuBLAS stream");
    checked(cublasSetWorkspace(handle_,vendor_workspace_,VendorWorkspaceBytes),"Original Q8 owned library workspace");
    checked(cublasSetPointerMode(handle_,CUBLAS_POINTER_MODE_DEVICE),"Original Q8 device coefficients");
    const int tile=int(weight_values_/columns);
    for(int first=0;first<rows;first+=tile) {
        const int n=rows-first<tile?rows-first:tile;
        const auto* raw=static_cast<const unsigned char*>(blocks)+size_t(first)*(columns/32)*34;
        decode_q8_f32<<<unsigned((size_t(n)*columns+255)/256),256,0,stream>>>(raw,static_cast<float*>(weights_),n,columns);
        check_cuda(cudaGetLastError(),"Original Q8 exact GPU decode for library");
        if(cutlass_) {
            original_cutlass_float(static_cast<const float*>(weights_),input,output+first,
                n,columns,tokens,rows,constants_,constants_+3,stream);continue;
        }
        checked(cublasSgemm(handle_,CUBLAS_OP_T,CUBLAS_OP_N,n,tokens,columns,constants_,
            static_cast<const float*>(weights_),columns,input,columns,constants_+3,output+first,rows),
            "Original Q8 cuBLAS full-work projection");
    }
}

}
