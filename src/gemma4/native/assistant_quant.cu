#include "strata/gemma4/native/assistant_quant.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include <cuda_fp16.h>
#include <algorithm>
#include <cstdint>
#include <stdexcept>
namespace strata::gemma4::native::assistant {
namespace {
void blas(cublasStatus_t status){if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Assistant F32 cuBLAS projection");}
bool overlap(const void* a,size_t na,const void* b,size_t nb){
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return x<=y?y-x<na:x-y<nb;
}
__global__ void decode(Q40Matrix m,float* out,int first,int rows){
    const size_t i=size_t(blockIdx.x)*256+threadIdx.x;if(i>=size_t(rows)*m.columns)return;
    const size_t at=size_t(first)*m.columns+i;const auto* block=static_cast<const unsigned char*>(m.data)+(at/32)*18;
    const int j=at%32;const unsigned q=(block[2+j%16]>>(j<16?0:4))&15u;
    out[i]=__half2float(*reinterpret_cast<const half*>(block))*float(int(q)-8);
}
__global__ void concat(const float* embedding,const float* hidden,float* out,unsigned* invalid){
    const int i=int(blockIdx.x*256+threadIdx.x);if(i>=10752)return;
    const float x=i<5376?embedding[i]:hidden[i-5376];out[i]=x;
    if(!isfinite(x))atomicOr(invalid,1u);
}
}
void q40_decode_rows(Q40Matrix m,float* out,int first,int rows,cudaStream_t stream){
    if(!m.data||!out||!stream||m.rows<1||m.rows>262144||m.columns<32||m.columns>16384||m.columns%32||first<0||rows<1||first>m.rows-rows||
       reinterpret_cast<uintptr_t>(m.data)%2||reinterpret_cast<uintptr_t>(out)%alignof(float)||
       overlap(m.data,size_t(m.rows)*(m.columns/32)*18,out,size_t(rows)*m.columns*4))
        throw std::invalid_argument("Assistant Q4_0 decode lease");
    decode<<<unsigned((size_t(rows)*m.columns+255)/256),256,0,stream>>>(m,out,first,rows);
    strata::qwen35::native::check_cuda(cudaGetLastError(),"Assistant Q4_0 F32 decode");
}
void q40_project(Q40Matrix m,const float* input,float* output,float* decoded,size_t decoded_values,
    cublasHandle_t handle,void* workspace,size_t workspace_bytes,cudaStream_t stream,int row_chunk){
    if(!m.data||!input||!output||!decoded||!handle||!workspace||!stream||m.rows<1||m.rows>262144||m.columns<32||m.columns>16384||m.columns%32||
       row_chunk<1||row_chunk>32768||workspace_bytes<256||decoded_values>SIZE_MAX/4||
       reinterpret_cast<uintptr_t>(workspace)%256||reinterpret_cast<uintptr_t>(input)%4||reinterpret_cast<uintptr_t>(output)%4||
       decoded_values/size_t(m.columns)<size_t((std::min)(m.rows,row_chunk)))
        throw std::invalid_argument("Assistant Q4_0 SGEMM extent");
    const void* pointers[]{m.data,input,output,decoded,workspace};
    const size_t bytes[]{size_t(m.rows)*(m.columns/32)*18,size_t(m.columns)*4,size_t(m.rows)*4,decoded_values*4,workspace_bytes};
    for(int i=0;i<5;++i)for(int j=i+1;j<5;++j)if(overlap(pointers[i],bytes[i],pointers[j],bytes[j]))
        throw std::invalid_argument("Assistant projection alias");
    blas(cublasSetStream(handle,stream));blas(cublasSetWorkspace(handle,workspace,workspace_bytes));
    blas(cublasSetPointerMode(handle,CUBLAS_POINTER_MODE_HOST));blas(cublasSetMathMode(handle,CUBLAS_DEFAULT_MATH));
    const float one=1,zero=0;
    for(int first=0;first<m.rows;first+=row_chunk){
        const int rows=(std::min)(row_chunk,m.rows-first);q40_decode_rows(m,decoded,first,rows,stream);
        blas(cublasSgemm(handle,CUBLAS_OP_T,CUBLAS_OP_N,rows,1,m.columns,&one,decoded,m.columns,
            input,m.columns,&zero,output+first,rows));
    }
}
void condition_concat(const float* embedding,const float* hidden,float* out,unsigned* invalid,cudaStream_t stream){
    if(!embedding||!hidden||!out||!invalid||!stream||reinterpret_cast<uintptr_t>(embedding)%4||reinterpret_cast<uintptr_t>(hidden)%4||
       reinterpret_cast<uintptr_t>(out)%4||reinterpret_cast<uintptr_t>(invalid)%4||overlap(out,10752*4,embedding,5376*4)||
       overlap(out,10752*4,hidden,5376*4)||overlap(out,10752*4,invalid,4)||
       overlap(embedding,5376*4,invalid,4)||overlap(hidden,5376*4,invalid,4))throw std::invalid_argument("Assistant concat lease");
    concat<<<42,256,0,stream>>>(embedding,hidden,out,invalid);
    strata::qwen35::native::check_cuda(cudaGetLastError(),"Assistant main embedding/seed concat");
}
}
