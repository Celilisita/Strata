#include "strata/qwen35/completed_v100/native/raw_q8_projection.hpp"
#include "strata/qwen35/completed_v100/native/ffn.hpp"
#include <math_constants.h>
#include <cuda_fp16.h>
namespace strata::completed_v100::qwen35::native {
namespace {
template<int Group,int Tokens> __global__ void prepare_q8(const float* input,int* output,int groups) {
    const unsigned group=(blockIdx.x*blockDim.x+threadIdx.x)/Group,lane=threadIdx.x&(Group-1);
    if(group>=unsigned(groups))return;
    float residual=input[size_t(blockIdx.y)*groups*Group+group*Group+lane];
    const bool invalid=__any_sync(0xffffffff,!isfinite(residual));
    constexpr int Words=Group/4,Planes=3*Words+3;
    auto* item=output+size_t(group/32)*Tokens*Planes*32+size_t(blockIdx.y)*Planes*32+(group&31);
    #pragma unroll
    for(int term=0;term<3;++term) {
        float maximum=fabsf(residual);
        #pragma unroll
        for(int d=Group/2;d;d/=2)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d,Group));
        const float scale=maximum>0.f&&maximum/127.f>0.f?maximum/127.f:1.f;
        const int quant=__float2int_rn(residual/scale);
        unsigned word=unsigned(quant)&255;
        #pragma unroll
        for(int j=1;j<4;++j)word|=(unsigned(__shfl_down_sync(0xffffffff,quant,j,Group))&255)<<(8*j);
        if(!(lane&3))item[(term*Words+lane/4)*32]=int(word);
        if(!lane)reinterpret_cast<float*>(item)[(3*Words+term)*32]=invalid?CUDART_NAN_F:scale;
        residual=fmaf(-float(quant),scale,residual);
    }
}
// A Q8 block starts at a two-byte boundary. Two aligned u16 reads preserve
// every signed weight byte without undefined/misaligned u32 loads.
__device__ __forceinline__ int packed_word(const unsigned short* payload) {
    return int(unsigned(__ldg(payload))|(unsigned(__ldg(payload+1))<<16));
}
template<int Tokens,int Group> __global__ void project_q8_packed(
    const unsigned char* raw,const int* input,float* output,int rows,int columns) {
    const unsigned row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x&31;
    if(row>=unsigned(rows))return;
    constexpr int Words=Group/4,Planes=3*Words+3;
    const unsigned groups=unsigned(columns)/Group;float sum[Tokens]{};
    const auto* block=raw+size_t(row)*(columns/32)*34+(Group==32?lane:lane/2)*34;
    const int* x=input+lane;
    for(unsigned group=lane;group<groups;group+=32) {
        const float delta=__half2float(__ldg(reinterpret_cast<const half*>(block)));
        const auto* payload=reinterpret_cast<const unsigned short*>(block+2+(Group==16?(lane&1)*16:0));
        int weight[Group/4];
        #pragma unroll
        for(int j=0;j<Group/4;++j)weight[j]=packed_word(payload+j*2);
        #pragma unroll
        for(int token=0;token<Tokens;++token) {
            const auto* value=x+size_t(token)*Planes*32;float total=0;
            #pragma unroll
            for(int term=0;term<3;++term) {
                int dot=0;
                #pragma unroll
                for(int j=0;j<Group/4;++j)dot=__dp4a(weight[j],__ldg(value+(term*Words+j)*32),dot);
                total=fmaf(float(dot),__ldg(reinterpret_cast<const float*>(value)+(3*Words+term)*32),total);
            }
            sum[token]=fmaf(total,delta,sum[token]);
        }
        block+=Group*34;x+=Tokens*Planes*32;
    }
    #pragma unroll
    for(int token=0;token<Tokens;++token) {
        #pragma unroll
        for(int d=16;d;d/=2)sum[token]+=__shfl_down_sync(0xffffffff,sum[token],d);
        if(!lane)output[size_t(token)*rows+row]=sum[token];
    }
}
template<int Group,int Tokens> void launch(Matrix matrix,const float* x,float* y,int tokens,void* scratch,cudaStream_t stream) {
    const unsigned groups=unsigned(matrix.columns)/Group;
    prepare_q8<Group,Tokens><<<dim3((groups*Group+127)/128,unsigned(tokens)),128,0,stream>>>(x,static_cast<int*>(scratch),groups);
    check_cuda(cudaGetLastError(),"Original Q8 three-residual GPU preparation");
    const auto* raw=static_cast<const unsigned char*>(matrix.data);const auto* input=static_cast<const int*>(scratch);
    const unsigned blocks=(matrix.rows+3)/4;
    project_q8_packed<Tokens,Group><<<blocks,128,0,stream>>>(raw,input,y,matrix.rows,matrix.columns);
}
}
void raw_q8_blocked_projection(Matrix matrix,const float* input,float* output,int tokens,int group,void* workspace,cudaStream_t stream) {
    original_matrix_bytes(matrix);
    if(matrix.quant!=Quant::Q8||matrix.layout!=BlockLayout::Gguf||!input||!output||!workspace||
       input==output||tokens<1||tokens>3||(group!=16&&group!=32)||matrix.columns%(group*32))
        throw std::invalid_argument("Original Q8 packed projection format/extent");
    if(group==16) {
        if(tokens==1)launch<16,1>(matrix,input,output,tokens,workspace,stream);
        else if(tokens==2)launch<16,2>(matrix,input,output,tokens,workspace,stream);
        else launch<16,3>(matrix,input,output,tokens,workspace,stream);
    }else {
        if(tokens==1)launch<32,1>(matrix,input,output,tokens,workspace,stream);
        else if(tokens==2)launch<32,2>(matrix,input,output,tokens,workspace,stream);
        else launch<32,3>(matrix,input,output,tokens,workspace,stream);
    }
    check_cuda(cudaGetLastError(),"Original Q8 packed DP4A GPU projection");
}
}
