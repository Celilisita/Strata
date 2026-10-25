#include "strata/qwen35/native/original_fused.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "original_fused_kernel.cuh"
#include <algorithm>
#include "nvfp4_wmma.cuh"

namespace strata::qwen35::native {
namespace {
template<bool Q8>
__global__ void combined_input(const float* input,half* packed,int columns,int tokens) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x,count=size_t(columns)*tokens;
    if(i<count) {
        const size_t source=Q8?i:(i/columns)*columns+original_fused::nvfp4_k(int(i%columns));
        const float value=input[source];const half high=__float2half_rn(value);
        packed[i]=high;packed[count+i]=__float2half_rn((value-__half2float(high))*4096.f);
    }
}
template<bool Q8>
__global__ void combine(const float* partial,float* output,const float* scale,
    int rows,int tokens,int splits) {
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(rows)*tokens)return;
    const int row=i%rows,token=i/rows,n=Q8?2*rows:rows;
    const size_t stride=size_t(2)*tokens*n;
    float hh=0.f,lh=0.f,hl=0.f,ll=0.f;
    for(int z=0;z<splits;++z) {
        const auto* p=partial+size_t(z)*stride;
        if constexpr(Q8) {
            hh+=p[size_t(token)*n+2*row];lh+=p[size_t(token+tokens)*n+2*row];
            hl+=p[size_t(token)*n+2*row+1];ll+=p[size_t(token+tokens)*n+2*row+1];
        }else {hh+=p[size_t(token)*n+row];lh+=p[size_t(token+tokens)*n+row];}
    }
    if constexpr(Q8)output[i]=fmaf(ll,1.f/(4096.f*4096.f),fmaf(lh+hl,1.f/4096.f,hh));
    else output[i]=fmaf(lh,1.f/4096.f,hh)*scale[0];
}
template<bool Q8,bool Tiled=false>
void project(const void* payload,const float* scale,const float* input,float* output,
    int rows,int columns,int tokens,void* packed,size_t input_capacity,float* partial,
    size_t partial_capacity,int sms,int tile,cudaStream_t stream) {
    const size_t n=Q8?size_t(2)*rows:size_t(rows);
    const int limit=int((std::min)(input_capacity/(2*columns),partial_capacity/(2*n)));
    if(limit<1)throw std::invalid_argument("Fused CUTLASS original-format scratch budget");
    for(int first=0;first<tokens;first+=limit) {
        const int count=(std::min)(tokens-first,limit);int splits=1;
        if(count<=3) {
            const int blocks=int((n+(tile==1?127:63))/(tile==1?128:64));
            while(splits<32&&blocks*splits<sms*8&&
                size_t(splits*2)*2*count*n<=partial_capacity&&columns/(splits*2)>=64)splits*=2;
        }
        combined_input<Q8><<<unsigned((size_t(count)*columns+255)/256),256,0,stream>>>(
            input+size_t(first)*columns,static_cast<half*>(packed),columns,count);
        check_cuda(cudaGetLastError(),"Fused CUTLASS combined high/low input");
        using namespace original_fused;
        if(count<=3&&tile==5) {
            if constexpr(Q8)throw std::invalid_argument("Compact WMMA Q8 is not admitted");
            else nvfp4_wmma_partial(Tiled,payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        }else if constexpr(Tiled) {
            if(tile==4)FusedKernel<false,32,64,32,32,true,1>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
            else if(tile==2)FusedKernel<false,64,64,32,32,true>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
            else FusedKernel<false,32,64,32,32,true>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        }else if(count<=3&&tile==4) {
            if constexpr(Q8)throw std::invalid_argument("SingleStage Q8 is not admitted");
            else FusedKernel<false,32,64,32,32,false,1>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        }
        else if(count>=128)FusedKernel<Q8,128,128,64,64>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        else if(tile==3)FusedKernel<Q8,128,64,32,32>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        else if(tile==2)FusedKernel<Q8,64,64,32,32>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        else if(tile==1)FusedKernel<Q8,32,128,32,64>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        else FusedKernel<Q8,32,64,32,32>::run(payload,static_cast<half*>(packed),partial,rows,columns,count,splits,stream);
        check_cuda(cudaGetLastError(),"Fused CUTLASS quantized library GEMM");
        combine<Q8><<<unsigned((size_t(count)*rows+255)/256),256,0,stream>>>(
            partial,output+size_t(first)*rows,scale,rows,count,splits);
        check_cuda(cudaGetLastError(),"Fused CUTLASS high/low and parallel-K output");
    }
}
}
void original_fused_projection(bool q8,const void* payload,const float* scale,const float* input,
    float* output,int rows,int columns,int tokens,void* packed,size_t input_capacity,
    float* partial,size_t partial_capacity,int sms,int tile,cudaStream_t stream) {
    if(!payload||(!q8&&!scale)||!input||!output||input==output||!packed||!partial||!stream||
        rows<1||columns<64||columns%64||tokens<1||sms<1||tile<0||tile>5)
        throw std::invalid_argument("Fused CUTLASS original-format owner/geometry");
    if(q8)project<true>(payload,scale,input,output,rows,columns,tokens,packed,input_capacity,partial,partial_capacity,sms,tile,stream);
    else project<false>(payload,scale,input,output,rows,columns,tokens,packed,input_capacity,partial,partial_capacity,sms,tile,stream);
}
void original_fused_tiled_projection(const void* payload,const float* scale,const float* input,
    float* output,int rows,int columns,int tokens,void* packed,size_t input_capacity,
    float* partial,size_t partial_capacity,int sms,int tile,cudaStream_t stream) {
    if(!payload||!scale||!input||!output||input==output||!packed||!partial||!stream||
       rows<8||rows%8||columns<64||columns%64||tokens<1||tokens>3||sms<1||(tile!=0&&tile!=2&&tile!=4&&tile!=5))
        throw std::invalid_argument("Tiled NVFP4 CUTLASS owned lossless layout/extent");
    project<false,true>(payload,scale,input,output,rows,columns,tokens,packed,input_capacity,partial,partial_capacity,sms,tile,stream);
}

}
