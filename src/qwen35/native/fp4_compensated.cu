#include "fp4_compensated.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"
#include <cmath>
#include "fp4_tiny_fused.hpp"
#include "fp4_gu8_policy.hpp"
#include "fp4_compensated_codec.cuh"

namespace strata::qwen35::native {
namespace {
__device__ void mma(float (&c)[4],const unsigned (&a)[4],const unsigned (&b)[2],unsigned sa,unsigned sb){
    asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, %10, {0,0}, %11, {0,0};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(sa),"r"(sb));
}
__device__ void load_a(Nvfp4View w,int row,int first,int lane,unsigned (&a)[4],unsigned &sf){
    const int local=lane%4,group=lane/4;
    const size_t tile=size_t(row/16)*(w.columns/64)+first/64;
    const uint4 values=static_cast<const uint4*>(w.values)[tile*32+lane];
    a[0]=values.x;a[1]=values.y;a[2]=values.z;a[3]=values.w;
    sf=0;
    if(local<2){const int r=row+group+8*local;
        sf=*reinterpret_cast<const unsigned *>(static_cast<const unsigned char *>(w.scales)+
            nvfp4_detail::scale_index(r,first/16,w.columns));}
}
__device__ void load_b(CompensatedFp4Input x,int columns,int first,int lane,int tokens,unsigned (&b)[2],unsigned &sf){
    const int physical=lane/4,local=lane%4;
    b[0]=b[1]=sf=0;
    if(physical<2*tokens){
        const bool residual=physical>=tokens;const int token=residual?physical-tokens:physical;
        const auto plane=residual?x.low:x.high;
        const size_t at=size_t(token)*(columns/8)+first/8+local;
        const auto *words=static_cast<const unsigned *>(plane.values);b[0]=words[at];b[1]=words[at+4];
        if(!local)sf=*reinterpret_cast<const unsigned *>(static_cast<const unsigned char *>(plane.scales)+
            nvfp4_detail::scale_index(token,first/16,columns));
    }
}
struct ProjectionResult {float gate=0,up=0;};
template<bool Pair,int Warps> __device__ __forceinline__ ProjectionResult reduce_projection(
        const float (&partial)[Pair?2:1][Warps][4][32],Nvfp4View g,Nvfp4View u,int i,int lane,int tokens){
    ProjectionResult result;const int token=(lane%4)*2+(i&1);
    const int low_token=token+tokens,low_lane=(lane/4)*4+low_token/2;
    const int low_component=(i/2)*2+low_token%2;
    float one=partial[0][0][i][lane],low_one=partial[0][0][low_component][low_lane],two=0,low_two=0;
    if constexpr(Pair){two=partial[1][0][i][lane];low_two=partial[1][0][low_component][low_lane];}
    #pragma unroll
    for(int w=1;w<Warps;++w){
        one=__fadd_rn(one,partial[0][w][i][lane]);low_one=__fadd_rn(low_one,partial[0][w][low_component][low_lane]);
        if constexpr(Pair){two=__fadd_rn(two,partial[1][w][i][lane]);low_two=__fadd_rn(low_two,partial[1][w][low_component][low_lane]);}
    }
    const float gv=__fmul_rn(__fadd_rn(one,__fmul_rn(low_one,CompensatedLowRatio)),__fmul_rn(g.weight_scale,g.input_scale));result.gate=gv;
    if constexpr(Pair){const float uv=__fmul_rn(__fadd_rn(two,__fmul_rn(low_two,CompensatedLowRatio)),__fmul_rn(u.weight_scale,u.input_scale));
        result.up=uv;}
    return result;
}
template<bool Pair,int Warps,bool Fused=false,bool Debug=false> __global__ void project(Nvfp4View g,Nvfp4View u,CompensatedFp4Input x,
        float *gate,float *up,float *activation,int tokens,float down_scale=1.f){
    __shared__ float partial[Pair?2:1][Warps][4][32];
    const int lane=threadIdx.x%32,warp=threadIdx.x/32,row=int(blockIdx.x)*16;
    float first[4]{},second[4]{};
    unsigned a[4]{},b[2]{},sa=0,sb=0,ua[4]{},us=0;
    if(warp*64<g.columns){
        load_b(x,g.columns,warp*64,lane,tokens,b,sb);load_a(g,row,warp*64,lane,a,sa);
        if constexpr(Pair)load_a(u,row,warp*64,lane,ua,us);
    }
    for(int k=warp*64;k<g.columns;k+=Warps*64){
        unsigned next_a[4]{},next_b[2]{},next_sa=0,next_sb=0,next_ua[4]{},next_us=0;
        const int next=k+Warps*64;
        if(next<g.columns){
            load_b(x,g.columns,next,lane,tokens,next_b,next_sb);load_a(g,row,next,lane,next_a,next_sa);
            if constexpr(Pair)load_a(u,row,next,lane,next_ua,next_us);
        }
        asm volatile("" ::: "memory");
        mma(first,a,b,sa,sb);if constexpr(Pair)mma(second,ua,b,us,sb);
        #pragma unroll
        for(int i=0;i<4;++i){a[i]=next_a[i];if constexpr(Pair)ua[i]=next_ua[i];}
        b[0]=next_b[0];b[1]=next_b[1];sa=next_sa;sb=next_sb;us=next_us;
    }
    #pragma unroll
    for(int i=0;i<4;++i){partial[0][warp][i][lane]=first[i];
        if constexpr(Pair)partial[1][warp][i][lane]=second[i];}
    __syncthreads();
    if constexpr(Fused){
    if(!warp){
        const size_t span=size_t(8)*g.rows;
        CompensatedFp4Input out{{static_cast<unsigned char*>(x.high.values)+span,
            static_cast<unsigned char*>(x.high.scales)+span,16,g.rows},
            {static_cast<unsigned char*>(x.high.values)+2*span,
            static_cast<unsigned char*>(x.high.scales)+2*span,16,g.rows}};
        const int local=lane%4,feature=lane/4;
        const unsigned mask=0x11111111u<<local;
        #pragma unroll
        for(int parity=0;parity<2;++parity){
            const int token=2*local+parity;
            if(token<tokens){
                float value[2];
                #pragma unroll
                for(int half=0;half<2;++half){
                    const int i=2*half+parity,output_row=row+feature+8*half;
                    const auto sum=reduce_projection<true,Warps>(partial,g,u,i,lane,tokens);
                    const float gv=sum.gate,uv=sum.up;
                    value[half]=swiglu_value(gv,uv);
                    if constexpr(Debug){const size_t at=size_t(token)*g.rows+output_row;
                        gate[at]=gv;up[at]=uv;activation[at]=value[half];}
                }
                const int source=((2*feature)%8)*4+local;
                const float xl=__shfl_sync(mask,value[0],source),yl=__shfl_sync(mask,value[0],source+4);
                const float xh=__shfl_sync(mask,value[1],source),yh=__shfl_sync(mask,value[1],source+4);
                compensated_codec_detail::encode_pair<4,true>(feature<4?xl:xh,feature<4?yl:yh,out,token,row+2*feature,g.rows,down_scale,mask);
            }
        }
    }
    }else{
    if(!warp){
        #pragma unroll
        for(int i=0;i<4;++i){
            const int token=(lane%4)*2+(i&1),output_row=row+lane/4+(i/2)*8;
            if(token<tokens){
                const auto sum=reduce_projection<Pair,Warps>(partial,g,u,i,lane,tokens);
                const size_t at=size_t(token)*g.rows+output_row;gate[at]=sum.gate;
                if constexpr(Pair){up[at]=sum.up;activation[at]=swiglu_value(sum.gate,sum.up);}
            }
        }
    }
    }
}
template<bool Pair>void launch(Nvfp4View g,Nvfp4View u,CompensatedFp4Input x,float *a,float *b,float *act,
                               int tokens,int warps,cudaStream_t stream){
    if(!g.values||!g.scales||!x.high.values||!x.high.scales||!x.low.values||!x.low.scales||!a||g.rows<16||g.rows%16||g.columns%256||
        tokens<1||tokens>3||x.high.capacity_rows<tokens||x.high.capacity_columns<g.columns||x.low.capacity_rows<tokens||x.low.capacity_columns<g.columns||warps!=16)
        throw std::invalid_argument("compensated FP4 warp operand geometry");
    if constexpr(Pair){if(!u.values||!u.scales||!b||!act||g.rows!=u.rows||g.columns!=u.columns||g.input_scale!=u.input_scale)
        throw std::invalid_argument("compensated FP4 warp paired geometry/activation scale");}
    project<Pair,16><<<g.rows/16,512,0,stream>>>(g,u,x,a,b,act,tokens);
    check_cuda(cudaGetLastError(),"compensated FP4 warp projection");
}
}
namespace {
__global__ void encode(const float *input,CompensatedFp4Input out,int tokens,int columns,float global){
    const size_t pair=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(pair>=size_t(tokens)*columns/2)return;
    const size_t element=2*pair;const int row=int(element/columns),col=int(element%columns);
    const unsigned mask=0xffu<<((threadIdx.x&31)&~7);
    compensated_codec_detail::encode_pair<1,true>(input[element],input[element+1],out,row,col,columns,global,mask);
}
__global__ void decoder_probe(float *out){if(threadIdx.x<16)out[threadIdx.x]=compensated_codec_detail::decode_fp4(threadIdx.x);}
__global__ void compare_codec(CompensatedFp4Input a,CompensatedFp4Input b,int tokens,int columns,unsigned *errors){
    const size_t at=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(at>=size_t(tokens)*columns/2)return;
    for(int plane=0;plane<2;++plane){const auto x=plane?a.low:a.high,y=plane?b.low:b.high;
        if(static_cast<const unsigned char*>(x.values)[at]!=static_cast<const unsigned char*>(y.values)[at])atomicAdd(errors,1u);
        if(at%8==0){const int row=int(2*at/columns),col=int((2*at)%columns);
            const auto sf=nvfp4_detail::scale_index(row,col/16,columns);
            if(static_cast<const unsigned char*>(x.scales)[sf]!=static_cast<const unsigned char*>(y.scales)[sf])atomicAdd(errors,1u);}
    }
}
}
void fp4_compensated_quantize(const float *input,CompensatedFp4Input out,int tokens,int columns,float global,cudaStream_t stream){
    if(!input||!out.high.values||!out.high.scales||!out.low.values||!out.low.scales||tokens<1||tokens>3||columns<64||columns%64||
       out.high.capacity_rows<tokens||out.low.capacity_rows<tokens||out.high.capacity_columns<columns||out.low.capacity_columns<columns||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("compensated FP4 encode geometry/scales");
    const size_t pairs=size_t(tokens)*columns/2;encode<<<(pairs+255)/256,256,0,stream>>>(input,out,tokens,columns,global);
    check_cuda(cudaGetLastError(),"compensated highlow activation encode");
}
PackedFp4Resources fp4_compensated_resources(bool pair){
    cudaFuncAttributes a{};int blocks=0;
    if(pair){check_cuda(cudaFuncGetAttributes(&a,project<true,16>),"compensated pair attributes");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<true,16>,512,0),"compensated pair occupancy");}
    else{check_cuda(cudaFuncGetAttributes(&a,project<false,16>),"compensated linear attributes");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<false,16>,512,0),"compensated linear occupancy");}
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void fp4_compensated_linear(PackedFp4 w,CompensatedFp4Input x,float *out,int tokens,cudaStream_t stream){
    launch<false>(w.view,w.view,x,out,nullptr,nullptr,tokens,16,stream);
}
void fp4_compensated_pair(PackedFp4 g,PackedFp4 u,CompensatedFp4Input x,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    launch<true>(g.view,u.view,x,first,second,act,tokens,16,stream);
}
CompensatedFp4Input fp4_tiny_fused_output(CompensatedFp4Input x,int columns){
    const size_t span=size_t(8)*columns;
    if(columns<=0||columns%256||nvfp4_value_bytes(x.high.capacity_rows,x.high.capacity_columns)<3*span||
        nvfp4_scale_bytes(x.high.capacity_rows,x.high.capacity_columns)<3*span)
        throw std::invalid_argument("fused Tiny codec arena capacity");
    return {{static_cast<unsigned char*>(x.high.values)+span,static_cast<unsigned char*>(x.high.scales)+span,16,columns},
        {static_cast<unsigned char*>(x.high.values)+2*span,static_cast<unsigned char*>(x.high.scales)+2*span,16,columns}};
}
void fp4_tiny_fused_decoder(float *out,cudaStream_t stream){
    decoder_probe<<<1,32,0,stream>>>(out);check_cuda(cudaGetLastError(),"Tiny FP4 decoder probe");
}
PackedFp4Resources fp4_tiny_fused_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,project<true,16,true,false>),"fused Tiny attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<true,16,true,false>,512,0),"fused Tiny occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
namespace {
void validate_fused_pair(PackedFp4 gate,PackedFp4 up,CompensatedFp4Input x,float global,int tokens){
    const auto g=gate.view,u=up.view;fp4_tiny_fused_output(x,g.rows);
    const size_t input_scale_extent=nvfp4_scale_bytes(16,g.columns),span=size_t(8)*g.rows;
    if(!g.values||!u.values||!g.scales||!u.scales||g.rows%16||g.columns%256||g.rows!=u.rows||g.columns!=u.columns||
        g.input_scale!=u.input_scale||tokens<1||tokens>3||x.high.capacity_columns<g.columns||x.low.capacity_columns<g.columns||
        x.low.capacity_rows<tokens||size_t(tokens)*g.columns/2>span||input_scale_extent>span||!(global>0)||!std::isfinite(global))
        throw std::invalid_argument("fused Tiny input/weight/output lease");
}
}
void fp4_tiny_fused_pair(PackedFp4 gate,PackedFp4 up,CompensatedFp4Input x,float global,int tokens,cudaStream_t stream,
        float *first,float *second,float *activation){
    validate_fused_pair(gate,up,x,global,tokens);const auto g=gate.view,u=up.view;
    if(first&&second&&activation)project<true,16,true,true><<<g.rows/16,512,0,stream>>>(g,u,x,first,second,activation,tokens,global);
    else if(!first&&!second&&!activation)project<true,16,true,false><<<g.rows/16,512,0,stream>>>(g,u,x,nullptr,nullptr,nullptr,tokens,global);
    else throw std::invalid_argument("fused Tiny diagnostic output lease");
    check_cuda(cudaGetLastError(),"fused Tiny pair/codec");
}
PackedFp4Resources fp4_gu8_resources(){
    cudaFuncAttributes a{};int blocks=0;
    check_cuda(cudaFuncGetAttributes(&a,project<true,8,true>),"GU8 attributes");
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<true,8,true>,256,0),"GU8 occupancy");
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void fp4_tiny_gu8_pair(PackedFp4 gate,PackedFp4 up,CompensatedFp4Input x,float global,int tokens,cudaStream_t stream){
    const auto g=gate.view,u=up.view;
    if(tokens!=3||g.rows!=17408||g.columns!=5120)throw std::invalid_argument("GU8 measured T3 geometry");
    validate_fused_pair(gate,up,x,global,tokens);
    project<true,8,true><<<g.rows/16,256,0,stream>>>(g,u,x,nullptr,nullptr,nullptr,tokens,global);
    check_cuda(cudaGetLastError(),"GU8 T3 pair/codec");
}
void fp4_tiny_fused_compare(CompensatedFp4Input a,CompensatedFp4Input b,int tokens,int columns,unsigned *errors,cudaStream_t stream){
    const size_t n=size_t(tokens)*columns/2;
    compare_codec<<<(n+255)/256,256,0,stream>>>(a,b,tokens,columns,errors);
    check_cuda(cudaGetLastError(),"fused Tiny active code/SF compare");
}
}
