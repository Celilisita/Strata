#include "fp4_packed.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "nvfp4_layout.cuh"
#include "math.cuh"

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
__device__ void load_b(Nvfp4Input x,int columns,int first,int lane,int tokens,unsigned (&b)[2],unsigned &sf){
    const int token=lane/4,local=lane%4;
    b[0]=b[1]=sf=0;
    if(token<tokens){
        const size_t at=size_t(token)*(columns/8)+first/8+local;
        const auto *words=static_cast<const unsigned *>(x.values);b[0]=words[at];b[1]=words[at+4];
        if(!local)sf=*reinterpret_cast<const unsigned *>(static_cast<const unsigned char *>(x.scales)+
            nvfp4_detail::scale_index(token,first/16,columns));
    }
}
template<bool Pair,int Warps> __global__ void project(Nvfp4View g,Nvfp4View u,Nvfp4Input x,
        float *gate,float *up,float *activation,int tokens){
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
    if(!warp){
        #pragma unroll
        for(int i=0;i<4;++i){
            float one=partial[0][0][i][lane],two=0;if constexpr(Pair)two=partial[1][0][i][lane];
            #pragma unroll
            for(int w=1;w<Warps;++w){one=__fadd_rn(one,partial[0][w][i][lane]);
                if constexpr(Pair)two=__fadd_rn(two,partial[1][w][i][lane]);}
            const int token=(lane%4)*2+(i&1),output_row=row+lane/4+(i/2)*8;
            if(token<tokens){
                const size_t at=size_t(token)*g.rows+output_row;
                const float gv=__fmul_rn(one,__fmul_rn(g.weight_scale,g.input_scale));gate[at]=gv;
                if constexpr(Pair){const float uv=__fmul_rn(two,__fmul_rn(u.weight_scale,u.input_scale));
                    up[at]=uv;activation[at]=swiglu_value(gv,uv);}
            }
        }
    }
}
template<bool Pair>void launch(Nvfp4View g,Nvfp4View u,Nvfp4Input x,float *a,float *b,float *act,
                               int tokens,int warps,cudaStream_t stream){
    if(!g.values||!g.scales||!x.values||!x.scales||!a||g.rows<16||g.rows%16||g.columns%256||
        tokens<1||tokens>3||x.capacity_rows<tokens||x.capacity_columns<g.columns||warps!=16)
        throw std::invalid_argument("native FP4 warp operand geometry");
    if constexpr(Pair){if(!u.values||!u.scales||!b||!act||g.rows!=u.rows||g.columns!=u.columns||g.input_scale!=u.input_scale)
        throw std::invalid_argument("native FP4 warp paired geometry/activation scale");}
    project<Pair,16><<<g.rows/16,512,0,stream>>>(g,u,x,a,b,act,tokens);
    check_cuda(cudaGetLastError(),"native FP4 warp projection");
}
}
namespace {
template<bool Unpack>__global__ void transpose_values(const unsigned *source,unsigned *output,int rows,int columns){
    const size_t index=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const size_t count=size_t(rows/16)*(columns/64)*32;if(index>=count)return;
    const int lane=int(index%32),row=int((index/32)/(columns/64))*16,first=int((index/32)%(columns/64))*64;
    const size_t at=size_t(row+lane/4)*(columns/8)+first/8+lane%4,step=columns;
    if constexpr(Unpack){
        const uint4 v=reinterpret_cast<const uint4*>(source)[index];
        output[at]=v.x;output[at+step]=v.y;output[at+4]=v.z;output[at+step+4]=v.w;
    }else reinterpret_cast<uint4*>(output)[index]=uint4{source[at],source[at+step],source[at+4],source[at+step+4]};
}
void check_planes(Nvfp4View w,const void *out){
    if(!w.values||!w.scales||!out||w.rows<16||w.rows%16||w.columns<64||w.columns%64||
       reinterpret_cast<size_t>(out)%16||reinterpret_cast<size_t>(w.values)%16)throw std::invalid_argument("packed FP4 plane geometry/alignment");
    const size_t bytes=size_t(w.rows)*w.columns/2;
    const auto a=reinterpret_cast<uintptr_t>(w.values),b=reinterpret_cast<uintptr_t>(out);
    if((a<=b?b-a:a-b)<bytes)throw std::invalid_argument("packed FP4 immutable plane overlap");
}
}
PackedFp4 fp4_pack_values(Nvfp4View w,void *out,cudaStream_t stream){
    check_planes(w,out);const size_t count=size_t(w.rows/16)*(w.columns/64)*32;
    transpose_values<false><<<(count+127)/128,128,0,stream>>>(static_cast<const unsigned*>(w.values),static_cast<unsigned*>(out),w.rows,w.columns);
    check_cuda(cudaGetLastError(),"FP4 warp-fragment pack");w.values=out;return {w};
}
void fp4_unpack_values(PackedFp4 p,void *out,cudaStream_t stream){
    const auto w=p.view;check_planes(w,out);const size_t count=size_t(w.rows/16)*(w.columns/64)*32;
    transpose_values<true><<<(count+127)/128,128,0,stream>>>(static_cast<const unsigned*>(w.values),static_cast<unsigned*>(out),w.rows,w.columns);
    check_cuda(cudaGetLastError(),"FP4 canonical PP materialization");
}
PackedFp4Resources fp4_packed_resources(bool pair){
    cudaFuncAttributes a{};int blocks=0;
    if(pair){check_cuda(cudaFuncGetAttributes(&a,project<true,16>),"packed pair attributes");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<true,16>,512,0),"packed pair occupancy");}
    else{check_cuda(cudaFuncGetAttributes(&a,project<false,16>),"packed linear attributes");
        check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,project<false,16>,512,0),"packed linear occupancy");}
    return {a.numRegs,blocks,a.sharedSizeBytes,a.localSizeBytes};
}
void fp4_packed_linear(PackedFp4 w,Nvfp4Input x,float *out,int tokens,cudaStream_t stream){
    launch<false>(w.view,w.view,x,out,nullptr,nullptr,tokens,16,stream);
}
void fp4_packed_pair(PackedFp4 g,PackedFp4 u,Nvfp4Input x,float *first,float *second,float *act,int tokens,cudaStream_t stream){
    launch<true>(g.view,u.view,x,first,second,act,tokens,16,stream);
}
}
