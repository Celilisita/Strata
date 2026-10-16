#include "strata/qwen35/native/decode_graph_patch.hpp"
#include "strata/qwen35/native/token_ops.hpp"
#include "strata/qwen35/native/ffn.hpp"
#include "codec.cuh"
#include <math_constants.h>
#include <climits>

namespace strata::qwen35::native {
namespace {
__global__ void embedding(Matrix m,const int *ids,float *out,size_t count){
    const size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=count)return;
    const int token=ids[i/m.columns],column=i%m.columns;
    if(token<0 || token>=m.rows){out[i]=CUDART_NAN_F;return;}
    const size_t block=size_t(token)*(m.columns/256)+column/256;
    out[i]=block_element(static_cast<const unsigned char *>(m.data)+block*(m.quant==Quant::Q4K?144:210),m.quant,column%256,m.layout);
}
__global__ void positions(int *out,int first,int tokens){
    const int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<tokens)out[i]=first+i;
}
__global__ void init_cursor(DecodeCursor *cursor,int first){if(!threadIdx.x)*cursor={first,1,1,0};}
__global__ void cursor_position(int *out,DecodeCursor *cursor,int heads,DecodeRounding rounding){
    if(!threadIdx.x){
        if(cursor->position<0||cursor->position>=262144){cursor->invalid=1;cursor->position=0;}
        *out=cursor->position;cursor->partitions=decode_partitions_unchecked(cursor->position+1,heads,rounding);
    }
}
__global__ void cursor_record(const int *picked,const unsigned *invalid,DecodeCapture control){
    if(!threadIdx.x){
        const int index=control.cursor->index;
        if(index<1||index>=control.capacity)control.cursor->invalid=1;
        else{control.ids[index]=*picked;control.flags[index]=*invalid;}
        ++control.cursor->position;++control.cursor->index;
    }
}
__device__ GreedyPartial better(GreedyPartial a,GreedyPartial b){
    const unsigned invalid=a.invalid|b.invalid;
    if(b.value>a.value || (b.value==a.value && b.token<a.token))a=b;
    a.invalid=invalid;return a;
}
__device__ GreedyPartial warp_pick(GreedyPartial a){
    for(int d=16;d;d/=2){
        GreedyPartial b{__shfl_xor_sync(0xffffffff,a.value,d),__shfl_xor_sync(0xffffffff,a.token,d),
                        __shfl_xor_sync(0xffffffff,a.invalid,d)};a=better(a,b);
    }return a;
}
template<bool Batch,bool Filtered> __global__ void pick_parts(const float *x,int n,GreedyPartial *out,TokenMask mask){
    if constexpr(Batch){x+=size_t(blockIdx.y)*n;out+=size_t(blockIdx.y)*((n+255)/256);}
    __shared__ GreedyPartial part[8];const int i=blockIdx.x*256+threadIdx.x;
    GreedyPartial a{-CUDART_INF_F,INT_MAX,0};
    if(i<n){a={x[i],i,unsigned(!isfinite(x[i]))};if(a.invalid)a.value=-CUDART_INF_F;}
    if constexpr(Filtered){
        if(i>=mask.first&&i<=mask.last){
            for(int t=0;t<mask.count;++t)if(i==mask.ids[t])a.value=-CUDART_INF_F;
        }
    }
    a=warp_pick(a);if(threadIdx.x%32==0)part[threadIdx.x/32]=a;__syncthreads();
    if(threadIdx.x<32){a=threadIdx.x<8?part[threadIdx.x]:GreedyPartial{-CUDART_INF_F,INT_MAX,0};
        a=warp_pick(a);if(!threadIdx.x)out[blockIdx.x]=a;}
}
template<bool Batch> __global__ void pick_final(const GreedyPartial *parts,int n,int *token,unsigned *invalid){
    if constexpr(Batch){parts+=size_t(blockIdx.x)*n;token+=blockIdx.x;invalid+=blockIdx.x;}
    __shared__ GreedyPartial shared[8];GreedyPartial a{-CUDART_INF_F,INT_MAX,0};
    for(int i=threadIdx.x;i<n;i+=256)a=better(a,parts[i]);
    a=warp_pick(a);if(threadIdx.x%32==0)shared[threadIdx.x/32]=a;__syncthreads();
    if(threadIdx.x<32){a=threadIdx.x<8?shared[threadIdx.x]:GreedyPartial{-CUDART_INF_F,INT_MAX,0};
        a=warp_pick(a);if(!threadIdx.x){*token=a.token;*invalid=a.invalid;}}
}
}
void gather_embedding(Matrix m,const int *ids,float *out,int tokens,cudaStream_t s){
    matrix_bytes(m);if(!ids || !out || tokens<1 || tokens>8192)throw std::invalid_argument("invalid own embedding invocation");
    const size_t n=size_t(tokens)*m.columns;embedding<<<unsigned((n+255)/256),256,0,s>>>(m,ids,out,n);
    check_cuda(cudaGetLastError(),"own device embedding gather");
}
void set_positions(int *out,int first,int tokens,cudaStream_t s){
    if(!out || first<0 || tokens<1 || first>262144-tokens)throw std::invalid_argument("invalid own text positions");
    positions<<<(tokens+255)/256,256,0,s>>>(out,first,tokens);check_cuda(cudaGetLastError(),"own device text positions");
}
size_t greedy_workspace(int n){if(n<1 || n>1048576)throw std::invalid_argument("invalid own vocabulary");return size_t((n+255)/256)*sizeof(GreedyPartial);}
void greedy_pick(const float *x,int n,void *work,size_t bytes,int *token,unsigned *invalid,cudaStream_t s,TokenMask mask){
    if(!x || !work || !token || !invalid || bytes<greedy_workspace(n))throw std::invalid_argument("invalid own greedy buffers");
    if(mask.count<0||mask.count>16)throw std::invalid_argument("invalid greedy mask");
    const int parts=(n+255)/256;
    if(mask.count)pick_parts<false,true><<<parts,256,0,s>>>(x,n,static_cast<GreedyPartial *>(work),mask);
    else pick_parts<false,false><<<parts,256,0,s>>>(x,n,static_cast<GreedyPartial *>(work),mask);
    check_cuda(cudaGetLastError(),"own greedy partial reduction");
    pick_final<false><<<1,256,0,s>>>(static_cast<const GreedyPartial *>(work),parts,token,invalid);
    check_cuda(cudaGetLastError(),"own greedy finite output selection");
}
void greedy_pick_batch(const float *x,int n,int rows,void *work,size_t bytes,int *ids,unsigned *invalid,cudaStream_t stream,TokenMask mask){
    if(!x||!work||!ids||!invalid||rows<2||rows>3||bytes<size_t(rows)*greedy_workspace(n))
        throw std::invalid_argument("invalid owned batch greedy workspace");
    const int parts=(n+255)/256;
    if(mask.count<0||mask.count>16)throw std::invalid_argument("invalid batch greedy mask");
    if(mask.count)pick_parts<true,true><<<dim3(parts,rows),256,0,stream>>>(x,n,static_cast<GreedyPartial *>(work),mask);
    else pick_parts<true,false><<<dim3(parts,rows),256,0,stream>>>(x,n,static_cast<GreedyPartial *>(work),mask);
    check_cuda(cudaGetLastError(),"own batch greedy parts");
    pick_final<true><<<rows,256,0,stream>>>(static_cast<const GreedyPartial *>(work),parts,ids,invalid);
    check_cuda(cudaGetLastError(),"own batch greedy final");
}
DecodeNode decode_token_node(void *function){
    if(function==reinterpret_cast<void *>(positions)||function==reinterpret_cast<void *>(cursor_position))return DecodeNode::Position;
    return function==reinterpret_cast<void *>(cursor_record)?DecodeNode::Record:DecodeNode::None;
}
void update_decode_token(cudaGraphExec_t exec,cudaGraphNode_t node,const cudaKernelNodeParams &original,int position){
    auto args=borrowed_graph_arguments<3>(original);int tokens=*static_cast<const int *>(args[2]);
    if(tokens!=1||position<0||position>=262144)throw std::invalid_argument("invalid captured single text position");
    args[1]=&position;auto p=original;p.kernelParams=args.data();
    check_cuda(cudaGraphExecKernelNodeSetParams(exec,node,&p),"update own graph text position");
}
void initialize_decode_cursor(DecodeCursor *cursor,int first,cudaStream_t stream){
    if(!cursor||first<1||first>262142)throw std::invalid_argument("invalid decode cursor initialization");
    init_cursor<<<1,32,0,stream>>>(cursor,first);check_cuda(cudaGetLastError(),"own decode cursor initialization");
}
void set_decode_position(int *out,DecodeCursor *cursor,int heads,DecodeRounding rounding,cudaStream_t stream){
    decode_partition_count(1,heads,rounding);
    if(!out||!cursor)throw std::invalid_argument("invalid graph position cursor");
    cursor_position<<<1,32,0,stream>>>(out,cursor,heads,rounding);check_cuda(cudaGetLastError(),"own graph position and partition producer");
}
void record_decode_token(const int *picked,const unsigned *invalid,DecodeCapture control,cudaStream_t stream){
    if(!picked||!invalid||!control.cursor||!control.ids||!control.flags||control.capacity<3)
        throw std::invalid_argument("invalid graph own output records");
    cursor_record<<<1,32,0,stream>>>(picked,invalid,control);check_cuda(cudaGetLastError(),"own graph token record and cursor advance");
}

}
